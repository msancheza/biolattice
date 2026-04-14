"""
Central configuration for paths, extraction, training, and inference.
Run scripts from the repository root, or rely on PROJECT_ROOT for absolute paths.
"""
from __future__ import annotations

import os

# --- Layout ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def repo_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


# --- Data paths ---
PATH_RAW = repo_path("datasets", "raw_data")
PATH_ANNOTATION_BOXES = repo_path("datasets", "Annotation_Boxes.xlsx")
PATH_MICRO_CUBES = repo_path("datasets", "micro_cubes")
PATH_CLINICAL = repo_path("datasets", "Clinical_and_Other_Features.xlsx")
PATH_MODEL_DIR = repo_path("datasets", "model")
MODEL_WEIGHTS_FILENAME = "biolattice_3dresnet_binary.pth"
PATH_MODEL_WEIGHTS = repo_path("datasets", "model", MODEL_WEIGHTS_FILENAME)
PATH_LOGS = repo_path("dashboard", "extraction_log.txt")
PATH_EXTRACTION_AUDIT_DIR = repo_path("dashboard", "extraction_audits")

LATTICE_FILE_SUFFIX = "_lattice.pt"

# --- DICOM / series pairing (Duke-oriented) ---
DICOM_EXTENSION = ".dcm"
POST_SERIES_SUBSTRINGS = ("1st", "post_1", "ph1", "phase 1", "fase 1")


def series_is_pre_contrast(desc_lower: str) -> bool:
    """PRE: substring 'pre', or 'dyn'/'vibrant' without numbered phases."""
    is_pre = ("pre" in desc_lower)
    is_dynamic = ("dyn" in desc_lower) or ("vibrant" in desc_lower)
    
    # Check for numbered phases (ph1, ph 2, phase 1, etc.)
    import re
    has_numbered_phase = bool(re.search(r"(ph|phase)\s?\d", desc_lower))
    
    return is_pre or (is_dynamic and not has_numbered_phase)


def series_is_post_contrast(desc_lower: str) -> bool:
    # Explicit post-contrast keywords
    has_post_tag = any(x in desc_lower for x in POST_SERIES_SUBSTRINGS)
    
    # Also catch numbered phases (ph2, phase 2, dyn 1, etc.)
    import re
    has_numbered_phase = bool(re.search(r"(ph|phase|dyn|dynamic|fase)\s?[1-9]", desc_lower))
    
    return has_post_tag or has_numbered_phase


# --- Micro-cube construction ---
MICRO_CUBE_SIZE = 64
ROI_PADDING_FRACTION = 0.20
PRE_POST_INTERPOLATE_MODE = "trilinear"
SHOW_VISUALIZER_AFTER_SAVE = False

# --- v2 specific: Registration & Hybrid Logic ---
REGISTRATION_MIN_CORRELATION = 0.80  # Reduced threshold to mitigate safe warnings
HYBRID_STRUCTURAL_RATIO = 0.5        # Mix of MaxPool and AvgPool for Channel 1
VAR_DENOISING_SIGMA = 0.5           # Gaussian kernel sigma for Channel 2
# Local neighborhood for heterogeneity variance (Channel 2).
# Must be odd to preserve centered neighborhood.
C2_LOCAL_VAR_KERNEL = 3
AUDIT_WRITE_JSONL = True
TRAIN_FILTER_BY_AUDIT = True
TRAIN_ALLOWED_AUDIT_STATUSES = ("OK", "WARNING")

# --- Clinical table (Excel) ---
# Duke TCIA companion spreadsheets expect pandas read_excel(..., header=1).
CLINICAL_EXCEL_HEADER_ROW = 1
COL_PATIENT_ID = "Patient ID"
COL_MOL_SUBTYPE = "Mol Subtype"
MOL_SUBTYPE_POSITIVE_THRESHOLD = 0.0  # label 1.0 if value > this threshold

# --- Training ---
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
EPOCHS = 100
TRAIN_VAL_SPLIT_FRACTION = 0.8
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-3
EARLY_STOPPING_START_EPOCH = 10
FOCAL_LOSS_ALPHA = 0.75
FOCAL_LOSS_GAMMA = 2.0
ADAMW_WEIGHT_DECAY = 1e-2
ONECYCLE_MAX_LR = 5e-4
NORMALIZE_EPS = 1e-8

# --- Model (BioLattice3DResNet) ---
INPUT_CHANNELS = 3
STEM_CHANNELS = 32
RES_BLOCK1_IN = 32
RES_BLOCK1_OUT = 64
RES_BLOCK2_IN = 64
RES_BLOCK2_OUT = 128
CLASSIFIER_AVG_POOL_KERNEL = 8
CLASSIFIER_AVG_POOL_STRIDE = 8
# flattened spatial map after pools; tied to MICRO_CUBE_SIZE=64 and blocks above.
CLASSIFIER_LINEAR_IN = 128 * 8
CLASSIFIER_HIDDEN = 256
CLASSIFIER_DROPOUT = 0.5
POOL_KERNEL = 2
POOL_STRIDE = 2

# --- Inference / metrics ---
# Default operating point for current model stage.
# Recalibrate after each training cycle using validation threshold sweep (Youden report).
MALIGNANCY_PROB_THRESHOLD = 0.50
INFERENCE_DEVICE = "cpu"

# --- Multi-Modal Engineering ---
META_FEATURE_DIM = 3
META_MLP_OUT = 16
META_NORM_SHIFT_XY = -1.0  # Added to in-plane PixelSpacing components (mm).
META_NORM_SHIFT_Z = -2.0  # Added to SliceThickness (mm).

def normalize_metadata(spacing, thickness):
    """
    Build the 3-vector fed to the metadata MLP from DICOM spacing and thickness.

    Uses ``META_NORM_SHIFT_XY`` and ``META_NORM_SHIFT_Z`` so train and inference stay
    aligned; tune those constants in config instead of editing ``train`` / ``predict``.
    """
    import torch
    tensor = torch.tensor([spacing[0] + META_NORM_SHIFT_XY, 
                           spacing[1] + META_NORM_SHIFT_XY, 
                           thickness + META_NORM_SHIFT_Z], dtype=torch.float32)
    assert len(tensor) == META_FEATURE_DIM, f"CRITICAL: Metadata array length {len(tensor)} != Config DIM {META_FEATURE_DIM}"
    return tensor
