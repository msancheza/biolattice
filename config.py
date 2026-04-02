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
PATH_MICRO_CUBOS = repo_path("datasets", "micro_cubos")
PATH_CLINICAL = repo_path("datasets", "Clinical_and_Other_Features.xlsx")
PATH_MODEL_DIR = repo_path("datasets", "modelo")
MODEL_WEIGHTS_FILENAME = "biolattice_3dresnet_binary.pth"
PATH_MODEL_WEIGHTS = repo_path("datasets", "modelo", MODEL_WEIGHTS_FILENAME)

LATTICE_FILE_SUFFIX = "_lattice.pt"

# --- DICOM / series pairing (Duke-oriented) ---
DICOM_EXTENSION = ".dcm"
POST_SERIES_SUBSTRINGS = ("1st", "post_1", "ph1", "phase 1", "fase 1")


def series_is_pre_contrast(desc_lower: str) -> bool:
    """PRE: substring 'pre', or 'dyn' without phase substrings."""
    return ("pre" in desc_lower) or (
        "dyn" in desc_lower and "ph" not in desc_lower and "phase" not in desc_lower
    )


def series_is_post_contrast(desc_lower: str) -> bool:
    return any(x in desc_lower for x in POST_SERIES_SUBSTRINGS)


# --- Micro-cube construction ---
MICRO_CUBE_SIZE = 32
ROI_PADDING_FRACTION = 0.20
PRE_POST_INTERPOLATE_MODE = "trilinear"
SHOW_VISUALIZER_AFTER_SAVE = False

# --- Clinical table (Excel) ---
# Duke TCIA companion spreadsheets expect pandas read_excel(..., header=1).
CLINICAL_EXCEL_HEADER_ROW = 1
COL_PATIENT_ID = "Patient ID"
COL_MOL_SUBTYPE = "Mol Subtype"
MOL_SUBTYPE_POSITIVE_THRESHOLD = 0.0  # label 1.0 if value > this threshold

# --- Training ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
TRAIN_VAL_SPLIT_FRACTION = 0.8
RANDOM_SEED = 42
BCE_POS_WEIGHT = 3.0
ADAMW_WEIGHT_DECAY = 1e-2
ONECYCLE_MAX_LR = 1e-3
NORMALIZE_EPS = 1e-8

# --- Model (BioLattice3DResNet) ---
INPUT_CHANNELS = 3
STEM_CHANNELS = 32
RES_BLOCK1_IN = 32
RES_BLOCK1_OUT = 64
RES_BLOCK2_IN = 64
RES_BLOCK2_OUT = 128
CLASSIFIER_AVG_POOL_KERNEL = 4
CLASSIFIER_AVG_POOL_STRIDE = 4
# Must match flattened spatial map after pools; tied to MICRO_CUBE_SIZE=32 and blocks above.
CLASSIFIER_LINEAR_IN = 128 * 8
CLASSIFIER_HIDDEN = 256
CLASSIFIER_DROPOUT = 0.4
POOL_KERNEL = 2
POOL_STRIDE = 2

# --- Inference / metrics ---
MALIGNANCY_PROB_THRESHOLD = 0.75  # sigmoid output; higher → fewer FP, more FN
INFERENCE_DEVICE = "cpu"
