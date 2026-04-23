"""
Central configuration: constants and parameters only (no functions).
DICOM heuristics and ``normalize_metadata`` live in ``helper``.
Run scripts from the repository root, or rely on PROJECT_ROOT for absolute paths.
"""
from __future__ import annotations

import os

# =============================================================================
# 1. Layout (repository root)
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 2. Data paths, logs, and artifact filenames
# =============================================================================
PATH_RAW = os.path.join(PROJECT_ROOT, "datasets", "raw_data")
PATH_ANNOTATION_BOXES = os.path.join(PROJECT_ROOT, "datasets", "Annotation_Boxes.xlsx")
PATH_MICRO_CUBES = os.path.join(PROJECT_ROOT, "datasets", "micro_cubes")
PATH_CLINICAL = os.path.join(PROJECT_ROOT, "datasets", "Clinical_and_Other_Features.xlsx")
PATH_MODEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "model")
MODEL_WEIGHTS_FILENAME = "biolattice_3dresnet_binary.pth"
PATH_MODEL_WEIGHTS = os.path.join(PROJECT_ROOT, "datasets", "model", MODEL_WEIGHTS_FILENAME)
NORMALIZATION_PROFILE_STATS_FILENAME = "normalization_profile_stats.json"
PATH_NORMALIZATION_PROFILE_STATS = os.path.join(
    PATH_MODEL_DIR, NORMALIZATION_PROFILE_STATS_FILENAME
)
PATH_LOGS = os.path.join(PROJECT_ROOT, "dashboard", "extraction_log.txt")
PATH_EXTRACTION_AUDIT_DIR = os.path.join(PROJECT_ROOT, "dashboard", "extraction_audits")
PATH_TRAINING_LOG_DIR = os.path.join(PROJECT_ROOT, "dashboard", "training_logs")

LATTICE_FILE_SUFFIX = "_lattice.pt"

# =============================================================================
# 3. DICOM & series pairing (Duke-oriented heuristics)
# =============================================================================
DICOM_EXTENSION = ".dcm"
# (Note: SEMANTIC_* constants were migrated to core/classifier_rules.json in v2.6)

# =============================================================================
# 4. Clinical spreadsheet & supervised label (Excel)
# =============================================================================
# Duke TCIA companion spreadsheets expect pandas read_excel(..., header=1).
CLINICAL_EXCEL_HEADER_ROW = 1
COL_PATIENT_ID = "Patient ID"
COL_MOL_SUBTYPE = "Mol Subtype"
MOL_SUBTYPE_POSITIVE_THRESHOLD = 0.0  # label 1.0 if value > this threshold

# =============================================================================
# 5. Micro-cube construction (ROI → lattice tensor)
# =============================================================================
MICRO_CUBE_SIZE = 64
ROI_PADDING_FRACTION = 0.20
PRE_POST_INTERPOLATE_MODE = "trilinear"
# DEPRECATED v2.2: The popup visualizer was removed from main.py. This flag has no effect.
# SHOW_VISUALIZER_AFTER_SAVE = False

# =============================================================================
# 6. Registration, hybrid lattice (v2), and extraction audit
# =============================================================================
REGISTRATION_MIN_CORRELATION = 0.70  # Threshold balanced for Breast MRI FFT (keeps usable cases like 0.73)
# DEPRECATED v2.5: Replaced by 4-channel split (C1=Avg, C4=Max).
# HYBRID_STRUCTURAL_RATIO = 0.5 
# Local neighborhood for heterogeneity variance (Channel 2). Must be odd.
C2_LOCAL_VAR_KERNEL = 3
C4_LOCAL_PEAK_KERNEL = 3
# --- Bio-Lattice v2.5 Extraction Policy ---
EXTRACTION_KINETICS_CLAMP = 5.0    # Protects from spikes/noise (Log scale)
ROI_MIN_FRAC_FOR_REFLECT = 0.4     # Min ratio of ROI vs Cube to allow 'reflect' padding
EXTRACTION_MAX_RELATIVE_SHIFT = 0.05 # Threshold to switch from 'nearest' to 'constant' shift
AUDIT_WRITE_JSONL = True

# =============================================================================
# 7. Training loop (data quality gates, optimization, imbalance)
# =============================================================================
TRAIN_FILTER_BY_AUDIT = True           # Only train on audited-quality patients (OK + WARNING)
TRAIN_ALLOWED_AUDIT_STATUSES = ("OK", "WARNING")
# Per-run training summary (.txt): host, timestamps, sample counts, optional early-stop stats.
TRAIN_LOG_WRITE_TXT = True
# Optional external experiment tracking. Disabled by default so local/CPU runs
# and Kaggle notebooks work without installing wandb.
TRAIN_USE_WANDB = os.getenv("TRAIN_USE_WANDB", "0").strip().lower() in {"1", "true", "yes", "on"}
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "microcube")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "").strip()
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "").strip()

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 100
TRAIN_VAL_SPLIT_FRACTION = 0.8
RANDOM_SEED = 42
# (High Resilience Training) - Give model more time with new v2.5 Micro-cubes
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 0.0005
EARLY_STOPPING_START_EPOCH = 20
FOCAL_LOSS_ALPHA = 0.75
FOCAL_LOSS_GAMMA = 2.0
# Class imbalance (training only; same micro-cubes and labels). Prefer one strategy at a time.
TRAIN_STRATIFIED_SPLIT = True  # patient-level train/val preserving ~label ratio per side
TRAIN_USE_WEIGHTED_SAMPLER = False  # oversample minority class each epoch (DataLoader sampler)
TRAIN_USE_CLASS_POS_WEIGHT = True  # BCE pos_weight = n_neg / n_pos inside FocalLoss
TRAIN_NUM_WORKERS = 4              # Parallel data loading workers (Efficiency / Green AI)
ADAMW_WEIGHT_DECAY = 1e-2
ONECYCLE_MAX_LR = 1.5e-4
# Model-input normalization mode.
TRAIN_NORMALIZATION_MODE = "profile_conditioned"
# Baseline per-channel scaling switch.
TRAIN_NORMALIZE_CUBE_PER_CHANNEL = True
NORMALIZE_EPS = 1e-8
# High-percentile clipping used by conservative normalization paths.
NORMALIZATION_CLIP_PERCENTILE = 0.99
# Minimum robust scale denominator.
ROBUST_SCALE_MIN_IQR = 0.05

# =============================================================================
# 8. Model architecture (BioLattice3DResNet)
# =============================================================================
INPUT_CHANNELS = 4  # v2.5 Split Engine: C1=Avg, C2=Het, C3=Kin, C4=Max
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

# =============================================================================
# 9. Inference & reporting
# =============================================================================
# Default operating point for current model stage.
# Recalibrate after each training cycle using validation threshold sweep (Youden report).
MALIGNANCY_PROB_THRESHOLD = 0.50
# 'auto' resolves to CUDA > MPS > CPU at runtime via helper.get_device().
# Force a specific device by setting this to 'cpu', 'cuda', or 'mps'.
INFERENCE_DEVICE = "auto"

# =============================================================================
# 10. Metadata branch (spacing / thickness → MLP)
# =============================================================================
META_FEATURE_DIM = 3
META_MLP_OUT = 16
META_NORM_SHIFT_XY = -1.0  # Added to in-plane PixelSpacing components (mm).
META_NORM_SHIFT_Z = -2.0  # Added to SliceThickness (mm).
