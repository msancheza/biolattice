"""
Project-wide helpers: DICOM series heuristics and metadata tensors for train/inference.
Tunable values live in ``config`` only.
"""
from __future__ import annotations

import re
import os
import glob
import json
import random

import torch
from torch.utils.data import Subset, Dataset

import config

def normalize_metadata(spacing, thickness):
    """
    Build the 3-vector fed to the metadata MLP from DICOM spacing and thickness.
    Uses Z-score scaling based on Duke dataset approximate populations.
    """
    tensor = torch.tensor(
        [
            (spacing[0] - 0.75) / 0.2, # X-pixel spacing scaling
            (spacing[1] - 0.75) / 0.2, # Y-pixel spacing scaling
            (thickness - 3.0) / 1.0,   # Z-slice thickness scaling
        ],
        dtype=torch.float32,
    )
    assert len(tensor) == config.META_FEATURE_DIM, (
        f"CRITICAL: Metadata array length {len(tensor)} != Config DIM {config.META_FEATURE_DIM}"
    )
    return tensor


def normalize_cube_per_channel(cube: torch.Tensor) -> torch.Tensor:
    """
    Applies the same per-channel Z-score normalization used across training and inference.
    Returns a clone so callers do not mutate the loaded artifact in place.
    """
    cube = cube.clone()
    for c in range(cube.shape[0]):
        ch = cube[c]
        std = torch.std(ch)
        if std > config.NORMALIZE_EPS:
            cube[c] = (ch - torch.mean(ch)) / std
    return cube


def get_device():
    """Detects best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_allowed_patient_ids_from_audits():
    """
    Reads all extraction audit JSONL files, resolves to the latest status per patient,
    and returns allowed patient IDs. Used to filter the dataset based on quality gates.
    """
    if not getattr(config, "TRAIN_FILTER_BY_AUDIT", False):
        return None, []

    pattern = os.path.join(config.PATH_EXTRACTION_AUDIT_DIR, "extraction_audit_*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("Audit filter enabled, but no extraction audit files found. Proceeding without filtering.")
        return None, []

    allowed_statuses = set(getattr(config, "TRAIN_ALLOWED_AUDIT_STATUSES", ("OK", "WARNING")))
    all_patients = {}  # dict to keep the *latest* status (since paths are sorted by datetime)

    for fpath in paths:
        try:
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = rec.get("patient_id")
                    if pid:
                        all_patients[pid] = str(rec.get("status", "")).upper()
        except Exception as e:
            print(f"Could not parse extraction audit '{fpath}': {e}.")

    allowed_ids = []
    excluded_ids = []

    for pid, status in all_patients.items():
        if status in allowed_statuses:
            allowed_ids.append(pid)
        else:
            excluded_ids.append(pid)

    # SECURE GATE: Also check for the Quality Blacklist (Low signal, artifacts, or excessive padding).
    # This list allows for surgical exclusion of patients without re-running the full audit.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    blacklist_path = os.path.join(base_dir, "dashboard", "data_quality_audit", "cube_blacklist.txt")
    
    if os.path.exists(blacklist_path):
        try:
            with open(blacklist_path, "r") as f:
                quality_blacklist = [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            
            allowed_ids_set = set(allowed_ids)
            initial_count = len(allowed_ids_set)
            
            # Remove bad quality IDs from allowed
            for q_pid in quality_blacklist:
                if q_pid in allowed_ids_set:
                    allowed_ids_set.remove(q_pid)
                # Add to excluded list for log traceability
                if q_pid not in excluded_ids:
                    excluded_ids.append(q_pid)
            
            purged_by_quality = initial_count - len(allowed_ids_set)
            if purged_by_quality > 0:
                print(f"Quality Gate: Purged {purged_by_quality} suspicious patients from training set.")
            
            allowed_ids = list(allowed_ids_set)
        except Exception as e:
            print(f"Warning: Could not read quality blacklist: {e}")

    print(f"Audit filter active (from {len(paths)} files) | Final allowed: {len(allowed_ids)}")
    return set(allowed_ids), excluded_ids


class BioLatticeDataset(Dataset):
    def __init__(self, excel_file, folder, augment=True, allowed_patient_ids=None):
        import pandas as pd
        self.data_info = pd.read_excel(excel_file, header=config.CLINICAL_EXCEL_HEADER_ROW)
        self.folder = folder
        self.augment = augment

        suffix = config.LATTICE_FILE_SUFFIX
        patient_ids_with_tensor = [
            f.replace(suffix, "") for f in os.listdir(folder) if f.endswith(suffix)
        ]
        self.data_info = self.data_info[
            self.data_info[config.COL_PATIENT_ID].isin(patient_ids_with_tensor)
        ]

        self.data_info = self.data_info[self.data_info[config.COL_MOL_SUBTYPE].notna()].copy()
        self.data_info = self.data_info.drop_duplicates(subset=[config.COL_PATIENT_ID])

        if allowed_patient_ids is not None:
            self.data_info = self.data_info[
                self.data_info[config.COL_PATIENT_ID].isin(allowed_patient_ids)
            ].copy()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        p_id = row[config.COL_PATIENT_ID]

        label = binary_mol_subtype_label(row)

        data_obj = torch.load(
            os.path.join(self.folder, f"{p_id}{config.LATTICE_FILE_SUFFIX}"),
            weights_only=True
        )
        cube = data_obj['tensor']

        meta = data_obj.get('meta', {})
        spacing = meta.get('voxel_spacing', [1.0, 1.0])
        thickness = meta.get('slice_thickness', 1.0)
        meta_tensor = normalize_metadata(spacing, thickness)

        if self.augment:
            # --- Geometric augmentations (safe for 3D MRI) ---
            # Axial flip: anterior-posterior symmetry
            if random.random() > 0.5:
                cube = torch.flip(cube, dims=[2])
            # Coronal flip: left-right breast symmetry
            if random.random() > 0.5:
                cube = torch.flip(cube, dims=[3])
            # In-plane rotation (0°, 90°, 180°, 270°)
            k = random.randint(0, 3)
            cube = torch.rot90(cube, k, dims=[2, 3])

            # --- Intensity augmentations (simulate MRI gain variability) ---
            # Scale: simulates different RF gain / receive coil response (±10%)
            if random.random() > 0.5:
                scale = 1.0 + (random.random() - 0.5) * 0.20  # [0.90, 1.10]
                cube = cube * scale
            # Additive noise: simulates thermal noise floor in MRI
            if random.random() > 0.7:
                noise = torch.randn_like(cube) * 0.02
                cube = cube + noise

        # Bio-Lattice v2.6 Fix: Per-channel Z-score normalization.
        # This prevents high-magnitude anatomical channels (C1, C4) from 
        # mathematically "drowning out" functional kinetics/texture (C2, C3).
        cube = normalize_cube_per_channel(cube)
        
        return cube, meta_tensor, torch.tensor([label], dtype=torch.float32)


def binary_mol_subtype_label(row) -> float:
    """Standardized logic for binary label assignment (Malignancy/Aggressiveness)."""
    v = float(row[config.COL_MOL_SUBTYPE])
    return 1.0 if v > config.MOL_SUBTYPE_POSITIVE_THRESHOLD else 0.0


def build_patient_level_split(dataset, train_fraction, seed):
    """Deterministic patient-level split (train/val) using shuffled dataset indices."""
    total = len(dataset)
    train_size = int(train_fraction * total)
    train_size = max(0, min(train_size, total))

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), len(train_indices), len(val_indices)


def build_stratified_train_val_indices(dataset, train_fraction: float, seed: int):
    """Patient-level train/val index lists with approximate label balance."""
    labels = [binary_mol_subtype_label(dataset.data_info.iloc[i]) for i in range(len(dataset))]
    rng = random.Random(seed)
    pos_idx = [i for i, y in enumerate(labels) if y >= 0.5]
    neg_idx = [i for i, y in enumerate(labels) if y < 0.5]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split_stratum(idxs):
        n = len(idxs)
        if n <= 1:
            return (idxs, []) if train_fraction >= 0.5 else ([], idxs)
        n_tr = max(1, min(int(round(train_fraction * n)), n - 1))
        return idxs[:n_tr], idxs[n_tr:]

    tr_p, va_p = split_stratum(pos_idx)
    tr_n, va_n = split_stratum(neg_idx)
    train_indices = tr_p + tr_n
    val_indices = va_p + va_n
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices, len(train_indices), len(val_indices)


def build_train_val_subsets(dataset_train, dataset_val, train_fraction, seed, stratified=None):
    """Builds train/val subsets using the configured split strategy."""
    if stratified is None:
        stratified = getattr(config, "TRAIN_STRATIFIED_SPLIT", False)

    if stratified:
        tr_idx, va_idx, train_size, val_size = build_stratified_train_val_indices(
            dataset_train, train_fraction, seed
        )
        train_ds = Subset(dataset_train, tr_idx)
        val_ds = Subset(dataset_val, va_idx)
        return train_ds, val_ds, train_size, val_size

    train_ds, _, train_size, val_size = build_patient_level_split(
        dataset_train, train_fraction, seed
    )
    _, val_ds, _, _ = build_patient_level_split(dataset_val, train_fraction, seed)
    return train_ds, val_ds, train_size, val_size


def labels_for_subset(subset: Subset) -> list[float]:
    """Extracts binary labels for a Subset of a BioLatticeDataset."""
    base = subset.dataset
    return [binary_mol_subtype_label(base.data_info.iloc[i]) for i in subset.indices]
