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


def series_is_pre_contrast(desc_lower: str) -> bool:
    """PRE: substring 'pre', or 'dyn'/'vibrant' without numbered phases."""
    is_pre = "pre" in desc_lower
    is_dynamic = ("dyn" in desc_lower) or ("vibrant" in desc_lower)
    has_numbered_phase = bool(re.search(r"(ph|phase)\s?\d", desc_lower))
    return is_pre or (is_dynamic and not has_numbered_phase)


def series_is_post_contrast(desc_lower: str) -> bool:
    has_post_tag = any(x in desc_lower for x in config.POST_SERIES_SUBSTRINGS)
    has_numbered_phase = bool(
        re.search(r"(ph|phase|dyn|dynamic|fase)\s?[1-9]", desc_lower)
    )
    return has_post_tag or has_numbered_phase


def normalize_metadata(spacing, thickness):
    """
    Build the 3-vector fed to the metadata MLP from DICOM spacing and thickness.

    Uses ``config.META_NORM_SHIFT_XY`` and ``config.META_NORM_SHIFT_Z`` so train and
    inference stay aligned; tune those constants in ``config`` only.
    """
    import torch

    tensor = torch.tensor(
        [
            spacing[0] + config.META_NORM_SHIFT_XY,
            spacing[1] + config.META_NORM_SHIFT_XY,
            thickness + config.META_NORM_SHIFT_Z,
        ],
        dtype=torch.float32,
    )
    assert len(tensor) == config.META_FEATURE_DIM, (
        f"CRITICAL: Metadata array length {len(tensor)} != Config DIM {config.META_FEATURE_DIM}"
    )
    return tensor


def get_device():
    """Detects best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_allowed_patient_ids_from_latest_audit():
    """
    Reads the latest extraction audit JSONL and returns allowed patient IDs.
    Used to filter the dataset based on quality gates in training/inference.
    """
    if not getattr(config, "TRAIN_FILTER_BY_AUDIT", False):
        return None, []

    pattern = os.path.join(config.PATH_EXTRACTION_AUDIT_DIR, "extraction_audit_*.jsonl")
    paths = glob.glob(pattern)
    if not paths:
        print("Audit filter enabled, but no extraction audit file found. Proceeding without filtering.")
        return None, []

    latest_path = max(paths, key=os.path.getmtime)
    allowed_statuses = set(getattr(config, "TRAIN_ALLOWED_AUDIT_STATUSES", ("OK", "WARNING")))
    allowed_ids = []
    excluded_ids = []

    try:
        with open(latest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = rec.get("patient_id")
                if not pid:
                    continue

                status = str(rec.get("status", "")).upper()
                if status in allowed_statuses:
                    allowed_ids.append(pid)
                else:
                    excluded_ids.append(pid)
    except Exception as e:
        print(f"Could not parse extraction audit '{latest_path}': {e}.")
        return None, []

    print(f"Audit filter active: {os.path.basename(latest_path)} | allowed: {len(allowed_ids)}")
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
            if random.random() > 0.5:
                cube = torch.flip(cube, dims=[2])
            k = random.randint(0, 3)
            cube = torch.rot90(cube, k, dims=[2, 3])
            if random.random() > 0.8:
                noise = torch.randn_like(cube) * 0.01
                cube = cube + noise

        std = torch.std(cube)
        cube = (
            (cube - torch.mean(cube)) / (std + config.NORMALIZE_EPS)
            if std > 0
            else cube
        )
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


def labels_for_subset(subset: Subset) -> list[float]:
    """Extracts binary labels for a Subset of a BioLatticeDataset."""
    base = subset.dataset
    return [binary_mol_subtype_label(base.data_info.iloc[i]) for i in subset.indices]
