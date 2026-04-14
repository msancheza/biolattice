import os
import math
import glob
import json
# Mandatory patch for Mac (Apple Silicon): Allows complex 3D operations like MaxPool3d to run softly
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

import config

# Re-export for callers that imported these from train (use config directly for new code).
PATH_CUBES = config.PATH_MICRO_CUBES
PATH_CLINICAL = config.PATH_CLINICAL
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS = config.EPOCHS

def build_patient_level_split(dataset, train_fraction, seed):
    """
    Deterministic patient-level split (train/val) using shuffled dataset indices.
    Assumes BioLatticeDataset is already deduplicated by Patient ID.
    """
    total = len(dataset)
    train_size = int(train_fraction * total)
    train_size = max(0, min(train_size, total))
    val_size = total - train_size

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_size, val_size

# --- 2. CUSTOM DATASET (AUGMENTED VERSION) ---
class BioLatticeDataset(Dataset):
    def __init__(self, excel_file, folder, augment=True, allowed_patient_ids=None):
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
        # Drop duplicates by Patient ID to ensure complete structural isolation in split
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

        label = (
            1.0
            if float(row[config.COL_MOL_SUBTYPE]) > config.MOL_SUBTYPE_POSITIVE_THRESHOLD
            else 0.0
        )

        data_obj = torch.load(
            os.path.join(self.folder, f"{p_id}{config.LATTICE_FILE_SUFFIX}")
        )
        cube = data_obj['tensor']
        
        # --- EXTRACT METADATA ---
        meta = data_obj.get('meta', {})
        spacing = meta.get('voxel_spacing', [1.0, 1.0])
        thickness = meta.get('slice_thickness', 1.0)
        # Use centralized config to prevent Training-Inference drift
        meta_tensor = config.normalize_metadata(spacing, thickness)

        
        # --- DATA AUGMENTATION (If enabled) ---
        if self.augment:
            # A. Random flip on Y-axis (right/left)
            if random.random() > 0.5:
                cube = torch.flip(cube, dims=[2])
            
            # B. Random rotation of 90, 180, or 270 degrees in the Axial plane
            k = random.randint(0, 3)
            cube = torch.rot90(cube, k, dims=[2, 3])
            
            # C. Add subtle Gaussian Noise to stimulate feature learning (Bio-Lattice v2.3)
            if random.random() > 0.8:
                noise = torch.randn_like(cube) * 0.01
                cube = cube + noise

        std = torch.std(cube)
        cube = (
            (cube - torch.mean(cube)) / (std + config.NORMALIZE_EPS)
            if std > 0
            else cube
        )
        
        return cube, meta_tensor, torch.tensor([label], dtype=torch.float32) # [1] shape


def load_allowed_patient_ids_from_latest_audit():
    """
    Reads the latest extraction audit JSONL and returns allowed/excluded patient IDs.
    If loading fails, returns (None, []) so training can proceed without filtering.
    """
    if not getattr(config, "TRAIN_FILTER_BY_AUDIT", False):
        return None, []

    pattern = os.path.join(config.PATH_EXTRACTION_AUDIT_DIR, "extraction_audit_*.jsonl")
    paths = glob.glob(pattern)
    if not paths:
        print("Audit filter enabled, but no extraction audit file found. Training without audit-based exclusions.")
        return None, []

    latest_path = max(paths, key=os.path.getmtime)
    allowed_statuses = set(getattr(config, "TRAIN_ALLOWED_AUDIT_STATUSES", ("OK", "WARNING")))
    latest_by_patient = {}

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
                latest_by_patient[pid] = rec
    except Exception as e:
        print(f"Could not parse extraction audit '{latest_path}': {e}. Training without audit-based exclusions.")
        return None, []

    if not latest_by_patient:
        print(f"Extraction audit '{latest_path}' has no patient records. Training without audit-based exclusions.")
        return None, []

    allowed_ids = []
    excluded_ids = []
    for pid, rec in latest_by_patient.items():
        status = str(rec.get("status", "")).upper()
        if status in allowed_statuses:
            allowed_ids.append(pid)
        else:
            excluded_ids.append(pid)

    print(
        f"Audit filter active from {os.path.basename(latest_path)} | "
        f"allowed: {len(allowed_ids)} | excluded: {len(excluded_ids)} | "
        f"allowed_statuses={sorted(list(allowed_statuses))}"
    )
    if excluded_ids:
        preview = ", ".join(sorted(excluded_ids)[:20])
        suffix = " ..." if len(excluded_ids) > 20 else ""
        print(f"Excluded patient IDs ({len(excluded_ids)}): {preview}{suffix}")

    return set(allowed_ids), excluded_ids

class ResidualBlock3D(nn.Module):
    """3D ResNet-style block."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.conv(x)) # Residual connection ('+') is critical

class BioLattice3DResNet(nn.Module):
    """3D-ResNet classifier for micro-cube tensors (spatial size follows config.MICRO_CUBE_SIZE, default 64³)."""

    def __init__(self):
        super().__init__()
        c = config
        self.prep = nn.Sequential(
            nn.Conv3d(c.INPUT_CHANNELS, c.STEM_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm3d(c.STEM_CHANNELS),
            nn.ReLU(),
        )
        self.block1 = ResidualBlock3D(c.RES_BLOCK1_IN, c.RES_BLOCK1_OUT)
        self.pool1 = nn.Conv3d(
            c.RES_BLOCK1_OUT,
            c.RES_BLOCK1_OUT,
            kernel_size=c.POOL_KERNEL,
            stride=c.POOL_STRIDE,
        )

        self.block2 = ResidualBlock3D(c.RES_BLOCK2_IN, c.RES_BLOCK2_OUT)
        self.pool2 = nn.Conv3d(
            c.RES_BLOCK2_OUT,
            c.RES_BLOCK2_OUT,
            kernel_size=c.POOL_KERNEL,
            stride=c.POOL_STRIDE,
        )

        self.avgpool_flatten = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=c.CLASSIFIER_AVG_POOL_KERNEL,
                stride=c.CLASSIFIER_AVG_POOL_STRIDE,
            ),
            nn.Flatten()
        )
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(c.META_FEATURE_DIM, c.META_MLP_OUT),
            nn.Mish(),
            nn.BatchNorm1d(c.META_MLP_OUT)
        )

        self.classifier = nn.Sequential(
            nn.Linear(c.CLASSIFIER_LINEAR_IN + c.META_MLP_OUT, c.CLASSIFIER_HIDDEN),
            nn.BatchNorm1d(c.CLASSIFIER_HIDDEN),
            nn.Mish(),
            nn.Dropout(c.CLASSIFIER_DROPOUT),
            nn.Linear(c.CLASSIFIER_HIDDEN, c.CLASSIFIER_HIDDEN // 2),
            nn.BatchNorm1d(c.CLASSIFIER_HIDDEN // 2),
            nn.Mish(),
            nn.Linear(c.CLASSIFIER_HIDDEN // 2, 1),
        )

    def forward(self, x, meta_features):
        x = self.prep(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))

        x_img = self.avgpool_flatten(x)
        m_feat = self.meta_mlp(meta_features)

        x_combined = torch.cat([x_img, m_feat], dim=1)

        return self.classifier(x_combined)

# --- 4. ROBUST TRAINING CYCLE ---
def train_model():
    allowed_patient_ids, _ = load_allowed_patient_ids_from_latest_audit()

    # Instantiate dataset handlers (True for Train, False for Val to prevent augmentation leakage)
    dataset_train = BioLatticeDataset(
        PATH_CLINICAL, PATH_CUBES, augment=True, allowed_patient_ids=allowed_patient_ids
    )
    dataset_val = BioLatticeDataset(
        PATH_CLINICAL, PATH_CUBES, augment=False, allowed_patient_ids=allowed_patient_ids
    )
    
    split = config.TRAIN_VAL_SPLIT_FRACTION
    train_dataset, _, train_size, val_size = build_patient_level_split(
        dataset_train, split, config.RANDOM_SEED
    )
    _, val_dataset, _, _ = build_patient_level_split(
        dataset_val, split, config.RANDOM_SEED
    )
    
    # Prevent BatchNorm1d crash on the last incomplete batch (size=1).
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    if num_train_batches == 0:
        print(
            "Error: training DataLoader has zero batches (dataset too small for BATCH_SIZE with drop_last=True). "
            "Lower BATCH_SIZE in config or add more samples."
        )
        return
    
    # BUGFIX 2: Active GPU Device (Supports Apple Silicon MPS and NVIDIA CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Sending Bio-Lattice 4D architecture to hardware accelerator: {device}")
    
    model = BioLattice3DResNet().to(device)

    # 1. Focal loss (emphasizes hard examples)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.75, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss) 
            return (self.alpha * ((1 - pt) ** self.gamma) * bce_loss).mean()

    criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)


    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=config.ADAMW_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.ONECYCLE_MAX_LR,
        steps_per_epoch=num_train_batches,
        epochs=EPOCHS,
    )
    
    print(f"-- Starting Phenotype Classification (High vs. Lower Risk) with {train_size} tensors (Train) and {val_size} (Val)...")
    
    best_val_loss = float('inf')
    early_patience = config.EARLY_STOPPING_PATIENCE
    early_min_delta = config.EARLY_STOPPING_MIN_DELTA
    early_start_epoch = config.EARLY_STOPPING_START_EPOCH
    epochs_without_improve = 0
    
    for epoch in range(EPOCHS): 
        model.train()
        running_loss = 0.0
        
        for i, (inputs, metas, labels) in enumerate(train_loader):
            inputs, metas, labels = inputs.to(device), metas.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, metas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # OneCycleLR steps forward per local micro-batch (Not per Epoch)
            scheduler.step() 
            
            running_loss += loss.item()
            
        train_loss = running_loss / num_train_batches
        
        # --- NEW: Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, metas, labels in val_loader:
                inputs, metas, labels = inputs.to(device), metas.to(device), labels.to(device)
                outputs = model(inputs, metas)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        if num_val_batches > 0:
            val_loss = val_loss / num_val_batches
        else:
            val_loss = float("nan")
            
        # Print telemetry
        val_disp = f"{val_loss:.4f}" if not math.isnan(val_loss) else "nan (no val batches)"
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_disp} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Save the best model natively via validation score (manual Early Stopping)
        improved = (not math.isnan(val_loss)) and (val_loss < (best_val_loss - early_min_delta))
        if improved:
            best_val_loss = val_loss
            epochs_without_improve = 0
            os.makedirs(config.PATH_MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.PATH_MODEL_WEIGHTS)
        elif not math.isnan(val_loss):
            epochs_without_improve += 1

        if (epoch + 1) >= early_start_epoch and epochs_without_improve >= early_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}: "
                f"no validation improvement > {early_min_delta:.4f} for {early_patience} consecutive epochs."
            )
            break
            
    print("3D-ResNet Training finished empirically and best weights saved locally.")

if __name__ == "__main__":
    train_model()