import os
# Mandatory patch for Mac (Apple Silicon): Allows complex 3D operations like MaxPool3d to run softly
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

import config

# Re-export for callers that imported these from train (use config directly for new code).
PATH_CUBOS = config.PATH_MICRO_CUBOS
PATH_CLINICAL = config.PATH_CLINICAL
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS = config.EPOCHS

# --- 2. CUSTOM DATASET (AUGMENTED VERSION) ---
class BioLatticeDataset(Dataset):
    def __init__(self, excel_file, folder, augment=True):
        self.data_info = pd.read_excel(excel_file, header=config.CLINICAL_EXCEL_HEADER_ROW)
        self.folder = folder
        self.augment = augment

        suffix = config.LATTICE_FILE_SUFFIX
        existentes = [
            f.replace(suffix, "") for f in os.listdir(folder) if f.endswith(suffix)
        ]
        self.data_info = self.data_info[
            self.data_info[config.COL_PATIENT_ID].isin(existentes)
        ]

        self.data_info = self.data_info[self.data_info[config.COL_MOL_SUBTYPE].notna()] 
        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        fila = self.data_info.iloc[idx]
        p_id = fila[config.COL_PATIENT_ID]

        label = (
            1.0
            if float(fila[config.COL_MOL_SUBTYPE]) > config.MOL_SUBTYPE_POSITIVE_THRESHOLD
            else 0.0
        )

        cubo = torch.load(
            os.path.join(self.folder, f"{p_id}{config.LATTICE_FILE_SUFFIX}")
        )
        
        # --- DATA AUGMENTATION (If enabled) ---
        if self.augment:
            # A. Random flip on Y-axis (right/left)
            if random.random() > 0.5:
                cubo = torch.flip(cubo, dims=[2])
            
            # B. Random rotation of 90, 180, or 270 degrees in the Axial plane
            k = random.randint(0, 3)
            cubo = torch.rot90(cubo, k, dims=[2, 3])

        std = torch.std(cubo)
        cubo = (
            (cubo - torch.mean(cubo)) / (std + config.NORMALIZE_EPS)
            if std > 0
            else cubo
        )
        
        return cubo, torch.tensor([label], dtype=torch.float32) # [1] shape

class ResidualBlock3D(nn.Module):
    """ 3D-ResNet Core: Prevents vanishing gradient degradation """
    def __init__(self, in_canales, out_canales=None):
        super().__init__()
        if out_canales is None:
            out_canales = in_canales
            
        self.conv = nn.Sequential(
            nn.Conv3d(in_canales, out_canales, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_canales),
            nn.ReLU(),
            nn.Conv3d(out_canales, out_canales, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_canales)
        )
        self.relu = nn.ReLU()
        
        # If the number of channels changes, we need to adjust the shortcut
        self.shortcut = nn.Sequential()
        if in_canales != out_canales:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_canales, out_canales, kernel_size=1),
                nn.BatchNorm3d(out_canales)
            )

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.conv(x)) # Residual connection ('+') is critical

class BioLattice3DResNet(nn.Module):
    """3D-ResNet classifier for micro-cube tensors (shape tuned to MICRO_CUBE_SIZE=32)."""

    def __init__(self):
        super().__init__()
        c = config
        self.prep = nn.Sequential(
            nn.Conv3d(c.INPUT_CHANNELS, c.STEM_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm3d(c.STEM_CHANNELS),
            nn.ReLU(),
        )
        self.capa1 = ResidualBlock3D(c.RES_BLOCK1_IN, c.RES_BLOCK1_OUT)
        self.pool1 = nn.Conv3d(
            c.RES_BLOCK1_OUT,
            c.RES_BLOCK1_OUT,
            kernel_size=c.POOL_KERNEL,
            stride=c.POOL_STRIDE,
        )

        self.capa2 = ResidualBlock3D(c.RES_BLOCK2_IN, c.RES_BLOCK2_OUT)
        self.pool2 = nn.Conv3d(
            c.RES_BLOCK2_OUT,
            c.RES_BLOCK2_OUT,
            kernel_size=c.POOL_KERNEL,
            stride=c.POOL_STRIDE,
        )

        self.clasificador = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=c.CLASSIFIER_AVG_POOL_KERNEL,
                stride=c.CLASSIFIER_AVG_POOL_STRIDE,
            ),
            nn.Flatten(),
            nn.Linear(c.CLASSIFIER_LINEAR_IN, c.CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(c.CLASSIFIER_DROPOUT),
            nn.Linear(c.CLASSIFIER_HIDDEN, 1),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.pool1(self.capa1(x))
        x = self.pool2(self.capa2(x))
        return self.clasificador(x)

# --- 4. ROBUST TRAINING CYCLE ---
def train_model():
    # Instantiate dataset handlers (True for Train, False for Val to prevent augmentation leakage)
    dataset_train = BioLatticeDataset(PATH_CLINICAL, PATH_CUBOS, augment=True)
    dataset_val = BioLatticeDataset(PATH_CLINICAL, PATH_CUBOS, augment=False)
    
    split = config.TRAIN_VAL_SPLIT_FRACTION
    train_size = int(split * len(dataset_train))
    val_size = len(dataset_train) - train_size

    g_train = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, _ = random_split(
        dataset_train, [train_size, val_size], generator=g_train
    )

    g_val = torch.Generator().manual_seed(config.RANDOM_SEED)
    _, val_dataset = random_split(
        dataset_val, [train_size, val_size], generator=g_val
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # BUGFIX 2: Active GPU Device (Supports Apple Silicon MPS and NVIDIA CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Sending Bio-Lattice 4D architecture to hardware accelerator: {device}")
    
    modelo = BioLattice3DResNet().to(device) # <--- Transfer Residual Architecture to VRAM
    
    # 1. Binary Loss Function (BCEWithLogitsLoss)
    # Mathematically penalize False Negatives (If misclassified benign, gradient punishes much harder)
    pos_weight = torch.tensor([config.BCE_POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        modelo.parameters(),
        lr=LEARNING_RATE,
        weight_decay=config.ADAMW_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.ONECYCLE_MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
    )
    
    print(f"-- Starting Binary Classification (Malignant vs Benign) with {train_size} tensors (Train) and {val_size} (Val)...")
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS): 
        modelo.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Send matrix tensors to the active graphic hardware
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = modelo(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # OneCycleLR steps forward per local micro-batch (Not per Epoch)
            scheduler.step() 
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        # --- NEW: Validation Phase ---
        modelo.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = modelo(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)
            
        # Print telemetry
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save the best model natively via validation score (manual Early Stopping)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config.PATH_MODEL_DIR, exist_ok=True)
            torch.save(modelo.state_dict(), config.PATH_MODEL_WEIGHTS)
            
    print("3D-ResNet Training finished empirically and best weights saved locally.")

if __name__ == "__main__":
    train_model()