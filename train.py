import os
# Mandatory patch for Mac (Apple Silicon): Allows complex 3D operations like MaxPool3d to run softly
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# --- 1. CONFIGURATION ---
PATH_CUBOS = 'datasets/micro_cubos/'
PATH_CLINICAL = 'datasets/Clinical_and_Other_Features.xlsx'
# --- OPTIMIZED PARAMETERS ---
# --- CONFIGURATION FOR BALANCED DATASET ---
BATCH_SIZE = 4 
LEARNING_RATE = 0.0001 # A fine step to prevent "skipping" error minimums
EPOCHS = 50            # With 40 patients, 50 epochs is the "sweet spot"

import random

# --- 2. CUSTOM DATASET (AUGMENTED VERSION) ---
class BioLatticeDataset(Dataset):
    def __init__(self, excel_file, folder, augment=True):
        # Read Excel (header=1 is crucial in Duke)
        self.data_info = pd.read_excel(excel_file, header=1)
        self.folder = folder
        self.augment = augment
        
        # 1. Clean: Only patients with generated .pt files
        existentes = [f.replace('_lattice.pt', '') for f in os.listdir(folder) if f.endswith('.pt')]
        self.data_info = self.data_info[self.data_info['Patient ID'].isin(existentes)]

        # 2. Ensure row has usable clinical metadata
        self.data_info = self.data_info[self.data_info['Mol Subtype'].notna()] 
        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        fila = self.data_info.iloc[idx]
        p_id = fila['Patient ID']
        
        # Binary Classification Logic (0: Benign, 1: Confirmed Malignant)
        # Assuming if 'Mol Subtype' > 0 it is a malignant cancer.
        # BCEWithLogits requires Float variables for tensor [1]
        label = 1.0 if float(fila['Mol Subtype']) > 0 else 0.0
        
        # Load the cube
        cubo = torch.load(os.path.join(self.folder, f"{p_id}_lattice.pt"))
        
        # --- DATA AUGMENTATION (If enabled) ---
        if self.augment:
            # A. Random flip on Y-axis (right/left)
            if random.random() > 0.5:
                cubo = torch.flip(cubo, dims=[2])
            
            # B. Random rotation of 90, 180, or 270 degrees in the Axial plane
            k = random.randint(0, 3)
            cubo = torch.rot90(cubo, k, dims=[2, 3])

        # Strict Z-Score Normalization (Prevents compressing values over bone/metal/clips)
        std = torch.std(cubo)
        cubo = (cubo - torch.mean(cubo)) / (std + 1e-8) if std > 0 else cubo
        
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
    """ Parametric architecture matching technical_pipeline.md (Optimized 3D-ResNet) """
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.capa1 = ResidualBlock3D(32, 64)
        # Substitution of MaxPool3d for Conv3d with stride to avoid MPS crash:
        self.pool1 = nn.Conv3d(64, 64, kernel_size=2, stride=2) 
        
        self.capa2 = ResidualBlock3D(64, 128)
        self.pool2 = nn.Conv3d(128, 128, kernel_size=2, stride=2) 
        
        # Classification tail prevents spatial structure loss
        self.clasificador = nn.Sequential(
            nn.AvgPool3d(kernel_size=4, stride=4), # 8x8x8 -> 2x2x2 (Explicitly avoids Apple MPS crash)
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # Stochastic dropout randomly disabling 40% neurons 
            nn.Linear(256, 1) # Returns exactly 1 raw Logit to BCEWithLogitsLoss
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
    
    # NEW: Validation Split (80% / 20%)
    train_size = int(0.8 * len(dataset_train))
    val_size = len(dataset_train) - train_size
    
    # Guarantees identical patient assignment across datasets using the same seed
    g_train = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(dataset_train, [train_size, val_size], generator=g_train)
    
    g_val = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(dataset_val, [train_size, val_size], generator=g_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # BUGFIX 2: Active GPU Device (Supports Apple Silicon MPS and NVIDIA CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Sending Bio-Lattice 4D architecture to hardware accelerator: {device}")
    
    modelo = BioLattice3DResNet().to(device) # <--- Transfer Residual Architecture to VRAM
    
    # 1. Binary Loss Function (BCEWithLogitsLoss)
    # Mathematically penalize False Negatives (If misclassified benign, gradient punishes much harder)
    pos_weight = torch.tensor([3.0]).to(device) # High threshold to force cancer pattern learning
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # BUGFIX 3: AdamW Optimizer (Clears static Weight Decay)
    optimizer = optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # BUGFIX 4: Fast Industrial Scheduler (OneCycleLR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    print(f"🚀 Starting Binary Classification (Malignant vs Benign) with {train_size} tensors (Train) and {val_size} (Val)...")
    
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
            os.makedirs("datasets/modelo", exist_ok=True)
            torch.save(modelo.state_dict(), "datasets/modelo/biolattice_3dresnet_binary.pth")
            
    print("3D-ResNet Training finished empirically and best weights saved locally.")

if __name__ == "__main__":
    train_model()