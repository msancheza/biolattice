import os
import torch

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, random_split

# Import the NEW robust architecture (3D-ResNet)
from train import RedMicroCubo3Ch, BioLatticeDataset, PATH_CLINICAL, PATH_CUBOS

# Threshold over sigmoid output (0–1). Higher reduces false positives; raises false negatives.
UMBRAL_MALIGNIDAD = 0.75


def predecir_paciente(p_id):
    # 1. File Paths (Pointing to the Binary version)
    PATH_CUBO = f'datasets/micro_cubos/{p_id}_lattice.pt'
    PATH_MODELO = 'datasets/modelo/biolattice_3dresnet_binary.pth'

    if not os.path.exists(PATH_CUBO):
        print(f"❌ Error: Tensor not found for patient {p_id}. Run main.py first.")
        return {"error": f"Medical tensor missing for patient {p_id}. Make sure Data Extraction ran."}
    if not os.path.exists(PATH_MODELO):
        print("❌ Error: Trained model not found. Run train.py first.")
        return {"error": "Trained 3D-ResNet model not found. Make sure Model Training completed."}

    # 2. Load the "Brain" (Model) to CPU for safe local inference
    device = torch.device("cpu")
    modelo = RedMicroCubo3Ch()
    # Use map_location='cpu' in case it was saved from MPS/CUDA
    modelo.load_state_dict(torch.load(PATH_MODELO, map_location=device, weights_only=True))
    modelo.eval() # Imperative: Turn off dynamic training layers like Dropouts
    modelo.to(device)

    # 3. Prepare the Patient Tensor
    cubo = torch.load(PATH_CUBO, map_location=device).unsqueeze(0) # [1, 3, 32, 32, 32]
    # BUGFIX: Strict Z-Score Normalization (identical to what the network saw in train.py)
    std = torch.std(cubo)
    cubo = (cubo - torch.mean(cubo)) / (std + 1e-8) if std > 0 else cubo

    # 4. Clinical Inference (Binary Virtual Biopsy)
    with torch.no_grad():
        # Our 3D-ResNet outputs a raw mathematical 'Logit'
        logit_crudo = modelo(cubo)
        
        p_mal = torch.sigmoid(logit_crudo).item()
        probabilidad_malignidad = p_mal * 100

        print(f"\n--- 📊 4D Oncological Evaluation for Patient: {p_id} ---")
        print(f"   (Positive threshold: ≥ {UMBRAL_MALIGNIDAD * 100:.0f}%)")

        if p_mal >= UMBRAL_MALIGNIDAD:
            print(f"🚦 AI DIAGNOSIS: POSITIVE (HIGH RISK OF MALIGNANT CANCER)")
        else:
            print(f"✅ AI DIAGNOSIS: NEGATIVE (LIKELY BENIGN TISSUE)")

        print(f"🔬 Malignancy Probability Index: {probabilidad_malignidad:.2f}%")
        print(f"----------------------------------------------------------------")
        
        return {
            "riesgo": probabilidad_malignidad,
            "positivo": p_mal >= UMBRAL_MALIGNIDAD,
            "umbral": UMBRAL_MALIGNIDAD * 100
        }

def evaluar_dataset():
    """ Evaluates the model on the full validation set checking ROC, Sensitivity, and Specificity. """
    device = torch.device("cpu")
    PATH_MODELO = 'datasets/modelo/biolattice_3dresnet_binary.pth'
    
    if not os.path.exists(PATH_MODELO):
        return {"error": "Trained model not found. Run Training first."}
        
    modelo = RedMicroCubo3Ch()
    modelo.load_state_dict(torch.load(PATH_MODELO, map_location=device, weights_only=True))
    modelo.eval()
    
    # Recreate the strict Val Loader without data augmentation
    dataset_val = BioLatticeDataset(PATH_CLINICAL, PATH_CUBOS, augment=False)
    train_size = int(0.8 * len(dataset_val))
    val_size = len(dataset_val) - train_size
    
    g_val = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(dataset_val, [train_size, val_size], generator=g_val)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = modelo(inputs)
            probs = torch.sigmoid(logits)
            
            # Convert to 1D numpy array
            probs_np = probs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            
            preds = (probs_np >= UMBRAL_MALIGNIDAD).astype(float)
            
            all_probs.extend(probs_np.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())
            
    # Calculate Metrics
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0 # Edge case where only one class is evaluated
        
    acc = accuracy_score(all_labels, all_preds)
    sensibilidad = recall_score(all_labels, all_preds, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "auc": auc,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "total": len(val_dataset),
        "matriz": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    }

if __name__ == "__main__":
    print("Starting Bio-Lattice 4D Virtual Biopsy Mode...")
    paciente_test = input("Enter the Patient ID to scan (e.g., Breast_MRI_001): ").strip()
    predecir_paciente(paciente_test)