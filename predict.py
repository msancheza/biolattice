import os
import torch

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, random_split

import config
from train import BioLattice3DResNet, BioLatticeDataset

# Threshold over sigmoid (0–1). Alias kept for Spanish-facing callers.
UMBRAL_MALIGNIDAD = config.MALIGNANCY_PROB_THRESHOLD


def predict_patient(p_id):
    path_cubo = os.path.join(
        config.PATH_MICRO_CUBOS, f"{p_id}{config.LATTICE_FILE_SUFFIX}"
    )
    path_modelo = config.PATH_MODEL_WEIGHTS

    if not os.path.exists(path_cubo):
        print(f"Error: Tensor not found for patient {p_id}. Run main.py first.")
        return {"error": f"Medical tensor missing for patient {p_id}. Make sure Data Extraction ran."}
    if not os.path.exists(path_modelo):
        print("Error: Trained model not found. Run train.py first.")
        return {"error": "Trained 3D-ResNet model not found. Make sure Model Training completed."}

    device = torch.device(config.INFERENCE_DEVICE)
    modelo = BioLattice3DResNet()
    modelo.load_state_dict(
        torch.load(path_modelo, map_location=device, weights_only=True)
    )
    modelo.eval() # Imperative: Turn off dynamic training layers like Dropouts
    modelo.to(device)

    # 3. Prepare the Patient Tensor
    cubo = torch.load(path_cubo, map_location=device).unsqueeze(0)
    std = torch.std(cubo)
    cubo = (
        (cubo - torch.mean(cubo)) / (std + config.NORMALIZE_EPS)
        if std > 0
        else cubo
    )

    # 4. Clinical Inference (Binary Virtual Biopsy)
    with torch.no_grad():
        # Our 3D-ResNet outputs a raw mathematical 'Logit'
        logit_crudo = modelo(cubo)
        
        p_mal = torch.sigmoid(logit_crudo).item()
        probabilidad_malignidad = p_mal * 100

        print(f"\n--- 4D Oncological Evaluation for Patient: {p_id} ---")
        print(f"   (Positive threshold: ≥ {UMBRAL_MALIGNIDAD * 100:.0f}%)")

        if p_mal >= UMBRAL_MALIGNIDAD:
            print(f"=> AI DIAGNOSIS: POSITIVE (HIGH RISK OF MALIGNANT CANCER)")
        else:
            print(f"=> AI DIAGNOSIS: NEGATIVE (LIKELY BENIGN TISSUE)")

        print(f"-- Malignancy Probability Index: {probabilidad_malignidad:.2f}%")
        print(f"----------------------------------------------------------------")
        
        return {
            "riesgo": probabilidad_malignidad,
            "positivo": p_mal >= UMBRAL_MALIGNIDAD,
            "umbral": UMBRAL_MALIGNIDAD * 100
        }

def evaluate_dataset():
    """ Evaluates the model on the full validation set checking ROC, Sensitivity, and Specificity. """
    device = torch.device(config.INFERENCE_DEVICE)
    path_modelo = config.PATH_MODEL_WEIGHTS

    if not os.path.exists(path_modelo):
        return {"error": "Trained model not found. Run Training first."}

    modelo = BioLattice3DResNet()
    modelo.load_state_dict(
        torch.load(path_modelo, map_location=device, weights_only=True)
    )
    modelo.eval()

    dataset_val = BioLatticeDataset(
        config.PATH_CLINICAL, config.PATH_MICRO_CUBOS, augment=False
    )
    split = config.TRAIN_VAL_SPLIT_FRACTION
    train_size = int(split * len(dataset_val))
    val_size = len(dataset_val) - train_size

    g_val = torch.Generator().manual_seed(config.RANDOM_SEED)
    _, val_dataset = random_split(
        dataset_val, [train_size, val_size], generator=g_val
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
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
    predict_patient(paciente_test)