import os
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

import config
from train import BioLattice3DResNet, BioLatticeDataset, build_patient_level_split

# Probability threshold on sigmoid output (0–1); mirrors `MALIGNANCY_PROB_THRESHOLD` in config.
INFERENCE_PROB_THRESHOLD = config.MALIGNANCY_PROB_THRESHOLD


def predict_patient(p_id):
    lattice_path = os.path.join(
        config.PATH_MICRO_CUBES, f"{p_id}{config.LATTICE_FILE_SUFFIX}"
    )
    path_weights = config.PATH_MODEL_WEIGHTS

    if not os.path.exists(lattice_path):
        print(f"Error: Tensor not found for patient {p_id}. Run main.py first.")
        return {"error": f"Medical tensor missing for patient {p_id}. Make sure Data Extraction ran."}
    if not os.path.exists(path_weights):
        print("Error: Trained model not found. Run train.py first.")
        return {"error": "Trained 3D-ResNet model not found. Make sure Model Training completed."}

    device = torch.device(config.INFERENCE_DEVICE)
    model = BioLattice3DResNet()
    model.load_state_dict(
        torch.load(path_weights, map_location=device, weights_only=True)
    )
    model.eval()
    model.to(device)

    # 3. Prepare the patient tensor
    data_obj = torch.load(lattice_path, map_location=device)
    cube = data_obj['tensor'].unsqueeze(0)
    # --- Metadata (same normalization as training) ---
    meta = data_obj.get('meta', {})
    spacing = meta.get('voxel_spacing', [1.0, 1.0])
    thickness = meta.get('slice_thickness', 1.0)
    meta_tensor = config.normalize_metadata(spacing, thickness).unsqueeze(0).to(device)

    std = torch.std(cube)
    cube = (
        (cube - torch.mean(cube)) / (std + config.NORMALIZE_EPS)
        if std > 0
        else cube
    )

    # 4. Clinical Inference (Binary Virtual Biopsy)
    with torch.no_grad():
        raw_logit = model(cube, meta_tensor)
        p_positive = torch.sigmoid(raw_logit).item()
        risk_percent = p_positive * 100

        print(f"\n--- 4D Oncological Evaluation (Risk Profiling): {p_id} ---")
        print(f"   (Aggressiveness threshold: ≥ {INFERENCE_PROB_THRESHOLD * 100:.0f}%)")

        if p_positive >= INFERENCE_PROB_THRESHOLD:
            print(f"=> AI DIAGNOSIS: HIGH RISK (NON-LUMINAL A PHENOTYPE)")
        else:
            print(f"=> AI DIAGNOSIS: LOWER RISK (LUMINAL A PHENOTYPE)")

        print(f"-- Risk Index: {risk_percent:.2f}%")
        print(f"----------------------------------------------------------------")
        
        return {
            "risk_percent": risk_percent,
            "high_risk": p_positive >= INFERENCE_PROB_THRESHOLD,
            "threshold_percent": INFERENCE_PROB_THRESHOLD * 100,
        }

def evaluate_dataset():
    """ Evaluates the model on the full validation set checking ROC, Sensitivity, and Specificity. """
    device = torch.device(config.INFERENCE_DEVICE)
    path_weights = config.PATH_MODEL_WEIGHTS

    if not os.path.exists(path_weights):
        return {"error": "Trained model not found. Run Training first."}

    model = BioLattice3DResNet()
    model.load_state_dict(
        torch.load(path_weights, map_location=device, weights_only=True)
    )
    model.eval()
    model.to(device)

    dataset_val = BioLatticeDataset(
        config.PATH_CLINICAL, config.PATH_MICRO_CUBES, augment=False
    )
    split = config.TRAIN_VAL_SPLIT_FRACTION
    _, val_dataset, _, _ = build_patient_level_split(
        dataset_val, split, config.RANDOM_SEED
    )
    
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Cannot perform quantitative evaluation.")
        return None

    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, metas, labels in val_loader:
            inputs = inputs.to(device)
            metas = metas.to(device)
            logits = model(inputs, metas)
            probs = torch.sigmoid(logits)
            
            # Convert to 1D numpy array
            probs_np = probs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            
            preds = (probs_np >= INFERENCE_PROB_THRESHOLD).astype(float)
            
            all_probs.extend(probs_np.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())
            
    # Calculate AUC once (threshold-independent metric)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0 # Edge case where only one class is evaluated

    # ROC curve points for dashboard visualization (single, high-value chart).
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_points = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
    except ValueError:
        roc_points = {"fpr": [], "tpr": []}

    # Sweep thresholds to find an operating point that balances sensitivity/specificity.
    # Criterion: maximize Youden's J = Sensitivity + Specificity - 1.
    best = {
        "threshold": INFERENCE_PROB_THRESHOLD,
        "youden_j": -1.0,
        "accuracy": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
    }

    labels_np_all = np.array(all_labels, dtype=float)
    probs_np_all = np.array(all_probs, dtype=float)
    for th in np.arange(0.05, 0.951, 0.05):
        preds_th = (probs_np_all >= th).astype(float)
        acc_th = accuracy_score(labels_np_all, preds_th)
        sens_th = recall_score(labels_np_all, preds_th, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(
            labels_np_all, preds_th, labels=[0, 1]
        ).ravel()
        esp_th = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden_j = sens_th + esp_th - 1.0

        if youden_j > best["youden_j"]:
            best = {
                "threshold": float(th),
                "youden_j": float(youden_j),
                "accuracy": float(acc_th),
                "sensitivity": float(sens_th),
                "specificity": float(esp_th),
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }

    # Metrics at configured threshold (current production-like operating point)
    acc = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "total": len(val_dataset),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": roc_points,
        "configured_threshold": INFERENCE_PROB_THRESHOLD,
        "best_threshold_youden": best,
    }

if __name__ == "__main__":
    print("Starting Bio-Lattice 4D Virtual Biopsy Mode...")
    patient_id = input("Enter the Patient ID to scan (e.g., Breast_MRI_001): ").strip()
    predict_patient(patient_id)