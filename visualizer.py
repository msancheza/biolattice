import os
# Mandatory patch for Mac (Apple Silicon): allows ops like AvgPool3d to fall back to CPU
# when not yet implemented on MPS (mirrors the same flag in train.py).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from core import helper

_STYLE = {
    "fig_bg": "#000000",
    "card_face": "#111111",
    "card_edge": "#333333",
    "title": "#ffffff",
    "subtitle": "#aaaaaa",
    "cmaps": ["gray", "hot", "viridis"],
    "accent": "#00f2ff" # Neon cyan
}

def compute_gradcam(patient_id, device="cpu"):
    """Computes Grad-CAM 3D for a specific patient using weights manually-loaded."""
    from train import BioLattice3DResNet
    
    lattice_path = os.path.join(config.PATH_MICRO_CUBES, f"{patient_id}{config.LATTICE_FILE_SUFFIX}")
    path_weights = config.PATH_MODEL_WEIGHTS
    
    if not os.path.exists(lattice_path) or not os.path.exists(path_weights):
        return None
    
    model = BioLattice3DResNet().to(device)
    model.load_state_dict(torch.load(path_weights, map_location=device, weights_only=True))
    model.eval()
    
    data_obj = torch.load(lattice_path, map_location=device, weights_only=True)
    raw_cube = data_obj['tensor']
    # Normalize for model input
    std = torch.std(raw_cube)
    norm_cube = (raw_cube - torch.mean(raw_cube)) / (std + config.NORMALIZE_EPS) if std > 0 else raw_cube
    model_input = norm_cube.unsqueeze(0).to(device)
    
    meta = data_obj.get('meta', {})
    meta_tensor = helper.normalize_metadata(
        meta.get('voxel_spacing', [1.0, 1.0]), 
        meta.get('slice_thickness', 1.0)
    ).unsqueeze(0).to(device)
    
    # Inference with Grad-CAM
    model.zero_grad()
    model_input.requires_grad = True
    logits = model(model_input, meta_tensor, return_cam=True)
    logits.backward()
    
    weights = model.get_gradcam_weights() # [1, C, 1, 1, 1]
    # Use the activations from the model (retained grad)
    activations = model.activations.detach()
    
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(cam, size=raw_cube.shape[1:], mode='trilinear').squeeze().detach().cpu().numpy()
    
    if cam.max() > 0:
        cam /= cam.max()
        
    return cam, raw_cube.detach().cpu().numpy()

def visualize_expert_analysis(patient_id):
    """Produces an expert clinical visualization with a 2x2 grid and diagnostic dark theme."""
    # Resolve device: 'auto' selects the best available (CUDA > MPS > CPU)
    _device_cfg = config.INFERENCE_DEVICE
    device = helper.get_device() if _device_cfg == "auto" else torch.device(_device_cfg)
    result = compute_gradcam(patient_id, device=device)
    if result is None:
        return None
        
    cam, cube = result
    # Select Z-slice by highest MEAN activation per slice (more robust than absolute max).
    # The absolute max voxel can be a peripheral outlier; the mean captures the dominant focus region.
    z_slice = int(np.argmax(np.mean(cam, axis=(1, 2))))
    
    # Large 2x3 grid with dark background
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor=_STYLE["fig_bg"])
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
    
    axes = axes.flatten()
    
    # 1. ANATOMY (Raw Input Reference)
    axes[0].imshow(cube[0, z_slice], cmap='gray')
    axes[0].set_title(f"1. ANATOMY (Z={z_slice})", fontsize=11, fontweight=700, color=_STYLE["title"])
    axes[0].axis('off')
    
    # 2. AI ATTENTION (Heatmap)
    axes[1].imshow(cube[0, z_slice], cmap='gray')
    axes[1].imshow(cam[z_slice], cmap='jet', alpha=0.65)
    axes[1].set_title("2. AI ATTENTION FOCUS", fontsize=11, fontweight=800, color=_STYLE["accent"])
    axes[1].axis('off')

    # 3. C4: SIGNAL PEAKS (Max)
    c4_slice = cube[3, z_slice]
    vmax_c4 = np.percentile(c4_slice, 99) if c4_slice.max() > 0 else 1.0
    axes[2].imshow(c4_slice, cmap='hot', vmax=vmax_c4)
    axes[2].set_title("3. SIGNAL PEAKS (MAX)", fontsize=10, color=_STYLE["subtitle"])
    axes[2].axis('off')

    # 4. C1: MORPHOLOGY (Avg)
    axes[3].imshow(cube[0, z_slice], cmap='bone')
    axes[3].set_title("4. MORPHOLOGY (AVG)", fontsize=10, color=_STYLE["subtitle"])
    axes[3].axis('off')

    # 5. C2: HETEROGENEITY (Variance)
    c2_slice = cube[1, z_slice]
    c2_active = c2_slice[c2_slice > 0.01]
    vmax_c2 = np.percentile(c2_active, 98) if len(c2_active) > 10 else (c2_slice.max() + 1e-9)
    axes[4].imshow(c2_slice, cmap='magma', vmax=vmax_c2)
    title_c2 = "5. HETEROGENEITY" if vmax_c2 > 50 else "5. HETEROGENEITY (Low Signal)"
    axes[4].set_title(title_c2, fontsize=10, color=_STYLE["subtitle"])
    axes[4].axis('off')

    # 6. C3: KINETICS (Ratio)
    c3_slice = cube[2, z_slice]
    c3_active = c3_slice[np.abs(c3_slice) > 0.01] 
    vmax_c3 = np.percentile(np.abs(c3_active), 98) if len(c3_active) > 10 else (c3_slice.max() + 1e-9)
    axes[5].imshow(c3_slice, cmap='cividis', vmin=-vmax_c3, vmax=vmax_c3)
    title_c3 = "6. KINETICS (RATIO)" if vmax_c3 > 1 else "6. KINETICS (Low Signal)"
    axes[5].set_title(title_c3, fontsize=10, color=_STYLE["subtitle"])
    axes[5].axis('off')
    
    fig.suptitle(f"Bio-Lattice Expert Console v2.5 · {patient_id}", 
                 fontsize=16, fontweight=700, color=_STYLE["title"])
    
    return fig

if __name__ == "__main__":
    # Test execution
    pid = "Breast_MRI_001"
    fig = visualize_expert_analysis(pid)
    if fig:
        plt.show()
    else:
        print("Required files not found.")
