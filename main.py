"""
Micro-cube factory: Duke-oriented DICOM pairing, shared ROI crop, and 32³ weave.

Registration is limited to trilinear resize of the pre ROI to the post ROI shape when
dims differ; see README *What main.py actually does* for series heuristics, ROI
indexing, and intensity handling.
"""
import os
import pandas as pd
import pydicom
import numpy as np
import torch
import torch.nn.functional as F

import config
from visualizer import visualize_micro_cube

if not os.path.exists(config.PATH_MICRO_CUBOS):
    os.makedirs(config.PATH_MICRO_CUBOS)

def get_3d_volume(ruta_serie):
    archivos_paths = [
        os.path.join(ruta_serie, f)
        for f in os.listdir(ruta_serie)
        if f.endswith(config.DICOM_EXTENSION)
    ]
    slices = [pydicom.dcmread(f) for f in archivos_paths]
    # Sort by Z position (crucial so the cube is not scrambled)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return np.stack([f.pixel_array for f in slices]).astype(np.float32)

def crop_roi_with_padding(volumen, coords, padding_pct=None):
    if padding_pct is None:
        padding_pct = config.ROI_PADDING_FRACTION
    """ Isolates the tumor but extracts an extra 20% to capture spicules (Peritumoral Halo). """
    z1, z2, y1, y2, x1, x2 = coords
    
    dz, dy, dx = max(1, z2 - z1), max(1, y2 - y1), max(1, x2 - x1)
    
    pad_z, pad_y, pad_x = int(dz * padding_pct), int(dy * padding_pct), int(dx * padding_pct)
    
    # Safe boundary cropping + Padding
    z1_p, z2_p = max(0, z1 - pad_z), min(volumen.shape[0], z2 + pad_z)
    y1_p, y2_p = max(0, y1 - pad_y), min(volumen.shape[1], y2 + pad_y)
    x1_p, x2_p = max(0, x1 - pad_x), min(volumen.shape[2], x2 + pad_x)
    
    recorte = volumen[z1_p:z2_p, y1_p:y2_p, x1_p:x2_p]
    # Return pure 5D tensor [Batch, Channels, Depth, Height, Width]
    return torch.from_numpy(recorte).unsqueeze(0).unsqueeze(0)

def weave_4d_micro_cube(t_pre, t_post, size=None):
    """Compresses to MICRO_CUBE_SIZE^3 with the three functional channels."""
    if size is None:
        size = config.MICRO_CUBE_SIZE
    if t_pre.shape != t_post.shape:
        t_pre = F.interpolate(
            t_pre,
            size=t_post.shape[2:],
            mode=config.PRE_POST_INTERPOLATE_MODE,
            align_corners=False,
        )
        
    # CHANNEL 1: STRUCTURE (Adaptive Max Pooling: Preserves clinical radiation high peaks)
    c1 = F.adaptive_max_pool3d(t_post, output_size=(size, size, size))
    
    # CHANNEL 2: REAL MATHEMATICAL VARIANCE/CHAOS (E[X^2] - E[X]^2)
    mean = F.adaptive_avg_pool3d(t_post, output_size=(size, size, size))
    mean_sq = F.adaptive_avg_pool3d(t_post**2, output_size=(size, size, size))
    c2 = torch.relu(mean_sq - (mean**2)) # ReLU eliminates floating point negative noise
    
    # CHANNEL 3: CONTRAST KINETICS (Exact spatial wash-in)
    c3 = F.adaptive_avg_pool3d(t_post - t_pre, output_size=(size, size, size))
    
    return torch.cat([c1, c2, c3], dim=1).squeeze(0) # Dense tensor [3, 32, 32, 32]

def process_dataset():
    df_boxes = pd.read_excel(config.PATH_ANNOTATION_BOXES)
    successes = 0
    
    for _, fila in df_boxes.iterrows():
        p_id = fila[config.COL_PATIENT_ID]
        folder_patient = os.path.join(config.PATH_RAW, p_id)
        
        if not os.path.exists(folder_patient): continue
            
        print(f"Analyzing {p_id}...")
        
        # 1. Map all subfolders containing DICOMs and their descriptions
        path_pre, path_post = None, None
        series_found = []

        for root, _, files in os.walk(folder_patient):
            dicoms = [f for f in files if f.endswith('.dcm')]
            if dicoms:
                ds = pydicom.dcmread(os.path.join(root, dicoms[0]))
                desc = getattr(ds, 'SeriesDescription', '').lower()
                series_found.append((root, desc))

        for d_path, desc in series_found:
            if config.series_is_pre_contrast(desc):
                path_pre = d_path
            if config.series_is_post_contrast(desc):
                path_post = d_path

        # 3. Processing if both phases were found
        if path_pre and path_post:
            try:
                v_pre = get_3d_volume(path_pre)
                v_post = get_3d_volume(path_post)
                
                # Coordinates (0-indexed adjustment)
                coords = (int(fila['Start Slice'])-1, int(fila['End Slice']), 
                          int(fila['Start Row'])-1, int(fila['End Row']), 
                          int(fila['Start Column'])-1, int(fila['End Column']))

                # 1. Anatomical extraction (Pre-aligned local Crops)
                t_post_roi = crop_roi_with_padding(v_post, coords)
                t_pre_roi = crop_roi_with_padding(v_pre, coords)

                # 2. Mathematical construction of the 4D Bio-Lattice
                micro_cubo = weave_4d_micro_cube(t_pre_roi, t_post_roi)

                output_path = os.path.join(
                    config.PATH_MICRO_CUBOS, f"{p_id}{config.LATTICE_FILE_SUFFIX}"
                )
                torch.save(micro_cubo, output_path)
                successes += 1
                print(f"Micro-cube successfully generated for {p_id}")
                
                if config.SHOW_VISUALIZER_AFTER_SAVE:
                    visualize_micro_cube(output_path)
                
            except Exception as e:
                print(f"Error processing volumes for {p_id}: {e}")
        else:
            reason = "Missing PRE" if not path_pre else "Missing POST"
            if not path_pre and not path_post: reason = "Missing BOTH"
            print(f"Skipped {p_id}: {reason} (Descriptions found: {[s[1] for s in series_found]})")

    print(f"\n✨ Process finished. {successes} micro-cubes generated in {config.PATH_MICRO_CUBOS}")

if __name__ == "__main__":
    process_dataset()