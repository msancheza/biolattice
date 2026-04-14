"""
Micro-cube factory: Duke-oriented DICOM pairing, shared ROI crop, and 64³ weave (see config.MICRO_CUBE_SIZE).

Registration now includes a 3D Phase Correlation (FFT) step to align pre and post
volumes before ROI extraction, ensuring accurate kinetics (Channel 3).
See README *What main.py actually does* for series heuristics, ROI
indexing, and intensity handling.
"""
import os
import json
from datetime import datetime
import pandas as pd
import pydicom
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

import config
from visualizer import visualize_micro_cube

if not os.path.exists(config.PATH_MICRO_CUBES):
    os.makedirs(config.PATH_MICRO_CUBES)

def log_event(message):
    """ Appends a message to the extraction_log.txt file. """
    with open(config.PATH_LOGS, "a") as f:
        f.write(f"{message}\n")


def build_extraction_audit_path():
    """Returns a timestamped JSONL path for this extraction run."""
    os.makedirs(config.PATH_EXTRACTION_AUDIT_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(config.PATH_EXTRACTION_AUDIT_DIR, f"extraction_audit_{run_id}.jsonl")


def append_audit_record(audit_path, record):
    """Appends one structured record (JSONL) for auditability."""
    if not config.AUDIT_WRITE_JSONL:
        return
    record = dict(record)
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(audit_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def to_repo_relative_path(path_value):
    """Return repo-relative path when possible (avoids machine-specific absolute paths)."""
    if not path_value:
        return ""
    try:
        rel = os.path.relpath(path_value, config.PROJECT_ROOT)
        if not rel.startswith(".."):
            return rel
    except Exception:
        pass
    return path_value


def tensor_channel_stats(micro_cube):
    """Returns lightweight per-channel stats for quick QC."""
    stats = {}
    for idx in range(micro_cube.shape[0]):
        channel = micro_cube[idx]
        stats[f"c{idx + 1}_mean"] = float(torch.mean(channel).item())
        stats[f"c{idx + 1}_std"] = float(torch.std(channel).item())
        stats[f"c{idx + 1}_min"] = float(torch.min(channel).item())
        stats[f"c{idx + 1}_max"] = float(torch.max(channel).item())
    return stats


def metadata_physical_is_valid(spacing, thickness):
    """
    Lightweight sanity check for physical metadata used in weave.
    Keeps the same pipeline, but flags impossible values for review.
    """
    if spacing is None or len(spacing) < 2:
        return False, "invalid PixelSpacing shape"
    if thickness is None:
        return False, "missing SliceThickness"
    sx, sy = float(spacing[0]), float(spacing[1])
    sz = float(thickness)
    if sx <= 0 or sy <= 0 or sz <= 0:
        return False, "non-positive spacing/thickness"
    if sx > 10 or sy > 10 or sz > 20:
        return False, "outlier spacing/thickness"
    return True, ""

def get_3d_volume(series_dir):
    dicom_paths = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.endswith(config.DICOM_EXTENSION)
    ]
    slices = [pydicom.dcmread(f) for f in dicom_paths]
    # Sort by Z position (crucial so the cube is not scrambled)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # --- Z-Integrity Check ---
    if len(slices) > 1:
        z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
        z_diffs = np.diff(z_positions)
        median_diff = np.median(z_diffs)
        max_deviation = np.max(np.abs(z_diffs - median_diff))
        
        # Threshold: deviation > 50% of the median slice thickness implies a missing slice
        if median_diff > 0.05 and max_deviation > (0.5 * median_diff):
            log_event(f"[{os.path.basename(os.path.dirname(series_dir))} WARNING] DICOM Z-Gap inconsistency. Missing slice likely! Dev: {max_deviation:.2f}mm")

    # Apply Rescale Slope / Intercept (Ensures PV = SV * Slope + Intercept)
    vols = []
    for s in slices:
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        vols.append(s.pixel_array.astype(np.float32) * slope + intercept)
        
    return np.stack(vols)


def resample_volume_to_shape(volume: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    Trilinear resampling of a (D, H, W) volume to ``target_shape``.
    Returns ``float32`` contiguous array (matches ``PRE_POST_INTERPOLATE_MODE`` in config).
    """
    if volume.shape == target_shape:
        return np.ascontiguousarray(volume.astype(np.float32, copy=False))
    t = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t_out = F.interpolate(
        t,
        size=target_shape,
        mode=config.PRE_POST_INTERPOLATE_MODE,
        align_corners=False,
    )
    return t_out.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def register_volumes_fft(v_pre, v_post, patient_id=None):
    """
    Finds the 3D translation (shift) between v_pre and v_post using Phase Correlation.
    Aligns v_pre (moving) to v_post (fixed).

    If full-volume shapes differ, v_pre is resampled to v_post's shape **before** FFT
    so phase correlation is well-defined. Resampling is logged to ``PATH_LOGS``.
    """
    shape_pre = tuple(v_pre.shape)
    shape_post = tuple(v_post.shape)
    if shape_pre != shape_post:
        tag = patient_id if patient_id is not None else "unknown"
        msg = (
            f"[REG RESAMPLE] {tag} | v_pre {shape_pre} -> {shape_post} "
            f"(trilinear, fixed=post, align_corners=False)"
        )
        print(f"  {msg}")
        log_event(msg)
        v_pre = resample_volume_to_shape(v_pre, shape_post)

    # 1. Convert to torch for FFT
    t_pre = torch.from_numpy(v_pre).float()
    t_post = torch.from_numpy(v_post).float()

    # 2. Compute 3D FFTs
    f_pre = torch.fft.fftn(t_pre)
    f_post = torch.fft.fftn(t_post)

    # 3. Phase Correlation: cross-power spectrum
    cps = (f_post * torch.conj(f_pre))
    cps /= (torch.abs(cps) + 1e-8)
    
    # 4. Inverse FFT to find the impulse response
    r = torch.fft.ifftn(cps).real
    
    # 5. Locate the peak
    peak_idx = torch.argmax(r)
    coords = np.unravel_index(peak_idx.item(), r.shape)
    
    # 6. Convert to signed shifts (handling wrap-around)
    shifts = []
    for i, dim_size in enumerate(r.shape):
        s = coords[i]
        if s > dim_size // 2:
            s -= dim_size
        shifts.append(int(s))
    
    # 7. Apply physical affine shift (non-circular)
    # Bio-Lattice v2.1: Prevents phantom noise from wrap-around in the border (Breast MRI context)
    v_pre_reg = scipy.ndimage.shift(v_pre, shifts, mode='constant', cval=0.0)
    
    # 8. Compute QA Metric: Correlation Coefficient
    # Flat correlation as a fast proxy for alignment quality
    corr = np.corrcoef(v_pre_reg.flatten(), v_post.flatten())[0, 1]
    
    return v_pre_reg, shifts, corr

def crop_roi_with_padding(volume, coords, padding_pct=None):
    """Crop ROI with optional peritumoral padding (halo) around the tumor box."""
    if padding_pct is None:
        padding_pct = config.ROI_PADDING_FRACTION
    z1, z2, y1, y2, x1, x2 = coords
    
    dz, dy, dx = max(1, z2 - z1), max(1, y2 - y1), max(1, x2 - x1)
    
    pad_z, pad_y, pad_x = int(dz * padding_pct), int(dy * padding_pct), int(dx * padding_pct)
    
    # Safe boundary cropping + Padding
    z1_p, z2_p = max(0, z1 - pad_z), min(volume.shape[0], z2 + pad_z)
    y1_p, y2_p = max(0, y1 - pad_y), min(volume.shape[1], y2 + pad_y)
    x1_p, x2_p = max(0, x1 - pad_x), min(volume.shape[2], x2 + pad_x)

    cropped = volume[z1_p:z2_p, y1_p:y2_p, x1_p:x2_p]
    # Return 5D tensor [N, C, D, H, W]
    return torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0)

def apply_denoising(t_vol, sigma=0.5):
    """ Simple Gaussian Blur 3D using a separable 3x3x3 kernel in Torch. """
    if sigma <= 0: return t_vol
    # Approximate a 3x3 kernel for sigma=0.5: [0.25, 0.5, 0.25]
    k = torch.tensor([0.25, 0.5, 0.25], device=t_vol.device)
    def blur_axis(t, axis):
        t = t.transpose(axis, -1)
        shape = t.shape
        t = t.reshape(-1, 1, shape[-1])
        t = F.conv1d(t, k.view(1, 1, 3), padding=1)
        return t.reshape(shape).transpose(axis, -1)
    
    t_vol = blur_axis(t_vol, 2)
    t_vol = blur_axis(t_vol, 3)
    t_vol = blur_axis(t_vol, 4)
    return t_vol

def weave_4d_micro_cube(t_pre, t_post, spacing, thickness, size=None):
    """
    Compresses to MICRO_CUBE_SIZE^3 with the three functional channels.
    Bio-Lattice v2.1: Preserves real physical aspect ratio using padding (fixed 'fake isotropy').
    """
    if size is None:
        size = config.MICRO_CUBE_SIZE
        
    # --- 1. Physical Scale Calculation ---
    dz, dy, dx = thickness, spacing[1], spacing[0]
    D, H, W = t_post.shape[2:]
    Lz, Ly, Lx = D * dz, H * dy, W * dx
    Lmax = max(Lz, Ly, Lx)
    
    # --- 2. Determine target shape for isotropic resampling ---
    sz = max(1, int(round(size * Lz / Lmax)))
    sy = max(1, int(round(size * Ly / Lmax)))
    sx = max(1, int(round(size * Lx / Lmax)))
    
    # --- 3. Resample PRE and POST to the intermediate isotropic shape ---
    t_pre = F.interpolate(
        t_pre,
        size=(sz, sy, sx),
        mode=config.PRE_POST_INTERPOLATE_MODE,
        align_corners=False,
    )
    t_post = F.interpolate(
        t_post,
        size=(sz, sy, sx),
        mode=config.PRE_POST_INTERPOLATE_MODE,
        align_corners=False,
    )
    
    # --- 4. Central Padding to reach exactly [size, size, size] ---
    pad_z = size - sz
    pad_y = size - sy
    pad_x = size - sx
    
    # (W_left, W_right, H_top, H_bottom, D_front, D_back)
    padding = (
        pad_x // 2, pad_x - (pad_x // 2),
        pad_y // 2, pad_y - (pad_y // 2),
        pad_z // 2, pad_z - (pad_z // 2)
    )
    t_pre_final = F.pad(t_pre, padding, mode='constant', value=0)
    t_post_final = F.pad(t_post, padding, mode='constant', value=0)
        
    # CHANNEL 1: HYBRID STRUCTURE (Mixed Max for peaks + Avg for mass)
    # Applied on the padded isotropic volume
    c1_max = F.adaptive_max_pool3d(t_post_final, output_size=(size, size, size))
    c1_avg = F.adaptive_avg_pool3d(t_post_final, output_size=(size, size, size))
    alpha = config.HYBRID_STRUCTURAL_RATIO
    c1 = alpha * c1_max + (1 - alpha) * c1_avg
    
    # CHANNEL 2: HETEROGENEITY with Denoising Pre-processing
    # Use local variance (sliding window), not identity pooling over same-size tensor.
    t_denoised = apply_denoising(t_post_final, sigma=config.VAR_DENOISING_SIGMA)
    k = int(getattr(config, "C2_LOCAL_VAR_KERNEL", 3))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    p = k // 2
    mean = F.avg_pool3d(t_denoised, kernel_size=k, stride=1, padding=p)
    mean_sq = F.avg_pool3d(t_denoised**2, kernel_size=k, stride=1, padding=p)
    c2 = torch.relu(mean_sq - (mean**2))
    
    # CHANNEL 3: CONTRAST KINETICS (Exact spatial wash-in)
    c3 = F.adaptive_avg_pool3d(t_post_final - t_pre_final, output_size=(size, size, size))
    
    return torch.cat([c1, c2, c3], dim=1).squeeze(0)  # [3, S, S, S], S = config.MICRO_CUBE_SIZE

def process_dataset():
    df_boxes = pd.read_excel(config.PATH_ANNOTATION_BOXES)
    successes = 0
    audit_path = build_extraction_audit_path()
    print(f"Audit log path: {audit_path}")
    
    for _, row in df_boxes.iterrows():
        p_id = row[config.COL_PATIENT_ID]
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
                s_num = getattr(ds, 'SeriesNumber', 0)
                series_found.append({'path': root, 'desc': desc, 'num': s_num})

        # Sort series by Number to maintain chronological order
        series_found.sort(key=lambda x: x['num'])
        
        # Heuristic matching
        for s in series_found:
            if config.series_is_pre_contrast(s['desc']):
                path_pre = s['path']
            if config.series_is_post_contrast(s['desc']):
                path_post = s['path']
        
        # Fallback: if names are ambiguous (all 'ax dyn'), use order
        if not path_post or not path_pre:
            dynamics = [s for s in series_found if ("dyn" in s['desc'] or "vibrant" in s['desc'])]
            if len(dynamics) >= 2:
                if not path_pre: path_pre = dynamics[0]['path']
                if not path_post: path_post = dynamics[1]['path']

        selected_pre_desc = next((s["desc"] for s in series_found if s["path"] == path_pre), "")
        selected_post_desc = next((s["desc"] for s in series_found if s["path"] == path_post), "")

        # 3. Processing if both phases were found
        if path_pre and path_post:
            try:
                v_pre = get_3d_volume(path_pre)
                v_post = get_3d_volume(path_post)
                
                # REGISTRATION: Align PRE to POST using Phase Correlation
                print(f"  Registering volumes for {p_id}...")
                v_pre, shifts, corr = register_volumes_fft(v_pre, v_post, patient_id=p_id)
                
                # Registration QA Gate
                status = "OK" if corr >= config.REGISTRATION_MIN_CORRELATION else "WARNING"
                log_msg = f"[{status}] {p_id} | Corr: {corr:.4f} | Shifts: {shifts}"
                print(f"  {log_msg}")
                
                if status == "WARNING":
                    log_event(log_msg)
                
                # Coordinates (0-indexed adjustment)
                coords = (int(row['Start Slice'])-1, int(row['End Slice']), 
                          int(row['Start Row'])-1, int(row['End Row']), 
                          int(row['Start Column'])-1, int(row['End Column']))

                # 1. Anatomical extraction (Pre-aligned local Crops)
                t_post_roi = crop_roi_with_padding(v_post, coords)
                t_pre_roi = crop_roi_with_padding(v_pre, coords)

                # 2. Mathematical construction of the 4D Bio-Lattice
                # Extracts PixelSpacing/Thickness metadata BEFORE weaving to ensure real aspect ratio.
                try:
                    dicom_files = [f for f in os.listdir(path_post) if f.endswith(config.DICOM_EXTENSION)]
                    ds_meta = pydicom.dcmread(os.path.join(path_post, dicom_files[0]))
                    pixel_spacing = [float(x) for x in getattr(ds_meta, "PixelSpacing", [1.0, 1.0])]
                    slice_thickness = float(getattr(ds_meta, "SliceThickness", 1.0))
                except Exception:
                    pixel_spacing = [1.0, 1.0]
                    slice_thickness = 1.0

                micro_cube = weave_4d_micro_cube(
                    t_pre_roi, t_post_roi, 
                    spacing=pixel_spacing, 
                    thickness=slice_thickness
                )
                channel_stats = tensor_channel_stats(micro_cube)
                meta_ok, meta_issue = metadata_physical_is_valid(pixel_spacing, slice_thickness)
                if not meta_ok:
                    status = "REVIEW"
                    issue_msg = f"[REVIEW] {p_id} | Metadata issue: {meta_issue} | spacing={pixel_spacing}, thickness={slice_thickness}"
                    print(f"  {issue_msg}")
                    log_event(issue_msg)
                same_series_pair = os.path.normpath(path_pre) == os.path.normpath(path_post)
                if same_series_pair:
                    status = "REVIEW"
                    pair_msg = f"[REVIEW] {p_id} | PRE and POST mapped to same series path"
                    print(f"  {pair_msg}")
                    log_event(pair_msg)
                
                lattice_obj = {
                    "tensor": micro_cube,
                    "meta": {
                        "voxel_spacing": pixel_spacing,
                        "slice_thickness": slice_thickness,
                        "orig_shape": list(v_post.shape),
                        "roi_coords": coords
                    }
                }

                output_path = os.path.join(
                    config.PATH_MICRO_CUBES, f"{p_id}{config.LATTICE_FILE_SUFFIX}"
                )
                torch.save(lattice_obj, output_path)
                successes += 1
                print(f"Micro-cube successfully generated for {p_id}")
                append_audit_record(
                    audit_path,
                    {
                        "patient_id": p_id,
                        "status": status,
                        "reason": "",
                        "path_pre": to_repo_relative_path(path_pre),
                        "path_post": to_repo_relative_path(path_post),
                        "same_series_pair": bool(same_series_pair),
                        "pre_desc": selected_pre_desc,
                        "post_desc": selected_post_desc,
                        "corr_global": float(corr),
                        "shifts": shifts,
                        "meta_valid": bool(meta_ok),
                        "meta_issue": meta_issue,
                        "micro_cube_path": output_path,
                        "micro_cube_shape": list(micro_cube.shape),
                        "orig_post_shape": list(v_post.shape),
                        "roi_coords": list(coords),
                        **channel_stats,
                    },
                )
                
                if config.SHOW_VISUALIZER_AFTER_SAVE:
                    visualize_micro_cube(output_path)
                
            except Exception as e:
                err_msg = f"[ERROR] {p_id} | Exception: {e}"
                print(f"  {err_msg}")
                log_event(err_msg)
                append_audit_record(
                    audit_path,
                    {
                        "patient_id": p_id,
                        "status": "REVIEW",
                        "reason": str(e),
                        "path_pre": to_repo_relative_path(path_pre),
                        "path_post": to_repo_relative_path(path_post),
                        "pre_desc": selected_pre_desc,
                        "post_desc": selected_post_desc,
                        "corr_global": None,
                        "shifts": [],
                        "micro_cube_path": "",
                        "micro_cube_shape": [],
                        "orig_post_shape": [],
                        "roi_coords": [],
                    },
                )
        else:
            reason = "Missing PRE" if not path_pre else "Missing POST"
            if not path_pre and not path_post: reason = "Missing BOTH"
            desc_str = str([s['desc'] for s in series_found])
            log_msg = f"[SKIP] {p_id} | {reason} | Found: {desc_str}"
            print(f"Skipped {p_id}: {reason}")
            log_event(log_msg)
            append_audit_record(
                audit_path,
                {
                    "patient_id": p_id,
                    "status": "REVIEW",
                    "reason": reason,
                    "path_pre": to_repo_relative_path(path_pre),
                    "path_post": to_repo_relative_path(path_post),
                    "pre_desc": selected_pre_desc,
                    "post_desc": selected_post_desc,
                    "corr_global": None,
                    "shifts": [],
                    "micro_cube_path": "",
                    "micro_cube_shape": [],
                    "orig_post_shape": [],
                    "roi_coords": [],
                },
            )

    print(f"\n✨ Process finished. {successes} micro-cubes generated in {config.PATH_MICRO_CUBES}")
    print(f"Extraction audit saved to: {audit_path}")

if __name__ == "__main__":
    process_dataset()