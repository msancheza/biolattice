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
from core import helper
from core.run_logs import RunLogWriter 

if not os.path.exists(config.PATH_MICRO_CUBES):
    os.makedirs(config.PATH_MICRO_CUBES)

_extraction_logger = RunLogWriter(config.PATH_LOGS)


def log_event(message):
    """Appends a line to ``PATH_LOGS`` (extraction / registration audit trail)."""
    _extraction_logger.event(message)


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
    # Bio-Lattice v2.7 Scientific Hygiene: Use 'nearest' for small anatomical drifts
    # to avoid blank slabs, but keep 'constant' for large shifts to trigger the Zero-Gate.
    max_dim = max(v_pre.shape)
    shift_magnitude = np.linalg.norm(shifts)
    rel_shift = shift_magnitude / max_dim
    is_small_drift = rel_shift < config.EXTRACTION_MAX_RELATIVE_SHIFT
    
    shift_mode = 'nearest' if is_small_drift else 'constant'
    v_pre_reg = scipy.ndimage.shift(v_pre, shifts, mode=shift_mode, cval=0.0, order=1)
    
    # 8. Compute QA Metric: Correlation Coefficient
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
    t_pre_iso = F.interpolate(
        t_pre,
        size=(sz, sy, sx),
        mode=config.PRE_POST_INTERPOLATE_MODE,
        align_corners=False,
    )
    t_post_iso = F.interpolate(
        t_post,
        size=(sz, sy, sx),
        mode=config.PRE_POST_INTERPOLATE_MODE,
        align_corners=False,
    )

    # 4.1 KINETICS (Percent Enhancement with Signed Log-Compression v2.5.3)
    eps = 1e-6
    # (Post - Pre) / Pre -> Capture the relative biological response
    ratio = (t_post_iso - t_pre_iso) / (torch.abs(t_pre_iso) + eps)
    # Bio-Lattice v2.5.3 Policy: Use signed log1p to compress noise while preserving 
    # the hierarchy of magnitudes (essential for distinguishing levels of malignancy).
    kinetics_raw = torch.sign(ratio) * torch.log1p(torch.abs(ratio))

    # --- Bio-Lattice v2.7.4 Authority Gate: ROI Zero-Masking ---
    # Detect if registration vacuum (zeros) entered the tissue volume.
    # We count zeros in the pre-interpolation T1w pre-contrast map.
    zero_mask = (t_pre_iso == 0).float()
    roi_zero_fraction = float(torch.mean(zero_mask).item())

    # Mandatory hygiene: Clamp kinetics to protect the gradient.
    c_limit = config.EXTRACTION_KINETICS_CLAMP
    kinetics_raw = torch.clamp(kinetics_raw, -c_limit, c_limit)
    
    # 4.2 HETEROGENEITY (Variance with Denoising)
    t_denoised_real = apply_denoising(t_post_iso, sigma=config.VAR_DENOISING_SIGMA)
    k = int(getattr(config, "C2_LOCAL_VAR_KERNEL", 3))
    p = k // 2
    mean_r = F.avg_pool3d(t_denoised_real, kernel_size=k, stride=1, padding=p)
    mean_sq_r = F.avg_pool3d(t_denoised_real**2, kernel_size=k, stride=1, padding=p)
    # Apply log1p to compress the dynamic range of variance (Texture normalization)
    c2_raw = torch.log1p(torch.relu(mean_sq_r - (mean_r**2)))

    # --- 5. Padding Logic (The Kaleidoscope Fix + Semantic Consistency) ---
    pad_z, pad_y, pad_x = size - sz, size - sy, size - sx
    padding = (
        pad_x // 2, pad_x - (pad_x // 2),
        pad_y // 2, pad_y - (pad_y // 2),
        pad_z // 2, pad_z - (pad_z // 2)
    )
    
    can_reflect = (
        padding[0] < t_post_iso.shape[4] and padding[1] < t_post_iso.shape[4] and
        padding[2] < t_post_iso.shape[3] and padding[3] < t_post_iso.shape[3] and
        padding[4] < t_post_iso.shape[2] and padding[5] < t_post_iso.shape[2]
    )

    # v2.5.2 Policy: Functional channels (C2, C3) MUST use constant padding.
    # C1 and C4 (Anatomy/Peaks) follow the Kaleidoscope safety policy.
    roi_fraction = (sz * sy * sx) / (size ** 3)
    struct_pad_mode = 'reflect' if (can_reflect and roi_fraction >= config.ROI_MIN_FRAC_FOR_REFLECT) else 'constant'

    # Apply padding
    t_post_final = F.pad(t_post_iso, padding, mode=struct_pad_mode)
    kinetics_final = F.pad(kinetics_raw, padding, mode='constant', value=0.0)
    c2_final = F.pad(c2_raw, padding, mode='constant', value=0.0)

        
    # --- 6. Channel Assembly (Final Adaptive Pooling) ---
    c1 = F.adaptive_avg_pool3d(t_post_final, output_size=(size, size, size))
    c2 = F.adaptive_avg_pool3d(c2_final, output_size=(size, size, size))
    c3 = F.adaptive_avg_pool3d(kinetics_final, output_size=(size, size, size))
    
    # Bio-Lattice v2.6.3 Orthogonality: Substract C1 (Avg) from MaxPool.
    # This isolates ONLY the Peak signals (vascular highlights) from the anatomical mass.
    # Added ReLU guard to ensure no floating-point noise produces negative values.
    c4_max = F.adaptive_max_pool3d(t_post_final, output_size=(size, size, size))
    c4 = torch.relu(c4_max - c1)
    
    cube = torch.cat([c1, c2, c3, c4], dim=1).squeeze(0)  # [4, S, S, S]
    return cube, roi_zero_fraction

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
        print("the series found are: ", series_found, " for: ", p_id)
        # Heuristic matching
        for s in series_found:
            if helper.series_is_pre_contrast(s['desc']):
                path_pre = s['path']
            # Bio-Lattice v2.9.1: Prioritize the FIRST post-contrast phase (peak enhancement)
            if helper.series_is_post_contrast(s['desc']) and path_post is None:
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

                micro_cube, zero_fraction = weave_4d_micro_cube(
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
                        "roi_zero_fraction": zero_fraction,
                        **channel_stats,
                    },
                )
                
                
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