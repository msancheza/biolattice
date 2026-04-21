"""
Volume registration helpers for the extraction pipeline.

Handles 3D alignment and resampling of medical volumes using phase correlation (FFT) 
and trilinear interpolation. This module is format-agnostic and operates on 
numerical arrays to ensure spatial consistency between different acquisition phases.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

import config


def resample_volume_to_shape(volume: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Trilinear resampling of a (D, H, W) volume to ``target_shape``."""
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


def register_volumes_fft(v_pre, v_post, patient_id=None, log_event=None):
    """
    Finds the 3D translation between v_pre and v_post using phase correlation.
    Aligns v_pre (moving) to v_post (fixed).
    """
    shape_pre = tuple(v_pre.shape)
    shape_post = tuple(v_post.shape)
    if shape_pre != shape_post:
        tag = patient_id if patient_id is not None else "unknown"
        msg = (
            f"[REG RESAMPLE] {tag} | v_pre {shape_pre} -> {shape_post} "
            f"(trilinear, fixed=post, align_corners=False)"
        )
        if log_event is not None:
            log_event(msg)
        v_pre = resample_volume_to_shape(v_pre, shape_post)

    t_pre = torch.from_numpy(v_pre).float()
    t_post = torch.from_numpy(v_post).float()

    f_pre = torch.fft.fftn(t_pre)
    f_post = torch.fft.fftn(t_post)

    cps = f_post * torch.conj(f_pre)
    cps /= (torch.abs(cps) + 1e-8)

    r = torch.fft.ifftn(cps).real
    peak_idx = torch.argmax(r)
    coords = np.unravel_index(peak_idx.item(), r.shape)

    shifts = []
    for i, dim_size in enumerate(r.shape):
        s = coords[i]
        if s > dim_size // 2:
            s -= dim_size
        shifts.append(int(s))

    max_dim = max(v_pre.shape)
    shift_magnitude = np.linalg.norm(shifts)
    rel_shift = shift_magnitude / max_dim
    is_small_drift = rel_shift < config.EXTRACTION_MAX_RELATIVE_SHIFT

    shift_mode = "nearest" if is_small_drift else "constant"
    v_pre_reg = scipy.ndimage.shift(v_pre, shifts, mode=shift_mode, cval=0.0, order=1)
    corr = np.corrcoef(v_pre_reg.flatten(), v_post.flatten())[0, 1]
    return v_pre_reg, shifts, corr
