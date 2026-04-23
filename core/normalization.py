"""
Shared normalization helpers for Bio-Lattice tensors.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import config


def normalize_cube_per_channel(cube: torch.Tensor) -> torch.Tensor:
    """
    Apply per-channel Z-score scaling without mutating the input tensor.
    """
    cube = cube.clone().float()
    for channel_idx in range(cube.shape[0]):
        channel = cube[channel_idx]
        std = torch.std(channel)
        if std > config.NORMALIZE_EPS:
            cube[channel_idx] = (channel - torch.mean(channel)) / std
    return cube


def _channel_quantile(channel: torch.Tensor, q: float) -> float:
    flat = channel.detach().reshape(-1).float().cpu()
    if flat.numel() == 0:
        return 0.0
    return float(torch.quantile(flat, torch.tensor(q, dtype=torch.float32)).item())


def _clip_channel(channel: torch.Tensor, percentile: float | None = None) -> torch.Tensor:
    clip_q = percentile if percentile is not None else config.NORMALIZATION_CLIP_PERCENTILE
    clip_q = min(max(float(clip_q), 0.50), 0.999)
    limit = abs(_channel_quantile(torch.abs(channel), clip_q))
    if limit <= config.NORMALIZE_EPS:
        return channel.clone().float()
    return torch.clamp(channel.float(), min=-limit, max=limit)


def apply_global_scale(channel: torch.Tensor, fitted_stats: dict[str, float] | None) -> torch.Tensor:
    """
    Apply fixed population-level scaling to one channel.
    """
    channel = channel.clone().float()
    if not fitted_stats:
        return channel
    center = float(fitted_stats.get("mean", 0.0))
    scale = abs(float(fitted_stats.get("std", 0.0)))
    if scale <= config.NORMALIZE_EPS:
        return channel
    return (channel - center) / scale


def apply_robust_scale(channel: torch.Tensor, fitted_stats: dict[str, float] | None = None) -> torch.Tensor:
    """
    Apply conservative robust scaling to one channel.
    """
    channel = _clip_channel(channel)
    if fitted_stats:
        center = float(fitted_stats.get("median", fitted_stats.get("mean", 0.0)))
        q1 = float(fitted_stats.get("q1", fitted_stats.get("p25", 0.0)))
        q3 = float(fitted_stats.get("q3", fitted_stats.get("p75", 0.0)))
        scale = abs(q3 - q1)
    else:
        center = _channel_quantile(channel, 0.50)
        q1 = _channel_quantile(channel, 0.25)
        q3 = _channel_quantile(channel, 0.75)
        scale = abs(q3 - q1)
    scale = max(scale, config.ROBUST_SCALE_MIN_IQR, config.NORMALIZE_EPS)
    return (channel - center) / scale


def apply_clip_only(channel: torch.Tensor) -> torch.Tensor:
    """
    Apply clipping without forcing variance scaling.
    """
    return _clip_channel(channel)


def apply_safe_fallback(channel: torch.Tensor) -> torch.Tensor:
    """
    Preserve the channel without aggressive scaling.
    """
    return channel.clone().float()


def _profile_stats_for_channel(
    normalization_stats: dict | None,
    channel_idx: int,
) -> dict[str, float] | None:
    if not normalization_stats:
        return None
    channels = normalization_stats.get("channels")
    if not isinstance(channels, dict):
        return None
    stats = channels.get(str(channel_idx))
    return stats if isinstance(stats, dict) else None


def load_normalization_profile_stats(
    path: str | Path | None = None,
) -> dict | None:
    """
    Load fitted normalization statistics when available.
    """
    stats_path = Path(path or config.PATH_NORMALIZATION_PROFILE_STATS)
    if not stats_path.exists():
        return None
    with stats_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def apply_profile_conditioned_normalization(
    cube: torch.Tensor,
    normalization_stats: dict | None = None,
) -> torch.Tensor:
    """
    Apply profile-conditioned normalization channel by channel.
    """
    cube = cube.clone().float()
    for channel_idx in range(cube.shape[0]):
        channel = cube[channel_idx]
        stats = channel_signal_stats(channel)
        profile = compute_channel_profile(stats, channel_idx=channel_idx)
        fitted_stats = _profile_stats_for_channel(normalization_stats, channel_idx)
        label = str(profile["profile"])

        if label == "global_scale_candidate":
            cube[channel_idx] = apply_global_scale(channel, fitted_stats)
        elif label == "robust_scale_candidate":
            cube[channel_idx] = apply_robust_scale(channel, fitted_stats)
        elif label == "clip_only":
            cube[channel_idx] = apply_clip_only(channel)
        else:
            cube[channel_idx] = apply_safe_fallback(channel)
    return cube


def prepare_cube_for_model(cube: torch.Tensor, normalization_stats: dict | None = None) -> torch.Tensor:
    """
    Apply the configured model-input normalization policy.
    """
    cube = cube.float()
    mode = getattr(config, "TRAIN_NORMALIZATION_MODE", "baseline_per_sample")
    if mode == "profile_conditioned":
        stats_payload = normalization_stats or load_normalization_profile_stats()
        return apply_profile_conditioned_normalization(cube, normalization_stats=stats_payload)
    if getattr(config, "TRAIN_NORMALIZE_CUBE_PER_CHANNEL", False):
        return normalize_cube_per_channel(cube)
    return cube


def channel_signal_stats(channel: torch.Tensor) -> dict[str, float]:
    """
    Summarize one channel for normalization diagnostics.
    """
    channel = channel.detach().float().cpu()
    total_voxels = int(channel.numel())
    if total_voxels == 0:
        return {
            "zero_fraction": 1.0,
            "nonzero_fraction": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "dynamic_range": 0.0,
            "energy": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "signal_mass_total": 0.0,
            "signal_mass_mean": 0.0,
            "usable_signal_fraction": 0.0,
            "high_tail_fraction": 0.0,
            "very_high_tail_fraction": 0.0,
            "spatial_coherence": 0.0,
            "border_dominance": 0.0,
            "interior_high_tail_fraction": 0.0,
        }

    flat = channel.reshape(-1)
    abs_flat = torch.abs(flat)
    zero_mask = flat == 0
    nonzero_mask = ~zero_mask
    nonzero_flat = abs_flat[nonzero_mask]
    nonzero_fraction = float(nonzero_mask.float().mean().item())
    std = float(torch.std(flat).item())
    max_abs = float(abs_flat.max().item()) if total_voxels else 0.0
    adaptive_floor = max(config.NORMALIZE_EPS, max_abs * 0.01)
    usable_signal_fraction = float((abs_flat >= adaptive_floor).float().mean().item())
    signal_mass_total = float(abs_flat.sum().item())
    signal_mass_mean = float(abs_flat.mean().item())
    high_tail_floor = max(config.NORMALIZE_EPS, max_abs * 0.25)
    very_high_tail_floor = max(config.NORMALIZE_EPS, max_abs * 0.50)
    high_tail_fraction = float((abs_flat >= high_tail_floor).float().mean().item())
    very_high_tail_fraction = float((abs_flat >= very_high_tail_floor).float().mean().item())
    high_tail_mask = (torch.abs(channel) >= high_tail_floor).float()

    if channel.ndim == 3:
        padded = high_tail_mask.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
        neighbor_counts = F.conv3d(padded, kernel, padding=1).squeeze(0).squeeze(0) - high_tail_mask
        supported_high_tail = ((high_tail_mask > 0) & (neighbor_counts >= 2)).float()
        high_tail_count = float(high_tail_mask.sum().item())
        raw_spatial_coherence = float(supported_high_tail.sum().item() / high_tail_count) if high_tail_count > 0 else 0.0

        border_mask = torch.zeros_like(high_tail_mask, dtype=torch.bool)
        border_width = 2
        border_mask[:border_width, :, :] = True
        border_mask[-border_width:, :, :] = True
        border_mask[:, :border_width, :] = True
        border_mask[:, -border_width:, :] = True
        border_mask[:, :, :border_width] = True
        border_mask[:, :, -border_width:] = True
        border_high_tail = high_tail_mask[border_mask]
        interior_high_tail = high_tail_mask[~border_mask]
        border_high_tail_count = float(border_high_tail.sum().item())
        border_dominance = border_high_tail_count / high_tail_count if high_tail_count > 0 else 0.0
        interior_high_tail_fraction = float(interior_high_tail.mean().item()) if interior_high_tail.numel() > 0 else 0.0
        spatial_coherence = raw_spatial_coherence * max(0.0, 1.0 - (border_dominance * 0.65))
    else:
        spatial_coherence = 0.0
        border_dominance = 0.0
        interior_high_tail_fraction = 0.0

    if nonzero_flat.numel() > 0:
        nonzero_quantiles = torch.quantile(
            nonzero_flat,
            torch.tensor([0.50, 0.90, 0.99], dtype=torch.float32),
        )
        nz_p50 = float(nonzero_quantiles[0].item())
        nz_p90 = float(nonzero_quantiles[1].item())
        nz_p99 = float(nonzero_quantiles[2].item())
    else:
        nz_p50 = 0.0
        nz_p90 = 0.0
        nz_p99 = 0.0

    quantiles = torch.quantile(
        flat,
        torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], dtype=torch.float32),
    )
    return {
        "zero_fraction": float(zero_mask.float().mean().item()),
        "nonzero_fraction": nonzero_fraction,
        "mean": float(torch.mean(flat).item()),
        "std": std,
        "min": float(torch.min(flat).item()),
        "max": float(torch.max(flat).item()),
        "dynamic_range": float((torch.max(flat) - torch.min(flat)).item()),
        "energy": float(torch.mean(flat * flat).item()),
        "signal_mass_total": signal_mass_total,
        "signal_mass_mean": signal_mass_mean,
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "nz_p50": nz_p50,
        "nz_p90": nz_p90,
        "nz_p99": nz_p99,
        "usable_signal_fraction": usable_signal_fraction,
        "high_tail_fraction": high_tail_fraction,
        "very_high_tail_fraction": very_high_tail_fraction,
        "spatial_coherence": spatial_coherence,
        "border_dominance": border_dominance,
        "interior_high_tail_fraction": interior_high_tail_fraction,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def channel_normalization_evidence(
    stats: dict[str, float],
    channel_idx: int | None = None,
) -> dict[str, float]:
    """
    Compute interpretable evidence for normalization suitability.
    """
    zero_fraction = stats.get("zero_fraction", 1.0)
    usable_signal_fraction = stats.get("usable_signal_fraction", 0.0)
    high_tail_fraction = stats.get("high_tail_fraction", 0.0)
    very_high_tail_fraction = stats.get("very_high_tail_fraction", 0.0)
    signal_mass_mean = stats.get("signal_mass_mean", 0.0)
    spatial_coherence = stats.get("spatial_coherence", 0.0)
    border_dominance = stats.get("border_dominance", 0.0)
    interior_high_tail_fraction = stats.get("interior_high_tail_fraction", 0.0)
    is_sparse_channel = channel_idx in {2, 3}

    signal_strength = _clamp01((usable_signal_fraction * 0.55) + (signal_mass_mean * 1.75))
    high_tail_support = _clamp01((high_tail_fraction * 3.5) + (very_high_tail_fraction * 10.0))
    coherence_support = _clamp01((spatial_coherence * 0.35) + (interior_high_tail_fraction * 8.0))
    background_burden = _clamp01((zero_fraction * 0.55) + ((1.0 - usable_signal_fraction) * 0.45))
    border_penalty = _clamp01((border_dominance * 1.15) + max(0.0, high_tail_fraction - (interior_high_tail_fraction * 1.5)))

    if is_sparse_channel:
        global_support = (
            0.14 * signal_strength
            + 0.24 * high_tail_support
            + 0.24 * coherence_support
            + 0.22 * (1.0 - background_burden)
            + 0.16 * (1.0 - border_penalty)
        )
        robust_support = (
            0.20 * signal_strength
            + 0.20 * high_tail_support
            + 0.18 * coherence_support
            + 0.24 * (1.0 - background_burden)
            + 0.18 * (1.0 - border_penalty)
        )
    else:
        global_support = (
            0.28 * signal_strength
            + 0.18 * high_tail_support
            + 0.22 * coherence_support
            + 0.22 * (1.0 - background_burden)
            + 0.10 * (1.0 - border_penalty)
        )
        robust_support = (
            0.25 * signal_strength
            + 0.20 * high_tail_support
            + 0.18 * coherence_support
            + 0.25 * (1.0 - background_burden)
            + 0.12 * (1.0 - border_penalty)
        )

    return {
        "signal_strength_score": round(_clamp01(signal_strength), 6),
        "high_tail_support_score": round(_clamp01(high_tail_support), 6),
        "coherence_support_score": round(_clamp01(coherence_support), 6),
        "background_burden_score": round(_clamp01(background_burden), 6),
        "border_penalty_score": round(_clamp01(border_penalty), 6),
        "global_support_score": round(_clamp01(global_support), 6),
        "robust_support_score": round(_clamp01(robust_support), 6),
    }


def compute_channel_profile(
    stats: dict[str, float],
    channel_idx: int | None = None,
) -> dict[str, float | str]:
    """
    Assign a normalization profile from channel statistics.
    """
    std = stats.get("std", 0.0)
    signal_mass_total = stats.get("signal_mass_total", 0.0)
    evidence = channel_normalization_evidence(stats, channel_idx=channel_idx)

    profile = "clip_only"
    decision_reason = "composite evidence favored conservative clipping"

    if std <= config.NORMALIZE_EPS:
        profile = "pass_through"
        decision_reason = "channel standard deviation is below epsilon"
    elif signal_mass_total <= config.NORMALIZE_EPS:
        profile = "pass_through"
        decision_reason = "channel signal mass is below epsilon"
    else:
        background_burden = float(evidence["background_burden_score"])
        border_penalty = float(evidence["border_penalty_score"])
        coherence_support = float(evidence["coherence_support_score"])
        global_support = float(evidence["global_support_score"])
        robust_support = float(evidence["robust_support_score"])
        high_tail_support = float(evidence["high_tail_support_score"])
        interior_high_tail_fraction = stats.get("interior_high_tail_fraction", 0.0)
        border_dominance = stats.get("border_dominance", 0.0)
        is_sparse_channel = channel_idx in {2, 3}

        if background_burden >= 0.72 and coherence_support < 0.20:
            profile = "normalization_risk"
            decision_reason = "high background burden with weak coherence support"
        elif border_penalty >= 0.58 and coherence_support < 0.40:
            profile = "clip_only"
            decision_reason = "border-dominant pattern with limited coherent support"
        elif is_sparse_channel and interior_high_tail_fraction < 0.010 and background_burden >= 0.52:
            profile = "normalization_risk"
            decision_reason = "sparse channel with weak interior support and elevated background burden"
        elif is_sparse_channel and border_penalty >= 0.48 and interior_high_tail_fraction < 0.020:
            profile = "clip_only"
            decision_reason = "sparse channel with edge-heavy tail support and low interior support"
        elif (
            is_sparse_channel
            and global_support >= 0.58
            and (
                interior_high_tail_fraction < 0.040
                or border_dominance >= 0.24
                or high_tail_support < 0.20
            )
        ):
            profile = "robust_scale_candidate"
            decision_reason = "sparse channel remains too borderline for global scaling"
        elif is_sparse_channel and interior_high_tail_fraction < 0.018 and global_support >= 0.58:
            profile = "robust_scale_candidate"
            decision_reason = "global support is present but interior support is too limited for global scaling"
        elif (
            is_sparse_channel
            and global_support >= 0.66
            and border_penalty < 0.50
            and interior_high_tail_fraction >= 0.040
            and border_dominance < 0.24
            and high_tail_support >= 0.20
        ):
            profile = "global_scale_candidate"
            decision_reason = "sparse channel shows sustained interior support with controlled border influence"
        elif global_support >= 0.58 and border_penalty < 0.58:
            profile = "global_scale_candidate"
            decision_reason = "global support is strong enough and border penalty remains controlled"
        elif robust_support >= 0.44:
            profile = "robust_scale_candidate"
            decision_reason = "composite evidence supports conservative scaling"
        elif background_burden >= 0.60:
            profile = "normalization_risk"
            decision_reason = "background burden remains too high for reliable scaling"

    profile_payload: dict[str, float | str] = {
        "profile": profile,
        "decision_reason": decision_reason,
    }
    profile_payload.update(evidence)
    return profile_payload


def recommended_channel_policy(stats: dict[str, float], channel_idx: int | None = None) -> str:
    """
    Suggest a conservative normalization action from the shared profile contract.
    """
    return str(compute_channel_profile(stats, channel_idx=channel_idx)["profile"])


def write_audit_records(path: str | Path, records: list[dict]) -> None:
    """
    Persist normalization audit records as JSON.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
