"""
Micro-cube construction module.

Encapsulates the tensor-building logic that was previously embedded in ``main.py``.
The implementation intentionally preserves current behavior so the class can be
introduced without changing extraction results yet.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import config


class MicroCube:
    """Builds a 4-channel Bio-Lattice micro-cube plus lightweight diagnostics."""

    def __init__(
        self,
        size: int | None = None,
        interpolate_mode: str | None = None,
        var_kernel: int | None = None,
        c4_peak_kernel: int | None = None,
        kinetics_clamp: float | None = None,
        min_roi_reflect_fraction: float | None = None,
    ):
        self.size = size if size is not None else config.MICRO_CUBE_SIZE
        self.interpolate_mode = interpolate_mode or config.PRE_POST_INTERPOLATE_MODE
        self.var_kernel = int(var_kernel if var_kernel is not None else getattr(config, "C2_LOCAL_VAR_KERNEL", 3))
        self.c4_peak_kernel = int(c4_peak_kernel if c4_peak_kernel is not None else getattr(config, "C4_LOCAL_PEAK_KERNEL", 3))
        self.kinetics_clamp = kinetics_clamp if kinetics_clamp is not None else config.EXTRACTION_KINETICS_CLAMP
        self.min_roi_reflect_fraction = (
            min_roi_reflect_fraction
            if min_roi_reflect_fraction is not None
            else config.ROI_MIN_FRAC_FOR_REFLECT
        )

        if self.var_kernel % 2 == 0:
            raise ValueError(f"var_kernel must be odd to maintain symmetric padding, got {self.var_kernel}")
        if self.c4_peak_kernel % 2 == 0:
            raise ValueError(f"c4_peak_kernel must be odd to maintain symmetric padding, got {self.c4_peak_kernel}")

        if self.size <= 0:
            raise ValueError(f"size must be > 0, got {self.size}")
        if self.kinetics_clamp <= 0:
            raise ValueError(
                f"kinetics_clamp must be > 0 to avoid invalid division, got {self.kinetics_clamp}"
            )

    def _robust_norm_01(self, t: torch.Tensor, q: float = 0.995) -> torch.Tensor:
        p = torch.quantile(t, q)
        t = torch.clamp(t, max=p)
        t_min, t_max = t.min(), t.max()
        return (t - t_min) / (t_max - t_min + 1e-8)

    def _normalize_cube(
        self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self._robust_norm_01(c1)
        c2 = self._robust_norm_01(c2)
        c4 = self._robust_norm_01(c4)
        c3 = torch.clamp(c3 / self.kinetics_clamp, -1.0, 1.0)
        return c1, c2, c3, c4
        
    def _target_shape(self, t_post: torch.Tensor, spacing: list[float] | tuple[float, ...], thickness: float) -> tuple[int, int, int]:
        dz, dy, dx = thickness, spacing[1], spacing[0]
        d, h, w = t_post.shape[2:]
        lz, ly, lx = d * dz, h * dy, w * dx
        lmax = max(lz, ly, lx)
        sz = max(1, int(round(self.size * lz / lmax)))
        sy = max(1, int(round(self.size * ly / lmax)))
        sx = max(1, int(round(self.size * lx / lmax)))
        return sz, sy, sx

    def _interpolate(self, t: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        kwargs = {"size": size, "mode": self.interpolate_mode}
        if self.interpolate_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(t, **kwargs)

    def _build_kinetics_channel(self, t_pre: torch.Tensor, t_post: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        # Background mask threshold to avoid zero-division explosion mapped to kinetics
        bg_threshold = 1e-4 
        tissue_mask = (torch.abs(t_pre) > bg_threshold).float()
        
        ratio = (t_post - t_pre) / (torch.abs(t_pre) + eps)
        kinetics_raw = torch.sign(ratio) * torch.log1p(torch.abs(ratio))
        kinetics_raw = torch.clamp(kinetics_raw, -self.kinetics_clamp, self.kinetics_clamp)
        return kinetics_raw * tissue_mask

    def _build_heterogeneity_channel(self, t_post: torch.Tensor) -> torch.Tensor:
        # Operate directly on native resolution preserving micro-textures (No Denoise)
        p = self.var_kernel // 2
        mean_r = F.avg_pool3d(t_post, kernel_size=self.var_kernel, stride=1, padding=p)
        mean_sq_r = F.avg_pool3d(t_post ** 2, kernel_size=self.var_kernel, stride=1, padding=p)
        return torch.log1p(torch.relu(mean_sq_r - (mean_r ** 2)))

    def _build_peak_channel(self, t_post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Local vascular highlights calculated on native matrix resolution
        to preserve ultra-fine vessel identification.
        """
        p = self.c4_peak_kernel // 2
        local_max = F.max_pool3d(t_post, kernel_size=self.c4_peak_kernel, stride=1, padding=p)
        local_mean = F.avg_pool3d(t_post, kernel_size=self.c4_peak_kernel, stride=1, padding=p)
        c4_raw = torch.relu(local_max - local_mean)
        return c4_raw, local_max, local_mean

    def _padding_and_mode(
        self, t_post_iso: torch.Tensor, target_shape: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int, int, int, int], str, float]:
        sz, sy, sx = target_shape
        pad_z, pad_y, pad_x = self.size - sz, self.size - sy, self.size - sx
        padding = (
            pad_x // 2, pad_x - (pad_x // 2),
            pad_y // 2, pad_y - (pad_y // 2),
            pad_z // 2, pad_z - (pad_z // 2),
        )
        can_reflect = (
            padding[0] < t_post_iso.shape[4] and padding[1] < t_post_iso.shape[4]
            and padding[2] < t_post_iso.shape[3] and padding[3] < t_post_iso.shape[3]
            and padding[4] < t_post_iso.shape[2] and padding[5] < t_post_iso.shape[2]
        )
        roi_fraction = (sz * sy * sx) / (self.size ** 3)
        struct_pad_mode = "reflect" if (can_reflect and roi_fraction >= self.min_roi_reflect_fraction) else "constant"
        return padding, struct_pad_mode, roi_fraction

    def _channel_stats(self, cube: torch.Tensor) -> dict[str, float]:
        stats = {}
        for idx in range(cube.shape[0]):
            channel = cube[idx]
            stats[f"c{idx + 1}_mean"] = float(torch.mean(channel).item())
            stats[f"c{idx + 1}_std"] = float(torch.std(channel, unbiased=False).item())
            stats[f"c{idx + 1}_min"] = float(torch.min(channel).item())
            stats[f"c{idx + 1}_max"] = float(torch.max(channel).item())
        return stats

    def build(
        self, t_pre: torch.Tensor, t_post: torch.Tensor, spacing: list[float] | tuple[float, ...], thickness: float
    ) -> tuple[torch.Tensor, dict[str, float | int | list | str]]:
        """Build the 4-channel micro-cube and return `(cube, diagnostics)`."""
        # 1. Native Resolution Extraction ("Features-First")
        kinetics_raw = self._build_kinetics_channel(t_pre, t_post)
        c2_raw = self._build_heterogeneity_channel(t_post)
        c4_raw, local_max, local_mean = self._build_peak_channel(t_post)
        
        # 2. Determine isometric target shape
        target_shape = self._target_shape(t_post, spacing, thickness)
        
        # 3. Resample to isometric topology
        t_post_iso = self._interpolate(t_post, size=target_shape)
        c1_iso = t_post_iso
        
        c2_iso = self._interpolate(c2_raw, size=target_shape)
        c3_iso = self._interpolate(kinetics_raw, size=target_shape)
        c4_iso = self._interpolate(c4_raw, size=target_shape)

        zero_mask = (t_pre == 0).float()
        pre_zero_fraction = float(torch.mean(zero_mask).item())

        # 4. Standardize Equitable Padding
        padding, struct_pad_mode, roi_fraction = self._padding_and_mode(t_post_iso, target_shape)
        
        c1_pad = F.pad(c1_iso, padding, mode=struct_pad_mode)
        c2_pad = F.pad(c2_iso, padding, mode=struct_pad_mode)
        c3_pad = F.pad(c3_iso, padding, mode=struct_pad_mode)
        c4_pad = F.pad(c4_iso, padding, mode=struct_pad_mode)

        # 5. Final Sizing to Micro-Cube
        c1 = F.adaptive_avg_pool3d(c1_pad, output_size=(self.size, self.size, self.size))
        c2 = F.adaptive_avg_pool3d(c2_pad, output_size=(self.size, self.size, self.size))
        c3 = F.adaptive_avg_pool3d(c3_pad, output_size=(self.size, self.size, self.size))
        c4 = F.adaptive_avg_pool3d(c4_pad, output_size=(self.size, self.size, self.size))
        
        # 6. Global Equitable Normalization
        c1, c2, c3, c4 = self._normalize_cube(c1, c2, c3, c4)
        cube = torch.cat([c1, c2, c3, c4], dim=1).squeeze(0)
            
        diagnostics = {
            "target_shape": list(target_shape),
            "pre_zero_fraction": pre_zero_fraction,
            "padding_mode": struct_pad_mode,
            "roi_fraction": float(roi_fraction),
            "c4_energy": float(torch.mean(torch.abs(c4)).item()),
            "peak_minus_avg_mean": float(torch.mean(torch.relu(local_max - local_mean)).item()),
            **self._channel_stats(cube),
        }
        return cube, diagnostics
