"""
DICOM IO helpers for the extraction pipeline.

Provides utilities for reading medical imaging data in DICOM format, involving 
series sorting by spatial coordinates, rescale slope/intercept application, 
and metadata extraction. It serves as the primary interface between the 
filesystem and the volumetric processing logic.
"""
from __future__ import annotations

import os

import numpy as np
import pydicom

import config


def get_3d_volume(series_dir, log_event=None):
    """Load a DICOM series into a sorted 3D float32 volume."""
    dicom_paths = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.endswith(config.DICOM_EXTENSION)
    ]
    slices = [pydicom.dcmread(f) for f in dicom_paths]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    if len(slices) > 1:
        z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
        z_diffs = np.diff(z_positions)
        median_diff = np.median(z_diffs)
        max_deviation = np.max(np.abs(z_diffs - median_diff))
        if median_diff > 0.05 and max_deviation > (0.5 * median_diff) and log_event is not None:
            log_event(
                f"[{os.path.basename(os.path.dirname(series_dir))} WARNING] "
                f"DICOM Z-Gap inconsistency. Missing slice likely! Dev: {max_deviation:.2f}mm"
            )

    vols = []
    for s in slices:
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        vols.append(s.pixel_array.astype(np.float32) * slope + intercept)

    return np.stack(vols)


def load_weave_metadata(path_post):
    """Read spacing metadata from the chosen POST series for cube weaving."""
    try:
        dicom_files = [f for f in os.listdir(path_post) if f.endswith(config.DICOM_EXTENSION)]
        ds_meta = pydicom.dcmread(os.path.join(path_post, dicom_files[0]))
        pixel_spacing = [float(x) for x in getattr(ds_meta, "PixelSpacing", [1.0, 1.0])]
        slice_thickness = float(getattr(ds_meta, "SliceThickness", 1.0))
    except Exception:
        pixel_spacing = [1.0, 1.0]
        slice_thickness = 1.0
    return pixel_spacing, slice_thickness
