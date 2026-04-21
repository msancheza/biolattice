"""
Micro-cube factory: Duke-oriented DICOM pairing, shared ROI crop, and 64³ weave (see config.MICRO_CUBE_SIZE).

Registration now includes a 3D Phase Correlation (FFT) step to align pre and post
volumes before ROI extraction, ensuring accurate kinetics (Channel 3).
See README *What main.py actually does* for series heuristics, ROI
indexing, and intensity handling.
"""
import os
import pandas as pd
import pydicom
import torch

import config
from core.clasificador import Clasificador
from core.dicom_utils import get_3d_volume, load_weave_metadata
from core.microcube import MicroCube
from core.registration import register_volumes_fft
from core.audit import RunLogWriter 

if not os.path.exists(config.PATH_MICRO_CUBES):
    os.makedirs(config.PATH_MICRO_CUBES)

_extraction_logger = RunLogWriter(config.PATH_LOGS)
_microcube_builder = MicroCube()


def log_event(message: str) -> None:
    """Appends a line to ``PATH_LOGS`` (extraction / registration audit trail)."""
    _extraction_logger.event(message)


def to_repo_relative_path(path_value: str | None) -> str:
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


def tensor_channel_stats(micro_cube: torch.Tensor) -> dict[str, float]:
    """Returns lightweight per-channel stats for quick QC."""
    stats = {}
    for idx in range(micro_cube.shape[0]):
        channel = micro_cube[idx]
        stats[f"c{idx + 1}_mean"] = float(torch.mean(channel).item())
        stats[f"c{idx + 1}_std"] = float(torch.std(channel, unbiased=False).item())
        stats[f"c{idx + 1}_min"] = float(torch.min(channel).item())
        stats[f"c{idx + 1}_max"] = float(torch.max(channel).item())
    return stats


def metadata_physical_is_valid(spacing: list[float] | tuple[float, ...], thickness: float) -> tuple[bool, str]:
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

def crop_roi_with_padding(volume: 'numpy.ndarray', coords: tuple[int, int, int, int, int, int], padding_pct: float | None = None) -> torch.Tensor:
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


def discover_series_for_patient(folder_patient: str) -> list[dict]:
    """Scan patient folders, classify candidate series, and sort them chronologically."""
    series_found = []
    for root, _, files in os.walk(folder_patient):
        dicoms = [f for f in files if f.endswith('.dcm')]
        if dicoms:
            ds = pydicom.dcmread(os.path.join(root, dicoms[0]))
            series_found.append(
                Clasificador.classify_series(
                    Clasificador.extract_series_metadata(ds, root, len(dicoms))
                )
            )

    series_found.sort(key=Clasificador.sort_key)
    return series_found


def select_semantic_pair(series_found: list[dict]) -> dict:
    """Resolve PRE/POST paths and semantic metadata from discovered series."""
    selected_pre, selected_post = Clasificador.select_pre_post(series_found)
    path_pre = selected_pre["path"] if selected_pre else None
    path_post = selected_post["path"] if selected_post else None
    selected_pre_desc = next((s["desc"] for s in series_found if s["path"] == path_pre), "")
    selected_post_desc = next((s["desc"] for s in series_found if s["path"] == path_post), "")
    return {
        "path_pre": path_pre,
        "path_post": path_post,
        "selected_pre_desc": selected_pre_desc,
        "selected_post_desc": selected_post_desc,
        "selected_pre_semantic": selected_pre,
        "selected_post_semantic": selected_post,
    }


def build_series_candidates_payload(series_found: list[dict]) -> list[dict]:
    """Compact audit payload for semantic candidate inspection."""
    return [
        {
            "path": to_repo_relative_path(s["path"]),
            "desc": s.get("desc", ""),
            "protocol": s.get("protocol", ""),
            "manufacturer": s.get("manufacturer", ""),
            "series_number": int(s.get("num", 0)),
            "series_time": s.get("series_time", ""),
            "pre_score": float(s.get("pre_score", 0.0)),
            "post_score": float(s.get("post_score", 0.0)),
            "candidate_type": s.get("candidate_type", "unknown"),
            "evidence": s.get("evidence", []),
            "exclusion_reasons": s.get("exclusion_reasons", []),
        }
        for s in series_found
    ]


def build_patient_audit_base(
    p_id: str,
    series_found: list[dict],
    path_pre: str | None,
    path_post: str | None,
    selected_pre_desc: str,
    selected_post_desc: str,
    selected_pre_semantic: dict | None = None,
    selected_post_semantic: dict | None = None,
) -> dict:
    """Shared audit fields used by success, failure, and skip records."""
    return {
        "patient_id": p_id,
        "path_pre": to_repo_relative_path(path_pre),
        "path_post": to_repo_relative_path(path_post),
        "pre_desc": selected_pre_desc,
        "post_desc": selected_post_desc,
        "pre_semantic_score": float(selected_pre_semantic["pre_score"]) if selected_pre_semantic else None,
        "post_semantic_score": float(selected_post_semantic["post_score"]) if selected_post_semantic else None,
        "pre_semantic_evidence": selected_pre_semantic["evidence"] if selected_pre_semantic else [],
        "post_semantic_evidence": selected_post_semantic["evidence"] if selected_post_semantic else [],
        "series_candidates": build_series_candidates_payload(series_found),
    }
def build_roi_coords(row: pd.Series) -> tuple[int, int, int, int, int, int]:
    """Convert spreadsheet coordinates to zero-indexed ROI tuple."""
    return (
        int(row['Start Slice']) - 1,
        int(row['End Slice']),
        int(row['Start Row']) - 1,
        int(row['End Row']),
        int(row['Start Column']) - 1,
        int(row['End Column']),
    )


def process_patient(p_id: str, row: pd.Series, audit_log: RunLogWriter) -> bool:
    """End-to-end extraction for one patient, including semantic series selection and audit."""
    folder_patient = os.path.join(config.PATH_RAW, p_id)
    if not os.path.exists(folder_patient):
        return False

    print(f"Analyzing {p_id}...")
    series_found = discover_series_for_patient(folder_patient)
    print("the series found are: ", series_found, " for: ", p_id)
    pair = select_semantic_pair(series_found)
    path_pre = pair["path_pre"]
    path_post = pair["path_post"]
    selected_pre_desc = pair["selected_pre_desc"]
    selected_post_desc = pair["selected_post_desc"]
    selected_pre_semantic = pair["selected_pre_semantic"]
    selected_post_semantic = pair["selected_post_semantic"]

    audit_base = build_patient_audit_base(
        p_id,
        series_found,
        path_pre,
        path_post,
        selected_pre_desc,
        selected_post_desc,
        selected_pre_semantic,
        selected_post_semantic,
    )

    if not (path_pre and path_post):
        reason = "Missing PRE" if not path_pre else "Missing POST"
        if not path_pre and not path_post:
            reason = "Missing BOTH"
        desc_str = str([s['desc'] for s in series_found])
        log_msg = f"[SKIP] {p_id} | {reason} | Found: {desc_str}"
        print(f"Skipped {p_id}: {reason}")
        log_event(log_msg)
        if config.AUDIT_WRITE_JSONL:
            audit_log.append_jsonl(
            {
                **audit_base,
                "status": "REVIEW",
                "reason": reason,
                "corr_global": None,
                "shifts": [],
                "micro_cube_path": "",
                "micro_cube_shape": [],
                "orig_post_shape": [],
                "roi_coords": [],
            },
            )
        return False

    try:
        v_pre = get_3d_volume(path_pre, log_event=log_event)
        v_post = get_3d_volume(path_post, log_event=log_event)

        print(f"  Registering volumes for {p_id}...")
        v_pre, shifts, corr = register_volumes_fft(v_pre, v_post, patient_id=p_id, log_event=log_event)

        status = "OK"
        if corr < config.REGISTRATION_MIN_CORRELATION:
            status = "REVIEW"
            log_msg = f"[REVIEW] {p_id} | Corr: {corr:.4f} | Shifts: {shifts} (Low correlation)"
            print(f"  {log_msg}")
            log_event(log_msg)
        else:
            log_msg = f"[{status}] {p_id} | Corr: {corr:.4f} | Shifts: {shifts}"
            print(f"  {log_msg}")

        coords = build_roi_coords(row)
        t_post_roi = crop_roi_with_padding(v_post, coords)
        t_pre_roi = crop_roi_with_padding(v_pre, coords)
        pixel_spacing, slice_thickness = load_weave_metadata(path_post)

        micro_cube, cube_diagnostics = _microcube_builder.build(
            t_pre_roi, t_post_roi, spacing=pixel_spacing, thickness=slice_thickness
        )
        channel_stats = tensor_channel_stats(micro_cube)
        zero_fraction = cube_diagnostics["pre_zero_fraction"]
        
        if zero_fraction > 0.3:
            status = "REVIEW"
            issue_msg = f"[REVIEW] {p_id} | Excessive pre_zero_fraction: {zero_fraction*100:.1f}%. Bounding box may be out of bounds."
            print(f"  {issue_msg}")
            log_event(issue_msg)
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
                "roi_coords": coords,
            },
        }
        output_path = os.path.join(config.PATH_MICRO_CUBES, f"{p_id}{config.LATTICE_FILE_SUFFIX}")
        torch.save(lattice_obj, output_path)
        print(f"Micro-cube successfully generated for {p_id}")

        if config.AUDIT_WRITE_JSONL:
            audit_log.append_jsonl(
            {
                **audit_base,
                "status": status,
                "reason": "",
                "same_series_pair": bool(same_series_pair),
                "corr_global": float(corr),
                "shifts": shifts,
                "meta_valid": bool(meta_ok),
                "meta_issue": meta_issue,
                "micro_cube_path": output_path,
                "micro_cube_shape": list(micro_cube.shape),
                "orig_post_shape": list(v_post.shape),
                "roi_coords": list(coords),
                "pre_zero_fraction": zero_fraction,
                "cube_diagnostics": cube_diagnostics,
                **channel_stats,
            },
            )
        return True
    except Exception as e:
        err_msg = f"[ERROR] {p_id} | Exception: {e}"
        print(f"  {err_msg}")
        log_event(err_msg)
        if config.AUDIT_WRITE_JSONL:
            audit_log.append_jsonl(
            {
                **audit_base,
                "status": "REVIEW",
                "reason": str(e),
                "corr_global": None,
                "shifts": [],
                "micro_cube_path": "",
                "micro_cube_shape": [],
                "orig_post_shape": [],
                "roi_coords": [],
            },
            )
        return False

def process_dataset() -> None:
    df_boxes = pd.read_excel(config.PATH_ANNOTATION_BOXES)
    successes = 0
    audit_log = RunLogWriter.create_new(kind="extraction_audit")
    print(f"Audit log path: {audit_log.path}")
    
    for _, row in df_boxes.iterrows():
        p_id = row[config.COL_PATIENT_ID]
        if process_patient(p_id, row, audit_log):
            successes += 1

    print(f"\n Process finished. {successes} micro-cubes generated in {config.PATH_MICRO_CUBES}")
    print(f"Extraction audit saved to: {audit_log.path}")

if __name__ == "__main__":
    process_dataset()
