"""
Semantic DICOM series classifier.

This module promotes series selection from plain regex matching into a metadata-aware
classification step with vendor-sensitive rules and interpretable audit output.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import config


@dataclass
class SemanticScore:
    pre_score: float = 0.0
    post_score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    exclusion_reasons: list[str] = field(default_factory=list)

    def add_pre(self, value: float, reason: str) -> None:
        self.pre_score += value
        self.evidence.append(f"pre:+{value:g} {reason}")

    def add_post(self, value: float, reason: str) -> None:
        self.post_score += value
        self.evidence.append(f"post:+{value:g} {reason}")

    def add_both(self, value: float, reason: str) -> None:
        self.pre_score += value
        self.post_score += value
        self.evidence.append(f"both:+{value:g} {reason}")

    def exclude(self, reason: str, penalty: float = 6.0) -> None:
        self.pre_score -= penalty
        self.post_score -= penalty
        self.exclusion_reasons.append(reason)


class SeriesClassifier:
    """Scores DICOM series metadata and selects PRE/POST candidates."""

    _RULES: dict[str, Any] = {}
    VENDOR_GE = ("ge", "general electric")
    VENDOR_SIEMENS = ("siemens",)
    VENDOR_PHILIPS = ("philips",)

    @classmethod
    def _load_rules(cls) -> None:
        if not cls._RULES:
            rules_path = os.path.join(os.path.dirname(__file__), "classifier_rules.json")
            try:
                with open(rules_path, "r") as f:
                    cls._RULES = json.load(f)
            except Exception as e:
                # Fallback or empty rules if file not found
                cls._RULES = {"min_selection_score": 2.0}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(k in text for k in keywords)

    @classmethod
    def _manufacturer_family(cls, manufacturer: str) -> str:
        maker = (manufacturer or "").lower()
        if any(x in maker for x in cls.VENDOR_GE):
            return "ge"
        if any(x in maker for x in cls.VENDOR_SIEMENS):
            return "siemens"
        if any(x in maker for x in cls.VENDOR_PHILIPS):
            return "philips"
        return "generic"

    @classmethod
    def _series_text_blob(cls, meta: dict[str, Any]) -> str:
        parts = [
            meta.get("desc", ""),
            meta.get("protocol", ""),
            meta.get("sequence_name", ""),
            meta.get("sequence_variant", ""),
            meta.get("scan_options", ""),
            meta.get("image_type", ""),
            meta.get("series_label", ""),
        ]
        return " ".join(str(p).lower() for p in parts if p)

    @classmethod
    def extract_series_metadata(cls, ds, root: str, num_dicoms: int) -> dict[str, Any]:
        """Collect representative metadata from one DICOM header for a series."""
        image_type = getattr(ds, "ImageType", [])
        if isinstance(image_type, (list, tuple)):
            image_type = " ".join(str(x) for x in image_type)
        series_time = (
            getattr(ds, "SeriesTime", None)
            or getattr(ds, "AcquisitionTime", None)
            or getattr(ds, "ContentTime", None)
            or ""
        )

        manufacturer = str(getattr(ds, "Manufacturer", "")).strip().lower()
        meta = {
            "path": root,
            "desc": str(getattr(ds, "SeriesDescription", "")).strip().lower(),
            "protocol": str(getattr(ds, "ProtocolName", "")).strip().lower(),
            "sequence_name": str(getattr(ds, "SequenceName", "")).strip().lower(),
            "sequence_variant": str(getattr(ds, "SequenceVariant", "")).strip().lower(),
            "scan_options": str(getattr(ds, "ScanOptions", "")).strip().lower(),
            "image_type": str(image_type).strip().lower(),
            "manufacturer": manufacturer,
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")).strip(),
            "contrast_bolus_agent": str(getattr(ds, "ContrastBolusAgent", "")).strip().lower(),
            "contrast_bolus_route": str(getattr(ds, "ContrastBolusRoute", "")).strip().lower(),
            "contrast_bolus_volume": cls._safe_float(getattr(ds, "ContrastBolusVolume", 0.0), 0.0),
            "series_date": str(getattr(ds, "SeriesDate", "")).strip(),
            "series_time": str(series_time).strip(),
            "num": cls._safe_int(getattr(ds, "SeriesNumber", 0), 0),
            "acquisition_number": cls._safe_int(getattr(ds, "AcquisitionNumber", 0), 0),
            "temporal_position": cls._safe_int(getattr(ds, "TemporalPositionIdentifier", 0), 0),
            "num_dicoms": int(num_dicoms),
            "pixel_spacing": [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])],
            "slice_thickness": cls._safe_float(getattr(ds, "SliceThickness", 1.0), 1.0),
            "series_label": os.path.basename(root).strip().lower(),
        }
        meta["manufacturer_family"] = cls._manufacturer_family(manufacturer)
        return meta

    @classmethod
    def _apply_base_rules(cls, meta: dict[str, Any], text: str, score: SemanticScore) -> None:
        cls._load_rules()
        
        # 1. Apply exclusions from JSON
        for rule in cls._RULES.get("base_exclusions", []):
            keywords = tuple(rule["keywords"])
            if cls._contains_any(text, keywords):
                # Special case for "exclude_if_not" (e.g. T2 unless T1 also exists)
                if "exclude_if_not" in rule:
                    required = tuple(rule["exclude_if_not"])
                    if not cls._contains_any(text, required):
                        score.exclude(rule["reason"])
                else:
                    score.exclude(rule["reason"])

        # 2. Apply scoring from JSON
        for rule in cls._RULES.get("base_scoring", []):
            keywords = tuple(rule["keywords"])
            if cls._contains_any(text, keywords):
                if rule["score_type"] == "pre":
                    score.add_pre(rule["value"], rule["reason"])
                elif rule["score_type"] == "post":
                    score.add_post(rule["value"], rule["reason"])
                elif rule["score_type"] == "both":
                    score.add_both(rule["value"], rule["reason"])

        # 3. Hardcoded heuristic: Late phase penalty for numbered passes (2nd, 3rd, 4th, etc.)
        _is_late_pass = bool(re.search(r"[2-9](nd|rd|th)", text))
        _has_pass_word = "pass" in text
        _has_pre_anchor = cls._contains_any(text, ("pre", "1st", "baseline"))
        if (_is_late_pass or _has_pass_word) and not _has_pre_anchor:
            score.exclude("late numbered pass", penalty=4.0)

        # 4. Hardcoded heuristic: Baseline PRE bias for phase-agnostic dynamic acquisitions
        _has_pure_dynamic = cls._contains_any(text, ("dyn", "vibrant"))
        _has_phase_marker = cls._contains_any(text, ("ph", "phase", "fase"))
        if _has_pure_dynamic and not _has_phase_marker:
            score.add_pre(1.0, "pure dynamic without phase marker → pre bias")

        # 5. Conflict guard: Resolve PRE/POST ambiguity in absence of numbered phases
        _has_pre_kw = any(k in text for rule in cls._RULES.get("base_scoring", []) 
                         if rule["score_type"] == "pre" for k in rule["keywords"])
        _has_post_kw = any(k in text for rule in cls._RULES.get("base_scoring", []) 
                          if rule["score_type"] == "post" for k in rule["keywords"])
        
        _has_numbered_phase = bool(re.search(r"(ph|phase|dyn|dynamic|fase|pass)\s?[1-9]", text)) \
                           or bool(re.search(r"[1-9](st|nd|rd|th)", text))
        
        if _has_pre_kw and not _has_post_kw and not _has_numbered_phase:
            score.pre_score += 1.0
            score.evidence.append("pre:+1.0 conflict-guard (pre keyword, no post signal)")

    @classmethod
    def _apply_contrast_rules(cls, meta: dict[str, Any], score: SemanticScore) -> None:
        if meta.get("contrast_bolus_agent") or meta.get("contrast_bolus_route"):
            score.add_post(3.0, "contrast bolus metadata")
        if meta.get("contrast_bolus_volume", 0.0) > 0:
            score.add_post(1.0, "contrast bolus volume")
        if meta.get("temporal_position", 0) > 0:
            score.add_post(0.75, "temporal position")
        if meta.get("acquisition_number", 0) > 1:
            score.add_post(0.5, "late acquisition")

    @classmethod
    def _apply_vendor_rules(cls, meta: dict[str, Any], text: str, score: SemanticScore) -> None:
        cls._load_rules()
        family = meta.get("manufacturer_family", "generic")
        vendor_data = cls._RULES.get("vendor_rules", {}).get(family)
        
        if not vendor_data:
            return

        # Apply family keywords
        if cls._contains_any(text, tuple(vendor_data.get("family", []))):
            score.add_both(1.0, f"{family.capitalize()} dynamic family")

        # Apply manufacturer-specific phase scoring
        for rule in vendor_data.get("scoring", []):
            if cls._contains_any(text, tuple(rule["keywords"])):
                if rule["score_type"] == "post":
                    score.add_post(rule["value"], rule["reason"])
                elif rule["score_type"] == "pre":
                    score.add_pre(rule["value"], rule["reason"])
                elif rule["score_type"] == "both":
                    score.add_both(rule["value"], rule["reason"])

    @classmethod
    def _resolve_candidate_type(cls, score: SemanticScore) -> tuple[str, float]:
        cls._load_rules()
        min_score = cls._RULES.get("min_selection_score", 2.0)
        
        candidate_type = "unknown"
        confidence = 0.0
        if (
            score.pre_score >= min_score
            or score.post_score >= min_score
        ):
            if score.post_score > score.pre_score:
                candidate_type = "post"
                confidence = score.post_score - score.pre_score
            elif score.pre_score > score.post_score:
                candidate_type = "pre"
                confidence = score.pre_score - score.post_score
            else:
                candidate_type = "ambiguous"
        return candidate_type, confidence

    @classmethod
    def classify_series(cls, meta: dict[str, Any]) -> dict[str, Any]:
        """Assign PRE/POST scores using DICOM metadata and manufacturer-sensitive rules."""
        text = cls._series_text_blob(meta)
        score = SemanticScore()

        cls._apply_base_rules(meta, text, score)
        cls._apply_contrast_rules(meta, score)
        cls._apply_vendor_rules(meta, text, score)

        candidate_type, confidence = cls._resolve_candidate_type(score)
        return {
            **meta,
            "semantic_text": text,
            "pre_score": round(score.pre_score, 3),
            "post_score": round(score.post_score, 3),
            "candidate_type": candidate_type,
            "confidence": round(confidence, 3),
            "exclusion_reasons": score.exclusion_reasons,
            "evidence": score.evidence,
        }

    @classmethod
    def sort_key(cls, series_meta: dict[str, Any]):
        return (
            str(series_meta.get("series_date", "")),
            str(series_meta.get("series_time", "")),
            cls._safe_int(series_meta.get("temporal_position", 0), 0),
            cls._safe_int(series_meta.get("acquisition_number", 0), 0),
            cls._safe_int(series_meta.get("num", 0), 0),
        )

    @classmethod
    def _dynamic_fallback(cls, ordered: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        cls._load_rules()
        dyn_kws = tuple(cls._RULES.get("dynamic_keywords", []))
        dynamic_like = [
            s for s in ordered if cls._contains_any(s.get("semantic_text", ""), dyn_kws)
        ]
        if len(dynamic_like) >= 2:
            return dynamic_like[0], dynamic_like[1]
        return None, None

    @classmethod
    def select_pre_post(cls, series_found: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Select PRE and first POST candidates using semantic scores plus chronology."""
        if not series_found:
            return None, None

        cls._load_rules()
        min_score = cls._RULES.get("min_selection_score", 2.0)
        
        ordered = sorted(series_found, key=cls.sort_key)
        pre_candidates = [s for s in ordered if s["pre_score"] >= min_score]
        post_candidates = [s for s in ordered if s["post_score"] >= min_score]

        pre_candidates.sort(key=lambda s: (-s["pre_score"], cls.sort_key(s)))
        best_pre = pre_candidates[0] if pre_candidates else None

        best_post = None
        if best_pre is not None:
            post_after_pre = [s for s in post_candidates if cls.sort_key(s) > cls.sort_key(best_pre)]
            if post_after_pre:
                post_after_pre.sort(key=lambda s: (cls.sort_key(s), -s["post_score"]))
                best_post = post_after_pre[0]

        if best_post is None and post_candidates:
            post_candidates.sort(key=lambda s: (-s["post_score"], cls.sort_key(s)))
            best_post = post_candidates[0]

        if best_pre is None:
            best_pre, fallback_post = cls._dynamic_fallback(ordered)
            if best_post is None:
                best_post = fallback_post

        if best_pre is not None and best_post is not None and best_pre["path"] == best_post["path"]:
            distinct_post = [s for s in post_candidates if s["path"] != best_pre["path"]]
            if distinct_post:
                distinct_post.sort(key=lambda s: (cls.sort_key(s), -s["post_score"]))
                best_post = distinct_post[0]

        return best_pre, best_post
