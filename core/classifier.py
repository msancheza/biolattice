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
    HARD_EXCLUSION_REASONS = {
        "derived/postprocessed series",
        "localizer/scout",
        "diffusion",
    }
    STRONG_PENALTY_REASONS = {
        "non-T1",
    }
    SOFT_PENALTY_REASONS = {
        "late numbered pass",
        "explicit non-fatsat",
    }

    @classmethod
    def _load_rules(cls) -> None:
        if not cls._RULES:
            rules_path = os.path.join(os.path.dirname(__file__), "classifier_rules.json")
            try:
                with open(rules_path, "r") as f:
                    loaded = json.load(f)
                cls._validate_rules(loaded)
                cls._RULES = loaded
            except Exception as e:
                # Fallback or empty rules if file not found
                cls._RULES = {"min_selection_score": 2.0}

    @classmethod
    def _validate_rules(cls, rules: dict[str, Any]) -> None:
        if not isinstance(rules, dict):
            raise ValueError("classifier_rules.json must contain a top-level object")

        for key in ("min_selection_score", "base_exclusions", "base_scoring", "vendor_rules"):
            if key not in rules:
                raise ValueError(f"classifier_rules.json missing required key: {key}")

        for idx, rule in enumerate(rules.get("base_exclusions", [])):
            if "keywords" not in rule or "reason" not in rule:
                raise ValueError(f"Invalid base_exclusions[{idx}] rule: requires keywords and reason")

        for idx, rule in enumerate(rules.get("base_scoring", [])):
            if "keywords" not in rule or "score_type" not in rule or "value" not in rule or "reason" not in rule:
                raise ValueError(
                    f"Invalid base_scoring[{idx}] rule: requires keywords, score_type, value, and reason"
                )

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

    @staticmethod
    def _relative_diff(a: float, b: float) -> float:
        denom = max(abs(float(a)), abs(float(b)), 1e-6)
        return abs(float(a) - float(b)) / denom

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

        # 3. Downweight later dynamic passes when the label does not indicate a baseline phase.
        _is_late_pass = bool(re.search(r"[2-9](nd|rd|th)", text))
        _has_pass_word = "pass" in text
        _has_pre_anchor = cls._contains_any(text, ("pre", "1st", "baseline"))
        if (_is_late_pass or _has_pass_word) and not _has_pre_anchor:
            score.exclude("late numbered pass", penalty=4.0)

        # 4. Give a small PRE preference to dynamic labels that do not carry an explicit phase marker.
        _has_pure_dynamic = cls._contains_any(text, ("dyn", "vibrant"))
        _has_phase_marker = cls._contains_any(text, ("ph", "phase", "fase"))
        if _has_pure_dynamic and not _has_phase_marker:
            score.add_pre(1.0, "pure dynamic without phase marker → pre bias")

        # 5. Reinforce PRE scoring when only PRE-style wording is present and no phase index is available.
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
    def _is_t1_like(cls, series_meta: dict[str, Any]) -> bool:
        text = str(series_meta.get("semantic_text", ""))
        return cls._contains_any(
            text,
            ("t1", "fl2d", "3d", "vibrant", "thrive", "lava", "flash", "vibe", "twist"),
        )

    @classmethod
    def _is_dynamic_like(cls, series_meta: dict[str, Any]) -> bool:
        cls._load_rules()
        text = str(series_meta.get("semantic_text", ""))
        dyn_kws = tuple(cls._RULES.get("dynamic_keywords", []))
        return cls._contains_any(text, dyn_kws)

    @classmethod
    def _is_derived_series(cls, series_meta: dict[str, Any]) -> bool:
        reason_hits = set(series_meta.get("exclusion_reasons", []))
        if "derived/postprocessed series" in reason_hits:
            return True
        text = str(series_meta.get("semantic_text", ""))
        return cls._contains_any(text, ("sub", "subtracted", "mip", "mips", "map", "maps", "parametric", "fusion"))

    @classmethod
    def _has_hard_exclusion(cls, series_meta: dict[str, Any]) -> bool:
        """Return True only for series that are never considered valid semantic pairing inputs."""
        reasons = set(series_meta.get("exclusion_reasons", []))
        return any(reason in cls.HARD_EXCLUSION_REASONS for reason in reasons)

    @classmethod
    def _series_penalty_level(cls, series_meta: dict[str, Any]) -> tuple[int, list[str]]:
        """
        Map exclusion reasons onto a ranking severity used during pair selection.

        Levels:
        0 = clean
        1 = soft penalty
        2 = strong penalty
        3 = hard exclusion
        """
        reasons = set(series_meta.get("exclusion_reasons", []))
        if any(reason in cls.HARD_EXCLUSION_REASONS for reason in reasons):
            return 3, sorted(reason for reason in reasons if reason in cls.HARD_EXCLUSION_REASONS)
        if any(reason in cls.STRONG_PENALTY_REASONS for reason in reasons):
            return 2, sorted(reason for reason in reasons if reason in cls.STRONG_PENALTY_REASONS)
        if any(reason in cls.SOFT_PENALTY_REASONS for reason in reasons):
            return 1, sorted(reason for reason in reasons if reason in cls.SOFT_PENALTY_REASONS)
        return 0, []

    @classmethod
    def _pairing_affinity(
        cls,
        pre: dict[str, Any] | None,
        post: dict[str, Any] | None,
    ) -> tuple[int, list[str]]:
        """Score methodological affinity between a selected PRE and a candidate POST."""
        if not pre or not post:
            return -10, ["missing_pair"]

        notes: list[str] = []
        affinity = 0
        pre_dynamic = cls._is_dynamic_like(pre)
        post_dynamic = cls._is_dynamic_like(post)

        if pre_dynamic and post_dynamic:
            affinity += 3
            notes.append("dynamic_pair_match")
        elif pre_dynamic and not post_dynamic:
            affinity -= 3
            notes.append("dynamic_pre_non_dynamic_post")

        pre_t1_like = cls._is_t1_like(pre)
        post_t1_like = cls._is_t1_like(post)
        if pre_t1_like and post_t1_like:
            affinity += 1
            notes.append("t1_family_match")

        if pre.get("manufacturer_family") == post.get("manufacturer_family"):
            affinity += 1
            notes.append("vendor_family_match")

        compat = cls._pairing_compatibility(pre, post)
        if compat["compatible"]:
            affinity += 2
            notes.append("geometry_match")
        else:
            affinity -= 4
            notes.extend(compat.get("reasons", []))

        penalty_level, penalty_notes = cls._series_penalty_level(post)
        if penalty_level == 2:
            affinity -= 3
            notes.append("strong_penalty_post")
        elif penalty_level == 1:
            affinity -= 1
            notes.append("soft_penalty_post")
        notes.extend(f"penalty:{reason}" for reason in penalty_notes)

        return affinity, notes

    @classmethod
    def _pair_selection_key(
        cls,
        pre: dict[str, Any],
        post: dict[str, Any],
    ) -> tuple[Any, ...]:
        """
        Rank PRE/POST candidates jointly using compatibility, affinity, and semantic strength.
        """
        compat = cls._pairing_compatibility(pre, post)
        affinity, _ = cls._pairing_affinity(pre, post)
        pre_dynamic = cls._is_dynamic_like(pre)
        post_dynamic = cls._is_dynamic_like(post)
        pre_penalty, _ = cls._series_penalty_level(pre)
        post_penalty, _ = cls._series_penalty_level(post)

        # Favor dynamic-to-dynamic pairs when both series belong to the same temporal family.
        dynamic_pair_bonus = 1 if pre_dynamic and post_dynamic else 0
        non_dynamic_pre_penalty = 1 if (post_dynamic and not pre_dynamic) else 0

        return (
            0 if compat["compatible"] else 1,
            -dynamic_pair_bonus,
            non_dynamic_pre_penalty,
            -affinity,
            pre_penalty + post_penalty,
            -(pre["pre_score"] + post["post_score"]),
            -post["post_score"],
            -pre["pre_score"],
            cls.sort_key(pre),
            cls.sort_key(post),
        )

    @classmethod
    def _filter_series_pool(
        cls,
        candidates: list[dict[str, Any]],
        *,
        prefer_t1: bool,
        prefer_dynamic: bool = False,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        notes: list[str] = []
        filtered = [s for s in candidates if not cls._is_derived_series(s)]
        removed_derived = len(candidates) - len(filtered)
        if removed_derived > 0:
            notes.append(f"removed_derived:{removed_derived}")

        if not filtered:
            filtered = list(candidates)

        if prefer_dynamic:
            dynamic_like = [s for s in filtered if cls._is_dynamic_like(s)]
            removed_non_dynamic = len(filtered) - len(dynamic_like)
            if dynamic_like and removed_non_dynamic > 0:
                notes.append(f"deprioritized_non_dynamic:{removed_non_dynamic}")
                filtered = dynamic_like

        if prefer_t1:
            t1_like = [s for s in filtered if cls._is_t1_like(s)]
            removed_non_t1 = len(filtered) - len(t1_like)
            if t1_like and removed_non_t1 > 0:
                notes.append(f"deprioritized_non_t1:{removed_non_t1}")
                filtered = t1_like

        return filtered, notes

    @classmethod
    def _pairing_compatibility(
        cls,
        pre: dict[str, Any] | None,
        post: dict[str, Any] | None,
    ) -> dict[str, Any]:
        diagnostics = {
            "compatible": True,
            "reasons": [],
            "pixel_spacing_pre": [],
            "pixel_spacing_post": [],
            "pixel_spacing_rel_diff": [],
            "slice_thickness_pre": None,
            "slice_thickness_post": None,
            "slice_thickness_rel_diff": None,
        }
        if not pre or not post:
            diagnostics["compatible"] = False
            diagnostics["reasons"].append("missing_pair")
            return diagnostics

        spacing_pre = [cls._safe_float(v, 0.0) for v in pre.get("pixel_spacing", [])[:2]]
        spacing_post = [cls._safe_float(v, 0.0) for v in post.get("pixel_spacing", [])[:2]]
        diagnostics["pixel_spacing_pre"] = spacing_pre
        diagnostics["pixel_spacing_post"] = spacing_post

        if len(spacing_pre) == 2 and len(spacing_post) == 2:
            spacing_rel = [cls._relative_diff(a, b) for a, b in zip(spacing_pre, spacing_post)]
            diagnostics["pixel_spacing_rel_diff"] = spacing_rel
            if any(diff > config.CLASSIFIER_MAX_PIXEL_SPACING_REL_DIFF for diff in spacing_rel):
                diagnostics["compatible"] = False
                diagnostics["reasons"].append("pixel_spacing_mismatch")

        thick_pre = cls._safe_float(pre.get("slice_thickness", 0.0), 0.0)
        thick_post = cls._safe_float(post.get("slice_thickness", 0.0), 0.0)
        diagnostics["slice_thickness_pre"] = thick_pre
        diagnostics["slice_thickness_post"] = thick_post
        diagnostics["slice_thickness_rel_diff"] = cls._relative_diff(thick_pre, thick_post)
        if diagnostics["slice_thickness_rel_diff"] > config.CLASSIFIER_MAX_SLICE_THICKNESS_REL_DIFF:
            diagnostics["compatible"] = False
            diagnostics["reasons"].append("slice_thickness_mismatch")

        dicoms_pre = cls._safe_int(pre.get("num_dicoms", 0), 0)
        dicoms_post = cls._safe_int(post.get("num_dicoms", 0), 0)
        if dicoms_pre > 0 and dicoms_post > 0:
            dicom_rel_diff = abs(dicoms_pre - dicoms_post) / max(dicoms_pre, dicoms_post)
            diagnostics["num_dicoms_pre"] = dicoms_pre
            diagnostics["num_dicoms_post"] = dicoms_post
            diagnostics["num_dicoms_rel_diff"] = round(dicom_rel_diff, 4)
            if dicom_rel_diff > config.CLASSIFIER_MAX_NUM_DICOMS_REL_DIFF:
                diagnostics["compatible"] = False
                diagnostics["reasons"].append("num_dicoms_mismatch")

        return diagnostics

    @classmethod
    def _attach_selection_metadata(
        cls,
        selected: dict[str, Any] | None,
        notes: list[str],
        pairing: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if selected is None:
            return None
        enriched = dict(selected)
        enriched["selection_notes"] = list(notes)
        if pairing is not None:
            enriched["pairing_compatibility"] = pairing
        return enriched

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
        pre_candidates = [s for s in ordered if s["pre_score"] >= min_score and not cls._has_hard_exclusion(s)]
        post_candidates = [s for s in ordered if s["post_score"] >= min_score and not cls._has_hard_exclusion(s)]

        pre_candidates, pre_notes = cls._filter_series_pool(pre_candidates, prefer_t1=True)
        post_candidates, post_notes = cls._filter_series_pool(post_candidates, prefer_t1=False)

        pair_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for pre in pre_candidates:
            posts_after_pre = [
                post
                for post in post_candidates
                if post["path"] != pre["path"] and cls.sort_key(post) > cls.sort_key(pre)
            ]
            for post in posts_after_pre:
                pair_candidates.append((pre, post))

        best_pre = None

        best_post = None

        if pair_candidates:
            best_pre, best_post = min(
                pair_candidates,
                key=lambda pair: cls._pair_selection_key(pair[0], pair[1]),
            )

        if best_pre is None and pre_candidates:
            pre_candidates.sort(key=lambda s: (-s["pre_score"], cls.sort_key(s)))
            best_pre = pre_candidates[0]

        if best_post is None and best_pre is not None and post_candidates:
            if cls._is_dynamic_like(best_pre):
                post_candidates, dynamic_post_notes = cls._filter_series_pool(
                    post_candidates,
                    prefer_t1=False,
                    prefer_dynamic=True,
                )
                post_notes.extend(dynamic_post_notes)

            ranked_post_candidates = [
                post for post in post_candidates
                if post["path"] != best_pre["path"] and cls.sort_key(post) > cls.sort_key(best_pre)
            ]
            if ranked_post_candidates:
                best_post = min(
                    ranked_post_candidates,
                    key=lambda s: cls._pair_selection_key(best_pre, s),
                )

        if best_pre is None or best_post is None:
            fallback_pre, fallback_post = cls._dynamic_fallback(ordered)
            if best_pre is None:
                best_pre = fallback_pre
            if best_post is None:
                best_post = fallback_post

        pairing = cls._pairing_compatibility(best_pre, best_post)
        if best_pre is not None and best_post is not None:
            affinity_score, affinity_notes = cls._pairing_affinity(best_pre, best_post)
            post_notes.extend(
                [note for note in affinity_notes if note not in post_notes]
            )
            post_notes.append(f"pairing_affinity:{affinity_score}")
        best_pre = cls._attach_selection_metadata(best_pre, pre_notes, pairing)
        best_post = cls._attach_selection_metadata(best_post, post_notes, pairing)
        return best_pre, best_post
