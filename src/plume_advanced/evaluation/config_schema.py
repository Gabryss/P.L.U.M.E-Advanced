"""Validate every scientific experiment setting before starting work."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

Validator = Callable[[Any, str], None]


def _text(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")


def _boolean(value: Any, path: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")


def _number(minimum: float, maximum: float = math.inf, *, integer=False,
            exclusive_min=False, exclusive_max=False) -> Validator:
    def validate(value: Any, path: str) -> None:
        types = (int,) if integer else (int, float)
        if type(value) not in types or not math.isfinite(value):
            kind = "integer" if integer else "number"
            raise ValueError(f"{path} must be a finite {kind}")
        if (value < minimum or value > maximum
                or (exclusive_min and value == minimum)
                or (exclusive_max and value == maximum)):
            raise ValueError(f"{path} is outside its allowed range")
    return validate


def _choice(*values: str) -> Validator:
    def validate(value: Any, path: str) -> None:
        _text(value, path)
        if value not in values:
            raise ValueError(f"{path} must be one of: {', '.join(values)}")
    return validate


def _array(item_validator: Validator) -> Validator:
    def validate(value: Any, path: str) -> None:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{path} must be a nonempty array")
        for index, item in enumerate(value):
            item_validator(item, f"{path}[{index}]")
        if len(set(value)) != len(value):
            raise ValueError(f"{path} must not contain duplicates")
    return validate


def _table(fields: dict[str, Validator]) -> Validator:
    def validate(value: Any, path: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a TOML table")
        unknown = set(value) - fields.keys()
        if unknown:
            raise ValueError("Unknown evaluation configuration keys: " + ", ".join(
                f"{path}.{key}" for key in sorted(unknown)))
        for key, item in value.items():
            fields[key](item, f"{path}.{key}")
    return validate


_BODY = _choice("earth", "mars", "moon")
_POSITIVE = _number(0, exclusive_min=True)
_FRACTION = _number(0, 1)
_METRIC = _choice("width_m", "height_m", "aspect_ratio", "area_m2", "compactness",
                  "floor_residual_norm", "roof_asymmetry_norm")
_COMMON = {"body": _BODY, "seed_file": _text}
_SCHEMA = _table({
    "schema_version": _number(1, 1, integer=True),
    "general": _table({
        "output_root": _text, "project_config": _text, "asset_directory": _text,
        "bootstrap_iterations": _number(1, integer=True),
        "confidence_level": _number(0, 1, exclusive_min=True, exclusive_max=True),
        "bootstrap_seed": _number(0, 2**32 - 1, integer=True),
    }),
    "datasets": _table({"pdc": _table({key: _text for key in (
        "title", "version", "zenodo_record", "doi", "recommended_citation_doi",
        "source_archive_md5", "path_env", "calibration_caves", "evaluation_caves",
    )})}),
    "morphometry": _table({
        **_COMMON, "reference_partition": _choice("calibration", "evaluation", "all"),
        "exclude_self_intersections": _boolean, "stop_after_stage": _choice("sections"),
        "baseline": _choice("ellipse"), "metrics": _array(_METRIC),
        "aggregate_metrics": _array(_METRIC),
    }),
    "controllability": _table({
        **_COMMON, "distributary_values": _array(_FRACTION),
        "inflation_values": _array(_FRACTION), "duration_values": _array(_POSITIVE),
        "supply_values": _array(_POSITIVE),
    }),
    "host_ablation": _table({**_COMMON, "conditions": _array(_choice(
        "full", "no_slope", "no_cover", "no_fracture", "no_capacity", "no_stability",
        "unconditioned"))}),
    "sampling_ablation": _table({**_COMMON, "reference_spacing_m": _POSITIVE}),
    "scalability": _table({
        **_COMMON, "route_lengths_m": _array(_POSITIVE),
        "storage_modes": _array(_choice("dense", "tiled")),
        "quality": _choice("preview", "standard", "production"),
        "timeout_s": _POSITIVE, "memory_limit_gib": _POSITIVE,
    }),
    "export_consistency": _table({"seed_file": _text, "targets": _array(_choice(
        "blender", "ue5", "unity", "gazebo", "omniverse"))}),
    "determinism": _table(_COMMON),
})


def validate_evaluation_config(raw: dict[str, Any]) -> None:
    _SCHEMA(raw, "config")
    if "schema_version" not in raw:
        raise ValueError("schema_version is required")
    morphology = raw.get("morphometry", {})
    if "metrics" in morphology and "aggregate_metrics" in morphology:
        if not set(morphology["aggregate_metrics"]) <= set(morphology["metrics"]):
            raise ValueError("morphometry.aggregate_metrics must be included in metrics")
