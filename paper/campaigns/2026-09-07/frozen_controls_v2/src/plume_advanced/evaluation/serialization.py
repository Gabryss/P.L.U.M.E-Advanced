"""Canonical JSON encoding for machine-readable evaluation reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        # Round-trip-stable representation; retain useful precision and avoid
        # platform-specific numpy scalar formatting.
        return float(format(value, ".15g"))
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def diagnostics_to_json(report: Any, *, indent: int | None = None) -> str:
    """Serialize a report with sorted keys and no non-standard JSON values."""

    return json.dumps(
        _clean(report),
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        ensure_ascii=True,
        allow_nan=False,
    )


serialize_diagnostics = diagnostics_to_json
