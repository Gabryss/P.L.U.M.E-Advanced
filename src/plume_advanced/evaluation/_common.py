"""Small dependency-free helpers shared by the evaluation modules."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def value(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute or mapping key from *obj*."""

    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def finite(value_: Any) -> float | None:
    try:
        result = float(value_)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quantile(values: Iterable[float], probability: float) -> float | None:
    """Linear-interpolated quantile with a stable, numpy-independent definition."""

    ordered = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not clean:
        return {
            "count": 0,
            "q1": None,
            "median": None,
            "q3": None,
            "iqr": None,
            "min": None,
            "max": None,
            "mean": None,
        }
    q1 = quantile(clean, 0.25)
    median = quantile(clean, 0.5)
    q3 = quantile(clean, 0.75)
    return {
        "count": len(clean),
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": q3 - q1 if q1 is not None and q3 is not None else None,
        "min": clean[0],
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
    }


def as_xy(point: Any) -> tuple[float, float] | None:
    x = finite(value(point, "x", "x_coord", "longitude"))
    y = finite(value(point, "y", "y_coord", "latitude"))
    if x is not None and y is not None:
        return x, y
    raw = value(point, "position", "coordinate", "coords")
    try:
        if raw is not None and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        pass
    return None


def arc_lengths(points: list[Any]) -> list[float]:
    supplied = [finite(value(point, "arc_length", "distance", "s")) for point in points]
    if len(supplied) > 1 and all(item is not None for item in supplied):
        result = [float(item) for item in supplied]  # type: ignore[arg-type]
        if all(result[index] > result[index - 1] for index in range(1, len(result))):
            return result
    result = [0.0]
    for first, second in zip(points, points[1:]):
        p0, p1 = as_xy(first), as_xy(second)
        result.append(result[-1] + (math.dist(p0, p1) if p0 and p1 else 1.0))
    return result


def elevations(points: list[Any]) -> list[float | None]:
    return [finite(value(point, "elevation", "z", "height")) for point in points]
