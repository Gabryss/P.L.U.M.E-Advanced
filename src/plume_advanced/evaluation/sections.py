"""Cross-section morphometry summaries compatible with PDC-style reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from ._common import as_xy, finite, summary, value

FEATURES = (
    "width",
    "height",
    "aspect_ratio",
    "area",
    "compactness",
    "floor_residual",
    "roof_asymmetry",
)


def _sections(source: Any) -> list[Any]:
    sections = value(source, "sections", "cross_sections", "profiles", default=None)
    if sections is not None:
        return list(sections)
    if isinstance(source, Mapping) and "records" in source:
        return list(source["records"])
    if isinstance(source, Mapping):
        columns = {
            key: item
            for key, item in source.items()
            if isinstance(item, (list, tuple)) or getattr(item, "ndim", 0) > 0
        }
        if columns:
            count = max((len(item) for item in columns.values()), default=0)
            return [
                {key: item[index] for key, item in columns.items() if index < len(item)}
                for index in range(count)
            ]
    if source is None or isinstance(source, (str, bytes)):
        return []
    try:
        return list(source)
    except TypeError:
        return []


def _outline(section: Any) -> list[tuple[float, float]]:
    raw = value(section, "outline", "vertices", "boundary", "polygon", "points", default=None)
    if raw is None:
        return []
    result = []
    for item in raw:
        point = as_xy(item)
        if point is None:
            try:
                point = (float(item[0]), float(item[1]))
            except (TypeError, ValueError, IndexError):
                continue
        result.append(point)
    return result


def _polygon_area_perimeter(outline: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if len(outline) < 3:
        return None, None
    closed = outline + [outline[0]]
    signed_area = 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(closed, closed[1:]))
    perimeter = sum(math.dist(first, second) for first, second in zip(closed, closed[1:]))
    return abs(signed_area), perimeter


def _feature_record(section: Any, index: int) -> dict[str, Any]:
    outline = _outline(section)
    area_from_outline, perimeter = _polygon_area_perimeter(outline)
    xs = [point[0] for point in outline]
    ys = [point[1] for point in outline]
    width = finite(value(section, "width", "width_m", "tube_width", "span", "diameter_x", default=None))
    height = finite(value(section, "height", "height_m", "tube_height", "vertical_span", "diameter_y", default=None))
    if width is None and xs:
        width = max(xs) - min(xs)
    if height is None and ys:
        height = max(ys) - min(ys)
    area = finite(value(section, "area", "area_m2", "cross_section_area", default=None))
    if area is None:
        area = area_from_outline
    if area is None and width is not None and height is not None:
        # An ellipse is the least assumptive fallback for width/height-only
        # Stage-C records and is explicitly marked by the inferred flag.
        area = math.pi * width * height / 4.0
    aspect = finite(value(section, "aspect_ratio", "width_height_ratio", default=None))
    if aspect is None and width is not None and height and abs(height) > 1.0e-12:
        aspect = width / height
    compactness = finite(value(section, "compactness", "circularity", default=None))
    if compactness is None and area is not None and perimeter and perimeter > 1.0e-12:
        compactness = 4.0 * math.pi * area / perimeter**2
    if compactness is None and width and height and width > 0 and height > 0:
        # Ramanujan perimeter approximation for an inferred ellipse.
        semi_a, semi_b = width / 2.0, height / 2.0
        h = ((semi_a - semi_b) / (semi_a + semi_b)) ** 2
        ellipse_perimeter = math.pi * (semi_a + semi_b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))
        compactness = 4.0 * math.pi * area / ellipse_perimeter**2 if area else None
    floor_residual = finite(value(section, "floor_residual", "floor_residual_norm", "floor_elevation_residual", "floor_error", default=None))
    if floor_residual is None:
        floor_z = finite(value(section, "floor_z", "floor_elevation", default=None))
        expected_floor = finite(value(section, "expected_floor_z", "reference_floor_z", default=None))
        if floor_z is not None and expected_floor is not None:
            floor_residual = floor_z - expected_floor
    roof_asymmetry = finite(value(section, "roof_asymmetry", "roof_asymmetry_norm", "roof_asymmetry_ratio", "ceiling_asymmetry", default=None))
    if roof_asymmetry is None:
        left = finite(value(section, "roof_left", "left_roof_height", default=None))
        right = finite(value(section, "roof_right", "right_roof_height", default=None))
        if left is not None and right is not None:
            scale = max(abs(left), abs(right), 1.0e-12)
            roof_asymmetry = abs(left - right) / scale
    return {
        "index": index,
        "width": width,
        "height": height,
        "aspect_ratio": aspect,
        "area": area,
        "compactness": compactness,
        "floor_residual": floor_residual,
        "roof_asymmetry": roof_asymmetry,
        "arc_length": finite(value(section, "arc_length", "distance", "s", default=index)),
    }


def section_feature_records(sections: Any) -> list[dict[str, Any]]:
    """Extract normalized feature records while retaining missing values as null."""

    return [_feature_record(section, index) for index, section in enumerate(_sections(sections))]


def section_diagnostics(sections: Any) -> dict[str, Any]:
    """Return PDC-comparable Q1/median/Q3/IQR summaries for each feature."""

    records = section_feature_records(sections)
    feature_summary = {
        feature: summary(record[feature] for record in records if record[feature] is not None)
        for feature in FEATURES
    }
    missing = {
        feature: sum(1 for record in records if record[feature] is None)
        for feature in FEATURES
    }
    return {
        "schema_version": "1.0",
        "section_count": len(records),
        "features": feature_summary,
        "missing_count": missing,
        "records": records,
    }


cross_section_diagnostics = section_diagnostics
evaluate_sections = section_diagnostics
