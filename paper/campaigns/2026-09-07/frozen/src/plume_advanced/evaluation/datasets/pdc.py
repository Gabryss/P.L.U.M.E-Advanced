"""Auditable recursive loader for Pyroduct Digital Catalog v2 TXT sections."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.evaluation.metrics.contours import (
    ContourError,
    clean_contour,
    self_intersection_count,
)
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry
from plume_advanced.evaluation.provenance import directory_identity

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class PDCSection:
    reference_cave_id: str
    reference_section_id: str
    relative_path: str
    raw_points: tuple[tuple[float, float], ...]
    contour: tuple[tuple[float, float], ...]
    was_open: bool
    trailing_cycle_point_count: int
    duplicate_point_count: int
    self_intersection_count: int


@dataclass(frozen=True)
class PDCRejection:
    relative_path: str
    reason: str
    detail: str


def load_pdc(
    data_root: str | Path,
    *,
    endpoint_tolerance_m: float = 0.05,
    relative_endpoint_tolerance: float = 0.02,
    implicit_closure: bool = False,
    cave_ids: Collection[str] | None = None,
) -> tuple[list[PDCSection], list[PDCRejection]]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"PDC data root does not exist: {root}")
    sections: list[PDCSection] = []
    rejections: list[PDCRejection] = []
    # PDC station files are named ``cross-section_1``, ``cross-section_2``,
    # ... ``cross-section_10``.  Plain lexical sorting places station 10
    # between stations 1 and 2 and silently corrupts any longitudinal
    # analysis.  Keep cave/path ordering deterministic while comparing every
    # digit run numerically.
    for path in sorted(
        root.rglob("*.txt"),
        key=lambda item: _natural_path_key(item, root),
    ):
        relative = path.relative_to(root).as_posix()
        cave_id, section_id = _infer_identifiers(path, root)
        # Filter before reading coordinates: calibration must not inspect
        # held-out geometry, even when both partitions share an archive.
        if cave_ids is not None and cave_id not in cave_ids:
            continue
        try:
            raw = _parse_numeric_rows(path)
            if raw.shape[0] < 3:
                raise ContourError("fewer than three numeric coordinate rows")
            duplicate_count = int(
                np.count_nonzero(np.linalg.norm(np.diff(raw, axis=0), axis=1) <= 1e-12)
            )
            source_cycle, explicitly_closed, trailing_cycle_points = _source_cycle(raw)
            # Validate geometry before applying the closure policy so a
            # collinear file is reported as degenerate rather than merely open.
            contour = clean_contour(source_cycle)
            extent = np.ptp(source_cycle, axis=0)
            diagonal = float(np.linalg.norm(extent))
            closing_distance = float(np.linalg.norm(source_cycle[0] - source_cycle[-1]))
            was_open = not explicitly_closed and closing_distance > 1e-12
            tolerance = max(endpoint_tolerance_m, relative_endpoint_tolerance * diagonal)
            if was_open and closing_distance > tolerance and not implicit_closure:
                raise ContourError(
                    f"open contour endpoints are {closing_distance:.6g} m apart "
                    f"(closure tolerance {tolerance:.6g} m)"
                )
            intersections = self_intersection_count(contour)
            sections.append(
                PDCSection(
                    reference_cave_id=cave_id,
                    reference_section_id=section_id,
                    relative_path=relative,
                    raw_points=tuple(map(tuple, raw.tolist())),
                    contour=tuple(map(tuple, contour.tolist())),
                    was_open=was_open,
                    trailing_cycle_point_count=trailing_cycle_points,
                    duplicate_point_count=duplicate_count,
                    self_intersection_count=intersections,
                )
            )
        except (OSError, ValueError, ContourError) as error:
            rejections.append(
                PDCRejection(relative_path=relative, reason=_reason(error), detail=str(error))
            )
    return sections, rejections


def audit_pdc(
    data_root: str | Path,
    output_root: str | Path,
    *,
    endpoint_tolerance_m: float = 0.05,
    relative_endpoint_tolerance: float = 0.02,
    implicit_closure: bool = False,
) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    sections, rejections = load_pdc(
        root,
        endpoint_tolerance_m=endpoint_tolerance_m,
        relative_endpoint_tolerance=relative_endpoint_tolerance,
        implicit_closure=implicit_closure,
    )
    inventory_rows: list[dict[str, Any]] = []
    widths: list[float] = []
    heights: list[float] = []
    point_counts: list[int] = []
    for section in sections:
        metrics = contour_morphometry(section.contour)
        widths.append(metrics["width_m"])
        heights.append(metrics["height_m"])
        point_counts.append(len(section.contour) - 1)
        inventory_rows.append(
            {
                "reference_cave_id": section.reference_cave_id,
                "reference_section_id": section.reference_section_id,
                "relative_path": section.relative_path,
                "point_count": len(section.contour) - 1,
                "was_open": section.was_open,
                "trailing_cycle_point_count": section.trailing_cycle_point_count,
                "duplicate_point_count": section.duplicate_point_count,
                "self_intersection_count": section.self_intersection_count,
                **metrics,
            }
        )
    audit = {
        "schema": "plume.pdc-audit.v1",
        "dataset": {
            "title": "Pyroduct Digital Catalog",
            "version": "2.0",
            "zenodo_record": "17750755",
            "doi": "10.5281/zenodo.17750755",
            "recommended_citation_doi": "10.5281/zenodo.14535885",
            "source_archive_md5": "cccad95bbf3ef56d1bd48ac75273682c",
            **directory_identity(root),
        },
        "total_candidate_files": len(sections) + len(rejections),
        "accepted_contours": len(sections),
        "rejected_contours": len(rejections),
        "unique_cave_ids": len({section.reference_cave_id for section in sections}),
        "point_count_distribution": _distribution(point_counts),
        "width_m_distribution": _distribution(widths),
        "height_m_distribution": _distribution(heights),
        "suspicious_scale_outlier_count": sum(
            width < 0.25 or height < 0.25 or width > 500.0 or height > 500.0
            for width, height in zip(widths, heights, strict=True)
        ),
        "open_contour_count": sum(section.was_open for section in sections),
        "trailing_cycle_point_count": sum(
            section.trailing_cycle_point_count for section in sections
        ),
        "duplicate_point_count": sum(section.duplicate_point_count for section in sections),
        "self_intersection_count": sum(section.self_intersection_count for section in sections),
        "closure_policy": {
            "endpoint_tolerance_m": endpoint_tolerance_m,
            "relative_endpoint_tolerance": relative_endpoint_tolerance,
            "implicit_closure": implicit_closure,
            "explicit_cycle_return": (
                "the first return to the initial vertex closes the contour; "
                "trailing repeated prefix vertices are audited and excluded"
            ),
        },
    }
    (output / "pdc_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "pdc_sections_inventory.csv", inventory_rows)
    _write_csv(
        output / "pdc_rejections.csv",
        [
            {
                "relative_path": rejection.relative_path,
                "reason": rejection.reason,
                "detail": rejection.detail,
            }
            for rejection in rejections
        ],
        fieldnames=("relative_path", "reason", "detail"),
    )
    return audit


def _parse_numeric_rows(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        values = _NUMBER.findall(line)
        if len(values) < 2:
            continue
        rows.append((float(values[0]), float(values[1])))
    return np.asarray(rows, dtype=float)


def _source_cycle(raw: np.ndarray) -> tuple[np.ndarray, bool, int]:
    """Extract one explicit source cycle without discarding the preserved raw rows.

    PDC v2 TXT files encode a closed polyline as ``A ... Z, A, B, C``.  The
    first repeated ``A`` is the closure marker; ``B, C`` are repeated prefix
    vertices emitted by the catalog export and must not become extra edges.
    """

    if raw.shape[0] < 4 or not bool(np.all(np.isfinite(raw))):
        return raw, False, 0
    returns = np.flatnonzero(np.linalg.norm(raw[3:] - raw[0], axis=1) <= 1e-12)
    if returns.size == 0:
        return raw, False, 0
    closure_index = int(returns[0]) + 3
    trailing_count = int(raw.shape[0] - closure_index - 1)
    return raw[: closure_index + 1], True, trailing_count


def _infer_identifiers(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    cave_id = relative.parts[-2] if len(relative.parts) > 1 else path.stem.split("_")[0]
    return cave_id.strip() or "unknown", path.stem


def _natural_path_key(
    path: Path,
    root: Path,
) -> tuple[tuple[tuple[int, str | int], ...], ...]:
    """Return a deterministic, numeric-aware key for a path below ``root``."""

    return tuple(
        tuple(
            (1, int(token)) if token.isdigit() else (0, token.casefold())
            for token in re.split(r"(\d+)", part)
            if token
        )
        for part in path.relative_to(root).parts
    )


def _reason(error: Exception) -> str:
    message = str(error).lower()
    if "open contour" in message:
        return "open_contour"
    if "numeric" in message or "three" in message:
        return "insufficient_numeric_rows"
    if "nan" in message or "infinite" in message:
        return "non_finite"
    if "degenerate" in message or "collinear" in message or "distinct" in message:
        return "degenerate"
    return "parse_error"


def _distribution(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q1": float(np.percentile(array, 25.0)),
        "median": float(np.median(array)),
        "q3": float(np.percentile(array, 75.0)),
        "maximum": float(np.max(array)),
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


__all__ = ["PDCRejection", "PDCSection", "audit_pdc", "load_pdc"]
