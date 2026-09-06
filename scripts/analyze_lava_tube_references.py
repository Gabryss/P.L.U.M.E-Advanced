#!/usr/bin/env python3
"""Analyze PDC calibration sections and the USGS Valentine Cave point cloud.

This script deliberately excludes the frozen PDC evaluation partition.  The
Valentine LiDAR product is treated as an independent planform/longitudinal
case study; its PCA-aligned spans are envelopes and must not be mistaken for
single-passage cross-sections where branches overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from plume_advanced.evaluation.datasets.pdc import load_pdc
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDC_ROOT = ROOT / "data/reference/pdc_v2/extracted/Pyroduct Digital Catalog in .txt"
DEFAULT_VALENTINE = ROOT / "data/reference/valentine/Valentine_TUBE_UTM_10cm.copc.laz"
DEFAULT_CALIBRATION_SPLIT = ROOT / "paper/splits/pdc_calibration_caves.txt"
DEFAULT_EVALUATION_SPLIT = ROOT / "paper/splits/pdc_evaluation_caves.txt"
DEFAULT_OUTPUT = ROOT / "outputs/reference_morphology"
DESCRIPTORS = ("width_m", "height_m", "aspect_ratio", "area_m2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdc-root", type=Path, default=DEFAULT_PDC_ROOT)
    parser.add_argument("--valentine-laz", type=Path, default=DEFAULT_VALENTINE)
    parser.add_argument("--calibration-split", type=Path, default=DEFAULT_CALIBRATION_SPLIT)
    parser.add_argument("--evaluation-split", type=Path, default=DEFAULT_EVALUATION_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--valentine-bin-m", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.valentine_bin_m <= 0.0:
        raise ValueError("--valentine-bin-m must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    calibration_ids = _read_partition(args.calibration_split)
    evaluation_ids = _read_partition(args.evaluation_split)
    overlap = calibration_ids & evaluation_ids
    if overlap:
        raise ValueError(f"PDC calibration/evaluation partitions overlap: {sorted(overlap)}")

    pdc_rows, pdc_longitudinal, pdc_audit = _analyze_pdc(
        args.pdc_root,
        calibration_ids,
        evaluation_ids,
    )
    valentine_rows, valentine_summary, valentine_points = _analyze_valentine(
        args.valentine_laz,
        args.valentine_bin_m,
    )

    _write_csv(args.output / "pdc_calibration_sections.csv", pdc_rows)
    _write_csv(args.output / "pdc_cave_longitudinal.csv", pdc_longitudinal)
    _write_csv(args.output / "valentine_longitudinal_envelope.csv", valentine_rows)
    _plot_pdc_distributions(pdc_rows, args.output / "pdc_cross_section_distributions.png")
    _plot_pdc_longitudinal(
        pdc_longitudinal,
        args.output / "pdc_longitudinal_variation.png",
    )
    _plot_valentine(
        valentine_points,
        valentine_rows,
        args.output / "valentine_planform_and_envelope.png",
    )
    _plot_comparison(
        pdc_rows,
        valentine_rows,
        args.output / "pdc_valentine_scale_comparison.png",
    )

    payload = {
        "schema": "plume.reference-morphology.v1",
        "scope": {
            "pdc_partition": "calibration_only",
            "pdc_evaluation_sections_used_in_metrics": 0,
            "valentine_role": "independent_planform_and_longitudinal_case_study",
            "valentine_envelope_warning": (
                "PCA-aligned spans can cover several parallel passages and are not "
                "single-passage cross-sections."
            ),
        },
        "pdc": {
            **pdc_audit,
            "descriptors": {
                descriptor: _distribution([float(row[descriptor]) for row in pdc_rows])
                for descriptor in DESCRIPTORS
            },
            "relative_width_to_cave_median": _distribution(
                [float(row["relative_width_to_cave_median"]) for row in pdc_rows]
            ),
            "longitudinal": {
                key: _distribution([float(row[key]) for row in pdc_longitudinal if row[key] != ""])
                for key in (
                    "width_lag1_correlation",
                    "height_lag1_correlation",
                    "median_abs_log_width_change",
                    "p90_abs_log_width_change",
                    "maximum_adjacent_width_ratio",
                )
            },
        },
        "valentine": valentine_summary,
        "outputs": [
            "pdc_calibration_sections.csv",
            "pdc_cave_longitudinal.csv",
            "pdc_cross_section_distributions.png",
            "pdc_longitudinal_variation.png",
            "pdc_valentine_scale_comparison.png",
            "reference_summary.json",
            "valentine_longitudinal_envelope.csv",
            "valentine_planform_and_envelope.png",
        ],
    }
    (args.output / "reference_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _read_partition(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _analyze_pdc(
    root: Path,
    calibration_ids: set[str],
    evaluation_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sections, rejections = load_pdc(root)
    encountered_evaluation = {
        section.reference_cave_id
        for section in sections
        if section.reference_cave_id in evaluation_ids
    }
    selected = [
        section
        for section in sections
        if section.reference_cave_id in calibration_ids and section.self_intersection_count == 0
    ]
    if not selected:
        raise ValueError("No valid PDC calibration sections were found")

    metrics_by_cave: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for section in selected:
        metrics = contour_morphometry(section.contour)
        row = {
            "reference_cave_id": section.reference_cave_id,
            "reference_section_id": section.reference_section_id,
            "relative_path": section.relative_path,
            **metrics,
        }
        rows.append(row)
        metrics_by_cave[section.reference_cave_id].append(row)

    cave_width_medians = {
        cave_id: float(np.median([float(row["width_m"]) for row in cave_rows]))
        for cave_id, cave_rows in metrics_by_cave.items()
    }
    for row in rows:
        median = cave_width_medians[str(row["reference_cave_id"])]
        row["relative_width_to_cave_median"] = float(row["width_m"]) / max(median, 1e-9)

    longitudinal = [
        _longitudinal_record(cave_id, cave_rows)
        for cave_id, cave_rows in sorted(metrics_by_cave.items())
        if len(cave_rows) >= 3
    ]
    audit = {
        "candidate_sections": len(sections) + len(rejections),
        "loader_rejections": len(rejections),
        "calibration_caves_declared": len(calibration_ids),
        "calibration_caves_present": len(metrics_by_cave),
        "calibration_sections": len(rows),
        "self_intersections_excluded": sum(
            section.reference_cave_id in calibration_ids and section.self_intersection_count > 0
            for section in sections
        ),
        "evaluation_caves_present_in_archive_but_excluded": len(encountered_evaluation),
    }
    return rows, longitudinal, audit


def _longitudinal_record(cave_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "reference_cave_id": cave_id,
        "section_count": len(rows),
    }
    for descriptor in ("width_m", "height_m", "area_m2", "aspect_ratio"):
        values = np.asarray([float(row[descriptor]) for row in rows], dtype=float)
        log_changes = np.abs(np.diff(np.log(np.maximum(values, 1e-9))))
        correlation = _lag1_correlation(np.log(np.maximum(values, 1e-9)))
        name = descriptor.removesuffix("_m").removesuffix("_m2")
        record[f"{name}_lag1_correlation"] = correlation
        record[f"median_abs_log_{name}_change"] = float(np.median(log_changes))
        record[f"p90_abs_log_{name}_change"] = float(np.percentile(log_changes, 90.0))
    widths = np.asarray([float(row["width_m"]) for row in rows], dtype=float)
    adjacent_ratios = np.maximum(
        widths[1:] / np.maximum(widths[:-1], 1e-9),
        widths[:-1] / np.maximum(widths[1:], 1e-9),
    )
    record["maximum_adjacent_width_ratio"] = float(np.max(adjacent_ratios))
    return record


def _lag1_correlation(values: np.ndarray) -> float | str:
    if values.size < 3 or float(np.std(values[:-1])) <= 1e-12:
        return ""
    if float(np.std(values[1:])) <= 1e-12:
        return ""
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _analyze_valentine(
    path: Path,
    bin_width_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    try:
        import laspy
    except ImportError as error:  # pragma: no cover - depends on optional paper extra
        raise RuntimeError(
            "Install the paper extra to read COPC LAZ: uv sync --extra paper"
        ) from error

    cloud = laspy.read(path)
    xy = np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y)))
    z = np.asarray(cloud.z, dtype=float)
    intensity = np.asarray(cloud.intensity, dtype=float)
    center = np.median(xy, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((xy - center).T))
    along_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if along_axis[1] < 0.0:
        along_axis = -along_axis
    cross_axis = np.asarray((-along_axis[1], along_axis[0]))
    along = (xy - center) @ along_axis
    cross = (xy - center) @ cross_axis
    along -= float(np.min(along))

    edges = np.arange(0.0, float(np.max(along)) + bin_width_m, bin_width_m)
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (along >= lower) & (along < upper)
        if int(np.count_nonzero(mask)) < 100:
            continue
        local_cross = cross[mask]
        local_z = z[mask]
        width = float(np.percentile(local_cross, 99.0) - np.percentile(local_cross, 1.0))
        height = float(np.percentile(local_z, 99.0) - np.percentile(local_z, 1.0))
        rows.append(
            {
                "along_m": 0.5 * (lower + upper),
                "envelope_width_m": width,
                "envelope_height_m": height,
                "envelope_aspect_ratio": width / max(height, 1e-9),
                "point_count": int(np.count_nonzero(mask)),
                "median_elevation_m": float(np.median(local_z)),
            }
        )

    widths = [float(row["envelope_width_m"]) for row in rows]
    heights = [float(row["envelope_height_m"]) for row in rows]
    aspects = [float(row["envelope_aspect_ratio"]) for row in rows]
    summary = {
        "source": "USGS NASA TubeX Valentine Cave 2018 Valentine LiDAR",
        "doi": "10.5066/P14AC3J5",
        "file": path.name,
        "point_count": int(xy.shape[0]),
        "robust_longitudinal_extent_m": float(np.ptp(along)),
        "robust_planform_cross_extent_m": float(
            np.percentile(cross, 99.0) - np.percentile(cross, 1.0)
        ),
        "elevation_extent_m": float(np.ptp(z)),
        "bin_width_m": bin_width_m,
        "envelope_width_m": _distribution(widths),
        "envelope_height_m": _distribution(heights),
        "envelope_aspect_ratio": _distribution(aspects),
        "wide_envelope_fraction_above_20m": float(np.mean(np.asarray(widths) >= 20.0)),
        "intensity": _distribution(intensity.tolist()),
    }
    points = {
        "along": along,
        "cross": cross,
        "z": z,
        "intensity": intensity,
    }
    return rows, summary, points


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q05": float(np.percentile(array, 5.0)),
        "q25": float(np.percentile(array, 25.0)),
        "median": float(np.median(array)),
        "q75": float(np.percentile(array, 75.0)),
        "q95": float(np.percentile(array, 95.0)),
        "q99": float(np.percentile(array, 99.0)),
        "maximum": float(np.max(array)),
    }


def _plot_pdc_distributions(rows: list[dict[str, Any]], output: Path) -> None:
    labels = {
        "width_m": "Width (m)",
        "height_m": "Height (m)",
        "aspect_ratio": "Width / height",
        "area_m2": "Area (m²)",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, descriptor in zip(axes.flat, DESCRIPTORS, strict=True):
        values = np.asarray([float(row[descriptor]) for row in rows], dtype=float)
        display_max = float(np.percentile(values, 99.0))
        axis.hist(values[values <= display_max], bins=35, color="#2563eb", alpha=0.82)
        axis.axvline(np.median(values), color="#f97316", linewidth=2, label="median")
        axis.axvline(np.percentile(values, 95.0), color="#dc2626", linestyle="--", label="95%")
        axis.set_xlabel(labels[descriptor])
        axis.set_ylabel("Calibration sections")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("PDC v2 — 76-cave calibration partition (held-out caves excluded)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_pdc_longitudinal(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for descriptor, color in (("width", "#2563eb"), ("height", "#f97316")):
        correlations = [
            float(row[f"{descriptor}_lag1_correlation"])
            for row in rows
            if row[f"{descriptor}_lag1_correlation"] != ""
        ]
        changes = [float(row[f"median_abs_log_{descriptor}_change"]) for row in rows]
        axes[0].hist(correlations, bins=18, alpha=0.58, label=descriptor, color=color)
        axes[1].hist(changes, bins=18, alpha=0.58, label=descriptor, color=color)
    ratios = [float(row["maximum_adjacent_width_ratio"]) for row in rows]
    axes[2].hist(np.minimum(ratios, np.percentile(ratios, 98.0)), bins=20, color="#7c3aed")
    axes[0].set_xlabel("Lag-1 log-dimension correlation")
    axes[1].set_xlabel("Median |log adjacent ratio|")
    axes[2].set_xlabel("Maximum adjacent width ratio (98% clipped)")
    for axis in axes:
        axis.set_ylabel("Caves")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("PDC calibration longitudinal heterogeneity (numeric station order)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_valentine(
    points: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    plan = figure.add_subplot(grid[:, 0])
    width_axis = figure.add_subplot(grid[0, 1])
    height_axis = figure.add_subplot(grid[1, 1])
    stride = max(1, len(points["along"]) // 120_000)
    plan.scatter(
        points["cross"][::stride],
        points["along"][::stride],
        c=points["z"][::stride],
        s=0.18,
        cmap="terrain",
        rasterized=True,
    )
    plan.set_aspect("equal")
    plan.set_xlabel("PCA cross-flow coordinate (m)")
    plan.set_ylabel("PCA along-flow coordinate (m)")
    plan.set_title("USGS Valentine 10 cm point cloud — planform")
    along = [float(row["along_m"]) for row in rows]
    widths = [float(row["envelope_width_m"]) for row in rows]
    heights = [float(row["envelope_height_m"]) for row in rows]
    aspects = [float(row["envelope_aspect_ratio"]) for row in rows]
    width_axis.plot(along, widths, color="#2563eb", linewidth=1.7)
    width_axis.axhline(20.0, color="#dc2626", linestyle="--", label="20 m wide envelope")
    width_axis.set_ylabel("Robust envelope width (m)")
    width_axis.legend(frameon=False)
    twin = height_axis.twinx()
    height_axis.plot(along, heights, color="#f97316", linewidth=1.7, label="height")
    twin.plot(along, aspects, color="#7c3aed", linewidth=1.2, alpha=0.8, label="aspect")
    height_axis.set_xlabel("PCA along-flow coordinate (m)")
    height_axis.set_ylabel("Robust envelope height (m)", color="#f97316")
    twin.set_ylabel("Envelope width / height", color="#7c3aed")
    for axis in (plan, width_axis, height_axis):
        axis.grid(alpha=0.2)
    figure.suptitle("Valentine cave: large rooms are localized, not a global scale increase")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_comparison(
    pdc_rows: list[dict[str, Any]],
    valentine_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    pairs = (
        ("width_m", "envelope_width_m", "Width (m)"),
        ("height_m", "envelope_height_m", "Height (m)"),
        ("aspect_ratio", "envelope_aspect_ratio", "Width / height"),
    )
    for axis, (pdc_key, valentine_key, label) in zip(axes, pairs, strict=True):
        pdc = np.sort(np.asarray([float(row[pdc_key]) for row in pdc_rows]))
        valentine = np.sort(np.asarray([float(row[valentine_key]) for row in valentine_rows]))
        axis.plot(pdc, np.linspace(0.0, 1.0, pdc.size), label="PDC calibration", color="#2563eb")
        axis.plot(
            valentine,
            np.linspace(0.0, 1.0, valentine.size),
            label="Valentine multi-route envelope",
            color="#dc2626",
        )
        axis.set_xlim(0.0, max(np.percentile(pdc, 99.0), np.percentile(valentine, 99.0)))
        axis.set_xlabel(label)
        axis.set_ylabel("Empirical cumulative fraction")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Scale context — Valentine envelope is not a PDC-style passage section")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plume-matplotlib")
    raise SystemExit(main())
