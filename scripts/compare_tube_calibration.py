#!/usr/bin/env python3
"""Compare the accepted Earth scenario with calibration-only surveyed sections.

This is a discrepancy assessment, not a fit to the evaluation partition or a
claim that a single deliberately low-roof scenario spans the global catalogue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plume_advanced.evaluation.datasets.pdc import load_pdc
from plume_advanced.evaluation.metrics.contours import clean_contour
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry

ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "width_m",
    "height_m",
    "aspect_ratio",
    "compactness",
    "solidity",
    "floor_residual_norm",
    "roof_asymmetry_norm",
)


def uniform_boundary(points, count=512):
    contour = clean_contour(points)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))]
    locations = np.linspace(0.0, distance[-1], count, endpoint=False)
    return np.column_stack([np.interp(locations, distance, contour[:, axis]) for axis in range(2)])


def metrics(points):
    result = contour_morphometry(points)
    # Use equal boundary sampling for point-weighted shape descriptors. Keep
    # exact source-polyline extrema/area/perimeter for the other measurements.
    uniform = contour_morphometry(uniform_boundary(points))
    for key in ("floor_residual_norm", "roof_asymmetry_norm"):
        result[key] = uniform[key]
    return result


def weighted_summary(rows, weights):
    weights = np.asarray(weights, float)
    weights /= weights.sum()
    result = {}
    for key in KEYS:
        values = np.array([r[key] for r in rows])
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        quantiles = values[order][
            np.minimum(np.searchsorted(cumulative, [0.05, 0.5, 0.95]), len(values) - 1)
        ]
        result[key] = dict(zip(("p05", "median", "p95"), map(float, quantiles)))
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=ROOT / "outputs/earth_tube_relief_seed2")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/geometry_validation/calibration"
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    split = ROOT / "paper/splits/pdc_calibration_caves.txt"
    evaluation = ROOT / "paper/splits/pdc_evaluation_caves.txt"

    def read_ids(path):
        return {
            s.strip() for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")
        }

    calibration_ids, evaluation_ids = read_ids(split), read_ids(evaluation)
    assert not calibration_ids & evaluation_ids
    root = ROOT / "data/reference/pdc_v2/extracted/Pyroduct Digital Catalog in .txt"
    selected, rejected = load_pdc(root, cave_ids=calibration_ids)
    sections = [s for s in selected if not s.self_intersection_count]
    counts = Counter(s.reference_cave_id for s in sections)
    reference = [
        {
            "cave_id": s.reference_cave_id,
            "station_id": s.reference_section_id,
            "relative_path": s.relative_path,
            **metrics(s.contour),
        }
        for s in sections
    ]
    ref_weights = np.array([1.0 / counts[s.reference_cave_id] for s in sections])
    ref_weights /= ref_weights.sum()
    data = np.load(args.generated / "sections.npz", allow_pickle=False)
    generated, weights = [], np.zeros(len(data["segment_id"]))
    offsets, points = data["profile_offsets"], data["profile_points"]
    for index, segment in enumerate(data["segment_id"]):
        generated.append(
            {
                "sample_index": index,
                "segment_id": int(segment),
                "arc_length_m": float(data["arc_length_m"][index]),
                **metrics(points[offsets[index] : offsets[index + 1]]),
            }
        )
    for segment in np.unique(data["segment_id"]):
        ids = np.flatnonzero(data["segment_id"] == segment)
        intervals = np.diff(data["arc_length_m"][ids])
        weights[ids[:-1]] += intervals * 0.5
        weights[ids[1:]] += intervals * 0.5
    weights /= weights.sum()
    longitudinal = []
    for label, rows, group_key in (
        ("PDC calibration", reference, "cave_id"),
        ("Generated profiles", generated, "segment_id"),
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[row[group_key]].append(row)
        for key, group in groups.items():
            if len(group) < 3:
                continue
            row = {"source": label, "group_id": key, "stations": len(group)}
            for metric in ("width_m", "height_m"):
                changes = np.abs(np.diff(np.log([r[metric] for r in group])))
                row[f"median_abs_log_{metric}_change"] = float(np.median(changes))
                row[f"p90_abs_log_{metric}_change"] = float(np.quantile(changes, 0.9))
            longitudinal.append(row)
    source_hash = hashlib.sha256()
    for section in sections:
        source_hash.update(section.relative_path.encode() + b"\0")
        source_hash.update((root / section.relative_path).read_bytes())
    summary = {
        "scope": "Calibration discrepancy assessment; no parameter fitting performed. Generated measurements are Stage C profiles before 3-D relief, not mesh clearances.",
        "pdc_caves": len(counts),
        "pdc_sections": len(reference),
        "loader_rejections": len(rejected),
        "self_intersections_excluded": len(selected) - len(sections),
        "evaluation_caves_reserved": len(evaluation_ids),
        "evaluation_coordinate_files_read": 0,
        "generated_sections": len(generated),
        "reference_weighting": "equal weight per cave, then equal per station",
        "generated_weighting": "arc length represented by each original sample; all branches included",
        "point_weighted_descriptors": "512 equal arc-length boundary samples; normalized floor residual and roof asymmetry describe section shape, not centimetre surface roughness",
        "pdc": weighted_summary(reference, ref_weights),
        "generated": weighted_summary(generated, weights),
        "generated_length_fraction_height_1_to_3_m": float(
            sum(w for r, w in zip(generated, weights) if 1 <= r["height_m"] <= 3)
        ),
        "longitudinal_limitation": "PDC station ordering is known; station spacing is generally unavailable. Adjacent log changes are descriptive and cannot calibrate a physical correlation length.",
        "surface_relief_limitation": "Survey contour scale and station spacing do not validate the frequency or amplitude of wall crust, floor lobes or roof drips. Matching broad envelopes is insufficient to call the surface calibrated.",
        "provenance": {
            "pdc_doi": "10.5281/zenodo.17750755",
            "pdc_selected_files_sha256": source_hash.hexdigest(),
            "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
            "generated_sections_sha256": hashlib.sha256(
                (args.generated / "sections.npz").read_bytes()
            ).hexdigest(),
            "generated_config_sha256": hashlib.sha256(
                (args.generated / "resolved_config.json").read_bytes()
            ).hexdigest(),
        },
    }
    for row, w in zip(reference, ref_weights):
        row["weight"] = float(w)
    for row, w in zip(generated, weights):
        row["weight"] = float(w)
    write_csv(out / "reference_sections.csv", reference)
    write_csv(out / "generated_profiles.csv", generated)
    write_csv(out / "longitudinal_station_changes.csv", longitudinal)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, key in zip(axes.flat, KEYS[:6]):
        for rows, w, label in (
            (reference, ref_weights, "Surveyed calibration caves"),
            (generated, weights, "Earth scenario profiles"),
        ):
            values = np.array([r[key] for r in rows])
            order = np.argsort(values)
            ax.step(values[order], np.cumsum(w[order]), where="post", label=label)
        if key in ("width_m", "height_m", "aspect_ratio"):
            ax.set_xscale("log")
        ax.set(xlabel=key.replace("_", " "), ylabel="Weighted cumulative fraction", ylim=(0, 1))
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Earth scenario versus surveyed caves — calibration partition only")
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
