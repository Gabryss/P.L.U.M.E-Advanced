#!/usr/bin/env python3
"""Matched-radius surface residuals: Valentine scan and before/after tube reach.

Local-plane residual combines relief, curvature, edges and acquisition effects.
It is a descriptive check at stated scales, not a calibrated naturalness score.
Requires the optional ``paper`` dependencies for the local LAZ reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def residuals(points, eligible, *, seed):
    random = np.random.default_rng(seed)
    tree = cKDTree(points)
    centers = random.choice(eligible, size=min(1500, len(eligible)), replace=False)
    results = []
    for radius in (0.4, 0.8, 1.6):
        neighborhoods = tree.query_ball_point(points[centers], radius)
        measured = []
        for ids in neighborhoods:
            if len(ids) < 32:
                continue
            # Equal point count avoids giving the denser source an easier fit.
            patch = points[random.choice(ids, size=24, replace=False)]
            patch -= patch.mean(axis=0)
            values = np.linalg.eigvalsh(patch.T @ patch / len(patch))
            measured.append(float(np.sqrt(max(values[0], 0.0))))
        results.append(
            {
                "radius_m": radius,
                "candidate_centers": len(centers),
                "accepted_centers": len(measured),
                "rms_plane_residual_m": measured,
                "p05_m": float(np.quantile(measured, 0.05)),
                "median_m": float(np.median(measured)),
                "p95_m": float(np.quantile(measured, 0.95)),
                "fraction_below_2cm": float(np.mean(np.asarray(measured) < 0.02)),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/geometry_validation/surface_scales"
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/reference/valentine/Valentine_TUBE_UTM_10cm.copc.laz"
    plan = {
        "neighborhood_radii_m": [0.4, 0.8, 1.6],
        "centers_per_source": 1500,
        "points_per_plane_fit": 24,
        "minimum_neighbors": 32,
        "smooth_patch_threshold_m": 0.02,
        "synthetic_sample_density_per_m2": 100,
        "random_seed": 701,
        "scope": "Descriptive local-plane residual, including curvature and edges. No geological fit or source-invariant roughness claim.",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    cloud = laspy.read(source)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    xyz -= xyz.mean(axis=0)
    results = {"Valentine 10 cm scan": residuals(xyz, np.arange(len(xyz)), seed=701)}
    provenance = {
        "Valentine 10 cm scan": {
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "point_count": len(xyz),
        }
    }
    study = ROOT / "outputs/surface_relief_study"
    data = np.load(study / "sections.npz", allow_pickle=False)
    centers = data["center_xyz_m"]
    axis = centers[-1] - centers[0]
    axis /= np.linalg.norm(axis)
    limits = (centers - centers[0]) @ axis
    for state in ("before", "after"):
        path = study / state / "lava_tube_geometry.glb"
        scene = trimesh.load(path, force="scene", process=False)
        mesh = next(iter(scene.geometry.values())).copy(include_cache=True)
        mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
        points, _ = trimesh.sample.sample_surface(mesh, int(mesh.area * 100), seed=701)
        along = (points - centers[0]) @ axis
        eligible = np.flatnonzero((along > limits.min() + 3.0) & (along < limits.max() - 3.0))
        label = f"Generated {state} relief"
        results[label] = residuals(points, eligible, seed=701)
        provenance[label] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "point_count": len(points),
            "excluded_end_margin_m": 3.0,
        }
    summary = {
        "plan": plan,
        "provenance": provenance,
        "results": results,
        "limitations": [
            "The reference is a 10 cm decimated scan; shadows and scan layout affect point coverage.",
            "Synthetic points are area-uniform; reference centers are point-weighted, so the spatial sampling schemes are not identical.",
            "Curved roofs, benches, junctions and broken edges raise plane residual even on a smooth material.",
            "Large neighborhoods can include opposing roof/floor surfaces in shallow passages; the 1.6 m radius must not be interpreted as isolated material roughness.",
            "The generated comparison is one 64 m reach at 10 cm mesh resolution; it is not the whole network or a centimetre-scale texture assessment.",
            "Roof, floor and wall feature types are not labeled in this test; their proportions remain uncalibrated.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for i, ax in enumerate(axes):
        for label, rows in results.items():
            values = np.sort(rows[i]["rms_plane_residual_m"]) * 100
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label)
        ax.set(
            title=f"Neighborhood radius {plan['neighborhood_radii_m'][i]:g} m",
            xlabel="RMS distance to fitted plane (cm)",
            ylabel="Fraction of accepted patches",
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Surface variation at matched scales — curvature and edges included")
    fig.savefig(out / "comparison.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            {
                label: [{k: v for k, v in r.items() if k != "rms_plane_residual_m"} for r in rows]
                for label, rows in results.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
