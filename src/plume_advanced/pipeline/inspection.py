"""Always-on evaluation artifacts for the normal generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plume_advanced.acceptance import AcceptancePolicy, enforce_acceptance, evaluate_acceptance
from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.identity import sha256_file
from plume_advanced.progress import report_progress
from plume_advanced.world import ExportConfig


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def evaluate_sections(sections, voxel_size: float, output: Path) -> dict:
    report_progress("Section resolution evaluation", detail="checking every input profile")
    report = section_resolution_report(sections, voxel_size)
    _write(output, report)
    return report


def complete_inspection(
    geometry, export_result, resolution: dict, output: Path, *,
    acceptance: AcceptancePolicy = AcceptancePolicy(), export_config: ExportConfig | None = None,
) -> tuple[Path, Path]:
    package_path = next(p for p in export_result.files if p.name == "pipeline_inspection.json")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if not package.get("passed") or not package.get("serialized", {}).get("passed"):
        raise ValueError("Export has no passing embedded inspection")
    for record in package["serialized"]["files"]:
        asset = package_path.parent / record["path"]
        if (
            not asset.resolve().is_relative_to(package_path.parent.resolve())
            or sha256_file(asset) != record["sha256"]
        ):
            raise ValueError(f"Inspected export changed before run completion: {record['path']}")
    warnings = list(package["visual"].get("warnings", ()))
    textures = package.get("textures")
    if not textures or not textures.get("passed"):
        raise ValueError("Export has no passing texture inspection")
    for record in textures["package_attempts"][-1]["checked_files"]:
        asset = package_path.parent / record["path"]
        if (not asset.resolve().is_relative_to(package_path.parent.resolve())
                or sha256_file(asset) != record["sha256"]):
            raise ValueError(f"Inspected texture package changed before completion: {record['path']}")
    texture_path = package_path.parent / "texture_recovery.json"
    if json.loads(texture_path.read_text()) != textures:
        raise ValueError("Texture recovery journal differs from inspected package")
    # Re-evaluate on every completion/resume; a saved passed flag is not sufficient.
    saved_acceptance = package.get("acceptance", {})
    evaluated = evaluate_acceptance(acceptance, geometry, package, resolution, export_config)
    enforce_acceptance(evaluated)
    if saved_acceptance != evaluated:
        raise ValueError("Acceptance policy/evidence changed since package inspection")
    if textures["outcome"] == "repaired":
        warnings.append("Texture maps or material package repaired; see texture_recovery.json for changes.")
    recovery_path = output / "pipeline_recovery.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.is_file() else None
    if recovery is not None:
        if not recovery.get("accepted"):
            raise ValueError("Upstream recovery has no accepted realization")
        if recovery["outcome"] != "unchanged":
            warnings.append(
                f"Upstream recovery {recovery['outcome']}; see pipeline_recovery.json for original failures and the accepted realization."
            )
    if resolution["under_resolved_count"]:
        warnings.append(
            f"{resolution['under_resolved_count']}/{resolution['section_count']} input profiles have fewer than eight samples across their smallest dimension; inspect local resolution convergence."
        )
    if len(package["visual_attempts"]) > 1:
        warnings.append("Visual smoothing/displacement was reduced to preserve inspected geometry.")
    if (geometry.effective_surface_relief_scale < 1 or geometry.effective_density_opening_voxels
            or geometry.effective_local_relief_regions):
        warnings.append(
            "Bounded base-surface repair changed detail/filter settings; see surface_repairs."
        )
    if package["collision"].get("used_raw_fallback"):
        warnings.append(
            "Collision simplification failed inspection; the original collider was retained."
        )
    report = dict(
        schema="plume.pipeline-quality.v1",
        passed=True,
        acceptance=evaluated,
        warnings=warnings,
        resolution=resolution,
        resolution_repair=dict(geometry.resolution_repair),
        required_route_clearance=package["visual"].get("traversal", {"enabled": False}),
        final_mesh_inspection=dict(geometry.mesh_inspection),
        surface_repairs=[dict(record) for record in geometry.surface_quality_records],
        upstream_recovery=recovery,
        texture_recovery=textures,
        export_inspection=package,
        asset=export_result.primary_asset.name,
        scope="Embedded numerical evaluation, mesh inspection and bounded repairs; not geological realism, ground-contact navigation, exhaustive self-intersection or native-engine certification",
    )
    report_path = _write(output / "pipeline_quality_report.json", report)
    figure = render_inspection(
        package["raw"], package["visual"], output / "pipeline_inspection.png"
    )
    return report_path, figure


def record_failure(output: Path, error: Exception | KeyboardInterrupt, stage: str) -> Path:
    return _write(
        output / "pipeline_quality_report.json",
        dict(
            schema="plume.pipeline-quality.v1",
            passed=False,
            stage=stage,
            error_type=type(error).__name__,
            error=str(error),
            inspection=getattr(error, "report", None),
            scope="Generation stopped before successful pipeline acceptance",
        ),
    )


def render_inspection(raw: dict, visual: dict, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_progress("Inspection figure", detail="plotting measured passage samples")
    rows = visual.get("measurements", [])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if rows:
        points = np.asarray([r["point_m"] for r in rows])
        heights = np.asarray(
            [r["clearance_m"] if r["clearance_m"] is not None else np.nan for r in rows]
        )
        valid = np.isfinite(heights)
        if valid.any():
            dots = axes[0].scatter(
                points[valid, 0],
                points[valid, 1],
                c=heights[valid],
                s=10,
                cmap="viridis",
                rasterized=True,
            )
            figure.colorbar(dots, ax=axes[0], label="Measured floor-to-roof distance (m)")
        if (~valid).any():
            axes[0].scatter(
                points[~valid, 0],
                points[~valid, 1],
                c="crimson",
                marker="x",
                label="Obstructed sample",
            )
            axes[0].legend()
        axes[0].set(xlabel="X (m)", ylabel="Y (m)", aspect="equal")
        axes[1].plot(heights, ".", ms=3, label="Exported visual surface")
        raw_rows = raw.get("measurements", [])
        if len(raw_rows) == len(rows):
            axes[1].plot(
                [r["clearance_m"] if r["clearance_m"] is not None else np.nan for r in raw_rows],
                ".",
                alpha=0.45,
                ms=3,
                label="Final generated mesh",
            )
        axes[1].set(
            xlabel="Route sample index (segment order)", ylabel="Floor-to-roof distance (m)"
        )
        axes[1].legend()
        axes[1].grid(alpha=0.2)
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No route centres supplied\nClearance inspection unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
    try:
        figure.savefig(output, dpi=140)
    finally:
        plt.close(figure)
    return output
