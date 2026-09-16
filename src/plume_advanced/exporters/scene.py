"""Canonical, format-neutral scene preparation shared by every exporter."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from plume_advanced.exporters.collision import (
    passage_deviation,
    simplify_surface,
    surface_deviation,
)
from plume_advanced.progress import report_progress
from plume_advanced.stages.geometry_export import (
    CavePrimitivePayload,
    build_cave_visual_surface,
    canonical_visual_to_gltf,
)
from plume_advanced.stages.geometry_types import CaveGeometry
from plume_advanced.stages.mesh_inspection import (
    MeshInspectionError,
    inspect_surface,
    route_inspection_arguments,
)
from plume_advanced.stages.triangle_queries import TriangleIndex


@dataclass(frozen=True)
class PreparedExportScene:
    """Expensive surface and collision products prepared exactly once per export."""

    geometry: CaveGeometry
    canonical_visual: CavePrimitivePayload
    gltf_visual: CavePrimitivePayload
    collision_vertices: np.ndarray
    collision_faces: np.ndarray
    inspection: dict = field(default_factory=dict)


def prepare_export_scene(
    cave_geometry: CaveGeometry, *, generate_collision: bool = True,
    max_visual_triangles: int = 0, visual_max_error_m: float = 0.0,
) -> PreparedExportScene:
    if not cave_geometry.assembled_vertices or not cave_geometry.assembled_faces:
        raise ValueError(
            "Cave export requires the assembled Stage-E mesh; generate geometry before exporting."
        )
    expected = (
        cave_geometry.expected_surface_genus if not cave_geometry.structural_event_ids else None
    )
    arguments: dict = dict(
        points=cave_geometry.route_centers,
        expected_genus=expected,
        require_centers=not cave_geometry.structural_event_ids,
        protected_points=cave_geometry.protected_route_points,
        **route_inspection_arguments(cave_geometry),
    )
    raw = inspect_surface(
        cave_geometry.assembled_vertices, cave_geometry.assembled_faces, **arguments
    )
    config = cave_geometry.config
    visual_geometry = cave_geometry
    reduction: dict = dict(enabled=False)
    raw_arrays = canonical_cave_mesh(cave_geometry)
    if max_visual_triangles and len(raw_arrays[1]) > max_visual_triangles:
        if visual_max_error_m <= 0:
            raise ValueError("Visual triangles exceed budget and measured reduction is disabled")
        reduction = dict(enabled=True)
        vertices, faces = simplify_surface(*raw_arrays, arguments,
            target_reduction=1-max_visual_triangles/len(raw_arrays[1]),
            max_error_m=visual_max_error_m, attempts=4, triangle_budget=max_visual_triangles,
            purpose="Visual", report=reduction)
        visual_geometry = replace(cave_geometry,
            assembled_vertices=tuple(map(tuple, vertices)), assembled_faces=tuple(map(tuple, faces)))
    candidates = [(config.cave_smoothing_iterations, config.cave_displacement_scale_m)]
    if config.cave_displacement_texture and config.cave_displacement_scale_m:
        candidates.extend(
            [
                (config.cave_smoothing_iterations, config.cave_displacement_scale_m * 0.5),
                (config.cave_smoothing_iterations, 0.0),
            ]
        )
    candidates.append((0, 0.0))
    candidates = list(dict.fromkeys(candidates))
    attempts = []
    for index, (smoothing, displacement) in enumerate(candidates):
        report_progress(
            "Visual surface acceptance",
            index,
            len(candidates),
            f"smoothing {smoothing}; displacement {displacement:g} m",
        )
        effective = replace(
            visual_geometry,
            config=replace(
                config, cave_smoothing_iterations=smoothing, cave_displacement_scale_m=displacement
            ),
        )
        attempt: dict = dict(smoothing_iterations=smoothing, displacement_scale_m=displacement)

        def preflight(positions):
            # Reject only proven violations on the exact eventual float32
            # positions, before costly UV work. Passing this subset is not
            # acceptance: the full bidirectional comparison still follows.
            sample_ids = np.linspace(0, len(positions)-1, min(4096, len(positions)), dtype=int)
            report_progress("Visual geometry preflight", detail="checking smoothing before UV preparation")
            distances = TriangleIndex(*raw_arrays).distances(positions[sample_ids])
            worst = float(distances.max(initial=0.))
            attempt["geometry_preflight"] = dict(passed=worst <= visual_max_error_m,
                sampled_vertices=len(sample_ids), max_sampled_error_m=worst,
                scope="One-way vertex subset for early rejection only; full final checks remain required")
            if worst > visual_max_error_m:
                raise MeshInspectionError(dict(passed=False,
                    failures=["Smoothing exceeds the surface-error budget before UV preparation"]))
            # The inexpensive subset can miss a small local defect. Perform
            # the full geometry checks before spending time on its UV atlas.
            # The final post-atlas checks below remain independent and required.
            preflight_report = attempt["geometry_preflight"]
            preflight_report["scope"] = "Early geometry inspection and full sampled fidelity; final prepared-mesh checks remain required"
            try:
                preflight_faces = np.asarray(visual_geometry.assembled_faces, dtype=np.int64)
                early_inspection = inspect_surface(positions, preflight_faces, **arguments)
                deviation = surface_deviation(raw_arrays, (positions, preflight_faces),
                                              visual_max_error_m, purpose="Visual preflight")
                preflight_report.update(surface_deviation=deviation,
                    max_passage_error_m=passage_deviation(raw, early_inspection))
                if (not deviation["passed"]
                        or preflight_report["max_passage_error_m"] > visual_max_error_m):
                    raise MeshInspectionError(dict(passed=False,
                        failures=["Smoothing exceeds the surface-error budget before UV preparation"]))
            except MeshInspectionError:
                preflight_report["passed"] = False
                raise

        try:
            canonical_visual = build_cave_visual_surface(effective, convert_to_gltf=False,
                geometry_preflight=preflight if visual_max_error_m and smoothing and not displacement else None)
            inspected = inspect_surface(
                canonical_visual["positions"],
                canonical_visual["faces"],
                weld_seams=True,
                **arguments,
            )
            # Even a standalone imported mesh without a network must keep its topology.
            if any(
                inspected["topology"][key] != raw["topology"][key]
                for key in ("components", "euler")
            ):
                inspected.update(
                    passed=False, failures=["Visual processing changed raw mesh topology"]
                )
                raise MeshInspectionError(inspected)
            if visual_max_error_m:
                report_progress("Visual fidelity", detail="checking the final smoothed surface against the raw mesh")
                deviation = surface_deviation(raw_arrays,
                    (canonical_visual["positions"], canonical_visual["faces"]),
                    visual_max_error_m, purpose="Visual")
                attempt["surface_deviation"] = deviation
                attempt["max_passage_error_m"] = passage_deviation(raw, inspected)
                if not deviation["passed"] or attempt["max_passage_error_m"] > visual_max_error_m:
                    raise MeshInspectionError(dict(passed=False,
                        failures=["Visual processing exceeds the declared surface-error budget"]))
        except MeshInspectionError as error:
            attempt.update(accepted=False, inspection=error.report)
            attempts.append(attempt)
            continue
        attempt["accepted"] = True
        attempts.append(attempt)
        break
    else:
        raise MeshInspectionError(
            dict(passed=False, failures=["All bounded visual repairs failed"], attempts=attempts)
        )
    report_progress(
        "Visual surface acceptance",
        len(candidates),
        len(candidates),
        f"accepted after {len(attempts)} attempt(s)",
    )
    collision_report: dict = dict(enabled=generate_collision, used_raw_fallback=False)
    if generate_collision:
        report_progress("Collision mesh", detail="reducing edges and measuring surface/passage error")
        collision_vertices, collision_faces = simplified_collision_arrays(
            cave_geometry, report=collision_report
        )
        try:
            collider = inspect_surface(collision_vertices, collision_faces, **arguments)
            if any(
                collider["topology"][key] != raw["topology"][key] for key in ("components", "euler")
            ):
                collider.update(
                    passed=False, failures=["Collision simplification changed cave topology"]
                )
                raise MeshInspectionError(collider)
        except MeshInspectionError as error:
            fallback_report: dict = {}
            collision_vertices, collision_faces = simplified_collision_arrays(
                replace(cave_geometry, config=replace(config, collision_repair_attempts=0)),
                report=fallback_report,
            )
            collision_report.update(fallback_report)
            collision_report.update(
                used_raw_fallback=True,
                fallback_reason="Simplified collider failed topology or passage inspection",
                rejected_inspection=error.report,
            )
            collider = fallback_report["inspection"]
        collision_report["inspection"] = collider
    else:
        collision_vertices = np.empty((0, 3), dtype=np.float64)
        collision_faces = np.empty((0, 3), dtype=np.int64)
    return PreparedExportScene(
        geometry=effective,
        canonical_visual=canonical_visual,
        gltf_visual=canonical_visual_to_gltf(canonical_visual),
        collision_vertices=collision_vertices,
        collision_faces=collision_faces,
        inspection=dict(
            schema="plume.export-inspection.v1",
            passed=True,
            raw=raw,
            visual=inspected,
            visual_attempts=attempts,
            visual_reduction=reduction,
            collision=collision_report,
            scope="Actual canonical and float32 visual mesh inspection before serialization",
        ),
    )


def canonical_cave_mesh(cave_geometry: CaveGeometry) -> tuple[np.ndarray, np.ndarray]:
    if cave_geometry.assembled_vertices and cave_geometry.assembled_faces:
        return (
            np.asarray(cave_geometry.assembled_vertices, dtype=np.float64),
            np.asarray(cave_geometry.assembled_faces, dtype=np.int64),
        )

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    offset = 0
    for chunk in cave_geometry.chunk_meshes:
        vertices.extend(chunk.vertices)
        faces.extend(
            (
                int(face[0]) + offset,
                int(face[1]) + offset,
                int(face[2]) + offset,
            )
            for face in chunk.faces
        )
        offset += len(chunk.vertices)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def simplified_collision_arrays(
    cave_geometry: CaveGeometry,
    *,
    report: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an independently inspected, error-limited collision surface."""
    from plume_advanced.exporters.collision import simplify_collision

    arguments = dict(
        points=cave_geometry.route_centers,
        expected_genus=cave_geometry.expected_surface_genus
        if not cave_geometry.structural_event_ids else None,
        require_centers=not cave_geometry.structural_event_ids,
        protected_points=cave_geometry.protected_route_points,
        **route_inspection_arguments(cave_geometry),
    )
    return simplify_collision(*canonical_cave_mesh(cave_geometry), cave_geometry.config,
                              arguments, report=report)


__all__ = [
    "PreparedExportScene",
    "canonical_cave_mesh",
    "prepare_export_scene",
    "simplified_collision_arrays",
]
