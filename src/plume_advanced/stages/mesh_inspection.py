"""Deterministic mesh inspection at generation and export boundaries.

Vertical intersections inspect the actual triangles, independently of the source
voxel field. This is a sampled cavity/clearance check, not a continuous navigation
or exhaustive self-intersection certificate. No optional ray-tracing dependency
or full mesh copy per sample is required.
"""

from __future__ import annotations

import hashlib

import numpy as np
import trimesh

from plume_advanced.progress import report_progress
from plume_advanced.stages.surface_topology import (
    SurfaceTopologyError,
    check_closed_surface_topology,
    component_count,
)
from plume_advanced.stages.triangle_queries import vertical_clearances


class MeshInspectionError(SurfaceTopologyError):
    """A mesh failed a measured invariant; carries the complete inspection."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("Mesh inspection failed: " + "; ".join(report["failures"]))


def surface_identity(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (vertices, faces):
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def inspect_surface(
    vertices,
    faces,
    *,
    points=(),
    expected_genus: int | None = None,
    require_centers: bool = True,
    weld_seams: bool = False,
    protected_points=(),
    required_paths=(),
    route_segment_ids=(),
    route_height_m: float = 0.,
    route_width_m: float = 0.,
    route_margin_m: float = 0.02,
    route_placement_repair_attempts: int = 3,
    route_placement_max_sweeps: int = 20000,
    ground_robot=None,
    ground_repair_attempts: int = 3,
    ground_max_queries: int = 200000,
) -> dict:
    """Raise on failed invariants, retaining measurements for bounded repairs.

    Export UV seams are joined only at identical positions, without rounding.
    A structural event may obstruct a route intentionally: those centres are
    reported as warnings when require_centers=False, never labelled as passing.
    """
    positions = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(faces)
    report: dict = dict(schema="plume.mesh-inspection.v1", passed=False, failures=[], warnings=[])

    def fail(message):
        report["failures"].append(message)
        raise MeshInspectionError(report)

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not len(positions)
        or indices.ndim != 2
        or indices.shape[1] != 3
        or not len(indices)
    ):
        fail("Expected a nonempty triangular surface")
    if not np.isfinite(positions).all() or not np.issubdtype(indices.dtype, np.integer):
        fail("Nonfinite coordinates or noninteger face indices")
    indices = indices.astype(np.int64, copy=False)
    if indices.min() < 0 or indices.max() >= len(positions):
        fail("Face index outside vertex buffer")
    report["surface_sha256"] = surface_identity(positions, indices)
    if weld_seams:
        positions, inverse = np.unique(positions, axis=0, return_inverse=True)
        indices = inverse[indices]
    report_progress("Mesh invariants", detail=f"checking {len(indices):,} triangles")
    mesh = trimesh.Trimesh(positions, indices, process=False)
    checks = dict(
        finite=True,
        positive_triangle_area=bool(np.all(mesh.area_faces > 0)),
        closed=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
    )
    report["checks"] = checks
    report["failures"].extend(name for name, passed in checks.items() if not passed)
    if report["failures"]:
        raise MeshInspectionError(report)
    count = component_count(indices)
    topology = dict(
        components=count,
        euler=int(mesh.euler_number),
        genus=float((2 * count - mesh.euler_number) / 2),
        expected_genus=expected_genus,
    )
    report["topology"] = topology
    if expected_genus is not None:
        try:
            check_closed_surface_topology(positions, indices, count, expected_genus)
        except SurfaceTopologyError as error:
            fail(str(error))
    centers = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(centers).all():
        fail("Nonfinite inspection centres")
    protected = {tuple(point) for point in np.asarray(protected_points, dtype=float).reshape(-1, 3)}
    known = set(map(tuple, centers))
    missing = sorted(protected - known)
    if missing:
        centers = np.vstack((centers, missing))
    if not np.isfinite(centers).all():
        fail("Nonfinite protected inspection centres")
    measurements = vertical_clearances(positions, indices, centers)
    for row in measurements:
        row["required"] = require_centers or tuple(row["point_m"]) in protected
    blocked = [row["sample_index"] for row in measurements if not row["inside"]]
    report.update(
        measurements=measurements,
        inspected_centers=len(centers),
        outside_centers=blocked,
        centers_required=require_centers,
    )
    if blocked:
        message = f"{len(blocked)}/{len(centers)} sampled centres lie outside the actual mesh cavity; first indices {blocked[:8]}"
        if any(row["required"] and not row["inside"] for row in measurements):
            fail(message)
        report["warnings"].append("Structural-event inspection: " + message)
    if not len(centers):
        report["warnings"].append(
            "No route centres supplied; passage-clearance inspection unavailable"
        )
    if route_height_m or route_width_m:
        from plume_advanced.stages.route_clearance import inspect_capsule_routes

        if not 0 < route_width_m <= route_height_m or route_margin_m < 0:
            fail("Invalid required-route capsule dimensions")
        route = inspect_capsule_routes(positions, indices, required_paths, route_segment_ids,
                                       height=route_height_m, width=route_width_m, margin=route_margin_m,
                                       placement_attempts=route_placement_repair_attempts,
                                       placement_max_sweeps=route_placement_max_sweeps)
        report["traversal"] = route
        if not route["passed"]:
            report["defect_regions"] = route["defect_regions"]
            fail("; ".join(route["failures"]))
    if ground_robot is not None:
        from plume_advanced.stages.ground_routes import inspect_ground_routes
        ground = inspect_ground_routes(positions, indices, required_paths, route_segment_ids,
            robot=ground_robot, repair_attempts=ground_repair_attempts, max_queries=ground_max_queries)
        report["ground_traversal"] = ground
        if not ground["passed"]:
            report["defect_regions"] = ground["defect_regions"]
            fail("; ".join(ground["failures"]))
    report["passed"] = True
    return report


def route_inspection_arguments(geometry):
    """One contract shared by base, visual, collision and serialized inspection."""
    from plume_advanced.stages.ground_routes import GroundRobot
    cfg = geometry.config
    ground = (GroundRobot(length_m=cfg.ground_robot_length_m,
                          width_m=cfg.required_route_width_m, height_m=cfg.required_route_height_m,
                          margin_m=cfg.route_clearance_margin_m, max_slope_deg=cfg.ground_max_slope_deg,
                          max_step_m=cfg.ground_max_step_m, support_spacing_m=cfg.ground_support_spacing_m)
              if cfg.ground_robot_length_m else None)
    return dict(required_paths=geometry.required_route_paths,
                route_segment_ids=geometry.route_path_segment_ids,
                route_height_m=geometry.config.required_route_height_m,
                route_width_m=geometry.config.required_route_width_m,
                route_margin_m=geometry.config.route_clearance_margin_m,
                route_placement_repair_attempts=geometry.config.route_placement_repair_attempts,
                route_placement_max_sweeps=geometry.config.route_placement_max_sweeps,
                ground_robot=ground, ground_repair_attempts=cfg.ground_repair_attempts,
                ground_max_queries=cfg.ground_max_queries)
