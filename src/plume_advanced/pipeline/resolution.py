"""Bounded refinement on consistent lattices, with measured convergence evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.progress import report_progress
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.mesh_inspection import _vertical_clearances, surface_identity
from plume_advanced.stages.route_clearance import _polyline
from plume_advanced.stages.surface_topology import SurfaceTopologyError


class ResolutionBudgetError(SurfaceTopologyError):
    """A required resolution cannot be verified within the declared budget."""


def compare_mesh_clearance(previous, current, points, tolerance):
    """Compare both floor and roof on identical world-space probe lines."""
    rows = []
    for mesh in (previous, current):
        rows.append(_vertical_clearances(np.asarray(mesh.assembled_vertices),
                                        np.asarray(mesh.assembled_faces), np.asarray(points)))
    errors = []
    blocked = []
    for index, (a, b) in enumerate(zip(*rows, strict=True)):
        if not a["inside"] or not b["inside"]:
            blocked.append(index)
        else:
            errors.append(max(abs(a[k]-b[k]) for k in ("floor_distance_m", "roof_distance_m")))
    maximum = max(errors, default=0.)
    return dict(passed=bool(errors) and not blocked and maximum <= tolerance,
                max_floor_roof_change_m=maximum, tolerance_m=tolerance,
                compared_samples=len(errors), blocked_samples=blocked,
                topology_equal=(previous.component_count == current.component_count
                                and previous.expected_surface_genus == current.expected_surface_genus))


def build_with_resolution_checks(network, sections, controls, progress=None):
    """Preserve network/sections/seed; record every attempted effective grid size.

    The eight-sample rule triggers a convergence study. Refinement keeps one
    resolution per candidate (dense or tiled), avoiding mixed-resolution seams.
    It accepts a finer mesh only after measured floor/roof convergence and the
    ordinary topology, stability and capsule gates. Resource exhaustion fails.
    """
    if not controls.resolution_refinement_attempts:
        return GeometryGenerator(controls).build_base_volume(network, sections, progress=progress)
    initial = section_resolution_report(sections, controls.voxel_size)
    journal = dict(schema="plume.resolution-repair.v1", passed=False,
                   requested_voxel_size_m=controls.voxel_size, attempts=[],
                   max_refinements=controls.resolution_refinement_attempts,
                   minimum_voxel_size_m=controls.resolution_min_voxel_size_m,
                   max_allocated_voxels=controls.resolution_max_allocated_voxels,
                   initial_under_resolved=initial["under_resolved_count"],
                   scope="Consistent-grid refinement; floor/roof convergence at identical section probes plus topology and required continuous capsule checks. Not exhaustive surface convergence.")
    points = tuple(p for field in sections.segment_fields
                   for p in _polyline([(s.x, s.y, s.z) for s in field.samples],
                                      controls.route_inspection_spacing_m))
    previous = None
    for level in range(controls.resolution_refinement_attempts+1):
        size = controls.voxel_size/(2**level)
        if size < controls.resolution_min_voxel_size_m-1e-12:
            journal["failure"] = "Next refinement would cross the minimum voxel size"
            break
        report_progress("Resolution acceptance", level, controls.resolution_refinement_attempts+1,
                        f"building {size:g} m grid; fixed host, network and section identities")
        config = replace(controls, voxel_size=size)
        record = dict(voxel_size_m=size, accepted=False)
        journal["attempts"].append(record)
        try:
            current = GeometryGenerator(config).build_base_volume(network, sections, progress=progress)
        except ResolutionBudgetError as error:
            record.update(failure=str(error), allocation=error.report)
            journal["failure"] = "Allocated-grid budget exhausted"
            break
        record.update(triangles=len(current.assembled_faces),
                      surface_sha256=surface_identity(np.asarray(current.assembled_vertices),
                                                     np.asarray(current.assembled_faces)),
                      under_resolved=section_resolution_report(sections, size)["under_resolved_count"])
        if not initial["under_resolved_count"]:
            record["accepted"] = True
            journal.update(passed=True, outcome="input_sampling_sufficient", effective_voxel_size_m=size)
            return replace(current, resolution_repair=tuple(journal.items()))
        if previous is not None:
            comparison = compare_mesh_clearance(previous, current, points, controls.resolution_convergence_m)
            record["comparison"] = comparison
            report_progress("Resolution comparison", level, controls.resolution_refinement_attempts,
                            f"maximum floor/roof change {comparison['max_floor_roof_change_m']:.4f} m; "
                            f"limit {controls.resolution_convergence_m:g} m; "
                            f"{len(comparison['blocked_samples'])} invalid probes")
            if comparison["passed"] and comparison["topology_equal"]:
                record["accepted"] = True
                journal.update(passed=True, outcome="converged", effective_voxel_size_m=size)
                return replace(current, resolution_repair=tuple(journal.items()))
        previous = current
    journal.setdefault("failure", "Floor/roof convergence did not pass within the refinement budget")
    raise ResolutionBudgetError(journal["failure"], report=journal)
