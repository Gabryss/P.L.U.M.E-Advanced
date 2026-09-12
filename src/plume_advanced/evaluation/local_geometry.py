"""Bounded mesh studies that retain the complete network's sweep context.

Region faces are artificial inspection caps. These meshes are independent
inspection assets, not automatically stitched adaptive replacements.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from matplotlib.path import Path as Polygon

from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.stages.network import CaveNetwork
from plume_advanced.stages.section_field import SectionField, SectionSample
from plume_advanced.stages.surface_relief import apply_surface_relief


def section_resolution_report(
    sections: SectionField,
    voxel_size_m: float,
    *,
    minimum_samples: int = 8,
) -> dict:
    """Flag under-sampled input envelopes; this is not a clearance certificate."""
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0 or minimum_samples < 2:
        raise ValueError("Resolution must be positive and at least two samples required")
    records = []
    for field in sections.segment_fields:
        for sample in field.samples:
            width, height = np.ptp(np.asarray(sample.profile_points), axis=0)
            limiting = float(min(width, height))
            records.append(
                {
                    "segment_id": sample.segment_id,
                    "arc_length_m": sample.segment_arc_length,
                    "width_m": float(width),
                    "height_m": float(height),
                    "samples_across_smallest_dimension": limiting / voxel_size_m,
                    "under_resolved": bool(limiting / voxel_size_m < minimum_samples),
                    "initial_local_voxel_size_m": limiting / minimum_samples,
                }
            )
    flagged = [row for row in records if row["under_resolved"]]
    return {
        "voxel_size_m": voxel_size_m,
        "minimum_samples": minimum_samples,
        "section_count": len(records),
        "under_resolved_count": len(flagged),
        "under_resolved_sections": flagged,
        "scope": "Input profile envelopes before relief. Eight samples is an initial screening heuristic; actual mesh clearances require local convergence checks.",
    }


def local_geometry(
    network: CaveNetwork,
    sections: SectionField,
    config: GeometryConfig,
    *,
    center: np.ndarray,
    half_extent: np.ndarray,
    lattice_origin: np.ndarray | None = None,
    max_voxels: int = 60_000_000,
) -> CaveGeometry:
    """Mesh a fixed world-space box with all nearby branches contributing."""
    center, half_extent = np.asarray(center, float), np.asarray(half_extent, float)
    if center.shape != (3,) or half_extent.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center and half_extent must be finite three-vectors")
    if not np.all(np.isfinite(half_extent)) or np.any(half_extent <= 0.0):
        raise ValueError("half_extent must be positive and finite")
    voxel = config.voxel_size
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("voxel_size must be positive and finite")
    anchor = np.zeros(3) if lattice_origin is None else np.asarray(lattice_origin, float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("lattice_origin must be a finite three-vector")
    lower = anchor + np.floor((center - half_extent - anchor) / voxel) * voxel
    upper = anchor + np.ceil((center + half_extent - anchor) / voxel) * voxel
    shape = np.rint((upper - lower) / voxel).astype(int) + 1
    if int(np.prod(shape)) > max_voxels:
        raise ValueError(f"Local study needs {int(np.prod(shape)):,} voxels; reduce its extent")
    generator = GeometryGenerator(config)
    chains = {
        f.segment_id: generator._refine_profile_chain(f.samples)
        for f in sections.segment_fields
        if f.samples
    }
    refined = replace(
        sections,
        segment_fields=tuple(
            replace(f, samples=chains.get(f.segment_id, ())) for f in sections.segment_fields
        ),
    )
    density = np.full(tuple(shape), -8.0, dtype=np.float32)
    for samples in chains.values():
        generator._stamp_network_chain(density=density, origin=lower, samples=samples)
    junctions = generator._junction_stamp_points(chains, network)
    grid = VoxelGrid(
        origin=(float(lower[0]), float(lower[1]), float(lower[2])),
        voxel_size=voxel,
        density=density,
        iso_level=config.iso_level,
    )
    generator._remove_small_solid_pockets(grid)
    apply_surface_relief(grid, config)
    if any(
        getattr(config, name) > 0.0
        for name in (
            "surface_wall_relief_m",
            "surface_roof_relief_m",
            "surface_floor_relief_m",
            "surface_crust_relief_m",
        )
    ):
        generator._remove_small_solid_pockets(grid, include_void=True)
    records = generator._enforce_roof_stability(grid, refined, junctions)
    # Close the region well away from the measured section. These caps must
    # never be interpreted as generated cave termini or used for clearance.
    for axis in range(3):
        for index in (0, -1):
            edge: list[slice | int] = [slice(None)] * 3
            edge[axis] = index
            density[tuple(edge)] = config.iso_level - 8.0
    base = CaveGeometry(
        config=config,
        voxel_grid=grid,
        chunk_meshes=(),
        assembled_vertices=(),
        assembled_faces=(),
        component_count=0,
        stamped_sample_count=sum(map(len, chains.values())),
        stamped_segment_ids=tuple(chains),
        stability_records=records,
    )
    return generator.finalize(base)


def section_contour(
    mesh,
    sample: SectionSample,
    *,
    region_bounds: np.ndarray | None = None,
    clip_margin_m: float = 0.0,
) -> np.ndarray:
    """Select the closed cut containing the actual profile's interior point."""
    center = np.array([sample.x, sample.y, sample.z])
    cut = mesh.section(plane_origin=center, plane_normal=sample.tangent)
    if cut is None:
        raise ValueError("Mesh has no intersection with the requested section")
    candidates = []
    for points in cut.discrete:
        q = np.column_stack(
            ((points - center) @ sample.normal, (points - center) @ sample.binormal)
        )
        if (
            len(q) >= 4
            and np.linalg.norm(q[0] - q[-1]) < 0.001
            and Polygon(q).contains_point((0.0, 0.0))
        ):
            candidates.append(q)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one closed contour around the section axis, got {len(candidates)}"
        )
    contour = candidates[0]
    if region_bounds is not None:
        bounds = np.asarray(region_bounds, float)
        points = (
            center + contour[:, 0, None] * sample.normal + contour[:, 1, None] * sample.binormal
        )
        if np.any(points <= bounds[0] + clip_margin_m) or np.any(
            points >= bounds[1] - clip_margin_m
        ):
            raise ValueError(
                "Measured contour reaches an artificial inspection boundary; enlarge the region"
            )
    return contour


def contour_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric vertex-to-polyline distance, in metres (sampling approximation)."""

    def directed(points, contour):
        a, b = contour[:-1], contour[1:]
        edges = b - a
        t = np.sum((points[:, None] - a) * edges, axis=2) / np.maximum(
            np.sum(edges * edges, axis=1), 1e-18
        )
        nearest = a + np.clip(t, 0.0, 1.0)[..., None] * edges
        return float(np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1).max())

    return max(directed(first, second), directed(second, first))
