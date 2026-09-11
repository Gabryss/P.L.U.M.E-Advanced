"""Canonical, format-neutral scene preparation shared by every exporter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from plume_advanced.stages.geometry_export import (
    CavePrimitivePayload,
    build_cave_visual_surface,
    canonical_visual_to_gltf,
)
from plume_advanced.stages.geometry_types import CaveGeometry


@dataclass(frozen=True)
class PreparedExportScene:
    """Expensive surface and collision products prepared exactly once per export."""

    geometry: CaveGeometry
    canonical_visual: CavePrimitivePayload
    gltf_visual: CavePrimitivePayload
    collision_vertices: np.ndarray
    collision_faces: np.ndarray


def prepare_export_scene(cave_geometry: CaveGeometry) -> PreparedExportScene:
    canonical_visual = build_cave_visual_surface(
        cave_geometry,
        convert_to_gltf=False,
    )
    collision_vertices, collision_faces = simplified_collision_arrays(cave_geometry)
    return PreparedExportScene(
        geometry=cave_geometry,
        canonical_visual=canonical_visual,
        gltf_visual=canonical_visual_to_gltf(canonical_visual),
        collision_vertices=collision_vertices,
        collision_faces=collision_faces,
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
) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = canonical_cave_mesh(cave_geometry)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Cannot create collision geometry from an empty cave mesh")
    bounds = np.ptp(vertices, axis=0)
    cell_size = max(
        cave_geometry.voxel_grid.voxel_size * 2.5,
        float(np.max(bounds)) / 240.0,
        1e-6,
    )
    keys = np.floor((vertices - vertices.min(axis=0)) / cell_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    clustered = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.bincount(inverse)
    for axis in range(3):
        clustered[:, axis] = np.bincount(
            inverse,
            weights=vertices[:, axis],
            minlength=len(unique_keys),
        ) / np.maximum(counts, 1)
    remapped = inverse[faces]
    valid = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 2] != remapped[:, 0])
    )
    simplified_faces: list[tuple[int, int, int]] = []
    seen_faces: set[tuple[int, int, int]] = set()
    for face in remapped[valid]:
        oriented = tuple(int(value) for value in face)
        if len(oriented) != 3:
            continue
        triangle = (oriented[0], oriented[1], oriented[2])
        sorted_triangle = sorted(triangle)
        signature = (
            sorted_triangle[0],
            sorted_triangle[1],
            sorted_triangle[2],
        )
        if signature in seen_faces:
            continue
        seen_faces.add(signature)
        simplified_faces.append(triangle)
    if not simplified_faces:
        return vertices, faces
    candidate_faces = np.asarray(simplified_faces, dtype=np.int64)
    candidate = trimesh.Trimesh(vertices=clustered, faces=candidate_faces, process=False)
    # Spatial clustering can join opposite walls or delete a narrow passage's
    # triangles. Preserve the canonical collider when that opens the surface.
    if not candidate.is_watertight or not candidate.is_winding_consistent:
        return vertices, faces
    return clustered, candidate_faces


__all__ = [
    "PreparedExportScene",
    "canonical_cave_mesh",
    "prepare_export_scene",
    "simplified_collision_arrays",
]
