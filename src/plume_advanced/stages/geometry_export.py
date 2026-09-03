"""Mesh export helpers for Stage D geometry review."""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import trimesh
import xatlas
from scipy.spatial import cKDTree

from plume_advanced.procedural import procedural_rng
from plume_advanced.stages.geometry_types import CaveGeometry, SurfaceTextureFrame

GLB_EMBEDDED_TEXTURE_MAX_SIZE = 1024
GLB_CAVE_TEXTURE_SCALE_METERS = 8.0
XATLAS_MAX_FACES_PER_BATCH = 25_000


class DisplacementMetadata(TypedDict):
    baked: bool
    scale_m: float
    midlevel: float
    minimum_offset_m: float
    maximum_offset_m: float
    mean_offset_m: float
    sample_standard_deviation_m: NotRequired[float]


class CavePrimitivePayload(TypedDict):
    positions: np.ndarray
    faces: np.ndarray
    texcoords: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    material_index: int
    displacement: DisplacementMetadata


class EventGlbPayload(TypedDict):
    positions: np.ndarray
    faces: np.ndarray
    material_index: int
    texcoords: np.ndarray | None
    normals: np.ndarray
    tangents: np.ndarray | None
    translation: tuple[float, float, float]


def export_geometry_obj(
    cave_geometry: CaveGeometry,
    output_path: str | Path,
    *,
    visual_surface: CavePrimitivePayload | None = None,
) -> Path:
    """Write the same processed visual cave used by GLB as Wavefront OBJ."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cave_payload = visual_surface or build_cave_visual_surface(
        cave_geometry,
        convert_to_gltf=False,
    )
    cave_vertices = cave_payload["positions"]
    cave_faces = cave_payload["faces"]
    cave_uvs = cave_payload["texcoords"]
    cave_normals = cave_payload["normals"]

    mtl_path = output.with_suffix(".mtl")
    _write_obj_mtl(
        mtl_path,
        cave_geometry,
        list(cave_geometry.event_meshes),
        output.parent,
    )

    with output.open("w", encoding="utf-8") as handle:
        handle.write("# PLUME-Advanced portable visual geometry export\n")
        handle.write("# Coordinates: right-handed Z-up metres\n")
        handle.write("# Cave displacement is already baked into vertex positions\n")
        handle.write(f"mtllib {mtl_path.name}\n")
        for key, value in cave_geometry.summary().items():
            handle.write(f"# {key}={value:.3f}\n")
        mesh = trimesh.Trimesh(
            vertices=cave_vertices,
            faces=cave_faces,
            process=False,
        )
        handle.write(f"# trimesh_is_watertight={float(mesh.is_watertight):.3f}\n")
        handle.write(f"# trimesh_euler_number={float(mesh.euler_number):.3f}\n")
        handle.write("o cave_wall\n")
        handle.write("usemtl cave_wall_material\n")
        handle.write("s 1\n")
        for vertex in cave_vertices:
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for u_coord, v_coord in cave_uvs:
            handle.write(f"vt {u_coord:.9f} {1.0 - v_coord:.9f}\n")
        for normal in cave_normals:
            handle.write(f"vn {normal[0]:.9f} {normal[1]:.9f} {normal[2]:.9f}\n")
        for face in cave_faces:
            a, b, c = (index + 1 for index in face)
            handle.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
        vertex_offset = len(cave_vertices)
        uv_vertex_offset = len(cave_uvs)
        for event_mesh in cave_geometry.event_meshes:
            handle.write(f"\no event_{event_mesh.event_id:04d}_{event_mesh.kind}\n")
            handle.write(f"# material_hint={event_mesh.material_hint}\n")
            handle.write(f"# source_generator={event_mesh.source_generator}\n")
            if event_mesh.source_shape_type:
                handle.write(f"# source_shape_type={event_mesh.source_shape_type}\n")
            material_name = _event_material_name(event_mesh)
            handle.write(f"usemtl {material_name}\n")
            for vertex in event_mesh.vertices:
                handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
            has_face_uvs = len(event_mesh.face_uvs) == len(event_mesh.faces)
            if has_face_uvs:
                for face_uvs in event_mesh.face_uvs:
                    for u, v in face_uvs:
                        handle.write(f"vt {u:.9f} {1.0 - v:.9f}\n")
            for face_index, face in enumerate(event_mesh.faces):
                a, b, c = (index + vertex_offset + 1 for index in face)
                if has_face_uvs:
                    au, bu, cu = (
                        uv_vertex_offset + face_index * 3 + 1,
                        uv_vertex_offset + face_index * 3 + 2,
                        uv_vertex_offset + face_index * 3 + 3,
                    )
                    handle.write(f"f {a}/{au} {b}/{bu} {c}/{cu}\n")
                else:
                    handle.write(f"f {a} {b} {c}\n")
            vertex_offset += len(event_mesh.vertices)
            if has_face_uvs:
                uv_vertex_offset += len(event_mesh.faces) * 3

    return output


def build_cave_visual_surface(
    cave_geometry: CaveGeometry,
    *,
    convert_to_gltf: bool = False,
    displacement_image=None,
    image_cache: dict[str, object] | None = None,
) -> CavePrimitivePayload:
    """Finalize smoothing, metric UVs, displacement, normals and tangents.

    This is the format-neutral Stage-F boundary. Geometry generation and
    structural events are already complete when it runs; GLB, OBJ and USD
    consume the same returned visual surface.
    """

    _validate_texture_dependencies(cave_geometry)
    cache = image_cache if image_cache is not None else {}
    if displacement_image is None:
        displacement_image = _load_displacement_image(
            cave_geometry.config.cave_displacement_texture,
            cache,
            max_size=cave_geometry.config.embedded_texture_max_size,
        )
    if (
        cave_geometry.config.strict_texture_loading
        and cave_geometry.config.cave_displacement_texture
        and displacement_image is None
    ):
        raise RuntimeError("Failed to decode configured cave displacement texture")
    if not cave_geometry.assembled_vertices or not cave_geometry.assembled_faces:
        source_vertices, source_faces = _assemble_export_chunks(
            cave_geometry.chunk_meshes
        )
    else:
        source_vertices = np.asarray(
            cave_geometry.assembled_vertices,
            dtype=np.float64,
        )
        source_faces = np.asarray(cave_geometry.assembled_faces, dtype=np.uint32)
    return _cave_primitive_payload(
        vertices=source_vertices,
        faces=source_faces,
        material_index=-1,
        texture_frames=cave_geometry.surface_texture_frames,
        smoothing_iterations=cave_geometry.config.cave_smoothing_iterations,
        variation_seed=cave_geometry.config.random_seed,
        roughness_frequency=cave_geometry.config.wall_roughness_frequency,
        displacement_image=displacement_image,
        displacement_scale_m=cave_geometry.config.cave_displacement_scale_m,
        displacement_midlevel=cave_geometry.config.cave_displacement_midlevel,
        convert_to_gltf=convert_to_gltf,
    )


def export_cave_texture_files(
    cave_geometry: CaveGeometry,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write the same portable PBR images used by GLB for external USD assets."""

    _validate_texture_dependencies(cave_geometry)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, object] = {}
    max_size = cave_geometry.config.embedded_texture_max_size
    images = {
        "diffuse": _load_texture_image(
            cave_geometry.config.cave_diffuse_texture,
            image_cache,
            max_size=max_size,
        ),
        "normal": _load_texture_image(
            cave_geometry.config.cave_normal_texture,
            image_cache,
            max_size=max_size,
        ),
        "metallic_roughness": _load_metallic_roughness_texture(
            cave_geometry.config.cave_roughness_texture,
            image_cache,
            max_size=max_size,
        ),
    }
    filenames = {
        "diffuse": "cave_base_color.png",
        "normal": "cave_normal.png",
        "metallic_roughness": "cave_metallic_roughness.png",
    }
    exported: dict[str, Path] = {}
    for role, image in images.items():
        if image is None:
            continue
        path = output / filenames[role]
        image.save(path, format="PNG")
        exported[role] = path
    return exported


def export_geometry_glb(
    cave_geometry: CaveGeometry,
    output_path: str | Path,
    *,
    visual_surface: CavePrimitivePayload | None = None,
) -> Path:
    """Write a drag-and-drop GLB scene with separately editable event nodes."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _validate_texture_dependencies(cave_geometry)
    builder = _StrictGlbBuilder()
    material_cache: dict[tuple[tuple[str, str], str], object] = {}
    image_cache: dict[str, object] = {}
    cave_material = _cave_strict_glb_material(
        cave_geometry,
        builder=builder,
        image_cache=image_cache,
    )
    displacement_image = _load_displacement_image(
        cave_geometry.config.cave_displacement_texture,
        image_cache,
        max_size=cave_geometry.config.embedded_texture_max_size,
    )
    if (
        cave_geometry.config.strict_texture_loading
        and cave_geometry.config.cave_displacement_texture
        and displacement_image is None
    ):
        raise RuntimeError("Failed to decode configured cave displacement texture")
    displacement = _add_cave_wall_to_strict_glb(
        builder,
        cave_geometry,
        cave_material,
        displacement_image=displacement_image,
        visual_surface=visual_surface,
    )

    for event_mesh in cave_geometry.event_meshes:
        geometry = _event_mesh_to_glb_payload(
            event_mesh,
            builder=builder,
            material_cache=material_cache,
            image_cache=image_cache,
            strict=cave_geometry.config.strict_texture_loading,
            max_size=cave_geometry.config.embedded_texture_max_size,
        )
        node_name = f"event_{event_mesh.event_id:04d}_{event_mesh.kind}"
        builder.mesh_node(
            name=node_name,
            positions=geometry["positions"],
            faces=geometry["faces"],
            material_index=geometry["material_index"],
            texcoords=geometry["texcoords"],
            normals=geometry["normals"],
            tangents=geometry["tangents"],
            translation=geometry["translation"],
            extras={
                "kind": event_mesh.kind,
                "material_hint": event_mesh.material_hint,
                "source_generator": event_mesh.source_generator,
                "source_shape_type": event_mesh.source_shape_type,
                "debris_family_id": event_mesh.debris_family_id,
                "family_anchor_event_id": event_mesh.family_anchor_event_id,
                "debris_role": event_mesh.debris_role,
                "displacement_texture": dict(event_mesh.material_maps).get("displacement", ""),
            },
        )

    output.write_bytes(builder.to_glb())
    _write_geometry_manifest(
        cave_geometry,
        output.with_suffix(".manifest.json"),
        displacement=displacement,
    )
    return output


def _validate_texture_dependencies(cave_geometry: CaveGeometry) -> None:
    """Fail explicitly when a requested material map cannot be embedded."""

    if not cave_geometry.config.strict_texture_loading:
        return
    configured = tuple(
        path
        for path in (
            cave_geometry.config.cave_diffuse_texture,
            cave_geometry.config.cave_normal_texture,
            cave_geometry.config.cave_roughness_texture,
            cave_geometry.config.cave_displacement_texture,
            *(
                path
                for event_mesh in cave_geometry.event_meshes
                for _kind, path in event_mesh.material_maps
            ),
        )
        if path
    )
    missing = [path for path in configured if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Configured textures do not exist: " + ", ".join(missing)
        )
    unsupported = [
        path
        for path in configured
        if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".exr"}
    ]
    if unsupported:
        raise ValueError(
            "Configured textures use unsupported formats: " + ", ".join(unsupported)
        )
    try:
        from PIL import Image as _Image  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Pillow is required for textured GLB export") from error
    if any(Path(path).suffix.lower() == ".exr" for path in configured):
        if shutil.which("convert") is None:
            raise RuntimeError(
                "ImageMagick 'convert' is required to embed configured EXR textures"
            )


def _add_cave_wall_to_strict_glb(
    builder,
    cave_geometry: CaveGeometry,
    cave_material: int,
    *,
    displacement_image=None,
    visual_surface: CavePrimitivePayload | None = None,
) -> DisplacementMetadata:
    if (
        (
            not cave_geometry.assembled_vertices
            or not cave_geometry.assembled_faces
        )
        and not cave_geometry.chunk_meshes
    ):
        return {
            "baked": False,
            "scale_m": 0.0,
            "midlevel": cave_geometry.config.cave_displacement_midlevel,
            "minimum_offset_m": 0.0,
            "maximum_offset_m": 0.0,
            "mean_offset_m": 0.0,
        }
    payload = visual_surface or build_cave_visual_surface(
        cave_geometry,
        convert_to_gltf=True,
        displacement_image=displacement_image,
    )
    builder.mesh_node(
        name="cave_wall",
        positions=payload["positions"],
        faces=payload["faces"],
        material_index=cave_material,
        texcoords=payload["texcoords"],
        normals=payload["normals"],
        tangents=payload["tangents"],
        extras={
            "source": "stage_d_assembled",
            "canonical_source_coordinates": "right-handed Z-up metres",
            "gltf_coordinates": "right-handed Y-up metres",
            "processing_chunk_count": len(cave_geometry.chunk_meshes),
            "structural_event_ids": list(cave_geometry.structural_event_ids),
            "displacement": payload["displacement"],
        },
    )
    return payload["displacement"]


def _assemble_export_chunks(chunk_meshes) -> tuple[np.ndarray, np.ndarray]:
    """Fallback assembly used only when a legacy geometry lacks a welded mesh."""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    offset = 0
    for chunk_mesh in chunk_meshes:
        vertices.extend(chunk_mesh.vertices)
        faces.extend(
            (
                int(face[0]) + offset,
                int(face[1]) + offset,
                int(face[2]) + offset,
            )
            for face in chunk_mesh.faces
        )
        offset += len(chunk_mesh.vertices)
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
    )


def _cave_primitive_payload(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    material_index: int,
    texture_frames: tuple[SurfaceTextureFrame, ...] = (),
    smoothing_iterations: int = 0,
    variation_seed: int | None = None,
    roughness_frequency: float = 0.16,
    displacement_image=None,
    displacement_scale_m: float = 0.0,
    displacement_midlevel: float = 0.5,
    convert_to_gltf: bool = True,
) -> CavePrimitivePayload:
    canonical_vertices = np.asarray(vertices, dtype=np.float64)
    face_indices = np.asarray(faces, dtype=np.uint32)
    canonical_vertices = _smooth_visual_surface(
        canonical_vertices,
        face_indices,
        iterations=smoothing_iterations,
        variation_seed=variation_seed,
        roughness_frequency=roughness_frequency,
    )
    face_indices = _orient_faces_toward_cave_interior(
        canonical_vertices,
        face_indices,
        texture_frames,
    )
    canonical_normals = _angle_weighted_vertex_normals(canonical_vertices, face_indices)
    vertex_mapping, atlas_faces, texcoords = _xatlas_metric_uvs(
        canonical_vertices,
        face_indices,
        canonical_normals,
        scale_m=GLB_CAVE_TEXTURE_SCALE_METERS,
    )
    canonical_vertices, displacement = _bake_seam_consistent_displacement(
        canonical_vertices,
        canonical_normals,
        vertex_mapping,
        texcoords,
        displacement_image,
        scale_m=displacement_scale_m,
        midlevel=displacement_midlevel,
        strength_weights=(
            0.28
            + 0.72
            * _surface_roughness_weights(
                canonical_vertices,
                variation_seed=variation_seed,
                roughness_frequency=roughness_frequency,
            )
        ),
    )
    canonical_normals = _angle_weighted_vertex_normals(
        canonical_vertices,
        face_indices,
    )
    canonical_vertices = canonical_vertices[vertex_mapping]
    canonical_normals = canonical_normals[vertex_mapping]
    face_indices = atlas_faces
    mesh_tangents = _mesh_tangents(
        canonical_vertices,
        face_indices,
        texcoords,
        canonical_normals,
    )
    canonical_tangents = mesh_tangents
    if convert_to_gltf:
        output_positions = _canonical_to_gltf_vectors(canonical_vertices)
        output_normals = _canonical_to_gltf_vectors(canonical_normals)
        output_tangents = _canonical_to_gltf_tangents(canonical_tangents)
    else:
        output_positions = canonical_vertices.astype(np.float32)
        output_normals = canonical_normals.astype(np.float32)
        output_tangents = canonical_tangents.astype(np.float32)
    return {
        "positions": output_positions,
        "faces": face_indices,
        "texcoords": texcoords.astype(np.float32),
        "normals": output_normals,
        "tangents": output_tangents,
        "material_index": material_index,
        "displacement": displacement,
    }


def _xatlas_metric_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    *,
    scale_m: float,
    max_faces_per_batch: int = XATLAS_MAX_FACES_PER_BATCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate seam-aware conformal charts with a repeatable metric density.

    Large voxel surfaces are divided into bounded, face-order-preserving
    batches. The source assembler emits spatially local face runs, and the
    material is repeat-wrapped, so independent atlas packing is both faster
    and harmless: only each chart's local shape and metric scale matter.
    """

    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    triangles = np.ascontiguousarray(faces, dtype=np.uint32)
    vertex_normals = np.ascontiguousarray(normals, dtype=np.float32)
    if len(positions) == 0 or len(triangles) == 0:
        return (
            np.arange(len(positions), dtype=np.uint32),
            triangles.copy(),
            np.zeros((len(positions), 2), dtype=np.float64),
        )

    batch_size = max(1, int(max_faces_per_batch))
    mappings: list[np.ndarray] = []
    face_batches: list[np.ndarray] = []
    uv_batches: list[np.ndarray] = []
    vertex_offset = 0
    for face_start in range(0, len(triangles), batch_size):
        source_faces = triangles[face_start : face_start + batch_size]
        source_vertex_indices = np.unique(source_faces)
        local_faces = np.searchsorted(
            source_vertex_indices,
            source_faces,
        ).astype(np.uint32)
        atlas = xatlas.Atlas()
        atlas.add_mesh(
            positions[source_vertex_indices],
            local_faces,
            vertex_normals[source_vertex_indices],
        )
        atlas.generate()
        local_mapping, local_atlas_faces, local_atlas_uvs = atlas[0]
        local_mapping = np.asarray(local_mapping, dtype=np.uint32)
        local_atlas_faces = np.asarray(local_atlas_faces, dtype=np.uint32)
        local_atlas_uvs = np.asarray(local_atlas_uvs, dtype=np.float64)
        atlas_dimensions = np.asarray(
            (atlas.width, atlas.height),
            dtype=np.float64,
        )
        if np.any(atlas_dimensions <= 0.0):
            raise RuntimeError("xatlas returned an invalid atlas resolution")
        local_atlas_uvs *= atlas_dimensions
        source_mapping = source_vertex_indices[local_mapping].astype(np.uint32)
        local_atlas_uvs = _scale_atlas_uvs_to_metric(
            np.asarray(vertices, dtype=np.float64)[source_mapping],
            local_atlas_faces,
            local_atlas_uvs,
            scale_m=scale_m,
        )
        mappings.append(source_mapping)
        face_batches.append(local_atlas_faces + vertex_offset)
        uv_batches.append(local_atlas_uvs)
        vertex_offset += len(source_mapping)

    vertex_mapping = np.concatenate(mappings)
    atlas_faces = np.concatenate(face_batches)
    atlas_uvs = np.concatenate(uv_batches)
    if (
        len(vertex_mapping) != len(atlas_uvs)
        or atlas_faces.shape != triangles.shape
        or int(vertex_mapping.max(initial=0)) >= len(positions)
        or int(atlas_faces.max(initial=0)) >= len(vertex_mapping)
        or not np.isfinite(atlas_uvs).all()
    ):
        raise RuntimeError("xatlas returned an invalid cave-wall parameterization")
    return vertex_mapping, atlas_faces, atlas_uvs


def _scale_atlas_uvs_to_metric(
    expanded_positions: np.ndarray,
    atlas_faces: np.ndarray,
    atlas_uvs: np.ndarray,
    *,
    scale_m: float,
) -> np.ndarray:
    """Rescale a normalized xatlas result to metres per repeated texture tile."""

    triangle_positions = expanded_positions[atlas_faces]
    triangle_uvs = atlas_uvs[atlas_faces]
    world_double_areas = np.linalg.norm(
        np.cross(
            triangle_positions[:, 1] - triangle_positions[:, 0],
            triangle_positions[:, 2] - triangle_positions[:, 0],
        ),
        axis=1,
    )
    uv_edge_a = triangle_uvs[:, 1] - triangle_uvs[:, 0]
    uv_edge_b = triangle_uvs[:, 2] - triangle_uvs[:, 0]
    uv_double_areas = np.abs(
        uv_edge_a[:, 0] * uv_edge_b[:, 1]
        - uv_edge_a[:, 1] * uv_edge_b[:, 0]
    )
    valid = (world_double_areas > 1e-12) & (uv_double_areas > 1e-12)
    if not np.any(valid):
        raise RuntimeError("xatlas produced no measurable cave-wall UV triangles")

    metres_per_uv = np.sqrt(
        world_double_areas[valid] / uv_double_areas[valid]
    )
    return (
        np.asarray(atlas_uvs, dtype=np.float64)
        * float(np.median(metres_per_uv))
        / max(float(scale_m), 1e-6)
    )


def _bake_seam_consistent_displacement(
    vertices: np.ndarray,
    normals: np.ndarray,
    vertex_mapping: np.ndarray,
    texcoords: np.ndarray,
    image,
    *,
    scale_m: float,
    midlevel: float,
    strength_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, DisplacementMetadata]:
    """Bake atlas-driven displacement without separating duplicated UV seams."""

    positions = np.asarray(vertices, dtype=np.float64)
    vertex_normals = np.asarray(normals, dtype=np.float64)
    mapping = np.asarray(vertex_mapping, dtype=np.int64)
    expanded_positions = positions[mapping]
    expanded_normals = vertex_normals[mapping]
    expanded_weights = (
        None
        if strength_weights is None
        else np.asarray(strength_weights, dtype=np.float64)[mapping]
    )
    expanded_displaced, metadata = _bake_vertex_displacement(
        expanded_positions,
        expanded_normals,
        texcoords,
        image,
        scale_m=scale_m,
        midlevel=midlevel,
        strength_weights=expanded_weights,
    )
    if not metadata["baked"]:
        return positions.copy(), metadata

    expanded_offsets = np.einsum(
        "ij,ij->i",
        expanded_displaced - expanded_positions,
        expanded_normals,
    )
    offset_sums = np.bincount(
        mapping,
        weights=expanded_offsets,
        minlength=len(positions),
    )
    offset_counts = np.bincount(mapping, minlength=len(positions))
    offsets = offset_sums / np.maximum(offset_counts, 1)
    displaced = positions + vertex_normals * offsets[:, None]
    metadata.update(
        {
            "minimum_offset_m": float(offsets.min()),
            "maximum_offset_m": float(offsets.max()),
            "mean_offset_m": float(offsets.mean()),
            "sample_standard_deviation_m": float(offsets.std()),
        }
    )
    return displaced, metadata


def _bake_vertex_displacement(
    vertices: np.ndarray,
    normals: np.ndarray,
    texcoords: np.ndarray,
    image,
    *,
    scale_m: float,
    midlevel: float,
    strength_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, DisplacementMetadata]:
    """Bake a repeat-wrapped height map into portable vertex positions."""

    positions = np.asarray(vertices, dtype=np.float64)
    metadata: DisplacementMetadata = {
        "baked": False,
        "scale_m": float(scale_m),
        "midlevel": float(midlevel),
        "minimum_offset_m": 0.0,
        "maximum_offset_m": 0.0,
        "mean_offset_m": 0.0,
    }
    if image is None or scale_m <= 0.0 or len(positions) == 0:
        return positions.copy(), metadata
    if not 0.0 < midlevel < 1.0:
        raise ValueError("Displacement midlevel must be in (0, 1)")

    samples = _sample_periodic_grayscale(image, texcoords)
    denominators = np.where(
        samples >= midlevel,
        max(1.0 - midlevel, 1e-9),
        max(midlevel, 1e-9),
    )
    normalized = (samples - midlevel) / denominators
    offsets = np.clip(normalized, -1.0, 1.0) * float(scale_m)
    if strength_weights is not None:
        weights = np.asarray(strength_weights, dtype=np.float64)
        if weights.shape != offsets.shape:
            raise ValueError("Displacement strength weights must match vertex count")
        offsets *= np.clip(weights, 0.0, 1.0)
    displaced = positions + np.asarray(normals, dtype=np.float64) * offsets[:, None]
    metadata.update(
        {
            "baked": True,
            "minimum_offset_m": float(offsets.min()),
            "maximum_offset_m": float(offsets.max()),
            "mean_offset_m": float(offsets.mean()),
            "sample_standard_deviation_m": float(offsets.std()),
        }
    )
    return displaced, metadata


def _sample_periodic_grayscale(image, texcoords: np.ndarray) -> np.ndarray:
    """Bilinearly sample an image using glTF repeat-wrapped UV coordinates."""

    pixels = np.asarray(image, dtype=np.float64)
    if pixels.ndim == 3:
        pixels = pixels[:, :, 0]
    if pixels.size and float(np.max(pixels)) > 1.0:
        maximum = 65_535.0 if float(np.max(pixels)) > 255.0 else 255.0
        pixels /= maximum
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError("Displacement image must contain grayscale pixels")
    height, width = pixels.shape
    uv = np.mod(np.asarray(texcoords, dtype=np.float64), 1.0)
    x_coord = uv[:, 0] * max(width - 1, 0)
    y_coord = uv[:, 1] * max(height - 1, 0)
    x0 = np.floor(x_coord).astype(np.int64)
    y0 = np.floor(y_coord).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x_weight = x_coord - x0
    y_weight = y_coord - y0
    top = pixels[y0, x0] * (1.0 - x_weight) + pixels[y0, x1] * x_weight
    bottom = pixels[y1, x0] * (1.0 - x_weight) + pixels[y1, x1] * x_weight
    return top * (1.0 - y_weight) + bottom * y_weight


def _smooth_visual_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int,
    variation_seed: int | None = None,
    roughness_frequency: float = 0.16,
) -> np.ndarray:
    """Remove voxel terraces while retaining coherent rough lava-flow zones."""

    positions = np.asarray(vertices, dtype=np.float64)
    if iterations <= 0 or len(positions) == 0:
        return positions.copy()

    def filtered(source: np.ndarray, count: int) -> np.ndarray:
        mesh = trimesh.Trimesh(
            vertices=source.copy(),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=0.50,
            nu=0.53,
            iterations=int(count),
        )
        return np.asarray(mesh.vertices, dtype=np.float64)

    smoothed = filtered(positions, int(iterations))
    if iterations > 2:
        lightly_smoothed = filtered(positions, 2)
        roughness = _surface_roughness_weights(
            positions,
            variation_seed=variation_seed,
            roughness_frequency=roughness_frequency,
        )
        extra_smoothing = 1.0 - 0.78 * roughness
        smoothed = lightly_smoothed + extra_smoothing[:, None] * (
            smoothed - lightly_smoothed
        )
    if smoothed.shape != positions.shape or not np.isfinite(smoothed).all():
        raise ValueError("Visual cave smoothing produced invalid vertices")
    return smoothed


def _surface_roughness_weights(
    vertices: np.ndarray,
    *,
    variation_seed: int | None,
    roughness_frequency: float,
) -> np.ndarray:
    """Build deterministic broad zones that alternate smooth and rough relief."""

    positions = np.asarray(vertices, dtype=np.float64)
    if len(positions) == 0:
        return np.empty(0, dtype=np.float64)
    rng = procedural_rng(variation_seed, "export-surface-detail")
    phase_a, phase_b, phase_c = rng.uniform(0.0, 2.0 * np.pi, size=3)
    frequency = max(float(roughness_frequency) * 0.28, 0.025)
    x_coord, y_coord, z_coord = positions.T
    field = (
        0.56 * np.sin(frequency * x_coord + phase_a)
        + 0.40 * np.cos(frequency * 0.83 * y_coord + phase_b)
        + 0.28
        * np.sin(
            frequency * 0.61 * (x_coord + y_coord + 0.45 * z_coord)
            + phase_c
        )
    )
    weights = np.clip(0.5 + 0.43 * field, 0.0, 1.0)
    return weights * weights * (3.0 - 2.0 * weights)


def _orient_faces_toward_cave_interior(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_frames: tuple[SurfaceTextureFrame, ...],
) -> np.ndarray:
    """Orient the cave boundary toward its route centres for backface culling."""

    triangles = np.asarray(faces, dtype=np.uint32).copy()
    if not texture_frames or len(triangles) == 0:
        return triangles
    positions = np.asarray(vertices, dtype=np.float64)
    triangle_positions = positions[triangles]
    face_centers = triangle_positions.mean(axis=1)
    route_centers = np.asarray(
        [frame.center for frame in texture_frames],
        dtype=np.float64,
    )
    _distances, frame_indices = cKDTree(route_centers).query(face_centers, k=1)
    toward_interior = route_centers[frame_indices] - face_centers
    face_normals = np.cross(
        triangle_positions[:, 1] - triangle_positions[:, 0],
        triangle_positions[:, 2] - triangle_positions[:, 0],
    )
    valid = (
        np.linalg.norm(face_normals, axis=1) > 1e-12
    ) & (
        np.linalg.norm(toward_interior, axis=1) > 1e-12
    )
    if np.any(valid):
        orientation = np.einsum(
            "ij,ij->i",
            face_normals[valid],
            toward_interior[valid],
        )
        if float(np.median(orientation)) < 0.0:
            triangles[:, [1, 2]] = triangles[:, [2, 1]]
    return triangles


def _angle_weighted_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Compute smooth normals after global welding."""

    positions = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(positions, dtype=np.float64)
    for face in triangles:
        triangle = positions[face]
        edge_a = triangle[1] - triangle[0]
        edge_b = triangle[2] - triangle[0]
        face_normal = np.cross(edge_a, edge_b)
        normal_length = float(np.linalg.norm(face_normal))
        if normal_length <= 1e-12:
            continue
        face_normal /= normal_length
        for corner in range(3):
            center = triangle[corner]
            vector_a = triangle[(corner + 1) % 3] - center
            vector_b = triangle[(corner + 2) % 3] - center
            length_a = float(np.linalg.norm(vector_a))
            length_b = float(np.linalg.norm(vector_b))
            if length_a <= 1e-12 or length_b <= 1e-12:
                continue
            cosine = float(np.dot(vector_a, vector_b) / (length_a * length_b))
            angle = math.acos(float(np.clip(cosine, -1.0, 1.0)))
            normals[face[corner]] += face_normal * angle

    lengths = np.linalg.norm(normals, axis=1)
    missing = lengths <= 1e-12
    normals[~missing] /= lengths[~missing, None]
    normals[missing] = np.array((0.0, 0.0, 1.0))
    return normals


def _mesh_tangents(
    vertices: np.ndarray,
    faces: np.ndarray,
    texcoords: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Generate orthonormal tangent frames compatible with glTF normal maps."""

    positions = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(texcoords, dtype=np.float64)
    vertex_normals = np.asarray(normals, dtype=np.float64)
    tangent_u = np.zeros_like(positions)
    tangent_v = np.zeros_like(positions)

    for face in triangles:
        p0, p1, p2 = positions[face]
        uv0, uv1, uv2 = uv[face]
        edge1 = p1 - p0
        edge2 = p2 - p0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0
        determinant = duv1[0] * duv2[1] - duv1[1] * duv2[0]
        if abs(float(determinant)) <= 1e-12:
            continue
        reciprocal = 1.0 / determinant
        s_direction = (edge1 * duv2[1] - edge2 * duv1[1]) * reciprocal
        t_direction = (edge2 * duv1[0] - edge1 * duv2[0]) * reciprocal
        for vertex_index in face:
            tangent_u[vertex_index] += s_direction
            tangent_v[vertex_index] += t_direction

    tangents = np.zeros((len(positions), 4), dtype=np.float64)
    for index, normal in enumerate(vertex_normals):
        tangent = tangent_u[index] - normal * float(np.dot(normal, tangent_u[index]))
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1e-12:
            reference = (
                np.array((0.0, 0.0, 1.0))
                if abs(float(normal[2])) < 0.9
                else np.array((1.0, 0.0, 0.0))
            )
            tangent = np.cross(reference, normal)
            tangent_length = max(float(np.linalg.norm(tangent)), 1e-12)
        tangent /= tangent_length
        handedness = (
            -1.0
            if float(np.dot(np.cross(normal, tangent), tangent_v[index])) < 0.0
            else 1.0
        )
        tangents[index, :3] = tangent
        tangents[index, 3] = handedness
    return tangents


def _canonical_to_gltf_vectors(values: np.ndarray) -> np.ndarray:
    """Rotate right-handed Z-up coordinates into glTF right-handed Y-up."""

    vectors = np.asarray(values, dtype=np.float64)
    transformed = np.column_stack(
        (vectors[:, 0], vectors[:, 2], -vectors[:, 1])
    )
    return transformed.astype(np.float32)


def _canonical_to_gltf_tangents(tangents: np.ndarray) -> np.ndarray:
    tangent_values = np.asarray(tangents, dtype=np.float64)
    transformed_xyz = _canonical_to_gltf_vectors(tangent_values[:, :3])
    return np.column_stack((transformed_xyz, tangent_values[:, 3])).astype(np.float32)


def canonical_visual_to_gltf(
    visual_surface: CavePrimitivePayload,
) -> CavePrimitivePayload:
    """Transform one prepared canonical surface without rebuilding its atlas."""

    return {
        "positions": _canonical_to_gltf_vectors(visual_surface["positions"]),
        "faces": visual_surface["faces"],
        "texcoords": visual_surface["texcoords"],
        "normals": _canonical_to_gltf_vectors(visual_surface["normals"]),
        "tangents": _canonical_to_gltf_tangents(visual_surface["tangents"]),
        "material_index": visual_surface["material_index"],
        "displacement": visual_surface["displacement"],
    }


def _canonical_to_gltf_translation(
    translation: np.ndarray,
) -> tuple[float, float, float]:
    vector = np.asarray(translation, dtype=np.float64)
    return (float(vector[0]), float(vector[2]), float(-vector[1]))


def _cave_strict_glb_material(cave_geometry: CaveGeometry, *, builder, image_cache: dict) -> int:
    texture_size = cave_geometry.config.embedded_texture_max_size
    diffuse_image = _load_texture_image(
        cave_geometry.config.cave_diffuse_texture,
        image_cache,
        max_size=texture_size,
    )
    normal_image = _load_texture_image(
        cave_geometry.config.cave_normal_texture,
        image_cache,
        max_size=texture_size,
    )
    roughness_image = _load_metallic_roughness_texture(
        cave_geometry.config.cave_roughness_texture,
        image_cache,
        max_size=texture_size,
    )
    for label, path, image in (
        ("diffuse", cave_geometry.config.cave_diffuse_texture, diffuse_image),
        ("normal", cave_geometry.config.cave_normal_texture, normal_image),
        ("roughness", cave_geometry.config.cave_roughness_texture, roughness_image),
    ):
        if cave_geometry.config.strict_texture_loading and path and image is None:
            raise RuntimeError(f"Failed to decode configured cave {label} texture: {path}")
    return builder.material(
        name="cave_wall_material",
        base_color_factor=(0.36, 0.35, 0.31, 1.0),
        base_color_texture=diffuse_image,
        normal_texture=normal_image,
        normal_scale=cave_geometry.config.cave_normal_scale,
        metallic_roughness_texture=roughness_image,
        metallic_factor=0.0,
        roughness_factor=0.92,
        double_sided=False,
    )


def _write_geometry_manifest(
    cave_geometry: CaveGeometry,
    output_path: Path,
    *,
    displacement: DisplacementMetadata,
) -> Path:
    payload = {
        "schema": "plume.geometry_manifest.v2",
        "coordinates": {
            "source": "right-handed Z-up metres",
            "asset": "glTF right-handed Y-up metres",
        },
        "cave": {
            "node": "cave_wall",
            "primitive_count": 1,
            "processing_chunk_count": len(cave_geometry.chunk_meshes),
            "attributes": ["POSITION", "NORMAL", "TANGENT", "TEXCOORD_0"],
            "material": {
                "diffuse": cave_geometry.config.cave_diffuse_texture,
                "normal": cave_geometry.config.cave_normal_texture,
                "roughness": cave_geometry.config.cave_roughness_texture,
                "displacement": cave_geometry.config.cave_displacement_texture,
                "uv_projection": "xatlas_metric_charts",
                "uv_scale_m": GLB_CAVE_TEXTURE_SCALE_METERS,
                "surface_order": [
                    "visual_smoothing",
                    "xatlas_chart_generation",
                    "metric_uv_rescaling",
                    "seam_consistent_vertex_displacement",
                    "final_normals_and_tangents",
                    "material_binding",
                ],
                "normal_scale": cave_geometry.config.cave_normal_scale,
                "visual_smoothing_iterations": (
                    cave_geometry.config.cave_smoothing_iterations
                ),
                "displacement_baked": bool(displacement.get("baked", False)),
                "displacement_scale_m": float(displacement.get("scale_m", 0.0)),
                "displacement_midlevel": float(
                    displacement.get(
                        "midlevel",
                        cave_geometry.config.cave_displacement_midlevel,
                    )
                ),
                "displacement_range_m": [
                    float(displacement.get("minimum_offset_m", 0.0)),
                    float(displacement.get("maximum_offset_m", 0.0)),
                ],
                "double_sided": False,
            },
        },
        "events": [
            {
                "node": f"event_{event_mesh.event_id:04d}_{event_mesh.kind}",
                "event_id": event_mesh.event_id,
                "kind": event_mesh.kind,
                "material_hint": event_mesh.material_hint,
                "source_generator": event_mesh.source_generator,
                "source_shape_type": event_mesh.source_shape_type,
                "debris_family_id": event_mesh.debris_family_id,
                "family_anchor_event_id": event_mesh.family_anchor_event_id,
                "debris_role": event_mesh.debris_role,
                "vertex_count": event_mesh.vertex_count,
                "face_count": event_mesh.face_count,
                "material_maps": dict(event_mesh.material_maps),
            }
            for event_mesh in cave_geometry.event_meshes
        ],
        "structural_event_ids": list(cave_geometry.structural_event_ids),
        "summary": cave_geometry.summary(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _event_mesh_to_glb_payload(
    event_mesh,
    *,
    builder,
    material_cache: dict,
    image_cache: dict,
    strict: bool,
    max_size: int,
) -> EventGlbPayload:
    vertices = np.array(event_mesh.vertices, dtype=np.float32)
    pivot = _event_pivot(vertices)
    local_vertices = vertices - pivot
    material_index = _event_strict_glb_material(
        event_mesh,
        builder=builder,
        material_cache=material_cache,
        image_cache=image_cache,
        strict=strict,
        max_size=max_size,
    )

    if event_mesh.face_uvs and len(event_mesh.face_uvs) == len(event_mesh.faces):
        expanded_vertices: list[tuple[float, float, float]] = []
        expanded_uvs: list[tuple[float, float]] = []
        expanded_faces: list[tuple[int, int, int]] = []
        for face, face_uvs in zip(event_mesh.faces, event_mesh.face_uvs, strict=True):
            start = len(expanded_vertices)
            for vertex_index, uv in zip(face, face_uvs, strict=True):
                vertex = local_vertices[vertex_index]
                expanded_vertices.append(
                    (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                )
                expanded_uvs.append((float(uv[0]), float(1.0 - uv[1])))
            expanded_faces.append((start, start + 1, start + 2))
        expanded_positions = np.array(expanded_vertices, dtype=np.float64)
        expanded_face_indices = np.array(expanded_faces, dtype=np.uint32)
        texcoords = np.array(expanded_uvs, dtype=np.float64)
        normals = _angle_weighted_vertex_normals(
            expanded_positions,
            expanded_face_indices,
        )
        tangents = _mesh_tangents(
            expanded_positions,
            expanded_face_indices,
            texcoords,
            normals,
        )
        return {
            "positions": _canonical_to_gltf_vectors(expanded_positions),
            "texcoords": texcoords.astype(np.float32),
            "normals": _canonical_to_gltf_vectors(normals),
            "tangents": _canonical_to_gltf_tangents(tangents),
            "faces": expanded_face_indices,
            "material_index": material_index,
            "translation": _canonical_to_gltf_translation(pivot),
        }

    faces = np.array(event_mesh.faces, dtype=np.uint32)
    normals = _angle_weighted_vertex_normals(local_vertices, faces)
    return {
        "positions": _canonical_to_gltf_vectors(local_vertices),
        "texcoords": None,
        "normals": _canonical_to_gltf_vectors(normals),
        "tangents": None,
        "faces": faces,
        "material_index": material_index,
        "translation": _canonical_to_gltf_translation(pivot),
    }


def _event_strict_glb_material(
    event_mesh,
    *,
    builder,
    material_cache: dict,
    image_cache: dict,
    strict: bool,
    max_size: int,
) -> int:
    cache_key = (tuple(event_mesh.material_maps), event_mesh.source_shape_type)
    if cache_key in material_cache:
        return material_cache[cache_key]

    material_maps = dict(event_mesh.material_maps)
    diffuse_image = _load_texture_image(
        material_maps.get("diffuse"),
        image_cache,
        max_size=max_size,
    )
    normal_image = _load_texture_image(
        material_maps.get("normal"),
        image_cache,
        max_size=max_size,
    )
    roughness_image = _load_metallic_roughness_texture(
        material_maps.get("roughness"),
        image_cache,
        max_size=max_size,
    )
    for label, image in (
        ("diffuse", diffuse_image),
        ("normal", normal_image),
        ("roughness", roughness_image),
    ):
        path = material_maps.get(label)
        if strict and path and image is None:
            raise RuntimeError(
                f"Failed to decode event {event_mesh.event_id} {label} texture: {path}"
            )
    material_index = builder.material(
        name=_event_shared_material_name(event_mesh),
        base_color_factor=(
            (1.0, 1.0, 1.0, 1.0)
            if diffuse_image is not None
            else _event_fallback_color(event_mesh)
        ),
        base_color_texture=diffuse_image,
        normal_texture=normal_image,
        metallic_roughness_texture=roughness_image,
        metallic_factor=0.0,
        roughness_factor=0.86,
        double_sided=True,
    )
    material_cache[cache_key] = material_index
    return material_index


def _event_fallback_color(event_mesh) -> tuple[float, float, float, float]:
    """Return a non-white basalt fallback for native, untextured props."""

    if event_mesh.kind == "boulder":
        return (0.18, 0.17, 0.15, 1.0)
    if event_mesh.kind == "rock":
        return (0.22, 0.20, 0.17, 1.0)
    return (0.25, 0.23, 0.20, 1.0)


class _StrictGlbBuilder:
    def __init__(self) -> None:
        self._binary = bytearray()
        self._buffer_views: list[dict[str, object]] = []
        self._accessors: list[dict[str, object]] = []
        self._images: list[dict[str, object]] = []
        self._textures: list[dict[str, object]] = []
        self._materials: list[dict[str, object]] = []
        self._meshes: list[dict[str, object]] = []
        self._nodes: list[dict[str, object]] = [{"name": "world", "children": []}]
        self._image_cache: dict[bytes, int] = {}
        self._texture_cache: dict[int, int] = {}

    def material(
        self,
        *,
        name: str,
        base_color_factor: tuple[float, float, float, float] | None = None,
        base_color_texture=None,
        normal_texture=None,
        normal_scale: float = 1.0,
        metallic_roughness_texture=None,
        metallic_factor: float = 0.0,
        roughness_factor: float = 0.86,
        double_sided: bool = True,
    ) -> int:
        pbr: dict[str, object] = {
            "metallicFactor": float(metallic_factor),
            "roughnessFactor": float(roughness_factor),
        }
        if base_color_factor is not None:
            pbr["baseColorFactor"] = [float(value) for value in base_color_factor]
        if base_color_texture is not None:
            pbr["baseColorTexture"] = {"index": self._texture(base_color_texture)}
        if metallic_roughness_texture is not None:
            pbr["metallicRoughnessTexture"] = {"index": self._texture(metallic_roughness_texture)}

        material: dict[str, object] = {
            "name": name,
            "pbrMetallicRoughness": pbr,
            "doubleSided": bool(double_sided),
        }
        if normal_texture is not None:
            material["normalTexture"] = {
                "index": self._texture(normal_texture),
                "scale": float(normal_scale),
            }
        self._materials.append(material)
        return len(self._materials) - 1

    def mesh_node(
        self,
        *,
        name: str,
        positions: np.ndarray,
        faces: np.ndarray,
        material_index: int,
        texcoords: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        tangents: np.ndarray | None = None,
        translation: tuple[float, float, float] | None = None,
        extras: dict[str, object] | None = None,
    ) -> int:
        positions = np.asarray(positions, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        if positions.size == 0 or faces.size == 0:
            raise ValueError(f"Cannot export empty mesh node {name!r}")

        attributes = {
            "POSITION": self._accessor(
                positions,
                component_type=5126,
                accessor_type="VEC3",
                target=34962,
                minimum=positions.min(axis=0).tolist(),
                maximum=positions.max(axis=0).tolist(),
            )
        }
        if texcoords is not None:
            texcoords = np.asarray(texcoords, dtype=np.float32)
            if len(texcoords) != len(positions):
                raise ValueError(f"Mesh node {name!r} has mismatched texture coordinates")
            attributes["TEXCOORD_0"] = self._accessor(
                texcoords,
                component_type=5126,
                accessor_type="VEC2",
                target=34962,
                minimum=texcoords.min(axis=0).tolist(),
                maximum=texcoords.max(axis=0).tolist(),
            )
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)
            if len(normals) != len(positions):
                raise ValueError(f"Mesh node {name!r} has mismatched normals")
            attributes["NORMAL"] = self._accessor(
                normals,
                component_type=5126,
                accessor_type="VEC3",
                target=34962,
                minimum=normals.min(axis=0).tolist(),
                maximum=normals.max(axis=0).tolist(),
            )
        if tangents is not None:
            tangents = np.asarray(tangents, dtype=np.float32)
            if len(tangents) != len(positions):
                raise ValueError(f"Mesh node {name!r} has mismatched tangents")
            attributes["TANGENT"] = self._accessor(
                tangents,
                component_type=5126,
                accessor_type="VEC4",
                target=34962,
                minimum=tangents.min(axis=0).tolist(),
                maximum=tangents.max(axis=0).tolist(),
            )

        max_index = int(faces.max())
        if max_index <= 65_535:
            index_data = faces.astype(np.uint16)
            component_type = 5123
        else:
            index_data = faces.astype(np.uint32)
            component_type = 5125
        indices = self._accessor(
            index_data.reshape(-1),
            component_type=component_type,
            accessor_type="SCALAR",
            target=34963,
            minimum=[0],
            maximum=[max_index],
        )

        mesh_index = len(self._meshes)
        self._meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": indices,
                        "material": material_index,
                        "mode": 4,
                    }
                ],
            }
        )
        node: dict[str, object] = {"name": name, "mesh": mesh_index}
        if translation is not None:
            node["translation"] = [float(value) for value in translation]
        if extras:
            node["extras"] = extras
        node_index = len(self._nodes)
        self._nodes.append(node)
        children = self._nodes[0].setdefault("children", [])
        if not isinstance(children, list):
            raise TypeError("Root glTF node children must be a list")
        children.append(node_index)
        return node_index

    def _primitive(
        self,
        *,
        positions: np.ndarray,
        faces: np.ndarray,
        material_index: int,
        texcoords: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        tangents: np.ndarray | None = None,
    ) -> dict[str, object]:
        attributes = {
            "POSITION": self._accessor(
                positions,
                component_type=5126,
                accessor_type="VEC3",
                target=34962,
                minimum=positions.min(axis=0).tolist(),
                maximum=positions.max(axis=0).tolist(),
            )
        }
        if texcoords is not None:
            texcoords = np.asarray(texcoords, dtype=np.float32)
            attributes["TEXCOORD_0"] = self._accessor(
                texcoords,
                component_type=5126,
                accessor_type="VEC2",
                target=34962,
                minimum=texcoords.min(axis=0).tolist(),
                maximum=texcoords.max(axis=0).tolist(),
            )
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)
            attributes["NORMAL"] = self._accessor(
                normals,
                component_type=5126,
                accessor_type="VEC3",
                target=34962,
                minimum=normals.min(axis=0).tolist(),
                maximum=normals.max(axis=0).tolist(),
            )
        if tangents is not None:
            tangents = np.asarray(tangents, dtype=np.float32)
            attributes["TANGENT"] = self._accessor(
                tangents,
                component_type=5126,
                accessor_type="VEC4",
                target=34962,
                minimum=tangents.min(axis=0).tolist(),
                maximum=tangents.max(axis=0).tolist(),
            )

        max_index = int(faces.max())
        if max_index <= 65_535:
            index_data = faces.astype(np.uint16)
            component_type = 5123
        else:
            index_data = faces.astype(np.uint32)
            component_type = 5125
        indices = self._accessor(
            index_data.reshape(-1),
            component_type=component_type,
            accessor_type="SCALAR",
            target=34963,
            minimum=[0],
            maximum=[max_index],
        )
        return {
            "attributes": attributes,
            "indices": indices,
            "material": material_index,
            "mode": 4,
        }

    def to_glb(self) -> bytes:
        document = {
            "asset": {
                "version": "2.0",
                "generator": "PLUME-Advanced portable GLB exporter",
                "extras": {
                    "content": "complete procedural cave scene",
                    "length_unit": "metre",
                    "visual_displacement": "baked into POSITION",
                },
            },
            "scene": 0,
            "scenes": [{"name": "PLUME portable cave scene", "nodes": [0]}],
            "nodes": self._nodes,
            "meshes": self._meshes,
            "materials": self._materials,
            "buffers": [{"byteLength": len(self._binary)}],
            "bufferViews": self._buffer_views,
            "accessors": self._accessors,
        }
        if self._images:
            document["images"] = self._images
        if self._textures:
            document["textures"] = self._textures

        json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
        json_bytes += b" " * ((_alignment_padding(len(json_bytes), 4)))
        bin_bytes = bytes(self._binary)
        bin_bytes += b"\x00" * _alignment_padding(len(bin_bytes), 4)
        length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
        return (
            struct.pack("<4sII", b"glTF", 2, length)
            + struct.pack("<II", len(json_bytes), 0x4E4F534A)
            + json_bytes
            + struct.pack("<II", len(bin_bytes), 0x004E4942)
            + bin_bytes
        )

    def _texture(self, image) -> int:
        image_index = self._image(image)
        if image_index not in self._texture_cache:
            self._texture_cache[image_index] = len(self._textures)
            self._textures.append({"source": image_index})
        return self._texture_cache[image_index]

    def _image(self, image) -> int:
        png_bytes = _image_to_png_bytes(image)
        if png_bytes in self._image_cache:
            return self._image_cache[png_bytes]
        buffer_view = self._buffer_view(png_bytes, target=None)
        image_index = len(self._images)
        self._images.append({"bufferView": buffer_view, "mimeType": "image/png"})
        self._image_cache[png_bytes] = image_index
        return image_index

    def _accessor(
        self,
        data: np.ndarray,
        *,
        component_type: int,
        accessor_type: str,
        target: int,
        minimum: list[float] | list[int],
        maximum: list[float] | list[int],
    ) -> int:
        contiguous = np.ascontiguousarray(data)
        buffer_view = self._buffer_view(contiguous.tobytes(), target=target)
        self._accessors.append(
            {
                "bufferView": buffer_view,
                "byteOffset": 0,
                "componentType": component_type,
                "count": int(len(contiguous)),
                "type": accessor_type,
                "min": [float(value) if component_type == 5126 else int(value) for value in minimum],
                "max": [float(value) if component_type == 5126 else int(value) for value in maximum],
            }
        )
        return len(self._accessors) - 1

    def _buffer_view(self, payload: bytes, *, target: int | None) -> int:
        padding = _alignment_padding(len(self._binary), 4)
        if padding:
            self._binary.extend(b"\x00" * padding)
        offset = len(self._binary)
        self._binary.extend(payload)
        view: dict[str, object] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        self._buffer_views.append(view)
        return len(self._buffer_views) - 1


def _image_to_png_bytes(image) -> bytes:
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def _alignment_padding(length: int, alignment: int) -> int:
    return (alignment - (length % alignment)) % alignment


def _event_pivot(vertices: np.ndarray) -> np.ndarray:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return np.array(
        (
            0.5 * (lower[0] + upper[0]),
            0.5 * (lower[1] + upper[1]),
            lower[2],
        ),
        dtype=np.float64,
    )


def _load_texture_image(
    path: str | None,
    image_cache: dict[str, object],
    *,
    max_size: int = GLB_EMBEDDED_TEXTURE_MAX_SIZE,
):
    if not path:
        return None
    texture_path = Path(path)
    if texture_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".exr"}:
        return None
    max_size = max(int(max_size), 1)
    cache_key = f"{texture_path.resolve()}@{max_size}"
    if cache_key in image_cache:
        return image_cache[cache_key]
    try:
        from PIL import Image

        if texture_path.suffix.lower() == ".exr":
            image = _convert_exr_to_image(texture_path, max_size=max_size)
        else:
            image = Image.open(texture_path).convert("RGBA")
        if image is None:
            return None
        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS,
        )
        image_cache[cache_key] = image.copy()
        return image_cache[cache_key]
    except (OSError, ImportError):
        return None


def _load_metallic_roughness_texture(
    path: str | None,
    image_cache: dict[str, object],
    *,
    max_size: int = GLB_EMBEDDED_TEXTURE_MAX_SIZE,
):
    roughness = _load_texture_image(path, image_cache, max_size=max_size)
    if roughness is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    roughness_channel = roughness.convert("L")
    zero = Image.new("L", roughness_channel.size, 0)
    one = Image.new("L", roughness_channel.size, 255)
    return Image.merge("RGBA", (one, roughness_channel, zero, one))


def _load_displacement_image(
    path: str | None,
    image_cache: dict[str, object],
    *,
    max_size: int = GLB_EMBEDDED_TEXTURE_MAX_SIZE,
):
    """Load a height map without saturating 16-bit displacement masters."""

    if not path:
        return None
    texture_path = Path(path)
    if texture_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".exr"}:
        return None
    max_size = max(int(max_size), 1)
    cache_key = f"{texture_path.resolve()}@displacement@{max_size}"
    if cache_key in image_cache:
        return image_cache[cache_key]
    try:
        from PIL import Image

        if texture_path.suffix.lower() == ".exr":
            source = _convert_exr_to_image(texture_path, max_size=max_size)
            if source is None:
                return None
            pixels = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
        else:
            source = Image.open(texture_path)
            pixels = np.asarray(source, dtype=np.float32)
            if pixels.ndim == 3:
                pixels = pixels[:, :, 0]
            maximum = 65_535.0 if pixels.size and float(pixels.max()) > 255.0 else 255.0
            pixels = np.clip(pixels / maximum, 0.0, 1.0)
        image = Image.fromarray(pixels, mode="F")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image_cache[cache_key] = image.copy()
        return image_cache[cache_key]
    except (OSError, ImportError, ValueError):
        return None


def _convert_exr_to_image(
    texture_path: Path,
    *,
    max_size: int = GLB_EMBEDDED_TEXTURE_MAX_SIZE,
):
    try:
        from PIL import Image
    except ImportError:
        return None

    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        command = [
            "convert",
            str(texture_path),
            "-resize",
            f"{max_size}x{max_size}>",
            temp_file.name,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return Image.open(temp_file.name).convert("RGBA").copy()


def _write_obj_mtl(
    event_mtl_path: Path,
    cave_geometry: CaveGeometry,
    event_meshes: list,
    obj_dir: Path,
) -> None:
    """Write cave and event materials without reapplying baked displacement."""

    cave_maps = {
        "diffuse": cave_geometry.config.cave_diffuse_texture,
        "normal": cave_geometry.config.cave_normal_texture,
        "roughness": cave_geometry.config.cave_roughness_texture,
    }
    lines: list[str] = [
        "newmtl cave_wall_material",
        "Ka 0.05 0.05 0.05",
        "Kd 0.36 0.35 0.31",
        "Ks 0.02 0.02 0.02",
        "Ns 8.0",
        "Pm 0.0",
        "Pr 0.92",
    ]
    if cave_maps["diffuse"]:
        lines.append(f"map_Kd {_relative_path(cave_maps['diffuse'], obj_dir)}")
    if cave_maps["normal"]:
        normal_reference = _relative_path(cave_maps["normal"], obj_dir)
        lines.append(
            f"map_Bump -bm {cave_geometry.config.cave_normal_scale:.6g} "
            f"{normal_reference}"
        )
        lines.append(f"norm {normal_reference}")
    if cave_maps["roughness"]:
        lines.append(f"map_Pr {_relative_path(cave_maps['roughness'], obj_dir)}")
    lines.extend(
        [
            "# The configured displacement map is baked into cave vertex positions.",
            "",
        ]
    )

    written: set[str] = set()
    for event_mesh in event_meshes:
        material_name = _event_material_name(event_mesh)
        if material_name in written:
            continue
        written.add(material_name)
        material_maps = dict(event_mesh.material_maps)
        red, green, blue, _alpha = _event_fallback_color(event_mesh)
        lines.extend(
            [
                f"newmtl {material_name}",
                f"Ka {0.2 * red:.6g} {0.2 * green:.6g} {0.2 * blue:.6g}",
                f"Kd {red:.6g} {green:.6g} {blue:.6g}",
                "Ks 0.04 0.04 0.04",
                "Ns 18.0",
                "Pm 0.0",
                "Pr 0.88",
            ]
        )
        if "diffuse" in material_maps:
            lines.append(f"map_Kd {_relative_path(material_maps['diffuse'], obj_dir)}")
        if "normal" in material_maps:
            lines.append(f"map_Bump {_relative_path(material_maps['normal'], obj_dir)}")
        if "roughness" in material_maps:
            lines.append(f"map_Pr {_relative_path(material_maps['roughness'], obj_dir)}")
        if "displacement" in material_maps:
            lines.append(f"disp {_relative_path(material_maps['displacement'], obj_dir)}")
        lines.append("")
    event_mtl_path.write_text("\n".join(lines), encoding="utf-8")


def _event_material_name(event_mesh) -> str:
    suffix = event_mesh.source_shape_type or event_mesh.material_hint or event_mesh.kind
    return f"event_{event_mesh.event_id:04d}_{_safe_material_token(suffix)}"


def _event_shared_material_name(event_mesh) -> str:
    material_maps = dict(event_mesh.material_maps)
    diffuse = Path(material_maps.get("diffuse", "")).stem
    suffix = "_".join(
        part
        for part in (
            event_mesh.source_shape_type,
            event_mesh.material_hint,
            diffuse,
        )
        if part
    )
    return f"rocky_{_safe_material_token(suffix)}"


def _safe_material_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "rock"


def _relative_path(path: str, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path).resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return Path(path).as_posix()
