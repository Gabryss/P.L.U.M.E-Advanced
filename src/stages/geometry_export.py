"""Mesh export helpers for Stage D geometry review."""

from __future__ import annotations

import os
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial, SimpleMaterial
from trimesh.visual.texture import TextureVisuals

from stages.geometry_types import CaveGeometry

GLB_EMBEDDED_TEXTURE_MAX_SIZE = 1024
GLB_MAX_PRIMITIVE_VERTICES = 65_000
GLB_CAVE_TEXTURE_SCALE_METERS = 8.0


def export_geometry_obj(cave_geometry: CaveGeometry, output_path: str | Path) -> Path:
    """Write the assembled Stage-D mesh as a Wavefront OBJ file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    textured_event_meshes = [
        event_mesh
        for event_mesh in cave_geometry.event_meshes
        if event_mesh.material_maps and event_mesh.face_uvs
    ]
    mtl_path = output.with_suffix(".mtl")
    if textured_event_meshes:
        _write_event_mtl(mtl_path, textured_event_meshes, output.parent)

    with output.open("w", encoding="utf-8") as handle:
        handle.write("# PLUME-Advanced Stage D geometry export\n")
        if textured_event_meshes:
            handle.write(f"mtllib {mtl_path.name}\n")
        for key, value in cave_geometry.summary().items():
            handle.write(f"# {key}={value:.3f}\n")
        mesh = trimesh.Trimesh(
            vertices=cave_geometry.assembled_vertices,
            faces=cave_geometry.assembled_faces,
            process=False,
        )
        handle.write(f"# trimesh_is_watertight={float(mesh.is_watertight):.3f}\n")
        handle.write(f"# trimesh_euler_number={float(mesh.euler_number):.3f}\n")
        handle.write("o cave_wall\n")
        for vertex in cave_geometry.assembled_vertices:
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in cave_geometry.assembled_faces:
            a, b, c = (index + 1 for index in face)
            handle.write(f"f {a} {b} {c}\n")
        vertex_offset = len(cave_geometry.assembled_vertices)
        uv_vertex_offset = 0
        for event_mesh in cave_geometry.event_meshes:
            handle.write(f"\no event_{event_mesh.event_id:04d}_{event_mesh.kind}\n")
            handle.write(f"# material_hint={event_mesh.material_hint}\n")
            handle.write(f"# source_generator={event_mesh.source_generator}\n")
            if event_mesh.source_shape_type:
                handle.write(f"# source_shape_type={event_mesh.source_shape_type}\n")
            material_name = _event_material_name(event_mesh)
            if event_mesh.material_maps:
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


def export_geometry_glb(cave_geometry: CaveGeometry, output_path: str | Path) -> Path:
    """Write a drag-and-drop GLB scene with separately editable event nodes."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    builder = _StrictGlbBuilder()
    material_cache: dict[tuple[tuple[str, str], str], object] = {}
    image_cache: dict[str, object] = {}
    cave_material = _cave_strict_glb_material(
        cave_geometry,
        builder=builder,
        image_cache=image_cache,
    )
    _add_cave_wall_to_strict_glb(builder, cave_geometry, cave_material)

    for event_mesh in cave_geometry.event_meshes:
        geometry = _event_mesh_to_glb_payload(
            event_mesh,
            builder=builder,
            material_cache=material_cache,
            image_cache=image_cache,
        )
        node_name = f"event_{event_mesh.event_id:04d}_{event_mesh.kind}"
        builder.mesh_node(
            name=node_name,
            positions=geometry["positions"],
            faces=geometry["faces"],
            material_index=geometry["material_index"],
            texcoords=geometry["texcoords"],
            translation=geometry["translation"],
            extras={
                "kind": event_mesh.kind,
                "material_hint": event_mesh.material_hint,
                "source_generator": event_mesh.source_generator,
                "source_shape_type": event_mesh.source_shape_type,
                "displacement_texture": dict(event_mesh.material_maps).get("displacement", ""),
            },
        )

    output.write_bytes(builder.to_glb())
    _write_geometry_manifest(cave_geometry, output.with_suffix(".manifest.json"))
    return output


def _add_cave_wall_to_strict_glb(
    builder,
    cave_geometry: CaveGeometry,
    cave_material: int,
) -> None:
    if cave_geometry.chunk_meshes:
        primitives: list[dict[str, np.ndarray | int]] = []
        for chunk_mesh in cave_geometry.chunk_meshes:
            if not chunk_mesh.vertices or not chunk_mesh.faces:
                continue
            primitives.append(
                _cave_primitive_payload(
                    vertices=np.array(chunk_mesh.vertices, dtype=np.float32),
                    faces=np.array(chunk_mesh.faces, dtype=np.uint32),
                    material_index=cave_material,
                )
            )
        if primitives:
            builder.multi_primitive_mesh_node(
                name="cave_wall",
                primitives=primitives,
                extras={
                    "source": "stage_d_chunks",
                    "chunk_count": len(primitives),
                    "displacement_texture": cave_geometry.config.cave_displacement_texture,
                },
            )
        return

    if not cave_geometry.assembled_vertices or not cave_geometry.assembled_faces:
        return
    vertices = np.array(cave_geometry.assembled_vertices, dtype=np.float32)
    faces = tuple(cave_geometry.assembled_faces)
    if len(vertices) <= GLB_MAX_PRIMITIVE_VERTICES:
        builder.mesh_node(
            name="cave_wall",
            positions=vertices,
            faces=np.array(faces, dtype=np.uint32),
            material_index=cave_material,
        )
        return
    builder.multi_primitive_mesh_node(
        name="cave_wall",
        primitives=[
            _cave_primitive_payload(
                vertices=part_vertices.astype(np.float32),
                faces=part_faces.astype(np.uint32),
                material_index=cave_material,
            )
            for part_vertices, part_faces in _split_mesh_faces(vertices, faces)
        ],
        extras={
            "source": "stage_d_assembled_split",
            "displacement_texture": cave_geometry.config.cave_displacement_texture,
        },
    )


def _cave_primitive_payload(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    material_index: int,
) -> dict[str, np.ndarray | int]:
    expanded_vertices: list[tuple[float, float, float]] = []
    expanded_uvs: list[tuple[float, float]] = []
    expanded_faces: list[tuple[int, int, int]] = []
    for face in faces:
        triangle = vertices[np.asarray(face, dtype=np.int64)]
        face_uvs = _project_cave_triangle_uvs(triangle)
        start = len(expanded_vertices)
        for vertex, uv in zip(triangle, face_uvs, strict=True):
            expanded_vertices.append(tuple(float(value) for value in vertex))
            expanded_uvs.append(uv)
        expanded_faces.append((start, start + 1, start + 2))
    return {
        "positions": np.array(expanded_vertices, dtype=np.float32),
        "faces": np.array(expanded_faces, dtype=np.uint32),
        "texcoords": np.array(expanded_uvs, dtype=np.float32),
        "material_index": material_index,
    }


def _project_cave_triangle_uvs(triangle: np.ndarray) -> list[tuple[float, float]]:
    edge_a = triangle[1] - triangle[0]
    edge_b = triangle[2] - triangle[0]
    normal = np.cross(edge_a, edge_b)
    normal_abs = np.abs(normal)
    axis = int(np.argmax(normal_abs))
    scale = max(GLB_CAVE_TEXTURE_SCALE_METERS, 1e-6)
    uvs: list[tuple[float, float]] = []
    for vertex in triangle:
        if axis == 0:
            u, v = vertex[1] / scale, vertex[2] / scale
        elif axis == 1:
            u, v = vertex[0] / scale, vertex[2] / scale
        else:
            u, v = vertex[0] / scale, vertex[1] / scale
        uvs.append((float(u), float(v)))
    return uvs


def _cave_strict_glb_material(cave_geometry: CaveGeometry, *, builder, image_cache: dict) -> int:
    diffuse_image = _load_texture_image(cave_geometry.config.cave_diffuse_texture, image_cache)
    normal_image = _load_texture_image(cave_geometry.config.cave_normal_texture, image_cache)
    roughness_image = _load_metallic_roughness_texture(
        cave_geometry.config.cave_roughness_texture,
        image_cache,
    )
    return builder.material(
        name="cave_wall_material",
        base_color_factor=(0.36, 0.35, 0.31, 1.0),
        base_color_texture=diffuse_image,
        normal_texture=normal_image,
        metallic_roughness_texture=roughness_image,
        metallic_factor=0.0,
        roughness_factor=0.92,
        double_sided=True,
    )


def _write_geometry_manifest(cave_geometry: CaveGeometry, output_path: Path) -> Path:
    payload = {
        "schema": "plume.geometry_manifest.v1",
        "cave": {
            "node": "cave_wall",
            "primitive_count": len(cave_geometry.chunk_meshes) if cave_geometry.chunk_meshes else 1,
            "material": {
                "diffuse": cave_geometry.config.cave_diffuse_texture,
                "normal": cave_geometry.config.cave_normal_texture,
                "roughness": cave_geometry.config.cave_roughness_texture,
                "displacement": cave_geometry.config.cave_displacement_texture,
                "uv_projection": "face_dominant_axis_planar",
                "uv_scale_m": GLB_CAVE_TEXTURE_SCALE_METERS,
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
                "vertex_count": event_mesh.vertex_count,
                "face_count": event_mesh.face_count,
                "material_maps": dict(event_mesh.material_maps),
            }
            for event_mesh in cave_geometry.event_meshes
        ],
        "summary": cave_geometry.summary(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _event_mesh_to_glb_payload(event_mesh, *, builder, material_cache: dict, image_cache: dict) -> dict[str, object]:
    vertices = np.array(event_mesh.vertices, dtype=np.float32)
    pivot = _event_pivot(vertices)
    local_vertices = vertices - pivot
    material_index = _event_strict_glb_material(
        event_mesh,
        builder=builder,
        material_cache=material_cache,
        image_cache=image_cache,
    )

    if event_mesh.face_uvs and len(event_mesh.face_uvs) == len(event_mesh.faces):
        expanded_vertices: list[tuple[float, float, float]] = []
        expanded_uvs: list[tuple[float, float]] = []
        expanded_faces: list[tuple[int, int, int]] = []
        for face, face_uvs in zip(event_mesh.faces, event_mesh.face_uvs, strict=True):
            start = len(expanded_vertices)
            for vertex_index, uv in zip(face, face_uvs, strict=True):
                expanded_vertices.append(tuple(float(value) for value in local_vertices[vertex_index]))
                expanded_uvs.append((float(uv[0]), float(1.0 - uv[1])))
            expanded_faces.append((start, start + 1, start + 2))
        return {
            "positions": np.array(expanded_vertices, dtype=np.float32),
            "texcoords": np.array(expanded_uvs, dtype=np.float32),
            "faces": np.array(expanded_faces, dtype=np.uint32),
            "material_index": material_index,
            "translation": tuple(float(value) for value in pivot),
        }

    return {
        "positions": local_vertices.astype(np.float32),
        "texcoords": None,
        "faces": np.array(event_mesh.faces, dtype=np.uint32),
        "material_index": material_index,
        "translation": tuple(float(value) for value in pivot),
    }


def _event_strict_glb_material(event_mesh, *, builder, material_cache: dict, image_cache: dict) -> int:
    cache_key = (tuple(event_mesh.material_maps), event_mesh.source_shape_type)
    if cache_key in material_cache:
        return material_cache[cache_key]

    material_maps = dict(event_mesh.material_maps)
    diffuse_image = _load_texture_image(material_maps.get("diffuse"), image_cache)
    normal_image = _load_texture_image(material_maps.get("normal"), image_cache)
    roughness_image = _load_metallic_roughness_texture(
        material_maps.get("roughness"),
        image_cache,
    )
    material_index = builder.material(
        name=_event_shared_material_name(event_mesh),
        base_color_texture=diffuse_image,
        normal_texture=normal_image,
        metallic_roughness_texture=roughness_image,
        metallic_factor=0.0,
        roughness_factor=0.86,
        double_sided=True,
    )
    material_cache[cache_key] = material_index
    return material_index


def _add_cave_wall_to_glb_scene(scene: trimesh.Scene, cave_geometry: CaveGeometry) -> None:
    cave_material = TextureVisuals(
        material=SimpleMaterial(
            diffuse=(92, 88, 80, 255),
            glossiness=0.08,
        )
    )
    if cave_geometry.chunk_meshes:
        for chunk_mesh in cave_geometry.chunk_meshes:
            if not chunk_mesh.vertices or not chunk_mesh.faces:
                continue
            mesh = trimesh.Trimesh(
                vertices=np.array(chunk_mesh.vertices, dtype=np.float64),
                faces=np.array(chunk_mesh.faces, dtype=np.int64),
                process=False,
                visual=cave_material,
            )
            name = f"cave_wall_chunk_{chunk_mesh.chunk_id:03d}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
        return

    if not cave_geometry.assembled_vertices or not cave_geometry.assembled_faces:
        return
    vertices = np.array(cave_geometry.assembled_vertices, dtype=np.float64)
    faces = tuple(cave_geometry.assembled_faces)
    if len(vertices) <= GLB_MAX_PRIMITIVE_VERTICES:
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.array(faces, dtype=np.int64),
            process=False,
            visual=cave_material,
        )
        scene.add_geometry(mesh, node_name="cave_wall", geom_name="cave_wall")
        return

    for index, (part_vertices, part_faces) in enumerate(_split_mesh_faces(vertices, faces)):
        mesh = trimesh.Trimesh(
            vertices=part_vertices,
            faces=part_faces,
            process=False,
            visual=cave_material,
        )
        name = f"cave_wall_part_{index:03d}"
        scene.add_geometry(mesh, node_name=name, geom_name=name)


def _split_mesh_faces(vertices: np.ndarray, faces: tuple[tuple[int, int, int], ...]):
    current_faces: list[tuple[int, int, int]] = []
    current_indices: dict[int, int] = {}
    for face in faces:
        missing = [vertex_index for vertex_index in face if vertex_index not in current_indices]
        if current_faces and len(current_indices) + len(missing) > GLB_MAX_PRIMITIVE_VERTICES:
            yield _remapped_mesh(vertices, current_faces, current_indices)
            current_faces = []
            current_indices = {}
        remapped_face: list[int] = []
        for vertex_index in face:
            if vertex_index not in current_indices:
                current_indices[vertex_index] = len(current_indices)
            remapped_face.append(current_indices[vertex_index])
        current_faces.append(tuple(remapped_face))
    if current_faces:
        yield _remapped_mesh(vertices, current_faces, current_indices)


def _remapped_mesh(
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
    index_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ordered_source_indices = sorted(index_map, key=index_map.__getitem__)
    return (
        vertices[np.array(ordered_source_indices, dtype=np.int64)],
        np.array(faces, dtype=np.int64),
    )


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
            material["normalTexture"] = {"index": self._texture(normal_texture)}
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
            attributes["TEXCOORD_0"] = self._accessor(
                texcoords,
                component_type=5126,
                accessor_type="VEC2",
                target=34962,
                minimum=texcoords.min(axis=0).tolist(),
                maximum=texcoords.max(axis=0).tolist(),
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
        self._nodes[0].setdefault("children", []).append(node_index)
        return node_index

    def multi_primitive_mesh_node(
        self,
        *,
        name: str,
        primitives: list[dict[str, object]],
        translation: tuple[float, float, float] | None = None,
        extras: dict[str, object] | None = None,
    ) -> int:
        gltf_primitives: list[dict[str, object]] = []
        for primitive in primitives:
            positions = np.asarray(primitive["positions"], dtype=np.float32)
            faces = np.asarray(primitive["faces"], dtype=np.uint32)
            if positions.size == 0 or faces.size == 0:
                continue
            gltf_primitives.append(
                self._primitive(
                    positions=positions,
                    faces=faces,
                    material_index=int(primitive["material_index"]),
                    texcoords=primitive.get("texcoords"),
                )
            )
        if not gltf_primitives:
            raise ValueError(f"Cannot export empty mesh node {name!r}")

        mesh_index = len(self._meshes)
        self._meshes.append({"name": name, "primitives": gltf_primitives})
        node: dict[str, object] = {"name": name, "mesh": mesh_index}
        if translation is not None:
            node["translation"] = [float(value) for value in translation]
        if extras:
            node["extras"] = extras
        node_index = len(self._nodes)
        self._nodes.append(node)
        self._nodes[0].setdefault("children", []).append(node_index)
        return node_index

    def _primitive(
        self,
        *,
        positions: np.ndarray,
        faces: np.ndarray,
        material_index: int,
        texcoords: np.ndarray | None = None,
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
            "asset": {"version": "2.0", "generator": "PLUME-Advanced strict GLB exporter"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
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


def _event_mesh_to_glb_geometry(event_mesh, *, material_cache: dict, image_cache: dict):
    vertices = np.array(event_mesh.vertices, dtype=np.float64)
    faces = np.array(event_mesh.faces, dtype=np.int64)
    pivot = _event_pivot(vertices)
    local_vertices = vertices - pivot
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = pivot
    material = _event_glb_material(
        event_mesh,
        material_cache=material_cache,
        image_cache=image_cache,
    )

    if event_mesh.face_uvs and len(event_mesh.face_uvs) == len(event_mesh.faces):
        expanded_vertices: list[tuple[float, float, float]] = []
        expanded_uvs: list[tuple[float, float]] = []
        expanded_faces: list[tuple[int, int, int]] = []
        for face, face_uvs in zip(event_mesh.faces, event_mesh.face_uvs, strict=True):
            start = len(expanded_vertices)
            for vertex_index, uv in zip(face, face_uvs, strict=True):
                expanded_vertices.append(tuple(float(value) for value in local_vertices[vertex_index]))
                expanded_uvs.append((float(uv[0]), float(1.0 - uv[1])))
            expanded_faces.append((start, start + 1, start + 2))
        visual = TextureVisuals(
            uv=np.array(expanded_uvs, dtype=np.float64),
            material=material,
        )
        mesh = trimesh.Trimesh(
            vertices=np.array(expanded_vertices, dtype=np.float64),
            faces=np.array(expanded_faces, dtype=np.int64),
            process=False,
            visual=visual,
        )
    else:
        mesh = trimesh.Trimesh(
            vertices=local_vertices,
            faces=faces,
            process=False,
            visual=TextureVisuals(material=material),
        )
    return mesh, transform


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


def _event_glb_material(event_mesh, *, material_cache: dict, image_cache: dict):
    cache_key = (tuple(event_mesh.material_maps), event_mesh.source_shape_type)
    if cache_key in material_cache:
        return material_cache[cache_key]

    material_maps = dict(event_mesh.material_maps)
    diffuse_image = _load_texture_image(material_maps.get("diffuse"), image_cache)
    normal_image = _load_texture_image(material_maps.get("normal"), image_cache)
    roughness_image = _load_metallic_roughness_texture(
        material_maps.get("roughness"),
        image_cache,
    )
    if diffuse_image is not None:
        material = PBRMaterial(
            name=_event_shared_material_name(event_mesh),
            baseColorTexture=diffuse_image,
            normalTexture=normal_image,
            metallicRoughnessTexture=roughness_image,
            metallicFactor=0.0,
            roughnessFactor=0.86,
            doubleSided=True,
        )
    else:
        material = SimpleMaterial(
            diffuse=(105, 100, 91, 255),
            glossiness=0.08,
        )
    material_cache[cache_key] = material
    return material


def _load_texture_image(path: str | None, image_cache: dict[str, object]):
    if not path:
        return None
    texture_path = Path(path)
    if texture_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".exr"}:
        return None
    cache_key = str(texture_path.resolve())
    if cache_key in image_cache:
        return image_cache[cache_key]
    try:
        from PIL import Image

        if texture_path.suffix.lower() == ".exr":
            image = _convert_exr_to_image(texture_path)
        else:
            image = Image.open(texture_path).convert("RGBA")
        if image is None:
            return None
        image.thumbnail(
            (GLB_EMBEDDED_TEXTURE_MAX_SIZE, GLB_EMBEDDED_TEXTURE_MAX_SIZE),
            Image.Resampling.LANCZOS,
        )
        image_cache[cache_key] = image.copy()
        return image_cache[cache_key]
    except (OSError, ImportError):
        return None


def _load_metallic_roughness_texture(path: str | None, image_cache: dict[str, object]):
    roughness = _load_texture_image(path, image_cache)
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


def _convert_exr_to_image(texture_path: Path):
    try:
        from PIL import Image
    except ImportError:
        return None

    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        command = [
            "convert",
            str(texture_path),
            "-resize",
            f"{GLB_EMBEDDED_TEXTURE_MAX_SIZE}x{GLB_EMBEDDED_TEXTURE_MAX_SIZE}>",
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


def _write_event_mtl(event_mtl_path: Path, event_meshes: list, obj_dir: Path) -> None:
    lines: list[str] = []
    written: set[str] = set()
    for event_mesh in event_meshes:
        material_name = _event_material_name(event_mesh)
        if material_name in written:
            continue
        written.add(material_name)
        material_maps = dict(event_mesh.material_maps)
        lines.extend(
            [
                f"newmtl {material_name}",
                "Ka 0.15 0.15 0.15",
                "Kd 0.82 0.82 0.82",
                "Ks 0.04 0.04 0.04",
                "Ns 18.0",
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
