"""Tests for Stage-D scene export helpers."""

import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

Image = pytest.importorskip("PIL.Image")
trimesh = pytest.importorskip("trimesh")

from plume_advanced.stages.events import GeologicalEventMesh
from plume_advanced.stages.geometry_export import (
    _bake_seam_consistent_displacement,
    _bake_vertex_displacement,
    _event_fallback_color,
    _load_displacement_image,
    _orient_faces_toward_cave_interior,
    _smooth_visual_surface,
    _xatlas_metric_uvs,
    export_geometry_glb,
    export_geometry_obj,
)
from plume_advanced.stages.geometry_types import (
    CaveGeometry,
    GeometryChunkMesh,
    GeometryConfig,
    SurfaceTextureFrame,
    VoxelGrid,
)


class GeometryExportTests(unittest.TestCase):
    def test_native_event_props_use_dark_nonwhite_fallback_materials(self) -> None:
        rock = GeologicalEventMesh(
            event_id=1,
            kind="rock",
            material_hint="floor_debris",
            vertices=(),
            faces=(),
        )
        boulder = replace(rock, kind="boulder")

        self.assertEqual(_event_fallback_color(rock), (0.22, 0.20, 0.17, 1.0))
        self.assertEqual(_event_fallback_color(boulder), (0.18, 0.17, 0.15, 1.0))

    def test_xatlas_uvs_are_seam_aware_and_metric(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
        vertices = np.array(
            [
                (4.0 * np.cos(angle), along, 4.0 * np.sin(angle))
                for along in (0.0, 4.0, 8.0)
                for angle in angles
            ],
            dtype=np.float64,
        )
        ring_size = len(angles)
        faces = []
        for ring in range(2):
            for index in range(ring_size):
                next_index = (index + 1) % ring_size
                lower = ring * ring_size + index
                lower_next = ring * ring_size + next_index
                upper = (ring + 1) * ring_size + index
                upper_next = (ring + 1) * ring_size + next_index
                faces.extend(
                    (
                        (lower, upper, upper_next),
                        (lower, upper_next, lower_next),
                    )
                )
        triangles = np.asarray(faces, dtype=np.uint32)
        normals = np.column_stack(
            (
                vertices[:, 0] / 4.0,
                np.zeros(len(vertices)),
                vertices[:, 2] / 4.0,
            )
        )

        mapping, atlas_faces, texcoords = _xatlas_metric_uvs(
            vertices,
            triangles,
            normals,
            scale_m=8.0,
            max_faces_per_batch=20,
        )

        self.assertEqual(atlas_faces.shape, triangles.shape)
        self.assertGreaterEqual(len(mapping), len(vertices))
        self.assertEqual(len(texcoords), len(mapping))
        self.assertTrue(np.isfinite(texcoords).all())
        positions = vertices[mapping]
        world_triangles = positions[atlas_faces]
        uv_triangles = texcoords[atlas_faces]
        world_edges = np.stack(
            (
                world_triangles[:, 1] - world_triangles[:, 0],
                world_triangles[:, 2] - world_triangles[:, 0],
            ),
            axis=2,
        )
        uv_edges = np.stack(
            (
                uv_triangles[:, 1] - uv_triangles[:, 0],
                uv_triangles[:, 2] - uv_triangles[:, 0],
            ),
            axis=2,
        )
        determinants = np.abs(np.linalg.det(uv_edges))
        self.assertTrue(np.all(determinants > 1e-10))
        singular_values = np.linalg.svd(
            world_edges @ np.linalg.inv(uv_edges),
            compute_uv=False,
        )
        metres_per_uv = np.sqrt(
            singular_values[:, 0] * singular_values[:, 1]
        )
        anisotropy = singular_values[:, 0] / singular_values[:, 1]
        self.assertAlmostEqual(float(np.median(metres_per_uv)), 8.0, delta=0.1)
        self.assertLess(float(np.quantile(anisotropy, 0.99)), 1.5)

    def test_xatlas_displacement_keeps_duplicate_seams_welded(self) -> None:
        vertices = np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        faces = np.array(
            ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
            dtype=np.uint32,
        )
        normals = np.array(
            (
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
            ),
            dtype=np.float64,
        )
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        mapping, _atlas_faces, texcoords = _xatlas_metric_uvs(
            vertices,
            faces,
            normals,
            scale_m=1.0,
        )
        image = Image.new("L", (2, 2))
        image.putdata((0, 255, 64, 192))

        displaced, metadata = _bake_seam_consistent_displacement(
            vertices,
            normals,
            mapping,
            texcoords,
            image,
            scale_m=0.2,
            midlevel=0.5,
        )

        self.assertTrue(metadata["baked"])
        expanded = displaced[mapping]
        for vertex_index in np.unique(mapping):
            copies = expanded[mapping == vertex_index]
            np.testing.assert_allclose(
                copies - copies[0],
                np.zeros_like(copies),
                atol=1e-12,
            )

    def test_visual_surface_smoothing_and_inward_orientation(self) -> None:
        vertices = np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 1.0),
                (2.0, 1.0, 0.0),
                (0.0, 2.0, 0.0),
                (1.0, 2.0, 0.0),
                (2.0, 2.0, 0.0),
            ),
            dtype=np.float64,
        )
        faces = np.array(
            (
                (0, 1, 4),
                (0, 4, 3),
                (1, 2, 5),
                (1, 5, 4),
                (3, 4, 7),
                (3, 7, 6),
                (4, 5, 8),
                (4, 8, 7),
            ),
            dtype=np.uint32,
        )
        smoothed = _smooth_visual_surface(vertices, faces, iterations=4)
        self.assertTrue(np.isfinite(smoothed).all())
        self.assertLess(smoothed[4, 2], vertices[4, 2])

        wall_vertices = np.array(
            ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        wall_faces = np.array(((0, 1, 2),), dtype=np.uint32)
        frame = SurfaceTextureFrame(
            segment_id=1,
            center=(0.0, 0.0, 0.0),
            tangent=(0.0, 1.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            binormal=(0.0, 0.0, 1.0),
            longitudinal_m=0.0,
        )
        oriented = _orient_faces_toward_cave_interior(
            wall_vertices,
            wall_faces,
            (frame,),
        )
        triangle = wall_vertices[oriented[0]]
        face_normal = np.cross(
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
        )
        self.assertLess(face_normal[0], 0.0)

    def test_displacement_is_baked_into_positions_with_bounded_metric_scale(
        self,
    ) -> None:
        vertices = np.zeros((4, 3), dtype=np.float64)
        normals = np.tile(np.array((1.0, 0.0, 0.0)), (4, 1))
        texcoords = np.array(
            ((0.0, 0.0), (0.999, 0.0), (0.0, 0.999), (0.999, 0.999)),
            dtype=np.float64,
        )
        image = Image.new("L", (2, 2))
        image.putdata((0, 255, 64, 192))

        displaced, metadata = _bake_vertex_displacement(
            vertices,
            normals,
            texcoords,
            image,
            scale_m=0.2,
            midlevel=0.5,
        )

        self.assertTrue(metadata["baked"])
        self.assertGreater(float(metadata["sample_standard_deviation_m"]), 0.0)
        self.assertGreaterEqual(float(metadata["minimum_offset_m"]), -0.2)
        self.assertLessEqual(float(metadata["maximum_offset_m"]), 0.2)
        self.assertTrue(np.allclose(displaced[:, 1:], 0.0))
        self.assertTrue(np.all(np.abs(displaced[:, 0]) <= 0.2 + 1e-12))

    def test_displacement_loader_preserves_sixteen_bit_height_variation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "height.png"
            pixels = np.array(
                ((0, 16_384), (32_768, 65_535)),
                dtype=np.uint16,
            )
            Image.fromarray(pixels).save(path)
            loaded = _load_displacement_image(
                str(path),
                {},
                max_size=2,
            )

            self.assertIsNotNone(loaded)
            values = np.asarray(loaded, dtype=np.float64)
            self.assertAlmostEqual(float(values.min()), 0.0, places=5)
            self.assertAlmostEqual(float(values.max()), 1.0, places=5)
            self.assertGreater(float(values.std()), 0.25)

    def test_glb_export_keeps_event_mesh_as_separate_textured_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            diffuse_path = temp_path / "rock_diffuse.png"
            Image.new("RGB", (4, 4), (92, 84, 72)).save(diffuse_path)
            cave_normal_path = temp_path / "cave_normal.png"
            Image.new("RGB", (4, 4), (128, 128, 255)).save(cave_normal_path)
            event_mesh = GeologicalEventMesh(
                event_id=1,
                kind="rock",
                material_hint="floor_debris",
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                faces=((0, 1, 2),),
                face_uvs=(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),),
                material_maps=(("diffuse", str(diffuse_path)),),
                source_generator="rocky",
                source_shape_type="angular_boulder",
            )
            cave_geometry = CaveGeometry(
                config=GeometryConfig(
                    cave_diffuse_texture="",
                    cave_normal_texture=str(cave_normal_path),
                    cave_roughness_texture="",
                    cave_displacement_texture="",
                    cave_smoothing_iterations=0,
                ),
                voxel_grid=VoxelGrid(
                    origin=(0.0, 0.0, 0.0),
                    voxel_size=1.0,
                    density=np.zeros((2, 2, 2), dtype=np.float32),
                    iso_level=0.0,
                ),
                chunk_meshes=(
                    GeometryChunkMesh(
                        chunk_id=0,
                        grid_bounds=(0, 1, 0, 1, 0, 1),
                        vertices=(
                            (0.0, 0.0, 0.0),
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                            (0.0, 0.0, 1.0),
                        ),
                        faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
                    ),
                ),
                assembled_vertices=(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                assembled_faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
                component_count=1,
                stamped_sample_count=0,
                stamped_segment_ids=(),
                event_meshes=(event_mesh,),
            )

            output_path = export_geometry_glb(cave_geometry, temp_path / "scene.glb")
            self.assertGreater(output_path.stat().st_size, 0)

            scene = trimesh.load(output_path, force="scene")
            self.assertIn("cave_wall", scene.graph.nodes_geometry)
            self.assertIn("event_0001_rock", scene.graph.nodes_geometry)

            document = self._read_glb_json(output_path)
            cave_node = next(node for node in document["nodes"] if node["name"] == "cave_wall")
            cave_mesh = document["meshes"][cave_node["mesh"]]
            self.assertEqual(len(cave_mesh["primitives"]), 1)
            attributes = cave_mesh["primitives"][0]["attributes"]
            self.assertIn("TEXCOORD_0", attributes)
            self.assertIn("NORMAL", attributes)
            self.assertIn("TANGENT", attributes)
            material = document["materials"][cave_mesh["primitives"][0]["material"]]
            self.assertAlmostEqual(material["normalTexture"]["scale"], 2.0)
            self.assertFalse(material["doubleSided"])
            position_accessor = document["accessors"][attributes["POSITION"]]
            self.assertEqual(position_accessor["min"], [0.0, 0.0, -1.0])
            self.assertEqual(position_accessor["max"], [1.0, 1.0, -0.0])

            manifest = json.loads(
                output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema"], "plume.geometry_manifest.v2")
            self.assertEqual(
                manifest["coordinates"]["asset"],
                "glTF right-handed Y-up metres",
            )
            self.assertEqual(manifest["cave"]["node"], "cave_wall")
            self.assertEqual(manifest["events"][0]["node"], "event_0001_rock")

            obj_path = export_geometry_obj(cave_geometry, temp_path / "scene.obj")
            obj_text = obj_path.read_text(encoding="utf-8")
            mtl_text = obj_path.with_suffix(".mtl").read_text(encoding="utf-8")
            self.assertIn("usemtl cave_wall_material", obj_text)
            self.assertIn("\nvt ", obj_text)
            self.assertIn("\nvn ", obj_text)
            self.assertRegex(obj_text, r"\nf \d+/\d+/\d+ ")
            self.assertIn("newmtl cave_wall_material", mtl_text)
            self.assertIn("map_Bump -bm 2", mtl_text)
            self.assertIn(
                "displacement map is baked into cave vertex positions",
                mtl_text,
            )

            broken_event = replace(
                event_mesh,
                material_maps=(("diffuse", str(temp_path / "missing.png")),),
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing.png"):
                export_geometry_glb(
                    replace(cave_geometry, event_meshes=(broken_event,)),
                    temp_path / "broken.glb",
                )

    @staticmethod
    def _read_glb_json(path: Path) -> dict:
        data = path.read_bytes()
        json_size = struct.unpack_from("<I", data, 12)[0]
        return json.loads(data[20 : 20 + json_size].rstrip(b" "))


if __name__ == "__main__":
    unittest.main()
