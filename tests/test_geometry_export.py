"""Tests for Stage-D scene export helpers."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

Image = pytest.importorskip("PIL.Image")
trimesh = pytest.importorskip("trimesh")

from stages.events import GeologicalEventMesh
from stages.geometry_export import export_geometry_glb
from stages.geometry_types import CaveGeometry, GeometryChunkMesh, GeometryConfig, VoxelGrid


class GeometryExportTests(unittest.TestCase):
    def test_glb_export_keeps_event_mesh_as_separate_textured_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            diffuse_path = temp_path / "rock_diffuse.png"
            Image.new("RGB", (4, 4), (92, 84, 72)).save(diffuse_path)
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
                    cave_normal_texture="",
                    cave_roughness_texture="",
                    cave_displacement_texture="",
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
                assembled_vertices=(),
                assembled_faces=(),
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

    @staticmethod
    def _read_glb_json(path: Path) -> dict:
        data = path.read_bytes()
        json_size = struct.unpack_from("<I", data, 12)[0]
        return json.loads(data[20 : 20 + json_size].rstrip(b" "))


if __name__ == "__main__":
    unittest.main()
