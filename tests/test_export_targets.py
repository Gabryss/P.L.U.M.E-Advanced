"""Target-adapter package tests using a tiny canonical cave."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.exporters import export_target_asset
from plume_advanced.stages.events import GeologicalEventMesh
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.world import ExportConfig


class TargetExporterTests(unittest.TestCase):
    def test_neutral_package_describes_self_contained_portable_glb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_target_asset(
                self._geometry(),
                ExportConfig(target="neutral", file_format="glb"),
                temp_dir,
                asset_name="plume_cave_scene",
            )

            self.assertEqual(result.primary_asset.name, "plume_cave_scene.glb")
            self.assertTrue(
                (Path(temp_dir) / "plume_cave_scene_fallback.obj").is_file()
            )
            self.assertTrue(
                (Path(temp_dir) / "plume_cave_scene_fallback.mtl").is_file()
            )
            descriptor = json.loads(
                (Path(temp_dir) / "plume_cave_scene.neutral.json").read_text(
                    encoding="utf-8"
                )
            )
            conventions = descriptor["target_conventions"]
            self.assertEqual(conventions["application_length_unit"], "metre")
            self.assertEqual(
                conventions["application_coordinates"],
                "glTF right-handed Y-up",
            )
            self.assertIn("baked into vertex positions", conventions["note"])

    def test_blender_package_has_valid_glb_fallback_and_import_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_target_asset(
                self._geometry(),
                ExportConfig(target="blender", file_format="glb"),
                temp_dir,
                asset_name="tube",
            )

            output = Path(temp_dir)
            self.assertEqual(result.primary_asset, output / "tube.glb")
            self.assertTrue((output / "tube_fallback.obj").is_file())
            collision_path = output / "tube_collision.obj"
            self.assertTrue(collision_path.is_file())
            collision = trimesh.load_mesh(collision_path, process=False)
            self.assertGreater(len(collision.faces), 0)
            self.assertTrue(collision.is_winding_consistent)
            self.assertTrue((output / "tube_import_blender.py").is_file())
            instructions = (output / "README_IMPORT_BLENDER.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("Do not use File > Open", instructions)
            self.assertIn("File > Import > glTF 2.0", instructions)
            validation = json.loads(
                (output / "tube.blender_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(validation["valid"])
            self.assertGreater(validation["geometry_count"], 0)
            descriptor = json.loads(
                (output / "tube.blender.json").read_text(encoding="utf-8")
            )
            self.assertEqual(descriptor["target"], "blender")
            self.assertIn("not File > Open", descriptor["target_conventions"]["note"])

    def test_ue5_package_uses_standard_glb_and_centimetre_import_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_target_asset(
                self._geometry(),
                ExportConfig(target="ue5", file_format="glb"),
                temp_dir,
                asset_name="tube",
            )

            self.assertEqual(result.primary_asset.suffix, ".glb")
            descriptor = json.loads(
                (Path(temp_dir) / "tube.ue5.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                descriptor["target_conventions"]["application_units_per_asset_metre"],
                100.0,
            )
            self.assertEqual(
                descriptor["target_conventions"]["recommended_import_uniform_scale"],
                1.0,
            )
            self.assertEqual(
                descriptor["asset_coordinates"],
                "glTF right-handed Y-up metres",
            )

    def test_gazebo_package_contains_relocatable_sdf_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_target_asset(
                self._geometry(),
                ExportConfig(target="gazebo", file_format="obj"),
                temp_dir,
                asset_name="tube",
            )

            package = Path(temp_dir) / "tube"
            self.assertEqual(result.primary_asset, package / "model.sdf")
            self.assertTrue((package / "model.config").is_file())
            self.assertTrue((package / "meshes" / "tube_collision.obj").is_file())
            sdf = (package / "model.sdf").read_text(encoding="utf-8")
            self.assertIn("model://tube/meshes/tube.obj", sdf)
            self.assertIn("model://tube/meshes/tube_collision.obj", sdf)
            self.assertIn("<static>true</static>", sdf)

    def test_omniverse_package_declares_usd_units_and_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            texture_dir = Path(temp_dir) / "source_textures"
            texture_dir.mkdir()
            diffuse = texture_dir / "diffuse.png"
            normal = texture_dir / "normal.png"
            roughness = texture_dir / "roughness.png"
            Image.new("RGB", (4, 4), (90, 82, 72)).save(diffuse)
            Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
            Image.new("L", (4, 4), 220).save(roughness)
            geometry = self._geometry()
            geometry = replace(
                geometry,
                config=replace(
                    geometry.config,
                    cave_diffuse_texture=str(diffuse),
                    cave_normal_texture=str(normal),
                    cave_roughness_texture=str(roughness),
                ),
                event_meshes=(
                    GeologicalEventMesh(
                        event_id=4,
                        kind="rock",
                        material_hint="floor_debris",
                        vertices=(
                            (0.1, 0.1, 0.0),
                            (0.4, 0.1, 0.0),
                            (0.1, 0.4, 0.0),
                        ),
                        faces=((0, 1, 2),),
                        face_uvs=(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),),
                        material_maps=(
                            ("diffuse", str(diffuse)),
                            ("normal", str(normal)),
                            ("roughness", str(roughness)),
                        ),
                        source_generator="rocky",
                        source_shape_type="angular_boulder",
                    ),
                ),
            )
            result = export_target_asset(
                geometry,
                ExportConfig(target="omniverse", file_format="usd"),
                temp_dir,
                asset_name="tube",
            )

            usda = result.primary_asset.read_text(encoding="utf-8")
            self.assertIn("metersPerUnit = 1", usda)
            self.assertIn('upAxis = "Z"', usda)
            self.assertIn('def Mesh "CaveWall"', usda)
            self.assertIn(
                'def Mesh "CaveWall" (\n'
                '        prepend apiSchemas = ["MaterialBindingAPI"]\n'
                "    )\n"
                "    {",
                usda,
            )
            self.assertIn("uniform bool doubleSided = false", usda)
            self.assertIn(
                'def Mesh "CaveCollision" (\n'
                '        prepend apiSchemas = ["PhysicsCollisionAPI"]\n'
                "    )",
                usda,
            )
            self.assertIn('uniform token purpose = "guide"', usda)
            self.assertIn('token visibility = "invisible"', usda)
            self.assertIn("bool physics:collisionEnabled = true", usda)
            self.assertIn("texCoord2f[] primvars:st", usda)
            self.assertIn('info:id = "UsdPreviewSurface"', usda)
            self.assertIn("rel material:binding", usda)
            self.assertIn("inputs:normal.connect", usda)
            self.assertIn("float4 inputs:scale = (4, 4, 2, 1)", usda)
            self.assertIn("float4 inputs:bias = (-2, -2, -1, 0)", usda)
            self.assertIn('def Mesh "Event_0004_rock" (', usda)
            self.assertIn(
                'custom string plume:sourceGenerator = "rocky"',
                usda,
            )
            self.assertIn(
                'custom string plume:sourceShapeType = "angular_boulder"',
                usda,
            )
            self.assertIn('interpolation = "faceVarying"', usda)
            self.assertGreaterEqual(usda.count("rel material:binding"), 2)
            self.assertTrue((Path(temp_dir) / "tube_textures" / "cave_base_color.png").is_file())
            self.assertTrue((Path(temp_dir) / "tube_textures" / "cave_normal.png").is_file())
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "tube_textures"
                    / "cave_metallic_roughness.png"
                ).is_file()
            )

    @staticmethod
    def _geometry() -> CaveGeometry:
        return CaveGeometry(
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
            chunk_meshes=(),
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
        )


if __name__ == "__main__":
    unittest.main()
