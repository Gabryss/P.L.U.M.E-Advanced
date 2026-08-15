"""Target-adapter package tests using a tiny canonical cave."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.exporters import export_target_asset
from plume_advanced.exporters import targets as target_module
from plume_advanced.stages.events import GeologicalEventMesh
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.world import ExportConfig


class TargetExporterTests(unittest.TestCase):
    def test_failed_staged_export_preserves_previous_complete_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "package"
            output.mkdir()
            marker = output / "previous-package.txt"
            marker.write_text("complete", encoding="utf-8")

            with patch.object(
                target_module,
                "_export_glb_or_obj",
                side_effect=RuntimeError("synthetic adapter failure"),
            ), self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
                export_target_asset(
                    self._geometry(),
                    ExportConfig(target="neutral", file_format="glb"),
                    output,
                    asset_name="tube",
                )

            self.assertEqual(tuple(output.iterdir()), (marker,))
            self.assertEqual(marker.read_text(encoding="utf-8"), "complete")
            self.assertFalse(tuple(output.parent.glob(".package.staging-*")))

    def test_successful_export_replaces_the_complete_previous_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "package"
            output.mkdir()
            stale = output / "stale.txt"
            stale.write_text("old", encoding="utf-8")

            result = export_target_asset(
                self._geometry(),
                ExportConfig(
                    target="neutral",
                    file_format="obj",
                    generate_collision=False,
                ),
                output,
                asset_name="tube",
            )

            self.assertTrue(result.primary_asset.is_file())
            self.assertFalse(stale.exists())
            self.assertFalse(tuple(output.parent.glob(".package.backup-*")))

    def test_exporter_rejects_invalid_direct_configs_before_writing_assets(self) -> None:
        cases = (
            (ExportConfig(target="unknown", file_format="glb"), "Unsupported export target"),
            (ExportConfig(target="all", file_format="glb"), "must be 'auto'"),
            (ExportConfig(target="neutral", file_format="usd"), "supports glb or obj"),
            (ExportConfig(target="gazebo", file_format="glb"), "must be 'obj'"),
            (ExportConfig(target="omniverse", file_format="glb"), "must be 'usd'"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (config, message) in enumerate(cases):
                output = Path(temp_dir) / str(index)
                output.mkdir()
                marker = output / "existing.txt"
                marker.write_text("preserve", encoding="utf-8")
                with self.subTest(target=config.target), self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    export_target_asset(
                        self._geometry(),
                        config,
                        output,
                        asset_name="tube",
                    )
                self.assertEqual(tuple(output.iterdir()), (marker,))
                self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

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
            self.assertTrue((Path(temp_dir) / "README_IMPORT_UE5.txt").is_file())

    def test_unity_package_declares_handedness_and_optional_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_target_asset(
                self._geometry(),
                ExportConfig(
                    target="unity",
                    file_format="glb",
                    generate_collision=False,
                ),
                temp_dir,
                asset_name="tube",
            )

            output = Path(temp_dir)
            self.assertEqual(result.primary_asset, output / "tube.glb")
            self.assertFalse((output / "tube_collision.obj").exists())
            descriptor = json.loads(
                (output / "tube.unity.json").read_text(encoding="utf-8")
            )
            conventions = descriptor["target_conventions"]
            self.assertEqual(conventions["application_coordinates"], "left-handed Y-up")
            self.assertEqual(conventions["application_length_unit"], "metre")
            self.assertEqual(conventions["recommended_import_uniform_scale"], 1.0)
            guide = (output / "README_IMPORT_UNITY.txt").read_text(encoding="utf-8")
            self.assertIn("Add collision in Unity", guide)

    def test_all_package_builds_every_application_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                target_module,
                "prepare_export_scene",
                wraps=target_module.prepare_export_scene,
            ) as prepare:
                result = export_target_asset(
                    self._geometry(),
                    ExportConfig(target="all", file_format="auto"),
                    temp_dir,
                    asset_name="tube",
                )

            self.assertEqual(prepare.call_count, 1)

            self.assertEqual(result.target, "all")
            manifest = json.loads(result.primary_asset.read_text(encoding="utf-8"))
            self.assertEqual(
                set(manifest["targets"]),
                {"blender", "ue5", "unity", "gazebo", "omniverse"},
            )
            self.assertTrue((Path(temp_dir) / "blender" / "tube.glb").is_file())
            self.assertTrue((Path(temp_dir) / "ue5" / "tube.glb").is_file())
            self.assertTrue((Path(temp_dir) / "unity" / "tube.glb").is_file())
            self.assertTrue(
                (Path(temp_dir) / "gazebo" / "tube" / "model.sdf").is_file()
            )
            self.assertTrue((Path(temp_dir) / "omniverse" / "tube.usd").is_file())
            recorded_files: set[Path] = {result.primary_asset}
            for target, record in manifest["targets"].items():
                primary = Path(temp_dir) / record["primary_asset"]
                files = [Path(temp_dir) / value for value in record["files"]]
                self.assertTrue(primary.is_file(), target)
                self.assertIn(primary, files, target)
                self.assertTrue(all(path.is_file() for path in files), target)
                self.assertEqual(len(files), len(set(files)), target)
                recorded_files.update(files)
            self.assertEqual(set(result.files), recorded_files)

    def test_gazebo_package_contains_relocatable_sdf_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_texture = Path(temp_dir) / "source_diffuse.png"
            Image.new("RGB", (4, 4), (90, 82, 72)).save(source_texture)
            geometry = self._geometry()
            geometry = replace(
                geometry,
                config=replace(
                    geometry.config,
                    cave_diffuse_texture=str(source_texture),
                ),
            )
            result = export_target_asset(
                geometry,
                ExportConfig(target="gazebo", file_format="obj"),
                temp_dir,
                asset_name="tube",
            )

            package = Path(temp_dir) / "tube"
            self.assertEqual(result.primary_asset, package / "model.sdf")
            self.assertTrue((package / "model.config").is_file())
            self.assertTrue((package / "meshes" / "tube_collision.obj").is_file())
            sdf = (package / "model.sdf").read_text(encoding="utf-8")
            self.assertIn('<sdf version="1.12">', sdf)
            self.assertIn("model://tube/meshes/tube.obj", sdf)
            self.assertIn("model://tube/meshes/tube_collision.obj", sdf)
            self.assertIn("<static>true</static>", sdf)
            self.assertTrue((Path(temp_dir) / "tube.world.sdf").is_file())
            descriptor = json.loads(
                (package / "plume_export.json").read_text(encoding="utf-8")
            )
            self.assertEqual(descriptor["gazebo_release"], "Jetty")
            self.assertEqual(descriptor["sdformat_major"], 16)
            copied_texture = package / "materials" / "textures" / source_texture.name
            self.assertTrue(copied_texture.is_file())
            material = (package / "meshes" / "tube.mtl").read_text(encoding="utf-8")
            self.assertIn("../materials/textures/source_diffuse.png", material)

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

    def test_collision_can_be_disabled_for_gazebo_and_omniverse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gazebo = export_target_asset(
                self._geometry(),
                ExportConfig(
                    target="gazebo",
                    file_format="obj",
                    generate_collision=False,
                ),
                root / "gazebo",
                asset_name="tube",
            )
            gazebo_sdf = gazebo.primary_asset.read_text(encoding="utf-8")
            self.assertFalse(
                (root / "gazebo" / "tube" / "meshes" / "tube_collision.obj").exists()
            )
            self.assertEqual(gazebo_sdf.count("model://tube/meshes/tube.obj"), 2)

            omniverse = export_target_asset(
                self._geometry(),
                ExportConfig(
                    target="omniverse",
                    file_format="usd",
                    generate_collision=False,
                ),
                root / "omniverse",
                asset_name="tube",
            )
            usda = omniverse.primary_asset.read_text(encoding="utf-8")
            self.assertNotIn('def Mesh "CaveCollision"', usda)
            self.assertNotIn("PhysicsCollisionAPI", usda)

    def test_asset_name_is_sanitized_and_cannot_escape_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "package"
            result = export_target_asset(
                self._geometry(),
                ExportConfig(
                    target="neutral",
                    file_format="obj",
                    generate_collision=False,
                ),
                output,
                asset_name="../../Tube A",
            )

            self.assertEqual(result.primary_asset, output / "Tube_A.obj")
            self.assertTrue(result.primary_asset.is_file())
            self.assertTrue(
                all(path.resolve().is_relative_to(output.resolve()) for path in result.files)
            )

            fallback = export_target_asset(
                self._geometry(),
                ExportConfig(
                    target="neutral",
                    file_format="obj",
                    generate_collision=False,
                ),
                output / "fallback",
                asset_name="../..",
            )
            self.assertEqual(fallback.primary_asset.name, "plume_cave.obj")

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
