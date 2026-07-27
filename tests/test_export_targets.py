"""Target-adapter package tests using a tiny canonical cave."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exporters import export_target_asset
from stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from world import ExportConfig


class TargetExporterTests(unittest.TestCase):
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
            sdf = (package / "model.sdf").read_text(encoding="utf-8")
            self.assertIn("model://tube/meshes/tube.obj", sdf)
            self.assertIn("<static>true</static>", sdf)

    def test_omniverse_package_declares_usd_units_and_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_target_asset(
                self._geometry(),
                ExportConfig(target="omniverse", file_format="usd"),
                temp_dir,
                asset_name="tube",
            )

            usda = result.primary_asset.read_text(encoding="utf-8")
            self.assertIn("metersPerUnit = 1", usda)
            self.assertIn('upAxis = "Z"', usda)
            self.assertIn('def Mesh "CaveWall"', usda)

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
