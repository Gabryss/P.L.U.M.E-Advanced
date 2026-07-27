"""Integration tests for the topology-aware cave-floor atlas."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.floor_map import FloorMapGenerator, export_floor_atlas
from stages.geometry import GeometryGenerator
from stages.host_field import HostFieldGenerator
from stages.network import CaveNetworkGenerator
from stages.section_field import SectionFieldGenerator


class FloorMapTests(unittest.TestCase):
    def test_floor_atlas_lifts_intrinsic_cells_to_generated_surface(self) -> None:
        config = load_project_config(ROOT / "config" / "project.toml")
        host = HostFieldGenerator(config.host_field).generate()
        network = CaveNetworkGenerator(config.network).generate(host)
        sections = SectionFieldGenerator(config.section_field).generate(network)
        base = GeometryGenerator(config.geometry).build_base_volume(network, sections)
        atlas = FloorMapGenerator(config.floor_map).generate(network, sections, base)

        self.assertGreater(len(atlas.cells), 0)
        self.assertEqual(
            len(atlas.cells),
            len({cell.cell_id for cell in atlas.cells}),
        )
        self.assertGreater(len(atlas.segment_band_offsets_m), 0)
        for cell in atlas.cells:
            self.assertTrue(cell.grounded)
            self.assertGreaterEqual(cell.clearance_m, config.floor_map.minimum_clearance_m)
            self.assertAlmostEqual(
                base.voxel_grid.sample_density(cell.position),
                base.voxel_grid.iso_level,
                delta=0.05,
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(cell.normal)),
                1.0,
                places=4,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path, json_path = export_floor_atlas(
                atlas,
                Path(temp_dir) / "cave_floor",
            )
            self.assertTrue(npz_path.is_file())
            self.assertTrue(json_path.is_file())
            with np.load(npz_path) as arrays:
                self.assertEqual(arrays["world_xyz_m"].shape, (len(atlas.cells), 3))
                self.assertEqual(arrays["atlas_xy_m"].shape, (len(atlas.cells), 2))
                self.assertEqual(
                    arrays["plan_occupancy"].shape,
                    arrays["plan_level_count"].shape,
                )
                self.assertGreater(
                    int(np.count_nonzero(arrays["plan_occupancy"])),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
