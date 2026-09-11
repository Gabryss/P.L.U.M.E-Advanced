"""Integration tests for the topology-aware cave-floor atlas."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.stages.floor_map import (
    FloorAtlas,
    FloorCell,
    FloorMapConfig,
    FloorMapGenerator,
    export_floor_atlas,
)
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


class FloorMapTests(unittest.TestCase):
    def test_summary_preserves_vertical_overlap_and_geology_counts(self) -> None:
        base = FloorCell(
            cell_id=0,
            segment_id=10,
            sample_index=3,
            z_level=0,
            distance_along_m=5.0,
            lateral_offset_m=1.0,
            atlas_x_m=5.0,
            atlas_y_m=1.0,
            x=20.0,
            y=30.0,
            z=4.0,
            normal_x=0.0,
            normal_y=0.0,
            normal_z=1.0,
            clearance_m=3.0,
            tube_width_m=6.0,
            grounded=True,
            geology_class="sediment",
            event_influence=0.5,
            sediment_thickness_m=0.2,
            is_chamber=True,
        )
        upper = replace(
            base,
            cell_id=1,
            z_level=1,
            z=14.0,
            lateral_offset_m=-1.0,
            geology_class="breakdown",
            is_chamber=False,
            is_terminus=True,
        )
        atlas = FloorAtlas(
            config=FloorMapConfig(plan_resolution_m=2.0),
            cells=(base, upper),
            segment_band_offsets_m=((10, 0.0),),
        )

        summary = atlas.summary()
        self.assertEqual(summary["vertical_overlap_pixel_count"], 1.0)
        self.assertEqual(summary["sediment_cell_count"], 1.0)
        self.assertEqual(summary["breakdown_cell_count"], 1.0)
        self.assertEqual(summary["geologically_influenced_cell_count"], 2.0)
        self.assertEqual(summary["chamber_cell_count"], 1.0)
        self.assertEqual(summary["terminus_cell_count"], 1.0)
        self.assertEqual(
            [cell.lateral_offset_m for cell in atlas.sample_lookup()[(10, 3)]],
            [-1.0, 1.0],
        )

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
                self.assertEqual(
                    arrays["plan_occupancy"].shape,
                    arrays["plan_geology_class"].shape,
                )
                self.assertGreater(
                    int(np.count_nonzero(arrays["plan_occupancy"])),
                    0,
                )
                self.assertEqual(arrays["geology_class"].shape, (len(atlas.cells),))


if __name__ == "__main__":
    unittest.main()
