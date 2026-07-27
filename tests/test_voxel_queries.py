"""Tests for cave-density sampling and final-surface contact queries."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stages.geometry_types import VoxelGrid


class VoxelQueryTests(unittest.TestCase):
    def test_raycast_finds_inside_to_solid_crossing_and_inward_normal(self) -> None:
        voxel_size = 0.2
        coordinates = np.linspace(-2.0, 2.0, 21)
        x_coord, y_coord, z_coord = np.meshgrid(
            coordinates,
            coordinates,
            coordinates,
            indexing="ij",
        )
        density = 1.0 - np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
        grid = VoxelGrid(
            origin=(-2.0, -2.0, -2.0),
            voxel_size=voxel_size,
            density=density.astype(np.float32),
            iso_level=0.0,
        )

        hit = grid.raycast_isosurface(
            origin=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, -1.0),
            max_distance=1.8,
        )

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit.position[2], -1.0, delta=0.02)
        self.assertAlmostEqual(grid.sample_density(hit.position), 0.0, delta=1e-3)
        self.assertGreater(hit.normal[2], 0.98)
        self.assertAlmostEqual(hit.distance, 1.0, delta=0.02)

    def test_outside_samples_are_solid(self) -> None:
        grid = VoxelGrid(
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            density=np.ones((2, 2, 2), dtype=np.float32),
            iso_level=0.0,
        )

        self.assertFalse(grid.contains((-1.0, 0.0, 0.0)))
        self.assertLess(grid.sample_density((-1.0, 0.0, 0.0)), grid.iso_level)


if __name__ == "__main__":
    unittest.main()
