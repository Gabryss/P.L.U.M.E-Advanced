"""Smoke tests for the stage-A host field."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stages import HostFieldGenerator


class HostFieldTests(unittest.TestCase):
    def test_host_field_has_consistent_shapes_and_ranges(self) -> None:
        host_field = HostFieldGenerator().generate()

        expected_shape = (host_field.y_coords.size, host_field.x_coords.size)

        self.assertEqual(host_field.elevation.shape, expected_shape)
        self.assertEqual(host_field.slope_degrees.shape, expected_shape)
        self.assertEqual(host_field.cover_thickness.shape, expected_shape)
        self.assertEqual(host_field.roof_competence.shape, expected_shape)
        self.assertEqual(host_field.growth_cost.shape, expected_shape)

        self.assertTrue(np.all(host_field.cover_thickness > 0.0))
        self.assertTrue(
            np.all((host_field.roof_competence >= 0.0) & (host_field.roof_competence <= 1.0))
        )
        self.assertTrue(
            np.all((host_field.growth_cost >= 0.0) & (host_field.growth_cost <= 1.0))
        )

    def test_host_field_supports_point_sampling(self) -> None:
        host_field = HostFieldGenerator().generate()
        sample = host_field.sample(0.0, 0.0)

        self.assertGreater(sample.cover_thickness, 0.0)
        self.assertGreaterEqual(sample.roof_competence, 0.0)
        self.assertLessEqual(sample.roof_competence, 1.0)
        self.assertGreaterEqual(sample.growth_cost, 0.0)
        self.assertLessEqual(sample.growth_cost, 1.0)
        self.assertIsInstance(sample.gradient_x, float)
        self.assertIsInstance(sample.gradient_y, float)


if __name__ == "__main__":
    unittest.main()
