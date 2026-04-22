"""Smoke tests for the stage-A host field."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stages import HostFieldGenerator
from stages.host_field import HostFieldConfig


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

    def test_seeded_host_field_is_reproducible_and_varies_with_seed(self) -> None:
        seeded_config = HostFieldConfig(random_seed=17)
        host_field_a = HostFieldGenerator(seeded_config).generate()
        host_field_b = HostFieldGenerator(seeded_config).generate()
        alternate_host_field = HostFieldGenerator(
            HostFieldConfig(random_seed=23)
        ).generate()

        self.assertTrue(np.allclose(host_field_a.elevation, host_field_b.elevation))
        self.assertTrue(
            np.allclose(host_field_a.roof_competence, host_field_b.roof_competence)
        )
        self.assertTrue(np.allclose(host_field_a.growth_cost, host_field_b.growth_cost))

        mean_elevation_difference = float(
            np.mean(np.abs(host_field_a.elevation - alternate_host_field.elevation))
        )
        mean_roof_difference = float(
            np.mean(
                np.abs(
                    host_field_a.roof_competence - alternate_host_field.roof_competence
                )
            )
        )

        self.assertGreater(mean_elevation_difference, 0.25)
        self.assertGreater(mean_roof_difference, 0.01)


if __name__ == "__main__":
    unittest.main()
