"""Smoke tests for the stage-A host field."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.host_field import HostFieldConfig, HostFieldGenerator


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

    def test_project_seed_resolves_high_level_host_ranges(self) -> None:
        config_text = (ROOT / "config" / "project.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            config_a_path = Path(temp_dir) / "seed_1.toml"
            config_b_path = Path(temp_dir) / "seed_2.toml"
            config_a_path.write_text(config_text, encoding="utf-8")
            config_b_path.write_text(
                config_text.replace("procedural_seed = 1", "procedural_seed = 2", 1),
                encoding="utf-8",
            )

            config_a = load_project_config(config_a_path).host_field
            config_b = load_project_config(config_b_path).host_field

        self.assertEqual(config_a.random_seed, 1)
        self.assertEqual(config_b.random_seed, 2)
        self.assertNotEqual(config_a.seed_point, config_b.seed_point)
        self.assertNotEqual(config_a.flow_angle_degrees, config_b.flow_angle_degrees)
        self.assertNotEqual(config_a.corridor_width, config_b.corridor_width)
        self.assertNotEqual(config_a.waves, config_b.waves)


if __name__ == "__main__":
    unittest.main()
