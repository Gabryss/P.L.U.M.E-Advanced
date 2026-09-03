"""Smoke tests for the stage-A host field."""

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator


class HostFieldTests(unittest.TestCase):
    def test_host_field_has_consistent_shapes_and_ranges(self) -> None:
        host_field = HostFieldGenerator().generate()

        expected_shape = (host_field.y_coords.size, host_field.x_coords.size)

        self.assertEqual(host_field.elevation.shape, expected_shape)
        self.assertEqual(host_field.slope_degrees.shape, expected_shape)
        self.assertEqual(host_field.cover_thickness.shape, expected_shape)
        self.assertEqual(host_field.roof_competence.shape, expected_shape)
        self.assertEqual(host_field.growth_cost.shape, expected_shape)
        for values in (
            host_field.emplacement_thickness,
            host_field.lithology_quality,
            host_field.fracture_intensity,
            host_field.cooling_index,
            host_field.flow_capacity,
            host_field.deposit_thickness,
            host_field.erosion_index,
            host_field.roof_stability,
        ):
            self.assertEqual(values.shape, expected_shape)

        self.assertTrue(np.all(host_field.cover_thickness > 0.0))
        self.assertTrue(
            np.all((host_field.roof_competence >= 0.0) & (host_field.roof_competence <= 1.0))
        )
        self.assertTrue(
            np.all((host_field.growth_cost >= 0.0) & (host_field.growth_cost <= 1.0))
        )
        for values in (
            host_field.lithology_quality,
            host_field.fracture_intensity,
            host_field.cooling_index,
            host_field.flow_capacity,
            host_field.erosion_index,
            host_field.roof_stability,
        ):
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
            self.assertGreater(float(np.std(values)), 1e-4)
        for influence in host_field.routing_influence_summary().values():
            self.assertGreater(influence["standard_deviation"], 1e-5)
            self.assertGreater(influence["mean_absolute_contribution"], 0.0)

    def test_host_field_supports_point_sampling(self) -> None:
        host_field = HostFieldGenerator().generate()
        sample = host_field.sample(0.0, 0.0)

        self.assertGreater(sample.cover_thickness, 0.0)
        self.assertGreaterEqual(sample.roof_competence, 0.0)
        self.assertLessEqual(sample.roof_competence, 1.0)
        self.assertGreaterEqual(sample.growth_cost, 0.0)
        self.assertLessEqual(sample.growth_cost, 1.0)
        self.assertGreater(sample.emplacement_thickness, 0.0)
        self.assertGreaterEqual(sample.roof_stability, 0.0)
        self.assertLessEqual(sample.roof_stability, 1.0)
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

    def test_omitted_seed_uses_the_reproducible_seed_zero_world(self) -> None:
        grid = GridConfig(width=300.0, height=240.0, nx=48, ny=40)
        implicit = HostFieldGenerator(HostFieldConfig(grid=grid)).generate()
        explicit = HostFieldGenerator(
            HostFieldConfig(grid=grid, random_seed=0)
        ).generate()

        self.assertTrue(np.array_equal(implicit.elevation, explicit.elevation))
        self.assertTrue(np.array_equal(implicit.roof_competence, explicit.roof_competence))
        self.assertTrue(np.array_equal(implicit.growth_cost, explicit.growth_cost))

    def test_project_seed_resolves_high_level_host_ranges(self) -> None:
        config_text = (ROOT / "config" / "project.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            config_a_path = Path(temp_dir) / "seed_1.toml"
            config_b_path = Path(temp_dir) / "seed_2.toml"
            config_a_path.write_text(
                re.sub(r"^procedural_seed = .*$", "procedural_seed = 1", config_text, count=1, flags=re.MULTILINE),
                encoding="utf-8",
            )
            config_b_path.write_text(
                re.sub(r"^procedural_seed = .*$", "procedural_seed = 2", config_text, count=1, flags=re.MULTILINE),
                encoding="utf-8",
            )

            config_a = load_project_config(config_a_path).host_field
            config_b = load_project_config(config_b_path).host_field

        self.assertIsInstance(config_a.random_seed, int)
        self.assertIsInstance(config_b.random_seed, int)
        self.assertNotEqual(config_a.random_seed, config_b.random_seed)
        self.assertNotEqual(config_a.seed_point, config_b.seed_point)
        self.assertNotEqual(config_a.flow_angle_degrees, config_b.flow_angle_degrees)
        self.assertNotEqual(config_a.corridor_width, config_b.corridor_width)
        self.assertNotEqual(config_a.waves, config_b.waves)


if __name__ == "__main__":
    unittest.main()
