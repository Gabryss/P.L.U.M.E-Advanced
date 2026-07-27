"""Configuration and world-profile regression tests."""

from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config, project_config_manifest
from stages.events import GeologicalEventGenerator
from stages.section_field import SectionField, SectionFieldConfig


class ProjectConfigurationTests(unittest.TestCase):
    def test_project_uses_resolved_earth_profile_and_named_seeds(self) -> None:
        config = load_project_config(ROOT / "config" / "project.toml")

        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.world.body.name, "earth")
        self.assertEqual(config.section_field.maximum_tube_width, 10.0)
        self.assertEqual(config.section_field.chamber_max_tube_width, 20.0)
        self.assertEqual(config.network.maximum_passage_radius, 5.0)
        self.assertEqual(config.export.target, "blender")
        self.assertEqual(config.export.file_format, "glb")
        self.assertLessEqual(
            config.host_field.grid.height,
            config.run.dev_max_route_length_m,
        )

        stage_seed_values = {
            config.stage_seeds.host,
            config.stage_seeds.network,
            config.stage_seeds.sections,
            config.stage_seeds.events,
            config.stage_seeds.geometry,
            config.stage_seeds.surface,
            config.stage_seeds.export,
        }
        self.assertEqual(len(stage_seed_values), 7)
        self.assertEqual(config.host_field.random_seed, config.stage_seeds.host)
        self.assertEqual(config.events.random_seed, config.stage_seeds.events)

    def test_body_selection_changes_generation_limits_and_stability(self) -> None:
        earth = self._load_minimal(
            """
            schema_version = 2
            procedural_seed = 7
            [world]
            body = "earth"
            """
        )
        moon = self._load_minimal(
            """
            schema_version = 2
            procedural_seed = 7
            [world]
            body = "moon"
            """
        )

        self.assertEqual(earth.section_field.maximum_tube_width, 10.0)
        self.assertEqual(moon.section_field.maximum_tube_width, 100.0)
        self.assertEqual(moon.section_field.chamber_max_tube_width, 200.0)
        self.assertGreater(
            moon.host_field.grid.width,
            earth.host_field.grid.width,
        )
        self.assertGreater(
            moon.network.braid_grammar.half_length_fraction[1],
            earth.network.braid_grammar.half_length_fraction[1],
        )
        self.assertLess(
            moon.world.roof_demand_ratio(10.0, 8.0),
            earth.world.roof_demand_ratio(10.0, 8.0),
        )
        self.assertEqual(earth.stage_seeds, moon.stage_seeds)

    def test_body_profiles_scale_host_extent_route_and_branch_persistence(self) -> None:
        configs = {
            body: load_project_config(
                ROOT / "config" / "project.toml",
                world_body=body,
            )
            for body in ("earth", "mars", "moon")
        }

        self.assertEqual(
            [configs[body].host_field.target_route_length_m for body in configs],
            [5_000.0, 15_000.0, 30_000.0],
        )
        self.assertLess(
            configs["earth"].host_field.grid.width,
            configs["mars"].host_field.grid.width,
        )
        self.assertLess(
            configs["mars"].host_field.grid.width,
            configs["moon"].host_field.grid.width,
        )
        self.assertLess(
            configs["earth"].host_field.grid.height,
            configs["mars"].host_field.grid.height,
        )
        self.assertLess(
            configs["mars"].host_field.grid.height,
            configs["moon"].host_field.grid.height,
        )
        self.assertLess(
            configs["earth"].network.braid_grammar.half_length_fraction[1],
            configs["mars"].network.braid_grammar.half_length_fraction[1],
        )
        self.assertLess(
            configs["mars"].network.braid_grammar.half_length_fraction[1],
            configs["moon"].network.braid_grammar.half_length_fraction[1],
        )

    def test_flow_regime_changes_morphology_without_changing_body_physics(self) -> None:
        baseline = self._load_minimal(
            """
            schema_version = 2
            procedural_seed = 7
            [world]
            body = "mars"
            [flow_regime]
            supply_rate_scale = 1.0
            duration_scale = 1.0
            inflation = 0.25
            distributary_tendency = 0.25
            cooling_rate_scale = 1.0
            """
        )
        sustained = self._load_minimal(
            """
            schema_version = 2
            procedural_seed = 7
            [world]
            body = "mars"
            [flow_regime]
            supply_rate_scale = 1.4
            duration_scale = 1.8
            inflation = 0.75
            distributary_tendency = 0.75
            cooling_rate_scale = 0.8
            """
        )

        self.assertEqual(baseline.world, sustained.world)
        self.assertGreater(
            sustained.host_field.target_route_length_m,
            baseline.host_field.target_route_length_m,
        )
        self.assertGreater(
            sustained.host_field.grid.width,
            baseline.host_field.grid.width,
        )
        self.assertGreater(
            sustained.network.chamber_radius_fraction,
            baseline.network.chamber_radius_fraction,
        )
        self.assertGreater(
            sustained.network.spur_count,
            baseline.network.spur_count,
        )
        self.assertGreater(
            sustained.network.braid_grammar.zone_count[1],
            baseline.network.braid_grammar.zone_count[1],
        )

    def test_world_body_override_uses_the_selected_bodys_default_material(self) -> None:
        moon = load_project_config(
            ROOT / "config" / "project.toml",
            world_body="moon",
        )

        self.assertEqual(moon.world.body.name, "moon")
        self.assertEqual(moon.world.material.name, "mare_basalt")
        self.assertEqual(moon.section_field.maximum_tube_width, 100.0)

    def test_run_config_can_explicitly_bypass_output_confirmation(self) -> None:
        config = self._load_minimal(
            """
            schema_version = 2
            [run]
            overwrite_outputs = true
            """
        )

        self.assertTrue(config.run.overwrite_outputs)

    def test_event_disable_returns_empty_field_without_optional_provider(self) -> None:
        config = self._load_minimal(
            """
            schema_version = 2
            [events]
            enabled = false
            use_rocky_meshes = true
            strict_optional_provider = true
            """
        )
        section_field = SectionField(
            config=SectionFieldConfig(),
            segment_fields=(),
            dominant_route_segment_ids=(),
        )

        result = GeologicalEventGenerator(config.events).generate(section_field)

        self.assertEqual(result.events, ())
        self.assertEqual(result.meshes, ())

    def test_manifest_records_canonical_coordinates_and_resolved_world(self) -> None:
        config = self._load_minimal(
            """
            schema_version = 2
            [world]
            body = "mars"
            [export]
            target = "gazebo"
            format = "dae"
            """
        )

        manifest = project_config_manifest(config)

        self.assertEqual(manifest["world"]["body"]["name"], "mars")
        self.assertEqual(manifest["flow_regime"]["supply_rate_scale"], 1.0)
        self.assertEqual(manifest["export"]["target"], "gazebo")
        self.assertEqual(manifest["canonical_coordinates"]["up_axis"], "Z")

    def test_schema_v1_minimal_configuration_still_loads(self) -> None:
        config = self._load_minimal("procedural_seed = 11")

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.world.body.name, "earth")
        self.assertFalse(config.run.dev_mode)

    def test_invalid_body_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "world.body must be one of"):
            self._load_minimal(
                """
                schema_version = 2
                [world]
                body = "venus"
                """
            )

    @staticmethod
    def _load_minimal(contents: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "project.toml"
            path.write_text(textwrap.dedent(contents), encoding="utf-8")
            return load_project_config(path)


if __name__ == "__main__":
    unittest.main()
