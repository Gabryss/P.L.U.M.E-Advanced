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
        self.assertLess(
            moon.world.roof_demand_ratio(10.0, 8.0),
            earth.world.roof_demand_ratio(10.0, 8.0),
        )
        self.assertEqual(earth.stage_seeds, moon.stage_seeds)

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
