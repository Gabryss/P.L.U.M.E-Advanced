"""Configuration and world-profile regression tests."""

import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.stages.events import GeologicalEventGenerator
from plume_advanced.stages.section_field import SectionField, SectionFieldConfig


class ProjectConfigurationTests(unittest.TestCase):
    def test_project_uses_resolved_earth_profile_and_named_seeds(self) -> None:
        config = load_project_config(ROOT / "config" / "project.toml")

        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.world.body.name, "earth")
        self.assertEqual(config.section_field.maximum_tube_width, 10.0)
        self.assertEqual(config.section_field.chamber_max_tube_width, 28.0)
        self.assertEqual(config.network.maximum_passage_radius, 5.0)
        self.assertEqual(config.network.growth_model, "hybrid_lobe")
        self.assertEqual(config.network.network_density, 3.0)
        self.assertEqual(config.network.lobe_growth.path_count, (6, 6))
        self.assertEqual(config.network.emplacement_history.phase_count, (3, 5))
        self.assertAlmostEqual(
            config.network.emplacement_history.phase_flux_budget_fraction,
            1.0,
        )
        self.assertGreaterEqual(
            config.network.emplacement_history.reoccupation_probability,
            0.0,
        )
        self.assertTrue(config.network.emplacement_history.drained_pool_enabled)
        self.assertEqual(config.network.emplacement_history.drained_pool_count, (1, 3))
        self.assertEqual(config.network.emplacement_history.drained_pool_max_width_m, 28.0)
        self.assertGreater(
            config.network.emplacement_history.stacked_lobe_fraction,
            0.0,
        )
        self.assertGreater(config.section_field.morphology_gradient_strength, 0.0)
        self.assertGreater(config.section_field.morphology_correlation_length, 0.0)
        self.assertGreaterEqual(config.section_field.maximum_uphill_grade, 0.0)
        self.assertGreater(
            config.network.lobe_growth.retirement_temperature_k,
            0.0,
        )
        self.assertEqual(config.geometry.resolution_policy, "body")
        self.assertEqual(config.geometry.resolution_quality, "standard")
        self.assertAlmostEqual(config.geometry.voxel_size, 0.6)
        self.assertAlmostEqual(config.geometry.cave_normal_scale, 2.0)
        self.assertEqual(config.geometry.cave_smoothing_iterations, 4)
        self.assertAlmostEqual(config.geometry.cave_displacement_scale_m, 0.12)
        self.assertAlmostEqual(config.geometry.cave_displacement_midlevel, 0.5)
        self.assertTrue(config.events.include_rock_props)
        self.assertAlmostEqual(config.events.rock_population_multiplier, 10.0)
        self.assertAlmostEqual(
            config.events.boulder_max_height_fraction,
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            config.geometry.characteristic_samples_across_passage,
            10.0 / 0.6,
        )
        self.assertEqual(config.export.target, "all")
        self.assertEqual(config.export.file_format, "auto")
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
        }
        self.assertEqual(len(stage_seed_values), 5)
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
        self.assertEqual(
            [configs[body].geometry.voxel_size for body in configs],
            [0.6, 1.2, 2.4],
        )
        self.assertTrue(
            all(
                configs[body].geometry.characteristic_samples_across_passage >= 10.0
                for body in configs
            )
        )

    def test_body_resolution_policy_tracks_run_quality(self) -> None:
        standard = self._load_minimal(
            """
            schema_version = 2
            [world]
            body = "earth"
            [run]
            quality = "standard"
            [geometry]
            resolution_policy = "body"
            """
        )
        production = self._load_minimal(
            """
            schema_version = 2
            [world]
            body = "earth"
            [run]
            quality = "production"
            [geometry]
            resolution_policy = "body"
            """
        )

        self.assertAlmostEqual(standard.geometry.voxel_size, 0.6)
        self.assertAlmostEqual(production.geometry.voxel_size, 0.5)
        self.assertGreater(
            production.geometry.characteristic_samples_across_passage,
            standard.geometry.characteristic_samples_across_passage,
        )

    def test_fixed_geometry_resolution_remains_available(self) -> None:
        fixed = self._load_minimal(
            """
            schema_version = 2
            [geometry]
            resolution_policy = "fixed"
            voxel_size = 2.25
            """
        )

        self.assertEqual(fixed.geometry.resolution_policy, "fixed")
        self.assertAlmostEqual(fixed.geometry.voxel_size, 2.25)

    def test_body_resolution_rejects_conflicting_fixed_voxel_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._load_minimal(
                """
                schema_version = 2
                [geometry]
                resolution_policy = "body"
                voxel_size = 2.0
                """
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

    def test_rocky_asset_paths_are_resolved_relative_to_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "scenario" / "project.toml"
            config_path.parent.mkdir()
            config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version = 2
                    [events]
                    enabled = false
                    rocky_source_path = "../vendor/Rocky/src"
                    rocky_texture_dir = "../textures"
                    rocky_output_dir = "../outputs/rocky"
                    """
                ),
                encoding="utf-8",
            )

            config = load_project_config(config_path)

            self.assertEqual(
                Path(config.events.rocky_source_path),
                Path(temp_dir) / "vendor" / "Rocky" / "src",
            )
            self.assertEqual(
                Path(config.events.rocky_texture_dir),
                Path(temp_dir) / "textures",
            )
            self.assertEqual(
                Path(config.events.rocky_output_dir),
                Path(temp_dir) / "outputs" / "rocky",
            )

    def test_manifest_records_canonical_coordinates_and_resolved_world(self) -> None:
        config = self._load_minimal(
            """
            schema_version = 2
            [world]
            body = "mars"
            [export]
            target = "gazebo"
            format = "obj"
            """
        )

        manifest = project_config_manifest(config)

        self.assertEqual(manifest["world"]["body"]["name"], "mars")
        self.assertEqual(manifest["flow_regime"]["supply_rate_scale"], 1.0)
        self.assertEqual(manifest["export"]["target"], "gazebo")
        self.assertEqual(manifest["geometry"]["resolution_policy"], "fixed")
        self.assertEqual(manifest["canonical_coordinates"]["up_axis"], "Z")

    def test_schema_v1_minimal_configuration_migrates_to_current(self) -> None:
        config = self._load_minimal("procedural_seed = 11")

        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.world.body.name, "earth")
        self.assertFalse(config.run.dev_mode)

    def test_schema_v2_removed_scaffolding_migrates_when_it_was_inactive(self) -> None:
        config = self._load_minimal(
            """
            schema_version = 2
            [export]
            quality = "preview"
            generate_visual = true
            generate_lods = false
            generate_wall_shell = false
            wall_thickness_m = 0.5
            """
        )

        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.export.target, "blender")

    def test_schema_v2_enabled_removed_capability_cannot_be_migrated(self) -> None:
        with self.assertRaisesRegex(ValueError, "generate_lods"):
            self._load_minimal(
                """
                schema_version = 2
                [export]
                generate_lods = true
                """
            )

    def test_invalid_body_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "world.body must be one of"):
            self._load_minimal(
                """
                schema_version = 2
                [world]
                body = "venus"
                """
            )

    def test_network_density_is_bounded(self) -> None:
        for value in (-0.1, 3.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"network\.network_density must be in \[0, 3\]",
                ):
                    self._load_minimal(
                        f"""
                        schema_version = 2
                        [network]
                        network_density = {value}
                        """
                    )

    def test_flux_breakout_controls_are_validated(self) -> None:
        invalid_controls = (
            (
                "minimum_viable_flux_fraction",
                0.0,
                "minimum_viable_flux_fraction must be in",
            ),
            (
                "coalescence_flux_return_fraction",
                1.1,
                "coalescence_flux_return_fraction must be in",
            ),
            (
                "deposition_feedback_m",
                -0.1,
                "deposition_feedback_m cannot be negative",
            ),
        )
        for name, value, message in invalid_controls:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    self._load_minimal(
                        f"""
                        schema_version = 2
                        [network.lobe_growth]
                        {name} = {value}
                        """
                    )

    def test_unknown_top_level_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown top-level"):
            self._load_minimal(
                """
                schema_version = 2
                [geomtry]
                voxel_size = 2.0
                """
            )

    def test_unknown_nested_keys_report_the_complete_configuration_path(self) -> None:
        with self.assertRaisesRegex(ValueError, r"geometry\.voxl_size"):
            self._load_minimal(
                """
                schema_version = 2
                [geometry]
                voxl_size = 2.0
                """
            )

        with self.assertRaisesRegex(ValueError, r"network\.braid_grammar\.zon_count"):
            self._load_minimal(
                """
                schema_version = 2
                [network.braid_grammar]
                zon_count = [2, 3]
                """
            )

        with self.assertRaisesRegex(ValueError, r"network\.lobe_growth\.path_cout"):
            self._load_minimal(
                """
                schema_version = 2
                [network.lobe_growth]
                path_cout = [6, 8]
                """
            )

        with self.assertRaisesRegex(
            ValueError,
            r"network\.emplacement_history\.phase_cout",
        ):
            self._load_minimal(
                """
                schema_version = 2
                [network.emplacement_history]
                phase_cout = [3, 5]
                """
            )

        with self.assertRaisesRegex(ValueError, r"host_field\.grid\.widht"):
            self._load_minimal(
                """
                schema_version = 2
                [host_field.grid]
                widht = 1200.0
                """
            )

    def test_incompatible_export_format_is_rejected_during_config_load(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports format"):
            self._load_minimal(
                """
                schema_version = 2
                [export]
                target = "blender"
                format = "usd"
                """
            )

    def test_all_export_target_uses_automatic_native_formats(self) -> None:
        config = self._load_minimal(
            """
            schema_version = 2
            [export]
            target = "all"
            format = "auto"
            """
        )

        self.assertEqual(config.export.target, "all")
        self.assertEqual(config.export.file_format, "auto")

    def test_invalid_rock_population_controls_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rock_population_multiplier"):
            self._load_minimal(
                """
                schema_version = 2
                [events]
                rock_population_multiplier = 0.0
                """
            )
        with self.assertRaisesRegex(ValueError, "boulder_max_height_fraction"):
            self._load_minimal(
                """
                schema_version = 2
                [events]
                boulder_max_height_fraction = 1.1
                """
            )

    def test_removed_export_scaffolding_is_rejected_during_config_load(self) -> None:
        for key, value in (
            ("quality", '"preview"'),
            ("generate_visual", "true"),
            ("generate_lods", "true"),
            ("generate_wall_shell", "true"),
            ("wall_thickness_m", "0.5"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self._load_minimal(
                    f"""
                    schema_version = 3
                    [export]
                    {key} = {value}
                    """
                )

    def test_removed_world_metadata_is_rejected_during_config_load(self) -> None:
        for key, value in (
            ("atmosphere", '"dense"'),
            ("erosion_regime", '"weathering"'),
            ("surface_deposit", '"basalt"'),
            ("cohesion_mpa", "12.0"),
            ("friction_angle_degrees", "38.0"),
            ("mean_joint_spacing_m", "1.8"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self._load_minimal(
                    f"""
                    schema_version = 3
                    [world]
                    {key} = {value}
                    """
                )

    @staticmethod
    def _load_minimal(contents: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "project.toml"
            path.write_text(textwrap.dedent(contents), encoding="utf-8")
            return load_project_config(path)

    def test_invalid_surface_relief_controls_are_rejected(self):
        for key, value in (
            ("surface_wall_relief_m", "-0.1"),
            ("surface_roof_relief_m", "nan"),
            ("surface_floor_relief_m", "inf"),
            ("surface_crust_relief_m", "-1.0"),
            ("surface_feature_scale_m", "0.0"),
            ("surface_normal_filter_voxels", "nan"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self._load_minimal(f"schema_version = 3\n[geometry]\n{key} = {value}\n")


if __name__ == "__main__":
    unittest.main()
