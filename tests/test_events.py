"""Smoke tests for the Stage-E geological event layer."""

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.stages.events import (
    GeologicalEvent,
    GeologicalEventConfig,
    GeologicalEventGenerator,
)
from plume_advanced.stages.floor_map import FloorCell, FloorMapGenerator
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator, SectionSample


def _section_sample(
    *,
    index: int = 0,
    arc: float = 0.0,
    width: float = 8.0,
    height: float = 5.0,
    roof_thickness: float = 8.0,
) -> SectionSample:
    return SectionSample(
        index=index,
        segment_id=0,
        segment_arc_length=arc,
        x=arc,
        y=0.0,
        z=0.0,
        surface_z=10.0,
        cover_thickness=roof_thickness,
        roof_thickness=roof_thickness,
        centerline_depth=10.0,
        tangent=(1.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        binormal=(0.0, 0.0, 1.0),
        tube_width=width,
        tube_height=height,
        floor_flatness=0.7,
        roof_arch=1.0,
        lateral_skew=0.0,
        junction_blend_weight=0.0,
        junction_influences=(),
        profile_points=((-0.5 * width, 0.0), (0.5 * width, 0.0)),
    )


def _floor_cell(
    sample: SectionSample,
    *,
    lateral: float = 0.0,
    clearance: float | None = None,
) -> FloorCell:
    return FloorCell(
        cell_id=sample.index,
        segment_id=sample.segment_id,
        sample_index=sample.index,
        z_level=0,
        distance_along_m=sample.segment_arc_length,
        lateral_offset_m=lateral,
        atlas_x_m=sample.segment_arc_length,
        atlas_y_m=lateral,
        x=sample.x,
        y=lateral,
        z=-0.5 * sample.tube_height,
        normal_x=0.0,
        normal_y=0.0,
        normal_z=1.0,
        clearance_m=clearance if clearance is not None else sample.tube_height,
        tube_width_m=sample.tube_width,
        grounded=True,
    )


def _compact_event_config(config: GeologicalEventConfig) -> GeologicalEventConfig:
    """Keep integration tests representative without building a full asset field."""

    return replace(
        config,
        rock_population_multiplier=1.0,
        rock_density_per_100m2=0.10,
        boulder_satellite_count_range=(3, 5),
        collapse_fragment_count_range=(5, 8),
        minor_cluster_density_per_1000m2=0.03,
        minor_cluster_count_range=(2, 3),
    )


class GeologicalEventTests(unittest.TestCase):
    def test_population_multiplier_scales_all_floor_density_counts(self) -> None:
        samples = [
            _section_sample(index=0, arc=0.0, width=10.0),
            _section_sample(index=1, arc=100.0, width=10.0),
        ]
        base = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                rock_population_multiplier=1.0,
                rock_density_per_100m2=1.0,
                boulder_density_per_100m2=0.1,
            )
        )._counts_from_density(samples)
        dense = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                rock_population_multiplier=10.0,
                rock_density_per_100m2=1.0,
                boulder_density_per_100m2=0.1,
            )
        )._counts_from_density(samples)
        walls_only = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                include_rock_props=False,
                rock_population_multiplier=10.0,
                rock_density_per_100m2=1.0,
                boulder_density_per_100m2=0.1,
            )
        )._counts_from_density(samples)

        self.assertEqual(base[:2], (10, 1))
        self.assertEqual(dense[:2], (100, 10))
        self.assertEqual(walls_only[:2], (0, 0))

    def test_wall_only_mode_does_not_load_rocky(self) -> None:
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                include_rock_props=False,
                use_rocky_meshes=True,
                strict_optional_provider=True,
                rocky_source_path="/provider/that/does/not/exist",
            )
        )
        self.assertIsNone(generator._rocky_api)

    def test_ground_contact_slots_bound_repeated_floor_raycasts(self) -> None:
        class CountingVoxelGrid:
            voxel_size = 0.6

            def __init__(self) -> None:
                self.raycast_count = 0

            def raycast_isosurface(self, origin, direction, max_distance):
                self.raycast_count += 1
                return SimpleNamespace(
                    position=tuple(float(value) for value in origin),
                    normal=(0.0, 0.0, 1.0),
                )

        sample = _section_sample()
        cell = _floor_cell(sample)
        voxel_grid = CountingVoxelGrid()
        generator = GeologicalEventGenerator(GeologicalEventConfig(enabled=False, random_seed=17))
        rng = np.random.default_rng(29)

        contacts = [
            generator._jitter_floor_contact(
                sample,
                cell,
                rng,
                voxel_grid,
            )[0]
            for _index in range(100)
        ]

        self.assertLessEqual(
            voxel_grid.raycast_count,
            generator._GROUND_CONTACT_SLOTS_PER_CELL,
        )
        self.assertGreater(voxel_grid.raycast_count, 1)
        self.assertLessEqual(
            len({tuple(float(value) for value in contact) for contact in contacts}),
            generator._GROUND_CONTACT_SLOTS_PER_CELL,
        )

    def test_rejected_family_slots_are_recovered_as_micro_debris(self) -> None:
        sample = _section_sample()
        cell = _floor_cell(sample)
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                minor_cluster_density_per_1000m2=0.0,
            )
        )
        template = GeologicalEvent(
            event_id=0,
            kind="rock",
            segment_id=0,
            sample_index=0,
            x=0.0,
            y=0.0,
            z=0.1,
            surface_z=0.0,
            floor_z=0.0,
            radius_x=0.05,
            radius_y=0.05,
            radius_z=0.05,
            angle=0.0,
            severity=0.5,
            material_hint="floor_debris",
            contact_point=(0.0, 0.0, 0.0),
            contact_normal=(0.0, 0.0, 1.0),
            grounded=True,
        )

        def choose(*args, **kwargs):
            if kwargs.get("debris_role") != "dense_micro_debris":
                return None
            return replace(
                template,
                event_id=len(args[3]),
                x=0.2 * len(args[3]),
                contact_point=(0.2 * len(args[3]), 0.0, 0.0),
                debris_role="dense_micro_debris",
            )

        progress_updates: list[tuple[str, int, int, str]] = []
        with patch.object(
            generator,
            "_choose_and_build_prop",
            side_effect=choose,
        ):
            placed = generator._place_rock_populations(
                background_count=4,
                candidates=[(1.0, sample, cell)],
                events=[],
                sample_lookup={(0, 0): sample},
                voxel_grid=None,
                rng=np.random.default_rng(9),
                material_hint="floor_debris",
                progress=lambda phase, current, total, message: progress_updates.append(
                    (phase, current, total, message)
                ),
            )

        self.assertEqual(len(placed), 4)
        self.assertTrue(all(event.debris_role == "dense_micro_debris" for event in placed))
        self.assertTrue(
            any("recovering rejected slots" in update[3] for update in progress_updates)
        )

    def test_floor_area_density_and_local_gallery_cap_scale_with_width(self) -> None:
        narrow = _section_sample(index=0, arc=0.0, width=6.0)
        narrow_end = _section_sample(index=1, arc=10.0, width=6.0)
        wide = _section_sample(
            index=2,
            arc=20.0,
            width=12.0,
            height=8.0,
            roof_thickness=16.0,
        )
        self.assertAlmostEqual(
            GeologicalEventGenerator._sampled_floor_area([narrow, narrow_end, wide]),
            150.0,
        )

        generator = GeologicalEventGenerator(GeologicalEventConfig(enabled=False))
        narrow_cap = generator._local_prop_size_cap(
            "boulder",
            narrow,
            _floor_cell(narrow),
            0.0,
        )
        wide_cap = generator._local_prop_size_cap(
            "boulder",
            wide,
            _floor_cell(wide),
            0.0,
        )
        edge_cap = generator._local_prop_size_cap(
            "boulder",
            wide,
            _floor_cell(wide, lateral=5.7),
            5.7,
        )
        self.assertGreater(wide_cap, narrow_cap)
        self.assertLess(edge_cap, wide_cap)

    def test_boulder_height_cap_reaches_two_thirds_of_local_clearance(self) -> None:
        sample = _section_sample(
            width=10.0,
            height=9.0,
            roof_thickness=100.0,
        )
        floor_cell = _floor_cell(sample, clearance=9.0)
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                rock_radius_range=(0.03, 4.0),
                boulder_radius_range=(0.3, 4.0),
                gallery_width_size_fraction=0.30,
                gallery_clearance_size_fraction=0.45,
                boulder_max_height_fraction=2.0 / 3.0,
                roof_block_size_fraction=1.0,
            )
        )

        boulder_cap = generator._local_prop_size_cap(
            "boulder",
            sample,
            floor_cell,
            0.0,
        )
        rock_cap = generator._local_prop_size_cap(
            "rock",
            sample,
            floor_cell,
            0.0,
        )

        self.assertAlmostEqual(boulder_cap, 6.0)
        self.assertAlmostEqual(rock_cap, 3.0)

    def test_rover_route_detects_a_cross_section_blockage(self) -> None:
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                rover_width_m=1.0,
                rover_side_margin_m=0.1,
            )
        )
        samples = [
            _section_sample(index=0, arc=0.0, width=3.0),
            _section_sample(index=1, arc=2.0, width=3.0),
        ]
        blocker = GeologicalEvent(
            event_id=0,
            kind="boulder",
            segment_id=0,
            sample_index=0,
            x=0.0,
            y=0.0,
            z=-1.0,
            surface_z=0.0,
            floor_z=-1.0,
            radius_x=1.0,
            radius_y=1.0,
            radius_z=0.7,
            angle=0.0,
            severity=0.5,
            material_hint="large_breakdown",
            contact_point=(0.0, 0.0, -1.5),
            contact_normal=(0.0, 0.0, 1.0),
            grounded=True,
        )
        route_valid, minimum_bypass = generator._segment_rover_route(
            samples,
            [blocker],
        )
        self.assertFalse(route_valid)
        self.assertAlmostEqual(minimum_bypass, 0.5)

    def test_radius_aware_spacing_allows_talus_but_rejects_overlap(self) -> None:
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                minimum_rock_spacing=0.05,
                collapse_cluster_spacing_scale=0.9,
                background_contact_spacing=1.15,
            )
        )
        first = GeologicalEvent(
            event_id=0,
            kind="rock",
            segment_id=0,
            sample_index=0,
            x=0.0,
            y=0.0,
            z=0.1,
            surface_z=0.0,
            floor_z=0.0,
            radius_x=0.1,
            radius_y=0.1,
            radius_z=0.1,
            angle=0.0,
            severity=0.5,
            material_hint="floor_debris",
            contact_point=(0.0, 0.0, 0.0),
            contact_normal=(0.0, 0.0, 1.0),
            grounded=True,
            cluster_parent_event_id=4,
        )
        touching = replace(
            first,
            event_id=1,
            x=0.19,
            contact_point=(0.19, 0.0, 0.0),
        )
        overlapping = replace(
            touching,
            x=0.10,
            contact_point=(0.10, 0.0, 0.0),
        )
        self.assertTrue(generator._prop_spacing_is_valid(touching, [first]))
        self.assertFalse(generator._prop_spacing_is_valid(overlapping, [first]))

    def test_family_child_can_touch_its_parent_boulder(self) -> None:
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                minimum_rock_spacing=0.05,
                minimum_boulder_spacing=0.05,
                collapse_cluster_spacing_scale=0.9,
                background_contact_spacing=1.15,
            )
        )
        anchor = GeologicalEvent(
            event_id=4,
            kind="boulder",
            segment_id=0,
            sample_index=0,
            x=0.0,
            y=0.0,
            z=0.5,
            surface_z=0.0,
            floor_z=0.0,
            radius_x=0.5,
            radius_y=0.5,
            radius_z=0.5,
            angle=0.0,
            severity=0.5,
            material_hint="large_breakdown",
            contact_point=(0.0, 0.0, 0.0),
            contact_normal=(0.0, 0.0, 1.0),
            grounded=True,
        )
        child = replace(
            anchor,
            event_id=5,
            kind="rock",
            x=0.55,
            radius_x=0.1,
            radius_y=0.1,
            radius_z=0.1,
            contact_point=(0.55, 0.0, 0.0),
            debris_family_id=anchor.event_id,
            family_anchor_event_id=anchor.event_id,
            debris_role="boulder_small_rubble",
        )
        unrelated = replace(
            child,
            debris_family_id=-1,
            family_anchor_event_id=-1,
            debris_role="background_scatter",
        )

        self.assertTrue(generator._prop_spacing_is_valid(child, [anchor]))
        self.assertFalse(generator._prop_spacing_is_valid(unrelated, [anchor]))

    def test_boulder_family_produces_compact_size_biased_satellite_roles(self) -> None:
        generator = GeologicalEventGenerator(GeologicalEventConfig(enabled=False, random_seed=5))
        anchor = GeologicalEvent(
            event_id=4,
            kind="boulder",
            segment_id=0,
            sample_index=0,
            x=0.0,
            y=0.0,
            z=-1.5,
            surface_z=0.0,
            floor_z=-2.5,
            radius_x=1.0,
            radius_y=0.8,
            radius_z=0.9,
            angle=0.0,
            severity=0.7,
            material_hint="large_breakdown",
            contact_point=(0.0, 0.0, -2.5),
            contact_normal=(0.0, 0.0, 1.0),
            grounded=True,
        )
        count = generator._boulder_satellite_count(
            anchor,
            np.random.default_rng(7),
        )
        roles = generator._role_sequence(
            count,
            (
                ("boulder_inner_rubble", 0.45),
                ("boulder_small_rubble", 0.30),
                ("boulder_companion", 0.20),
                ("boulder_runout", 0.05),
            ),
        )
        self.assertGreaterEqual(count, 25)
        self.assertLessEqual(count, 55)
        self.assertGreater(
            roles.count("boulder_inner_rubble"),
            roles.count("boulder_companion"),
        )
        inner_range = generator._role_diameter_range(
            "boulder_inner_rubble",
            anchor,
        )
        companion_range = generator._role_diameter_range(
            "boulder_companion",
            anchor,
        )
        self.assertLessEqual(inner_range[1], 0.16)
        self.assertGreater(companion_range[1], inner_range[1])

        candidates = []
        for index, distance in enumerate((1.0, 3.0, 7.0)):
            sample = _section_sample(index=index, arc=distance, width=8.0)
            candidates.append((1.0, sample, _floor_cell(sample)))
        family_candidates = generator._family_candidates(
            candidates,
            anchor,
            "boulder_companion",
        )
        selected_distances = {round(cell.x, 3) for _score, _sample, cell in family_candidates}
        self.assertIn(1.0, selected_distances)
        self.assertIn(3.0, selected_distances)
        self.assertNotIn(7.0, selected_distances)

    def test_minor_rubble_clusters_scale_with_floor_area(self) -> None:
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                minor_cluster_density_per_1000m2=2.0,
                rock_population_multiplier=1.0,
            )
        )
        samples = [
            _section_sample(index=0, arc=0.0, width=10.0),
            _section_sample(index=1, arc=100.0, width=10.0),
        ]
        self.assertEqual(generator._minor_cluster_count(samples), 2)
        roles = generator._role_sequence(
            10,
            (
                ("minor_cluster_rubble", 0.60),
                ("minor_cluster_companion", 0.30),
                ("minor_cluster_runout", 0.10),
            ),
        )
        self.assertEqual(roles.count("minor_cluster_rubble"), 6)
        self.assertEqual(roles.count("minor_cluster_companion"), 3)
        self.assertEqual(roles.count("minor_cluster_runout"), 1)

    def test_background_debris_uses_reproducible_clean_floor_patches(self) -> None:
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                enabled=False,
                random_seed=11,
                clean_floor_fraction=0.5,
                debris_patch_length_m=20.0,
            )
        )
        first = _section_sample(index=0, arc=2.0)
        same_patch = _section_sample(index=1, arc=18.0)
        next_patch = _section_sample(index=2, arc=22.0)
        self.assertEqual(
            generator._debris_patch_is_active(first),
            generator._debris_patch_is_active(same_patch),
        )
        patch_states = {
            generator._debris_patch_is_active(_section_sample(index=index, arc=20.0 * index + 1.0))
            for index in range(12)
        }
        self.assertEqual(patch_states, {False, True})
        self.assertIsInstance(
            generator._debris_patch_is_active(next_patch),
            bool,
        )

    def test_rocky_generates_detailed_grounded_deterministic_boulder(self) -> None:
        config = GeologicalEventConfig(
            random_seed=42,
            enabled_kinds=("rock", "boulder"),
            use_rocky_meshes=True,
            strict_optional_provider=True,
            rocky_source_path="",
            rocky_texture_dir=str(ROOT / "texture" / "dark_rock_8k" / "textures"),
            rocky_output_dir=str(ROOT / "outputs" / "rocky_stage_e"),
            rocky_resolution_scale=1.0,
        )
        event = GeologicalEvent(
            event_id=7,
            kind="boulder",
            segment_id=1,
            sample_index=9,
            x=5.0,
            y=7.0,
            z=1.0,
            surface_z=0.0,
            floor_z=0.0,
            radius_x=1.8,
            radius_y=1.25,
            radius_z=1.1,
            angle=0.4,
            severity=0.75,
            material_hint="large_breakdown",
            contact_point=(5.0, 7.0, 0.0),
            contact_normal=(0.0, 0.0, 1.0),
            grounded=True,
        )

        mesh = GeologicalEventGenerator(config)._build_event_mesh(event)
        repeated = GeologicalEventGenerator(config)._build_event_mesh(event)

        self.assertEqual(mesh.source_generator, "rocky")
        self.assertTrue(mesh.source_shape_type)
        self.assertGreater(mesh.vertex_count, 1_000)
        self.assertGreater(mesh.face_count, 2_000)
        self.assertEqual(len(mesh.face_uvs), mesh.face_count)
        self.assertIn("diffuse", dict(mesh.material_maps))
        self.assertEqual(mesh, repeated)
        signed_heights = [
            float(np.dot(np.subtract(vertex, event.contact_point), event.contact_normal))
            for vertex in mesh.vertices
        ]
        self.assertLess(min(signed_heights), 0.0)
        self.assertGreater(max(signed_heights), event.radius_z)

    def test_rocky_profile_selection_covers_distinct_lava_debris_families(self) -> None:
        generator = GeologicalEventGenerator(
            GeologicalEventConfig(
                random_seed=73,
                use_rocky_meshes=True,
                strict_optional_provider=True,
                rocky_source_path="",
            )
        )
        shapes = {
            generator._rocky_profile(
                GeologicalEvent(
                    event_id=index,
                    kind="rock",
                    segment_id=0,
                    sample_index=index * index + 3,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    surface_z=0.0,
                    floor_z=0.0,
                    radius_x=1.0,
                    radius_y=0.8,
                    radius_z=0.7,
                    angle=0.0,
                    severity=0.6,
                    material_hint="floor_debris",
                )
            )[0]
            for index in range(6)
        }

        self.assertGreaterEqual(len(shapes), 5)
        self.assertIn("angular_boulder", shapes)
        self.assertIn("flat_slab", shapes)
        self.assertIn("vesicular_chunk", shapes)

    def test_prop_size_sampling_is_varied_and_biased_toward_smaller_debris(self) -> None:
        config = GeologicalEventConfig(
            enabled=False,
            rock_radius_range=(0.25, 2.4),
            boulder_radius_range=(1.25, 6.2),
            rock_size_bias=2.2,
            boulder_size_bias=1.6,
        )
        generator = GeologicalEventGenerator(config)
        rock_rng = np.random.default_rng(18)
        boulder_rng = np.random.default_rng(19)
        rocks = np.asarray([generator._sample_radius("rock", rock_rng) for _ in range(1_000)])
        boulders = np.asarray(
            [generator._sample_radius("boulder", boulder_rng) for _ in range(1_000)]
        )

        self.assertGreater(float(np.ptp(rocks)), 1.8)
        self.assertGreater(float(np.ptp(boulders)), 4.0)
        self.assertLess(float(np.median(rocks)), sum(config.rock_radius_range) / 2.0)
        self.assertLess(
            float(np.median(boulders)),
            sum(config.boulder_radius_range) / 2.0,
        )

    def test_event_stage_places_seeded_mesh_events(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
        section_field = SectionFieldGenerator(project_config.section_field).generate(cave_network)
        base_geometry = GeometryGenerator(project_config.geometry).build_base_volume(
            cave_network, section_field
        )
        floor_atlas = FloorMapGenerator(project_config.floor_map).generate(
            cave_network,
            section_field,
            base_geometry,
        )

        event_config = _compact_event_config(project_config.events)
        progress_updates: list[tuple[str, int, int, str]] = []
        event_field = GeologicalEventGenerator(event_config).generate(
            section_field,
            base_geometry,
            floor_atlas,
            progress=lambda phase, current, total, message: progress_updates.append(
                (phase, current, total, message)
            ),
        )
        repeated_event_field = GeologicalEventGenerator(event_config).generate(
            section_field,
            base_geometry,
            floor_atlas,
        )

        self.assertEqual(event_field.events, repeated_event_field.events)
        summary = event_field.summary()
        self.assertGreater(int(summary["event_count"]), 0)
        self.assertGreater(int(summary["rock_count"]), 0)
        self.assertGreater(int(summary["boulder_count"]), 0)
        self.assertGreater(int(summary["collapse_count"]), 0)
        self.assertGreater(int(summary["choke_count"]), 0)
        self.assertGreater(int(summary["infill_count"]), 0)
        self.assertEqual(
            int(summary["event_mesh_count"]),
            int(summary["rock_count"] + summary["boulder_count"]),
        )
        self.assertEqual(
            int(summary["structural_modifier_count"]),
            int(summary["collapse_count"] + summary["choke_count"] + summary["infill_count"]),
        )
        self.assertEqual(
            int(summary["grounded_prop_count"]),
            int(summary["prop_count"]),
        )
        self.assertGreater(int(summary["clustered_prop_count"]), 0)
        self.assertGreater(int(summary["debris_family_count"]), 0)
        self.assertGreater(int(summary["family_prop_count"]), 0)
        self.assertGreater(int(summary["boulder_satellite_count"]), 0)
        self.assertGreater(int(summary["collapse_talus_prop_count"]), 0)
        self.assertGreater(int(summary["event_mesh_vertex_count"]), 0)
        self.assertGreater(int(summary["event_mesh_face_count"]), 0)
        self.assertIn("props", {update[0] for update in progress_updates})
        self.assertIn("meshes", {update[0] for update in progress_updates})
        self.assertTrue(
            all(
                0 <= current <= total and total > 0 and message
                for _phase, current, total, message in progress_updates
            )
        )

        segment_ids = {segment_field.segment_id for segment_field in section_field.segment_fields}
        sample_keys = {
            (sample.segment_id, sample.index)
            for segment_field in section_field.segment_fields
            for sample in segment_field.samples
        }
        for event in event_field.events:
            self.assertIn(event.segment_id, segment_ids)
            self.assertIn((event.segment_id, event.sample_index), sample_keys)
            self.assertGreater(event.radius_x, 0.0)
            self.assertGreater(event.radius_y, 0.0)
            self.assertGreater(event.radius_z, 0.0)
            self.assertGreaterEqual(event.severity, 0.0)
            self.assertLessEqual(event.severity, 1.0)
            self.assertIn(event.kind, {"rock", "boulder", "collapse", "choke", "infill"})
            self.assertAlmostEqual(
                sum(value * value for value in event.contact_normal),
                1.0,
                places=4,
            )
            if event.kind in {"rock", "boulder"}:
                self.assertTrue(event.grounded)
                self.assertGreaterEqual(event.floor_cell_id, 0)
                self.assertAlmostEqual(
                    base_geometry.voxel_grid.sample_density(event.contact_point),
                    base_geometry.voxel_grid.iso_level,
                    delta=0.05,
                )
            if event.cluster_parent_event_id >= 0:
                parent = event_field.events[event.cluster_parent_event_id]
                self.assertEqual(parent.kind, "collapse")
        event_lookup = {event.event_id: event for event in event_field.events}
        for mesh in event_field.meshes:
            self.assertTrue(mesh.vertices)
            self.assertTrue(mesh.faces)
            self.assertIn(mesh.kind, {"rock", "boulder"})
            self.assertEqual(mesh.source_generator, "rocky")
            self.assertTrue(mesh.source_shape_type)
            self.assertEqual(len(mesh.face_uvs), mesh.face_count)
            event = event_lookup[mesh.event_id]
            self.assertEqual(mesh.debris_family_id, event.debris_family_id)
            self.assertEqual(
                mesh.family_anchor_event_id,
                event.family_anchor_event_id,
            )
            self.assertEqual(mesh.debris_role, event.debris_role)
            contact = event.contact_point
            contact_normal = event.contact_normal
            signed_contact_distances = [
                sum((vertex[axis] - contact[axis]) * contact_normal[axis] for axis in range(3))
                for vertex in mesh.vertices
            ]
            self.assertLessEqual(
                min(signed_contact_distances),
                0.25 * base_geometry.voxel_grid.voxel_size,
                "prop mesh should touch or embed into the final floor",
            )
            self.assertGreater(max(signed_contact_distances), 0.0)

    def test_geometry_consumes_event_field(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
        section_field = SectionFieldGenerator(project_config.section_field).generate(cave_network)
        geometry_generator = GeometryGenerator(project_config.geometry)
        base_geometry = geometry_generator.build_base_volume(
            cave_network,
            section_field,
        )
        floor_atlas = FloorMapGenerator(project_config.floor_map).generate(
            cave_network,
            section_field,
            base_geometry,
        )
        event_field = GeologicalEventGenerator(
            _compact_event_config(project_config.events)
        ).generate(
            section_field,
            base_geometry,
            floor_atlas,
        )
        cave_geometry = geometry_generator.finalize(
            base_geometry,
            event_field,
        )
        final_floor_atlas = FloorMapGenerator(project_config.floor_map).revalidate(
            cave_network,
            section_field,
            cave_geometry,
            floor_atlas,
            event_field,
        )

        summary = cave_geometry.summary()
        self.assertGreater(int(summary["carved_voxel_count"]), 0)
        self.assertGreaterEqual(int(summary["vertex_count"]), 3)
        self.assertGreaterEqual(int(summary["face_count"]), 1)
        self.assertEqual(int(summary["event_mesh_count"]), len(event_field.meshes))
        self.assertGreater(int(summary["event_vertex_count"]), 0)
        self.assertGreater(int(summary["event_face_count"]), 0)
        self.assertEqual(
            int(summary["structural_event_count"]),
            int(event_field.summary()["structural_modifier_count"]),
        )
        self.assertLess(
            summary["carved_voxel_count"],
            base_geometry.summary()["carved_voxel_count"],
        )
        self.assertEqual(int(summary["voxel_component_count"]), 1)
        # One connected cave volume can legitimately have additional closed
        # surface shells around structural infill or choke obstacles.
        self.assertGreaterEqual(int(summary["component_count"]), 1)
        self.assertEqual(final_floor_atlas.generation_stage, "final")
        self.assertEqual(
            len(final_floor_atlas.cells) + len(final_floor_atlas.invalidated_cell_ids),
            len(floor_atlas.cells),
        )
        final_cell_ids = {cell.cell_id for cell in final_floor_atlas.cells}
        prop_cell_ids = {
            event.floor_cell_id for event in event_field.events if event.kind in {"rock", "boulder"}
        }
        self.assertTrue(prop_cell_ids.issubset(final_cell_ids))
        self.assertGreater(
            int(final_floor_atlas.summary()["geologically_influenced_cell_count"]),
            0,
        )
        self.assertGreater(
            int(final_floor_atlas.summary()["breakdown_cell_count"]),
            0,
        )
        self.assertGreater(
            int(final_floor_atlas.summary()["sediment_cell_count"]),
            0,
        )
        self.assertLess(
            int(final_floor_atlas.summary()["chamber_cell_count"]),
            len(final_floor_atlas.cells),
        )
        for cell in final_floor_atlas.cells:
            self.assertAlmostEqual(
                cave_geometry.voxel_grid.sample_density(cell.position),
                cave_geometry.voxel_grid.iso_level,
                delta=0.05,
            )


if __name__ == "__main__":
    unittest.main()
