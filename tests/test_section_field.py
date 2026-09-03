"""Smoke tests for the stage-C section field."""

import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


class SectionFieldTests(unittest.TestCase):
    def test_section_field_is_geometry_ready_and_junction_aware(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
        section_field = SectionFieldGenerator(project_config.section_field).generate(
            cave_network
        )
        junction_ids = {junction.junction_id for junction in cave_network.junctions}
        route_segment_ids = set(section_field.dominant_route_segment_ids)

        summary = section_field.summary()
        self.assertEqual(
            int(summary["segment_field_count"]),
            len(cave_network.segments),
        )
        self.assertGreater(int(summary["sample_count"]), len(cave_network.segments) * 4)
        self.assertGreater(summary["max_junction_blend_weight"], 0.2)
        self.assertTrue(section_field.dominant_route_segment_ids)

        sample_spacings: list[float] = []
        all_samples = []
        surface_offsets: list[float] = []
        for segment_field in section_field.segment_fields:
            self.assertTrue(segment_field.samples)
            for start, end in zip(segment_field.samples, segment_field.samples[1:]):
                sample_spacings.append(end.segment_arc_length - start.segment_arc_length)
            for sample in segment_field.samples:
                self.assertGreater(sample.tube_width, 0.0)
                self.assertGreater(sample.tube_height, 0.0)
                self.assertTrue(sample.profile_points)
                self.assertEqual(len(sample.tangent), 3)
                self.assertEqual(len(sample.normal), 3)
                self.assertEqual(len(sample.binormal), 3)
                self.assertGreater(sample.surface_z, sample.z)
                self.assertGreater(sample.centerline_depth, 0.5 * sample.tube_height)
                self.assertGreater(sample.roof_thickness, 0.0)
                self.assertGreater(sample.cover_thickness, sample.roof_thickness)
                self.assertTrue(np.allclose(sample.profile_points[0], sample.profile_points[-1]))
                tangent = np.array(sample.tangent, dtype=float)
                normal = np.array(sample.normal, dtype=float)
                binormal = np.array(sample.binormal, dtype=float)
                self.assertLess(abs(float(np.dot(tangent, normal))), 1e-5)
                self.assertLess(abs(float(np.dot(tangent, binormal))), 1e-5)
                self.assertLess(abs(float(np.dot(normal, binormal))), 1e-5)
                self.assertLess(abs(float(np.linalg.norm(tangent)) - 1.0), 1e-5)
                self.assertLess(abs(float(np.linalg.norm(normal)) - 1.0), 1e-5)
                self.assertLess(abs(float(np.linalg.norm(binormal)) - 1.0), 1e-5)
                self.assertGreaterEqual(
                    float(binormal[2]),
                    -1e-8,
                    "section vertical axes must never point below world horizontal",
                )
                profile = np.asarray(sample.profile_points, dtype=float)
                roof = profile[int(np.argmax(profile[:, 1]))]
                floor = profile[int(np.argmin(profile[:, 1]))]
                roof_world_z = sample.z + roof[0] * normal[2] + roof[1] * binormal[2]
                floor_world_z = (
                    sample.z + floor[0] * normal[2] + floor[1] * binormal[2]
                )
                self.assertGreater(
                    float(roof_world_z),
                    float(floor_world_z),
                    "profile roof must remain above its floor in world space",
                )
                for influence in sample.junction_influences:
                    self.assertIn(influence.junction_id, junction_ids)
                    self.assertGreaterEqual(influence.weight, 0.08)
                surface_offsets.append(sample.surface_z - sample.z)
                all_samples.append(sample)

        self.assertTrue(sample_spacings)
        self.assertGreater(max(sample_spacings) - min(sample_spacings), 1.0)
        self.assertTrue(any(sample.junction_blend_weight > 0.35 for sample in all_samples))
        self.assertTrue(any(sample.junction_influences for sample in all_samples))
        self.assertTrue(
            any(
                len(segment_field.connected_junction_ids) > 0
                for segment_field in section_field.segment_fields
            )
        )
        widths = np.array([sample.tube_width for sample in all_samples], dtype=float)
        heights = np.array([sample.tube_height for sample in all_samples], dtype=float)
        self.assertGreater(float(np.mean(widths)), float(np.mean(heights)))
        passage_cap = project_config.world.body.maximum_passage_width_m
        room_cap = project_config.world.body.maximum_room_width_m
        non_chamber_widths = [
            sample.tube_width
            for sample in all_samples
            if not any(
                influence.kind == "chamber"
                for influence in sample.junction_influences
            )
        ]
        self.assertTrue(non_chamber_widths)
        self.assertLessEqual(max(non_chamber_widths), passage_cap)
        self.assertTrue(np.any(widths > passage_cap))
        self.assertLessEqual(
            float(np.max(widths)),
            room_cap,
        )
        self.assertGreater(float(np.mean(surface_offsets)), 6.0)

        segment_field_lookup = {
            segment_field.segment_id: segment_field for segment_field in section_field.segment_fields
        }
        route_fields = [
            segment_field_lookup[segment_id]
            for segment_id in section_field.dominant_route_segment_ids
            if segment_id in route_segment_ids
        ]
        continuity_angles: list[float] = []
        for current_field, next_field in zip(route_fields, route_fields[1:]):
            current_sample = current_field.samples[-1]
            next_sample = next_field.samples[0]
            dot = float(
                np.clip(
                    np.dot(current_sample.normal, next_sample.normal),
                    -1.0,
                    1.0,
                )
            )
            continuity_angles.append(math.degrees(math.acos(dot)))
        if continuity_angles:
            self.assertLess(float(np.median(continuity_angles)), 95.0)

        underpasses = [
            segment
            for segment in cave_network.segments
            if segment.kind == "underpass" and segment.z_level != 0
        ]
        if underpasses:
            flat_network = replace(
                cave_network,
                segments=tuple(
                    replace(segment, z_level=0)
                    if segment.kind == "underpass"
                    else segment
                    for segment in cave_network.segments
                ),
            )
            flat_sections = SectionFieldGenerator(
                project_config.section_field
            ).generate(flat_network)
            flat_lookup = {
                field.segment_id: field
                for field in flat_sections.segment_fields
            }
            physical_lookup = {
                field.segment_id: field
                for field in section_field.segment_fields
            }
            separations = []
            for segment in underpasses:
                physical = physical_lookup[segment.segment_id].samples
                flat = flat_lookup[segment.segment_id].samples
                midpoint = len(physical) // 2
                separations.append(abs(physical[midpoint].z - flat[midpoint].z))
                # The shared attachment is now the physical floor. Reframing
                # a graded tube can slightly change its center's height while
                # preserving that exact floor connection.
                for endpoint in (0, -1):
                    self.assertAlmostEqual(
                        SectionFieldGenerator._sample_floor(physical[endpoint]),
                        SectionFieldGenerator._sample_floor(flat[endpoint]),
                        places=6,
                    )
            self.assertGreater(
                max(separations),
                project_config.section_field.minimum_vertical_clearance,
            )

    def test_frame_cannot_be_inverted_by_previous_segment_orientation(self) -> None:
        generator = SectionFieldGenerator()

        normal, binormal = generator._build_frame(
            tangent=(1.0, 0.0, 0.0),
            previous_normal=(0.0, -1.0, 0.0),
        )

        self.assertTrue(np.allclose(normal, (0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(binormal, (0.0, 0.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
