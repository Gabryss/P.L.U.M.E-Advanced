"""Smoke tests for the stage-C section field."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.host_field import HostFieldGenerator
from stages.network import CaveNetworkGenerator
from stages.section_field import SectionFieldGenerator


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
        self.assertGreater(float(np.mean(widths[widths <= 12.0])), 8.0)
        self.assertTrue(np.any(widths > 12.0))
        self.assertLessEqual(
            float(np.max(widths)),
            project_config.section_field.chamber_max_tube_width,
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


if __name__ == "__main__":
    unittest.main()
