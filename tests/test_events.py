"""Smoke tests for the Stage-E geological event layer."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.events import GeologicalEventGenerator
from stages.geometry import GeometryGenerator
from stages.host_field import HostFieldGenerator
from stages.network import CaveNetworkGenerator
from stages.section_field import SectionFieldGenerator


class GeologicalEventTests(unittest.TestCase):
    def test_event_stage_places_seeded_mesh_events(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
        section_field = SectionFieldGenerator(project_config.section_field).generate(
            cave_network
        )

        event_field = GeologicalEventGenerator(project_config.events).generate(section_field)
        repeated_event_field = GeologicalEventGenerator(project_config.events).generate(
            section_field
        )

        self.assertEqual(event_field.events, repeated_event_field.events)
        summary = event_field.summary()
        self.assertGreater(int(summary["event_count"]), 0)
        self.assertGreater(int(summary["rock_count"]), 0)
        self.assertGreater(int(summary["boulder_count"]), 0)
        self.assertGreater(int(summary["collapse_count"]), 0)
        self.assertGreater(int(summary["choke_count"]), 0)
        self.assertGreater(int(summary["infill_count"]), 0)
        self.assertEqual(int(summary["event_mesh_count"]), int(summary["event_count"]))
        self.assertGreater(int(summary["event_mesh_vertex_count"]), 0)
        self.assertGreater(int(summary["event_mesh_face_count"]), 0)

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
        for mesh in event_field.meshes:
            self.assertTrue(mesh.vertices)
            self.assertTrue(mesh.faces)
            self.assertIn(mesh.kind, {"rock", "boulder", "collapse", "choke", "infill"})

    def test_geometry_consumes_event_field(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
        section_field = SectionFieldGenerator(project_config.section_field).generate(
            cave_network
        )
        event_field = GeologicalEventGenerator(project_config.events).generate(section_field)
        cave_geometry = GeometryGenerator(project_config.geometry).generate(
            cave_network,
            section_field,
            event_field,
        )

        summary = cave_geometry.summary()
        self.assertGreater(int(summary["carved_voxel_count"]), 0)
        self.assertGreaterEqual(int(summary["vertex_count"]), 3)
        self.assertGreaterEqual(int(summary["face_count"]), 1)
        self.assertEqual(int(summary["event_mesh_count"]), len(event_field.meshes))
        self.assertGreater(int(summary["event_vertex_count"]), 0)
        self.assertGreater(int(summary["event_face_count"]), 0)
        self.assertEqual(int(summary["voxel_component_count"]), 1)
        self.assertEqual(int(summary["component_count"]), 1)


if __name__ == "__main__":
    unittest.main()
