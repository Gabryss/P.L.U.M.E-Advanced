"""Smoke tests for the Stage-E geological event layer."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.stages.events import GeologicalEventGenerator
from plume_advanced.stages.floor_map import FloorMapGenerator
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


class GeologicalEventTests(unittest.TestCase):
    def test_event_stage_places_seeded_mesh_events(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
        section_field = SectionFieldGenerator(project_config.section_field).generate(
            cave_network
        )
        base_geometry = GeometryGenerator(
            project_config.geometry
        ).build_base_volume(cave_network, section_field)
        floor_atlas = FloorMapGenerator(project_config.floor_map).generate(
            cave_network,
            section_field,
            base_geometry,
        )

        event_field = GeologicalEventGenerator(project_config.events).generate(
            section_field,
            base_geometry,
            floor_atlas,
        )
        repeated_event_field = GeologicalEventGenerator(project_config.events).generate(
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
            int(
                summary["collapse_count"]
                + summary["choke_count"]
                + summary["infill_count"]
            ),
        )
        self.assertEqual(
            int(summary["grounded_prop_count"]),
            int(summary["prop_count"]),
        )
        self.assertGreater(int(summary["clustered_prop_count"]), 0)
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
            event = event_lookup[mesh.event_id]
            contact = event.contact_point
            contact_normal = event.contact_normal
            signed_contact_distances = [
                sum(
                    (vertex[axis] - contact[axis]) * contact_normal[axis]
                    for axis in range(3)
                )
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
        section_field = SectionFieldGenerator(project_config.section_field).generate(
            cave_network
        )
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
        event_field = GeologicalEventGenerator(project_config.events).generate(
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
            event.floor_cell_id
            for event in event_field.events
            if event.kind in {"rock", "boulder"}
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
