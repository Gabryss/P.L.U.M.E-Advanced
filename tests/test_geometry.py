"""Smoke tests for the constrained stage-D geometry sweep."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.geometry import GeometryGenerator
from stages.host_field import HostFieldGenerator
from stages.network import CaveNetworkGenerator
from stages.section_field import SectionFieldGenerator


class GeometryTests(unittest.TestCase):
    def test_geometry_stage_sweeps_only_low_risk_segment_spans(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
        section_field = SectionFieldGenerator(project_config.section_field).generate(
            cave_network
        )
        cave_geometry = GeometryGenerator(project_config.geometry).generate(
            cave_network,
            section_field,
        )

        summary = cave_geometry.summary()
        self.assertGreaterEqual(int(summary["mesh_count"]), 1)
        self.assertGreaterEqual(int(summary["swept_segment_count"]), 1)
        self.assertGreaterEqual(int(summary["junction_patch_count"]), 1)
        self.assertEqual(int(summary["skylight_count"]), 1)
        self.assertEqual(int(summary["component_count"]), 1)
        self.assertGreaterEqual(int(summary["vertex_count"]), 3)
        self.assertGreaterEqual(int(summary["face_count"]), 2)
        self.assertGreaterEqual(int(summary["excluded_segment_count"]), 1)
        self.assertTrue(cave_geometry.assembled_vertices)
        self.assertTrue(cave_geometry.assembled_faces)

        segment_lookup = {segment.segment_id: segment for segment in cave_network.segments}
        field_lookup = {
            segment_field.segment_id: segment_field for segment_field in section_field.segment_fields
        }
        all_samples = [
            sample
            for segment_field in section_field.segment_fields
            for sample in segment_field.samples
            if not any(influence.kind == "crossing" for influence in sample.junction_influences)
        ]
        highest_sample = max(all_samples, key=lambda sample: sample.z)

        for mesh in cave_geometry.meshes:
            self.assertGreaterEqual(mesh.ring_vertex_count, 8)
            self.assertGreaterEqual(len(mesh.sample_indices), project_config.geometry.minimum_span_samples)
            self.assertEqual(len(mesh.vertices) % mesh.ring_vertex_count, 0)
            self.assertTrue(mesh.faces)

            max_vertex_index = len(mesh.vertices) - 1
            for face in mesh.faces:
                self.assertEqual(len(face), 3)
                self.assertTrue(all(0 <= index <= max_vertex_index for index in face))

            segment = segment_lookup[mesh.segment_id]
            self.assertNotEqual(segment.kind, "underpass")
            self.assertEqual(segment.z_level, 0)
            segment_field = field_lookup[mesh.segment_id]
            sample_lookup = {sample.index: sample for sample in segment_field.samples}
            span_samples = [sample_lookup[index] for index in mesh.sample_indices]
            if not mesh.is_connector:
                self.assertTrue(
                    all(
                        sample.junction_blend_weight
                        < project_config.geometry.junction_exclusion_weight
                        for sample in span_samples
                    )
                )
                self.assertTrue(
                    all(
                        all(influence.kind != "crossing" for influence in sample.junction_influences)
                        for sample in span_samples
                    )
                )

            vertex_array = np.array(mesh.vertices, dtype=float)
            self.assertTrue(np.isfinite(vertex_array).all())

        excluded_segments = {
            segment_lookup[segment_id].kind for segment_id in cave_geometry.excluded_segment_ids
        }
        self.assertIn("underpass", excluded_segments)

        self.assertTrue(cave_geometry.junction_patches)
        for patch in cave_geometry.junction_patches:
            self.assertGreaterEqual(len(patch.segment_ids), project_config.geometry.junction_min_segments)
            self.assertGreaterEqual(patch.ring_vertex_count, 8)
            self.assertTrue(patch.vertices)
            self.assertTrue(patch.faces)
            max_vertex_index = len(patch.vertices) - 1
            for face in patch.faces:
                self.assertEqual(len(face), 3)
                self.assertTrue(all(0 <= index <= max_vertex_index for index in face))
            self.assertTrue(np.isfinite(np.array(patch.vertices, dtype=float)).all())
            if patch.junction_kind == "crossing":
                self.assertTrue(
                    all(
                        segment_lookup[segment_id].kind != "underpass"
                        and segment_lookup[segment_id].z_level == 0
                        for segment_id in patch.segment_ids
                    )
                )

        self.assertTrue(cave_geometry.skylights)
        skylight = cave_geometry.skylights[0]
        self.assertEqual(skylight.anchor_segment_id, highest_sample.segment_id)
        self.assertEqual(skylight.anchor_sample_index, highest_sample.index)
        self.assertGreaterEqual(skylight.ring_vertex_count, 10)
        self.assertTrue(skylight.vertices)
        self.assertTrue(skylight.faces)
        self.assertIn(skylight.anchor_vertex, cave_geometry.assembled_vertices)
        self.assertGreater(skylight.top_center[2], highest_sample.surface_z)
        skylight_vertices = np.array(skylight.vertices, dtype=float)
        self.assertTrue(np.isfinite(skylight_vertices).all())
        top_ring = skylight_vertices[-skylight.ring_vertex_count :]
        top_center_xy = np.array(skylight.top_center[:2], dtype=float)
        top_radii = np.linalg.norm(top_ring[:, :2] - top_center_xy, axis=1)
        self.assertGreater(float(np.max(top_radii) - np.min(top_radii)), 0.2)


if __name__ == "__main__":
    unittest.main()
