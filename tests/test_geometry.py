"""Smoke tests for the voxel Stage-D geometry pipeline."""

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
    def test_geometry_stage_stamps_voxels_and_builds_isosurface_mesh(self) -> None:
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
        self.assertGreaterEqual(int(summary["chunk_mesh_count"]), 1)
        self.assertEqual(
            int(summary["stamped_segment_count"]),
            len(section_field.segment_fields),
        )
        self.assertGreater(int(summary["stamped_sample_count"]), 0)
        self.assertGreater(int(summary["voxel_count"]), 0)
        self.assertGreater(int(summary["carved_voxel_count"]), 0)
        self.assertEqual(int(summary["voxel_component_count"]), 1)
        self.assertEqual(int(summary["component_count"]), 1)
        self.assertLess(int(summary["carved_voxel_count"]), 10_000)
        self.assertGreaterEqual(int(summary["vertex_count"]), 3)
        self.assertGreaterEqual(int(summary["face_count"]), 1)

        density = cave_geometry.voxel_grid.density
        self.assertEqual(density.ndim, 3)
        self.assertTrue(np.isfinite(density).all())
        self.assertGreaterEqual(float(density.max()), project_config.geometry.iso_level)
        self.assertLess(float(density.min()), project_config.geometry.iso_level)

        vertices = np.array(cave_geometry.assembled_vertices, dtype=float)
        self.assertTrue(np.isfinite(vertices).all())
        for face in cave_geometry.assembled_faces:
            self.assertEqual(len(face), 3)
            self.assertEqual(len(set(face)), 3)
            self.assertTrue(all(0 <= index < len(vertices) for index in face))

        for mesh in cave_geometry.chunk_meshes:
            self.assertTrue(mesh.vertices)
            self.assertTrue(mesh.faces)
            x_start, x_end, y_start, y_end, z_start, z_end = mesh.grid_bounds
            self.assertLessEqual(x_start, x_end)
            self.assertLessEqual(y_start, y_end)
            self.assertLessEqual(z_start, z_end)


if __name__ == "__main__":
    unittest.main()
