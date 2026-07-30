"""Smoke tests for the voxel Stage-D geometry pipeline."""

import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


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
        self.assertAlmostEqual(
            summary["voxel_size_m"],
            project_config.geometry.voxel_size,
        )
        self.assertGreaterEqual(summary["characteristic_passage_samples"], 10.0)
        minimum_diameter_samples = (
            2.0
            * project_config.network.minimum_passage_radius
            / project_config.geometry.voxel_size
        )
        self.assertGreaterEqual(
            summary["minimum_section_width_samples"],
            minimum_diameter_samples,
        )
        self.assertGreater(summary["density_memory_mib"], 0.0)
        self.assertEqual(int(summary["voxel_component_count"]), 1)
        self.assertEqual(int(summary["component_count"]), 1)
        carved_ratio = summary["carved_voxel_count"] / summary["voxel_count"]
        self.assertGreater(carved_ratio, 0.0)
        self.assertLess(carved_ratio, 0.15)
        self.assertGreaterEqual(int(summary["vertex_count"]), 3)
        self.assertGreaterEqual(int(summary["face_count"]), 1)

        frames_by_segment = {}
        for frame in cave_geometry.surface_texture_frames:
            frames_by_segment.setdefault(frame.segment_id, []).append(frame)
            self.assertGreaterEqual(
                frame.binormal[2],
                -1e-7,
                "texture-frame vertical axes must point upward",
            )
            handed_binormal = np.cross(
                np.asarray(frame.tangent),
                np.asarray(frame.normal),
            )
            self.assertGreater(
                float(np.dot(handed_binormal, np.asarray(frame.binormal))),
                0.99,
                "texture frames must remain right-handed",
            )
        for frames in frames_by_segment.values():
            for first, second in zip(frames, frames[1:], strict=False):
                center_delta = np.asarray(second.center) - np.asarray(first.center)
                longitudinal_delta = (
                    second.longitudinal_m - first.longitudinal_m
                )
                if abs(longitudinal_delta) <= 1e-9:
                    continue
                tangent_progress = float(
                    np.dot(np.asarray(first.tangent), center_delta)
                )
                self.assertGreaterEqual(
                    tangent_progress * longitudinal_delta,
                    -1e-7,
                    "texture tangents must point toward increasing longitudinal UV",
                )

        if hasattr(cave_geometry.voxel_grid, "density"):
            density_tiles = (cave_geometry.voxel_grid.density,)
        else:
            density_tiles = tuple(cave_geometry.voxel_grid.tiles.values())
        self.assertTrue(density_tiles)
        self.assertTrue(all(tile.ndim == 3 for tile in density_tiles))
        self.assertTrue(all(np.isfinite(tile).all() for tile in density_tiles))
        self.assertGreaterEqual(
            max(float(tile.max()) for tile in density_tiles),
            project_config.geometry.iso_level,
        )
        self.assertLess(
            min(float(tile.min()) for tile in density_tiles),
            project_config.geometry.iso_level,
        )

        vertices = np.array(cave_geometry.assembled_vertices, dtype=float)
        self.assertTrue(np.isfinite(vertices).all())
        for face in cave_geometry.assembled_faces:
            self.assertEqual(len(face), 3)
            self.assertEqual(len(set(face)), 3)
            self.assertTrue(all(0 <= index < len(vertices) for index in face))
        edge_counts: Counter[tuple[int, int]] = Counter()
        directed_edges: Counter[tuple[int, int]] = Counter()
        for a, b, c in cave_geometry.assembled_faces:
            for first, second in ((a, b), (b, c), (c, a)):
                edge_counts[tuple(sorted((first, second)))] += 1
                directed_edges[(first, second)] += 1
        self.assertFalse(
            [edge for edge, count in edge_counts.items() if count == 1][:5],
            "assembled mesh should not contain open boundary edges",
        )
        self.assertFalse(
            [edge for edge, count in edge_counts.items() if count > 2][:5],
            "assembled mesh should not contain nonmanifold edges",
        )
        badly_oriented_edges = []
        for first, second in edge_counts:
            if directed_edges[(first, second)] != 1 or directed_edges[(second, first)] != 1:
                badly_oriented_edges.append((first, second))
                if len(badly_oriented_edges) >= 5:
                    break
        self.assertFalse(
            badly_oriented_edges,
            "shared edges should have opposite triangle winding",
        )

        for mesh in cave_geometry.chunk_meshes:
            self.assertTrue(mesh.vertices)
            self.assertTrue(mesh.faces)
            x_start, x_end, y_start, y_end, z_start, z_end = mesh.grid_bounds
            self.assertLessEqual(x_start, x_end)
            self.assertLessEqual(y_start, y_end)
            self.assertLessEqual(z_start, z_end)


if __name__ == "__main__":
    unittest.main()
