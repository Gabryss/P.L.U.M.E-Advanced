"""Smoke tests for the host-driven braided cave network."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.host_field import HostFieldConfig, HostFieldGenerator
from stages.network import CaveNetworkGenerator


class CaveNetworkTests(unittest.TestCase):
    def test_default_config_generates_host_driven_braided_network(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        self.assertEqual(project_config.procedural_seed, 17)
        self.assertIsInstance(project_config.host_field.seed_point, tuple)
        self.assertEqual(project_config.host_field.random_seed, project_config.procedural_seed)
        self.assertEqual(project_config.network.random_seed, project_config.procedural_seed)
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)

        summary = cave_network.summary()
        self.assertGreaterEqual(int(summary["node_count"]), 24)
        self.assertGreaterEqual(int(summary["segment_count"]), 22)
        self.assertGreaterEqual(int(summary["junction_count"]), 3)
        self.assertGreaterEqual(int(summary["loop_count"]), 1)
        self.assertGreaterEqual(int(summary["max_parallel_channels"]), 3)
        self.assertGreater(summary["dominant_route_length"], 5000.0)
        self.assertGreater(summary["occupied_cell_count"], 700.0)
        self.assertLessEqual(summary["mean_segment_width"], 12.0)

        entry_nodes = [node for node in cave_network.nodes if node.kind == "entry"]
        exit_nodes = [node for node in cave_network.nodes if node.kind == "exit"]
        self.assertEqual(len(entry_nodes), 1)
        self.assertEqual(len(exit_nodes), 1)
        self.assertGreater(len(cave_network.dominant_route_node_ids), 2)

        occupied_cells = np.argwhere(cave_network.occupancy)
        y0, x0 = occupied_cells.min(axis=0)
        y1, x1 = occupied_cells.max(axis=0)
        bbox_width = float(host_field.x_coords[x1] - host_field.x_coords[x0])
        bbox_height = float(host_field.y_coords[y1] - host_field.y_coords[y0])
        self.assertGreater(max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1e-6), 6.0)

        segment_kinds = {segment.kind for segment in cave_network.segments}
        self.assertIn("backbone", segment_kinds)
        self.assertIn("island_bypass", segment_kinds)
        self.assertIn("chamber_braid", segment_kinds)
        self.assertIn("ladder", segment_kinds)
        self.assertIn("spur", segment_kinds)
        self.assertIn("underpass", segment_kinds)
        self.assertTrue(any(segment.z_level != 0 for segment in cave_network.segments))
        self.assertIn("chamber", {node.kind for node in cave_network.nodes})
        self.assertIn("spur_terminal", {node.kind for node in cave_network.nodes})
        self.assertTrue(cave_network.junctions)
        self.assertTrue(any(junction.kind == "chamber" for junction in cave_network.junctions))
        self.assertTrue(
            any(
                junction.split_style == "pre_widen_then_split"
                for junction in cave_network.junctions
            )
        )

        for segment in cave_network.segments:
            self.assertIn("crossing_group_id", segment.metadata)
            self.assertIn("merge_behavior", segment.metadata)
            self.assertIn("island_id", segment.metadata)
            self.assertIn("chamber_id", segment.metadata)
            self.assertIn("formation_origin", segment.metadata)
            self.assertEqual(segment.metadata["formation_origin"], segment.kind)

        underpasses = [segment for segment in cave_network.segments if segment.kind == "underpass"]
        self.assertTrue(underpasses)
        self.assertTrue(all(segment.metadata["crossing_group_id"] is not None for segment in underpasses))
        self.assertTrue(
            all(segment.metadata["merge_behavior"] in {"cross_under", "cross_over"} for segment in underpasses)
        )

        island_segments = [segment for segment in cave_network.segments if segment.kind == "island_bypass"]
        self.assertTrue(island_segments)
        self.assertTrue(all(segment.metadata["island_id"] is not None for segment in island_segments))

        chamber_segments = [
            segment
            for segment in cave_network.segments
            if segment.kind in {"chamber_braid", "ladder"}
        ]
        self.assertTrue(chamber_segments)
        self.assertTrue(all(segment.metadata["chamber_id"] is not None for segment in chamber_segments))

        flow_angle = math.radians(project_config.host_field.flow_angle_degrees)
        flow_direction = (math.cos(flow_angle), math.sin(flow_angle))
        alignments: list[float] = []
        elevation_drops: list[float] = []
        for segment in cave_network.segments:
            if len(segment.points) < 2:
                continue
            elevation_drops.append(segment.points[0].elevation - segment.points[-1].elevation)
            for start, end in zip(segment.points, segment.points[1:]):
                dx = end.x - start.x
                dy = end.y - start.y
                length = math.hypot(dx, dy)
                if math.isclose(length, 0.0):
                    continue
                alignments.append((dx / length) * flow_direction[0] + (dy / length) * flow_direction[1])

        self.assertTrue(alignments)
        self.assertGreater(float(np.mean(alignments)), 0.7)
        self.assertGreater(float(np.quantile(alignments, 0.5)), 0.65)
        self.assertTrue(elevation_drops)
        self.assertGreater(float(np.quantile(elevation_drops, 0.5)), 0.0)

        stripped_host_config = HostFieldConfig(
            **{
                **project_config.host_field.__dict__,
                "waves": (),
                "corridor_depth": 0.0,
                "roof_competence_variation": 0.0,
            }
        )
        stripped_host = HostFieldGenerator(stripped_host_config).generate()
        stripped_network = CaveNetworkGenerator(project_config.network).generate(stripped_host)
        overlap = np.logical_and(cave_network.occupancy, stripped_network.occupancy).sum()
        union = np.logical_or(cave_network.occupancy, stripped_network.occupancy).sum()
        self.assertGreater(union, 0)
        self.assertLess(float(overlap / union), 0.35)


if __name__ == "__main__":
    unittest.main()
