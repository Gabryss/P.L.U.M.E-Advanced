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
from stages import CaveNetworkGenerator, HostFieldGenerator
from stages.host_field import HostFieldConfig


class CaveNetworkTests(unittest.TestCase):
    def test_default_config_generates_host_driven_braided_network(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)

        summary = cave_network.summary()
        self.assertGreaterEqual(int(summary["node_count"]), 200)
        self.assertGreaterEqual(int(summary["segment_count"]), 220)
        self.assertGreaterEqual(int(summary["loop_count"]), 20)
        self.assertGreaterEqual(int(summary["max_parallel_channels"]), 3)
        self.assertGreater(summary["dominant_route_length"], 1500.0)
        self.assertGreater(summary["occupied_cell_count"], 1000.0)

        entry_nodes = [node for node in cave_network.nodes if node.kind == "entry"]
        exit_nodes = [node for node in cave_network.nodes if node.kind == "exit"]
        self.assertEqual(len(entry_nodes), 1)
        self.assertEqual(len(exit_nodes), 1)
        self.assertGreater(len(cave_network.dominant_route_node_ids), 2)

        segment_kinds = {segment.kind for segment in cave_network.segments}
        self.assertIn("braid", segment_kinds)
        self.assertIn("spur", segment_kinds)

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
        self.assertGreater(float(np.mean(alignments)), 0.55)
        self.assertGreater(float(np.quantile(alignments, 0.5)), 0.55)
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
        self.assertLess(float(overlap / union), 0.45)


if __name__ == "__main__":
    unittest.main()
