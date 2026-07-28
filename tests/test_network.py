"""Smoke tests for the host-driven braided cave network."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import HostFieldConfig, HostFieldGenerator
from plume_advanced.stages.network import (
    CaveNetworkConfig,
    CaveNetworkGenerator,
    CaveNode,
    CavePoint,
    CaveSegment,
    export_network_report,
)


class CaveNetworkTests(unittest.TestCase):
    def test_flow_scaling_preserves_configured_small_passages(self) -> None:
        config = CaveNetworkConfig(
            source_flux=1.0,
            minimum_passage_radius=1.0,
            base_passage_radius=4.0,
            maximum_passage_radius=10.0,
        )
        nodes = [
            CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"),
            CaveNode(1, 10.0, 0.0, 10.0, 0.0, "exit"),
        ]
        points = tuple(
            CavePoint(
                index=index,
                x=float(index * 10),
                y=0.0,
                elevation=0.0,
                slope_degrees=0.0,
                cover_thickness=10.0,
                roof_competence=1.0,
                growth_cost=0.0,
                arc_length=float(index * 10),
                width=4.0,
            )
            for index in range(2)
        )
        segment = CaveSegment(
            segment_id=0,
            start_node_id=0,
            end_node_id=1,
            kind="backbone",
            z_level=0,
            points=points,
            metadata={},
        )

        resolved = CaveNetworkGenerator(config)._assign_conserved_flow(nodes, [segment])

        self.assertAlmostEqual(resolved[0].mean_width, 4.0)

    def test_default_config_generates_host_driven_braided_network(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        self.assertIsInstance(project_config.procedural_seed, int)
        self.assertIsInstance(project_config.host_field.seed_point, tuple)
        self.assertEqual(
            project_config.host_field.random_seed,
            project_config.stage_seeds.host,
        )
        self.assertEqual(
            project_config.network.random_seed,
            project_config.stage_seeds.network,
        )
        host_field = HostFieldGenerator(project_config.host_field).generate()
        cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)

        summary = cave_network.summary()
        self.assertGreaterEqual(int(summary["node_count"]), 12)
        self.assertGreaterEqual(int(summary["segment_count"]), 12)
        self.assertGreaterEqual(int(summary["junction_count"]), 1)
        self.assertGreaterEqual(int(summary["max_parallel_channels"]), 2)
        self.assertGreater(summary["total_length"], 1000.0)
        self.assertGreater(summary["dominant_route_length"], 800.0)
        self.assertGreater(summary["occupied_cell_count"], 50.0)
        self.assertLessEqual(
            summary["mean_segment_width"],
            project_config.world.body.maximum_passage_width_m,
        )
        self.assertGreater(summary["min_segment_width"], 0.0)
        self.assertGreaterEqual(summary["max_segment_width"], summary["mean_segment_width"])
        self.assertGreaterEqual(summary["max_visible_parallel_channels"], 2)
        self.assertGreater(summary["primary_branch_count"], 0)
        self.assertGreater(summary["mean_branch_persistence_widths"], 3.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = export_network_report(
                cave_network,
                Path(temp_dir) / "network.json",
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("plume.cave-network-diagnostics.v1", report)
            self.assertIn('"visible_channels"', report)

        entry_nodes = [node for node in cave_network.nodes if node.kind == "entry"]
        exit_nodes = [node for node in cave_network.nodes if node.kind == "exit"]
        self.assertGreaterEqual(len(entry_nodes), 2)
        self.assertLessEqual(len(entry_nodes), project_config.network.source_count)
        self.assertEqual(len(exit_nodes), 1)
        self.assertGreaterEqual(len(cave_network.dominant_route_node_ids), 2)
        self.assertGreater(summary["maximum_flux"], 0.0)
        self.assertGreater(summary["mean_temperature_k"], 273.15)
        self.assertLess(summary["max_flow_conservation_error"], 1e-8)
        self.assertTrue(
            all(
                point.flux >= 0.0
                and point.temperature_k >= 273.15
                and point.age_s >= 0.0
                for segment in cave_network.segments
                for point in segment.points
            )
        )

        occupied_cells = np.argwhere(cave_network.occupancy)
        y0, x0 = occupied_cells.min(axis=0)
        y1, x1 = occupied_cells.max(axis=0)
        bbox_width = float(host_field.x_coords[x1] - host_field.x_coords[x0])
        bbox_height = float(host_field.y_coords[y1] - host_field.y_coords[y0])
        self.assertGreater(
            max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1e-6),
            1.5,
        )

        segment_kinds = {segment.kind for segment in cave_network.segments}
        self.assertIn("source_feeder", segment_kinds)
        self.assertIn("backbone", segment_kinds)
        self.assertIn("island_bypass", segment_kinds)
        self.assertIn("chamber_braid", segment_kinds)
        self.assertIn("ladder", segment_kinds)
        self.assertIn("spur", segment_kinds)
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
        if underpasses:
            self.assertTrue(any(segment.z_level != 0 for segment in underpasses))
            self.assertTrue(
                all(segment.metadata["crossing_group_id"] is not None for segment in underpasses)
            )
            self.assertTrue(
                all(
                    segment.metadata["merge_behavior"] in {"cross_under", "cross_over"}
                    for segment in underpasses
                )
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
        self.assertGreater(float(np.mean(alignments)), 0.55)
        self.assertGreater(float(np.quantile(alignments, 0.5)), 0.50)
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
