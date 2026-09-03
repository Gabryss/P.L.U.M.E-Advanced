"""Smoke tests for the host-driven braided cave network."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.stages.host_field import HostFieldConfig, HostFieldGenerator
from plume_advanced.stages.network import (
    CaveNetworkConfig,
    CaveNetworkGenerator,
    CaveNode,
    CavePoint,
    CaveSegment,
    _SelectedPath,
    export_network_report,
)


class CaveNetworkTests(unittest.TestCase):
    def test_simplification_retains_branch_attachment_and_separate_underpass(self) -> None:
        paths = (
            _SelectedPath("backbone", ((0, 0), (0, 10))),
            _SelectedPath("source_feeder", ((4, 5), (0, 5))),
            _SelectedPath("underpass", ((-3, 2), (0, 2), (3, 2)), merge_shared_cells=False),
        )
        restored = CaveNetworkGenerator._restore_attachment_cells(paths)
        self.assertIn((0, 5), restored[0].path)
        self.assertEqual(restored[2], paths[2])

    def test_routes_round_grid_corners_and_keep_exact_attachment_positions(self) -> None:
        substrate = SimpleNamespace(elevation=10., slope_degrees=0., cover_thickness=20.,
                                    roof_competence=1., growth_cost=0.)
        host = SimpleNamespace(sample=lambda x, y: substrate)
        coordinates = ((0., 0.), (100., 0.), (100., 100.))
        points = tuple(CavePoint(
            index=i, x=x, y=y, elevation=10., slope_degrees=0., cover_thickness=20.,
            roof_competence=1., growth_cost=0., arc_length=100. * i, width=10.,
        ) for i, (x, y) in enumerate(coordinates))
        segment = CaveSegment(0, 0, 1, "backbone", 0, points, {})
        curved = CaveNetworkGenerator._smooth_graph_routes(host, [segment])[0]
        xy = np.asarray([(point.x, point.y) for point in curved.points])
        np.testing.assert_array_equal(xy[[0, -1]], np.asarray(coordinates)[[0, -1]])
        directions = np.diff(xy, axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        turns = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
        self.assertLess(float(np.max(turns)), math.radians(20.))
        self.assertTrue(np.all(np.diff([point.arc_length for point in curved.points]) > 0.))

    def test_split_and_merge_conserve_flux_and_advance_thermal_state(self) -> None:
        config = CaveNetworkConfig(
            source_flux=10.0,
            source_temperature_k=1400.0,
            cooling_k_per_m=1.0,
            nominal_flow_speed_m_s=2.0,
            minimum_passage_radius=0.1,
            maximum_passage_radius=100.0,
        )
        nodes = [
            CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"),
            CaveNode(1, 10.0, 0.0, 10.0, 0.0, "junction"),
            CaveNode(2, 20.0, -2.0, 20.0, -2.0, "junction"),
            CaveNode(3, 20.0, 2.0, 20.0, 2.0, "junction"),
            CaveNode(4, 30.0, 0.0, 30.0, 0.0, "junction"),
            CaveNode(5, 40.0, 0.0, 40.0, 0.0, "exit"),
        ]

        def segment(
            segment_id: int,
            start: int,
            end: int,
            width: float,
        ) -> CaveSegment:
            return CaveSegment(
                segment_id=segment_id,
                start_node_id=start,
                end_node_id=end,
                kind="backbone",
                z_level=0,
                points=tuple(
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
                        width=width,
                    )
                    for index in range(2)
                ),
                metadata={},
            )

        resolved = CaveNetworkGenerator(config)._assign_conserved_flow(
            nodes,
            [
                segment(0, 0, 1, 4.0),
                segment(1, 1, 2, 4.0),
                segment(2, 1, 3, 2.0),
                segment(3, 2, 4, 4.0),
                segment(4, 3, 4, 2.0),
                segment(5, 4, 5, 4.0),
            ],
        )
        by_id = {item.segment_id: item for item in resolved}

        self.assertAlmostEqual(by_id[0].mean_flux, 10.0)
        self.assertAlmostEqual(by_id[1].mean_flux, 8.0)
        self.assertAlmostEqual(by_id[2].mean_flux, 2.0)
        self.assertAlmostEqual(by_id[1].mean_flux + by_id[2].mean_flux, 10.0)
        self.assertAlmostEqual(by_id[3].mean_flux + by_id[4].mean_flux, 10.0)
        self.assertAlmostEqual(by_id[5].mean_flux, 10.0)
        self.assertLess(by_id[5].points[0].temperature_k, config.source_temperature_k)
        self.assertGreater(by_id[5].points[0].age_s, 0.0)
        for item in resolved:
            self.assertLessEqual(item.points[-1].temperature_k, item.points[0].temperature_k)
            self.assertGreaterEqual(item.points[-1].age_s, item.points[0].age_s)

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

    def test_spur_direction_is_authoritative_when_it_bends_upstream(self) -> None:
        config = CaveNetworkConfig(source_flux=9.0)
        nodes = [
            CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"),
            CaveNode(1, 10.0, 0.0, 10.0, 0.0, "junction"),
            CaveNode(2, 5.0, 5.0, 5.0, 5.0, "spur_terminal"),
            CaveNode(3, 20.0, 0.0, 20.0, 0.0, "exit"),
        ]

        def segment(segment_id: int, start: int, end: int, kind: str) -> CaveSegment:
            start_node = nodes[start]
            end_node = nodes[end]
            points = tuple(
                CavePoint(
                    index=index,
                    x=node.x,
                    y=node.y,
                    elevation=0.0,
                    slope_degrees=0.0,
                    cover_thickness=10.0,
                    roof_competence=1.0,
                    growth_cost=0.0,
                    arc_length=float(index * 10),
                    width=4.0,
                )
                for index, node in enumerate((start_node, end_node))
            )
            return CaveSegment(segment_id, start, end, kind, 0, points, {})

        resolved = CaveNetworkGenerator(config)._assign_conserved_flow(
            nodes,
            [
                segment(0, 0, 1, "backbone"),
                segment(1, 1, 2, "spur"),
                segment(2, 1, 3, "backbone"),
            ],
        )

        by_id = {item.segment_id: item for item in resolved}
        self.assertGreater(by_id[1].mean_flux, 0.0)
        self.assertGreater(by_id[2].mean_flux, 0.0)
        self.assertAlmostEqual(
            by_id[1].mean_flux + by_id[2].mean_flux,
            by_id[0].mean_flux,
        )

    def test_regression_seeds_produce_connected_positive_flow_graphs(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()

        for seed in (3, 5, 7, 9):
            with self.subTest(seed=seed):
                config = replace(project_config.network, random_seed=seed)
                cave_network = CaveNetworkGenerator(config).generate(host_field)
                metrics = network_metrics(cave_network)
                self.assertEqual(metrics["connected_component_count"], 1)
                self.assertEqual(metrics["source_unreachable_node_count"], 0)
                self.assertEqual(metrics["entries_without_exit_path_count"], 0)
                self.assertEqual(metrics["zero_flux_segment_count"], 0)
                self.assertLess(cave_network.max_flow_conservation_error(), 1e-8)

                counts = np.asarray(cave_network.slice_channel_counts)
                occupied_slices = np.flatnonzero(counts > 0)
                self.assertGreater(occupied_slices.size, 0)
                first, last = occupied_slices[[0, -1]]
                self.assertTrue(np.all(counts[first : last + 1] > 0))

        # Seed 12 previously selected a non-connectable underpass interior as
        # a spur anchor under the generic/default configuration.
        generic_host = HostFieldGenerator().generate()
        generic_network = CaveNetworkGenerator(
            replace(CaveNetworkConfig(), random_seed=12)
        ).generate(generic_host)
        generic_metrics = network_metrics(generic_network)
        self.assertEqual(generic_metrics["connected_component_count"], 1)
        self.assertEqual(generic_metrics["source_unreachable_node_count"], 0)
        self.assertEqual(generic_metrics["zero_flux_segment_count"], 0)

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

        adjacency = {node.node_id: set() for node in cave_network.nodes}
        for segment in cave_network.segments:
            adjacency[segment.start_node_id].add(segment.end_node_id)
            adjacency[segment.end_node_id].add(segment.start_node_id)
        seen = set()
        pending = [cave_network.nodes[0].node_id]
        while pending:
            node_id = pending.pop()
            if node_id not in seen:
                seen.add(node_id)
                pending.extend(adjacency[node_id] - seen)
        self.assertEqual(seen, set(adjacency), "Every branch must attach to the route graph")

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
        metrics = network_metrics(cave_network)
        self.assertEqual(metrics["connected_component_count"], 1)
        self.assertEqual(metrics["source_unreachable_node_count"], 0)
        self.assertEqual(metrics["entries_without_exit_path_count"], 0)
        self.assertEqual(metrics["zero_flux_segment_count"], 0)
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
