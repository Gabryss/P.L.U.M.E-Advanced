"""Smoke tests for the host-driven braided cave network."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.procedural import procedural_rng
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator
from plume_advanced.stages.network import (
    CaveNetworkConfig,
    CaveNetworkGenerator,
    CaveNode,
    CavePoint,
    CaveSegment,
    DownflowReferenceBackend,
    EmplacementProposal,
    FlowyBackend,
    _SelectedPath,
    export_network_report,
)


class CaveNetworkTests(unittest.TestCase):
    def test_downflow_reference_is_deterministic_and_terrain_sensitive(self) -> None:
        config = HostFieldConfig(
            grid=GridConfig(width=360.0, height=720.0, nx=36, ny=60),
            target_route_length_m=600.0,
            seed_point=(0.0, -300.0),
            flow_angle_degrees=90.0,
            random_seed=7,
        )
        host = HostFieldGenerator(config).generate()
        generator = CaveNetworkGenerator(
            CaveNetworkConfig(random_seed=13, target_route_length_m=600.0)
        )
        geometry = generator._build_flow_geometry(host)
        start = generator._world_to_cell(host, *config.seed_point)
        backend = DownflowReferenceBackend()
        first = backend.propose(
            host, geometry, start_cell=start, seed=13, steps=100, uphill_limit=1.2
        )
        repeated = backend.propose(
            host, geometry, start_cell=start, seed=13, steps=100, uphill_limit=1.2
        )
        self.assertEqual(first, repeated)
        altered = replace(host, elevation=host.elevation + 0.35 * host.y_coords[:, None])
        altered_proposal = backend.propose(
            altered, geometry, start_cell=start, seed=13, steps=100, uphill_limit=1.2
        )
        self.assertNotEqual(first.paths, altered_proposal.paths)
        changed_seed = backend.propose(
            host, geometry, start_cell=start, seed=14, steps=100, uphill_limit=1.2
        )
        self.assertNotEqual(first.paths, changed_seed.paths)
        self.assertEqual(first.backend, "downflow_reference")
        self.assertFalse(first.provenance["official_library"])
        self.assertEqual(first.provenance["ensemble_size"], 8)

    def test_flowy_adapter_requires_executable_and_reads_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "fake_flowy.py"
            script.write_text(
                """import pathlib, sys
out = pathlib.Path(sys.argv[-1]); out.mkdir(parents=True, exist_ok=True)
(out / "plume_backend_thickness_full.asc").write_text(
    "ncols 2\\nnrows 2\\nxllcorner 0\\nyllcorner 0\\ncellsize 1\\nNODATA_value -9999\\n1 1\\n1 1\\n"
)
(out / "lobes_0.csv").write_text(
    "centerx,centery,idx_parent,n_descendents,dist_n_lobes\\n0,0,-1,2,1\\n1,1,0,1,2\\n"
)
""",
                encoding="utf-8",
            )
            host = HostFieldGenerator(
                HostFieldConfig(grid=GridConfig(width=20.0, height=20.0, nx=8, ny=8))
            ).generate()
            generator = CaveNetworkGenerator(CaveNetworkConfig())
            geometry = generator._build_flow_geometry(host)
            proposal = FlowyBackend(str(script)).propose(
                host,
                geometry,
                start_cell=(0, 0),
                seed=4,
                steps=4,
                uphill_limit=1.2,
            )
            self.assertEqual(proposal.backend, "flowy")
            self.assertEqual(proposal.version, "external")
            self.assertEqual(len(proposal.paths), 1)

    def test_flowy_backend_does_not_silently_fallback_when_missing(self) -> None:
        host = HostFieldGenerator(
            HostFieldConfig(grid=GridConfig(width=20.0, height=20.0, nx=8, ny=8))
        ).generate()
        generator = CaveNetworkGenerator(CaveNetworkConfig())
        geometry = generator._build_flow_geometry(host)
        with self.assertRaises(FileNotFoundError):
            FlowyBackend("/definitely/missing/flowy").propose(
                host,
                geometry,
                start_cell=(0, 0),
                seed=4,
                steps=4,
                uphill_limit=1.2,
            )

    def test_proposal_rasterization_erases_chronological_loops(self) -> None:
        host = HostFieldGenerator(
            HostFieldConfig(grid=GridConfig(width=60.0, height=20.0, nx=7, ny=3))
        ).generate()
        generator = CaveNetworkGenerator(CaveNetworkConfig())
        cells = ((1, 1), (1, 2), (1, 3), (1, 2), (1, 4))
        path = tuple(generator._cell_to_world(host, cell) for cell in cells)

        rasterized = generator._proposal_path_to_cells(host, path)

        self.assertEqual(rasterized, [(1, 1), (1, 2), (1, 4)])

    def test_external_backbone_requires_material_downstream_progress(self) -> None:
        host = HostFieldGenerator(
            HostFieldConfig(
                grid=GridConfig(width=600.0, height=200.0, nx=61, ny=21),
                seed_point=(-250.0, 0.0),
                target_route_length_m=500.0,
                waves=(),
                corridor_depth=0.0,
            )
        ).generate()
        generator = CaveNetworkGenerator(
            CaveNetworkConfig(
                emplacement_backend="downflow_reference",
                target_route_length_m=500.0,
                network_density=0.0,
            )
        )
        start = generator._world_to_cell(host, -250.0, 0.0)
        cells = (start, (start[0], start[1] + 1), (start[0], start[1] + 2))
        proposal = EmplacementProposal(
            paths=(tuple(generator._cell_to_world(host, cell) for cell in cells),),
            backend="downflow_reference",
            version="test",
            provenance={},
        )

        with patch.object(generator, "_emplacement_proposal", return_value=proposal):
            with self.assertRaisesRegex(ValueError, "downstream progress"):
                generator.generate(host)

    def test_no_argument_generators_preserve_entry_to_exit_reachability(self) -> None:
        host_field = HostFieldGenerator().generate()
        network = CaveNetworkGenerator().generate(host_field)
        metrics = network_metrics(network)
        self.assertEqual(metrics["connected_component_count"], 1)
        self.assertEqual(metrics["source_unreachable_node_count"], 0)
        self.assertEqual(metrics["entries_without_exit_path_count"], 0)
        self.assertEqual(metrics["zero_flux_segment_count"], 0)
        self.assertEqual(network.backend_provenance["proposal_path_count"], 1)
        self.assertGreater(network.backend_provenance["proposal_cell_count"], 2)
        self.assertGreater(network.backend_provenance["proposal_downstream_progress_m"], 0.0)

    def test_downflow_perturbation_is_seeded_and_spatially_correlated(self) -> None:
        first = CaveNetworkGenerator._correlated_terrain_perturbation(
            (64, 64),
            amplitude_m=2.5,
            correlation_cells=4.0,
            rng=procedural_rng(19, "test-downflow"),
        )
        repeated = CaveNetworkGenerator._correlated_terrain_perturbation(
            (64, 64),
            amplitude_m=2.5,
            correlation_cells=4.0,
            rng=procedural_rng(19, "test-downflow"),
        )
        changed = CaveNetworkGenerator._correlated_terrain_perturbation(
            (64, 64),
            amplitude_m=2.5,
            correlation_cells=4.0,
            rng=procedural_rng(20, "test-downflow"),
        )

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, changed))
        self.assertAlmostEqual(float(np.std(first)), 2.5)
        self.assertLess(float(np.std(np.diff(first, axis=0))), float(np.std(first)))
        self.assertLess(float(np.std(np.diff(first, axis=1))), float(np.std(first)))


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
        substrate = SimpleNamespace(
            elevation=10.0,
            slope_degrees=0.0,
            cover_thickness=20.0,
            roof_competence=1.0,
            growth_cost=0.0,
        )
        host = SimpleNamespace(sample=lambda x, y: substrate)
        coordinates = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0))
        points = tuple(
            CavePoint(
                index=i,
                x=x,
                y=y,
                elevation=10.0,
                slope_degrees=0.0,
                cover_thickness=20.0,
                roof_competence=1.0,
                growth_cost=0.0,
                arc_length=100.0 * i,
                width=10.0,
            )
            for i, (x, y) in enumerate(coordinates)
        )
        segment = CaveSegment(0, 0, 1, "backbone", 0, points, {})
        curved = CaveNetworkGenerator._smooth_graph_routes(host, [segment])[0]
        xy = np.asarray([(point.x, point.y) for point in curved.points])
        np.testing.assert_array_equal(xy[[0, -1]], np.asarray(coordinates)[[0, -1]])
        directions = np.diff(xy, axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        turns = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
        self.assertLess(float(np.max(turns)), math.radians(20.0))
        self.assertTrue(np.all(np.diff([point.arc_length for point in curved.points]) > 0.0))

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

    def test_seeded_inlet_strengths_vary_but_preserve_total_supply(self) -> None:
        config = CaveNetworkConfig(random_seed=31, source_flux=5.0)
        nodes = [
            CaveNode(0, 0.0, -5.0, 0.0, -5.0, "entry"),
            CaveNode(1, 0.0, 5.0, 0.0, 5.0, "entry"),
            CaveNode(2, 10.0, 0.0, 10.0, 0.0, "junction"),
            CaveNode(3, 20.0, 0.0, 20.0, 0.0, "exit"),
        ]

        def segment(segment_id: int, start: int, end: int, width: float) -> CaveSegment:
            return CaveSegment(
                segment_id,
                start,
                end,
                "source_feeder" if start in {0, 1} else "backbone",
                0,
                tuple(
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
                {},
            )

        resolved = CaveNetworkGenerator(config)._assign_conserved_flow(
            nodes,
            [
                segment(0, 0, 2, 3.0),
                segment(1, 1, 2, 6.0),
                segment(2, 2, 3, 6.0),
            ],
        )
        by_id = {item.segment_id: item for item in resolved}
        self.assertNotAlmostEqual(by_id[0].mean_flux, by_id[1].mean_flux)
        self.assertAlmostEqual(by_id[0].mean_flux + by_id[1].mean_flux, 10.0)
        self.assertAlmostEqual(by_id[2].mean_flux, 10.0)

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

    def test_network_density_controls_lobes_anastomoses_and_loops(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        sparse = CaveNetworkGenerator(
            replace(project_config.network, network_density=0.5)
        ).generate(host_field)
        dense = CaveNetworkGenerator(replace(project_config.network, network_density=1.5)).generate(
            host_field
        )

        sparse_summary = sparse.summary()
        dense_summary = dense.summary()
        self.assertLess(
            sparse_summary["lobe_path_count"],
            dense_summary["lobe_path_count"],
        )
        self.assertLess(
            sparse_summary["anastomosis_count"],
            dense_summary["anastomosis_count"],
        )
        self.assertLess(sparse_summary["loop_count"], dense_summary["loop_count"])

    def test_density_zero_is_single_backbone_endmember(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        network = CaveNetworkGenerator(
            replace(project_config.network, network_density=0.0, random_seed=17)
        ).generate(host_field)
        self.assertTrue(network.segments)
        self.assertEqual({segment.kind for segment in network.segments}, {"backbone"})
        self.assertEqual(sum(node.kind == "entry" for node in network.nodes), 1)
        self.assertEqual(network.summary()["loop_count"], 0.0)
        self.assertLess(network.max_flow_conservation_error(), 1e-8)

    def test_backbone_curvature_controls_are_seeded(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        config = replace(
            project_config.network,
            network_density=0.0,
            random_seed=23,
            lobe_growth=replace(
                project_config.network.lobe_growth,
                backbone_curvature_fraction=0.20,
            ),
        )
        first = CaveNetworkGenerator(config).generate(host_field)
        repeated = CaveNetworkGenerator(config).generate(host_field)
        first_xy = [[(point.x, point.y) for point in segment.points] for segment in first.segments]
        repeated_xy = [
            [(point.x, point.y) for point in segment.points] for segment in repeated.segments
        ]
        self.assertEqual(first_xy, repeated_xy)
        self.assertGreater(
            network_metrics(first)["length_weighted_mean_sinuosity"],
            1.01,
        )

    def test_capture_probability_blocks_cross_level_merges(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        network = CaveNetworkGenerator(
            replace(project_config.network, network_density=3.0, capture_probability=0.0)
        ).generate(host_field)
        self.assertFalse(
            any(
                segment.z_level != 0
                and (
                    segment.kind == "anastomosis"
                    or bool(segment.metadata.get("vertical_capture", False))
                )
                for segment in network.segments
            )
        )

    def test_sustained_uphill_labels_are_process_scoped(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        network = CaveNetworkGenerator(project_config.network).generate(host_field)
        recognized = {"anastomosis", "underpass", "chamber_braid", "ladder"}
        for segment in network.segments:
            label = str(segment.metadata.get("grade_profile", ""))
            if label.startswith("process_uphill_"):
                self.assertIn(
                    segment.metadata.get("formation_origin", segment.kind),
                    recognized,
                )
            if segment.kind in {"abandoned_lobe", "stalled_lobe"}:
                self.assertFalse(label.startswith("process_uphill_"))

    def test_default_config_generates_host_driven_lobe_network(self) -> None:
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
        self.assertGreaterEqual(summary["emplacement_phase_count"], 3.0)
        self.assertGreater(summary["vertical_level_count"], 1.0)
        self.assertGreater(summary["stacked_segment_count"], 0.0)
        self.assertGreater(summary["vertical_capture_count"], 0.0)
        self.assertGreaterEqual(summary["drained_lava_pool_count"], 1.0)
        self.assertLessEqual(summary["drained_lava_pool_count"], 3.0)
        pool_segments = [
            segment
            for segment in cave_network.segments
            if segment.metadata.get("chamber_type") == "drained_lava_pool"
        ]
        self.assertTrue(pool_segments)
        self.assertTrue(
            all(segment.kind != "underpass" for segment in pool_segments),
            "drained rooms must not be stamped onto grade-separated crossings",
        )
        for segment in pool_segments:
            metadata = segment.metadata
            for key in (
                "process_cause",
                "pool_length_m",
                "pool_width_m",
                "pool_depth_m",
                "pool_aspect_ratio",
                "pool_inlet_count",
                "pool_outlet_count",
            ):
                self.assertIn(key, metadata)
            self.assertGreater(float(metadata["pool_length_m"]), 0.0)
            self.assertGreater(float(metadata["pool_width_m"]), 0.0)
            self.assertLessEqual(
                float(metadata["pool_width_m"]),
                project_config.network.emplacement_history.drained_pool_max_width_m + 1e-9,
            )
            self.assertAlmostEqual(
                float(metadata["pool_width_m"]),
                float(metadata["pool_outlet_width_m"]) * float(metadata["pool_outlet_ratio"]),
            )
        pool_junctions = [
            junction
            for junction in cave_network.junctions
            if junction.metadata.get("chamber_type") == "drained_lava_pool"
        ]
        self.assertEqual(len(pool_junctions), int(summary["drained_lava_pool_count"]))
        self.assertTrue(all(junction.kind == "chamber" for junction in pool_junctions))
        self.assertGreater(summary["reoccupied_path_count"], 0.0)
        self.assertGreater(summary["piracy_event_count"], 0.0)
        self.assertGreater(summary["flux_starved_retired_count"], 0.0)
        self.assertLessEqual(summary["max_phase_budget_utilization"], 1.0 + 1e-9)
        self.assertTrue(
            all(segment.z_level >= 0 for segment in cave_network.segments),
            "Preserved stacked lobes should sit above the younger arterial tube",
        )

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
                point.flux >= 0.0 and point.temperature_k >= 273.15 and point.age_s >= 0.0
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
        self.assertIn("anastomosis", segment_kinds)
        self.assertTrue(segment_kinds & {"abandoned_lobe", "stalled_lobe"})
        self.assertFalse(
            segment_kinds & {"island_bypass", "chamber_braid", "inner_bypass", "ladder"}
        )
        self.assertIn("chamber", {node.kind for node in cave_network.nodes})
        self.assertIn("terminal", {node.kind for node in cave_network.nodes})
        breakout_records = {
            str(segment.metadata["lobe_path_id"]): segment.metadata
            for segment in cave_network.segments
            if segment.metadata.get("branching_process") is not None
        }
        self.assertEqual(
            int(summary["breakout_event_count"]),
            len(breakout_records),
        )
        self.assertGreater(len(breakout_records), 0)
        self.assertGreaterEqual(
            len({int(metadata["birth_phase"]) for metadata in breakout_records.values()}),
            2,
            "default emplacement should span multiple eruptive phases",
        )
        phase_allocations: dict[int, float] = {}
        for metadata in breakout_records.values():
            self.assertEqual(
                metadata["branching_process"],
                "flux_breakout_avulsion",
            )
            self.assertIn(
                metadata["breakout_trigger"],
                {
                    "capacity_overflow",
                    "margin_avulsion",
                    "bend_overflow",
                    "seeded_blockage",
                },
            )
            initial_flux = float(metadata["initial_flux"])
            parent_before = float(metadata["parent_flux_before_split"])
            parent_after = float(metadata["parent_flux_after_split"])
            returned_flux = float(metadata["coalescence_returned_flux"])
            phase = int(metadata["birth_phase"])
            phase_allocations[phase] = phase_allocations.get(phase, 0.0) + initial_flux
            self.assertLessEqual(
                phase_allocations[phase],
                float(metadata["phase_flux_budget"]) + 1e-9,
            )
            self.assertIn(
                metadata["formation_state"],
                {
                    "coalesced",
                    "vertically_captured",
                    "pirated_breakout",
                    "reoccupied_passage",
                    "thermally_abandoned",
                    "flux_starved_retired",
                    "stranded",
                },
            )
            for key in (
                "new_path",
                "reoccupied",
                "reoccupied_path",
                "pirated",
                "coalesced",
                "stalled",
                "retired",
            ):
                self.assertIn(key, metadata)
            if metadata["reoccupied"]:
                self.assertIsNotNone(metadata["reoccupation_target_lobe_id"])
            self.assertIn(
                metadata["event_type"],
                {"new_breakout", "reoccupation", "pirated", "coalesced", "stalled", "retired"},
            )
            self.assertGreaterEqual(parent_before, initial_flux)
            self.assertAlmostEqual(parent_after, parent_before - initial_flux)
            self.assertGreater(float(metadata["deposition_feedback_m"]), 0.0)
            if metadata["termination"] == "coalesced":
                self.assertAlmostEqual(
                    returned_flux,
                    initial_flux
                    * project_config.network.lobe_growth.coalescence_flux_return_fraction,
                )
            else:
                self.assertEqual(returned_flux, 0.0)
            if metadata["coalesced"]:
                self.assertIn(
                    metadata["loop_mechanism"],
                    {
                        "vertical_capture",
                        "passage_reoccupation",
                        "obstacle_bypass",
                        "overflow_anastomosis",
                        "bend_bypass",
                        "lateral_avulsion",
                    },
                )
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
            self.assertIn("birth_phase", segment.metadata)
            self.assertIn("death_phase", segment.metadata)
            self.assertIn("formation_state", segment.metadata)
            self.assertIn("branch_order", segment.metadata)
            self.assertIn("emplacement_regime", segment.metadata)
            self.assertIn("roof_state", segment.metadata)
            self.assertIn("peak_formation_flux", segment.metadata)
            self.assertLessEqual(
                int(segment.metadata["birth_phase"]),
                int(segment.metadata["death_phase"]),
            )
        arterial_orders = {
            int(segment.metadata["branch_order"])
            for segment in cave_network.segments
            if segment.kind in {"backbone", "source_feeder"}
        }
        self.assertEqual(arterial_orders, {0})
        direct_breakouts = [
            segment
            for segment in cave_network.segments
            if segment.metadata.get("event_type") == "new_breakout"
        ]
        if direct_breakouts:
            self.assertTrue(all(int(s.metadata["branch_order"]) == 1 for s in direct_breakouts))

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

        chamber_segments = [
            segment for segment in cave_network.segments if segment.kind == "anastomosis"
        ]
        self.assertTrue(chamber_segments)
        self.assertTrue(
            any(segment.metadata["chamber_id"] is not None for segment in chamber_segments)
        )
        self.assertTrue(any(segment.metadata["chamber_id"] is None for segment in chamber_segments))
        self.assertTrue(
            all(
                (segment.metadata["chamber_id"] is not None)
                == bool(segment.metadata["chamber_forming"])
                for segment in chamber_segments
            )
        )
        self.assertTrue(
            all(
                segment.metadata.get("growth_model") == "hybrid_lobe"
                for segment in chamber_segments
            )
        )
        self.assertTrue(
            all(
                float(segment.metadata.get("initial_flux", 0.0)) > 0.0
                for segment in chamber_segments
            )
        )
        self.assertTrue(
            all(
                float(segment.metadata.get("final_temperature_k", 0.0))
                < project_config.network.source_temperature_k
                for segment in chamber_segments
            )
        )

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
                alignments.append(
                    (dx / length) * flow_direction[0] + (dy / length) * flow_direction[1]
                )

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
