"""Lift regional routing into optional host-bounded levels and descending ramps."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from plume_advanced.procedural import procedural_rng
from plume_advanced.progress import report_progress
from plume_advanced.stages.network_regional_routing import RegionalPlanner
from plume_advanced.stages.network_systems import GenerationDomainError


class LayeredRegionalPlanner(RegionalPlanner):
    def feeder_penalties(self, paths):
        """Reserve the swept volume of existing ramps, not just their end cells."""
        controls = self.config.layers
        scale = np.array(
            [
                1.12 * self.width,
                1.12 * self.width,
                controls.passage_height_m + controls.minimum_rock_m,
            ]
        )
        occupied = []
        existing = set()
        for path in paths:
            for a, b in zip(path, path[1:]):
                existing.add((a, b))
                length = np.linalg.norm(self.xy[b] - self.xy[a])
                t = np.linspace(0, 1, max(2, int(np.ceil(length / 2)) + 1))
                xy = (1 - t[:, None]) * self.xy[a] + t[:, None] * self.xy[b]
                depth = self.depths[self.layer_ids[a]] + (
                    self.depths[self.layer_ids[b]] - self.depths[self.layer_ids[a]]
                ) * t * t * (3 - 2 * t)
                occupied.append(np.column_stack((xy, self.sample_host(xy)[:, 0] - depth)))
        tree = cKDTree(np.concatenate(occupied) / scale)
        xyz = np.column_stack(
            (self.xy, self.sample_host(self.xy)[:, 0] - self.depths[self.layer_ids])
        )
        distance = tree.query(xyz / scale)[0]
        penalty = 1 + 20 * np.exp(-distance * distance / 2)
        weights = (penalty[self.rows] + penalty[self.cols]) / 2
        ramps = np.flatnonzero(self.layer_ids[self.rows] != self.layer_ids[self.cols])
        a, b = self.rows[ramps], self.cols[ramps]
        t = np.linspace(0.2, 0.8, 13)
        xy = self.xy[a, None, :] * (1 - t[None, :, None]) + self.xy[b, None, :] * t[None, :, None]
        depth = self.depths[self.layer_ids[a, None]] + (
            self.depths[self.layer_ids[b, None]] - self.depths[self.layer_ids[a, None]]
        ) * t * t * (3 - 2 * t)
        xyz = np.concatenate((xy, (self.sample_host(xy)[..., 0] - depth)[..., None]), axis=-1)
        distance = tree.query((xyz / scale).reshape(-1, 3))[0].reshape(len(a), len(t))
        weights[ramps] = np.maximum(
            weights[ramps], 1 + 40 * np.exp(-distance * distance / 2).max(axis=1)
        )
        # Once captured, a feeder may reuse the actual shared passage.
        shared = np.fromiter(
            ((int(a), int(b)) in existing for a, b in zip(self.rows, self.cols)),
            bool,
            len(self.rows),
        )
        weights[shared] = 1
        return weights

    def __init__(self, generator, host):
        cfg, layers = generator.config, generator.config.layers
        if cfg.systems.count < layers.count:
            raise GenerationDomainError(
                "Multi-layer growth needs at least one source per layer; increase network.systems.count"
            )
        # The base class checks this limit before allocating its routing graph.
        super().__init__(generator, host)
        base_xy, base_sources = self.xy.copy(), list(self.sources)
        n = len(base_xy)
        raw = self.graph.tocoo()
        controls, g = cfg.regional, self.geometry
        coords = [
            (base_xy[:, 1] - host.y_coords[0]) / np.diff(host.y_coords[:2])[0],
            (base_xy[:, 0] - host.x_coords[0]) / np.diff(host.x_coords[:2])[0],
        ]
        thickness = map_coordinates(host.emplacement_thickness, coords, order=1, mode="nearest")
        cover = map_coordinates(host.cover_thickness, coords, order=1, mode="nearest")
        layout_rng = procedural_rng(cfg.random_seed, "regional-layer-layout")
        gaps = layers.spacing_m * (
            1 + layers.spacing_variation * layout_rng.uniform(-0.4, 0.4, layers.count - 1)
        )
        gaps = np.maximum(gaps, layers.passage_height_m + layers.minimum_rock_m)
        self.depths = np.r_[layers.depth(0), layers.depth(0) + np.cumsum(gaps)]
        self.extents = np.full(layers.count, g.along_extent)
        if layers.minimum_extent_fraction < 1:
            self.extents[:-1] *= layout_rng.uniform(
                layers.minimum_extent_fraction, 1.0, layers.count - 1
            )
        along = (base_xy - [g.seed_x, g.seed_y]) @ [g.flow_x, g.flow_y]
        self.layer_goals = []
        for extent in self.extents:
            self.layer_goals.append(
                self.cell(np.array([g.seed_x, g.seed_y]) + extent * np.array([g.flow_x, g.flow_y]))
            )
        rows, cols, weights = [], [], []
        costs = []
        viable = []
        for layer in range(layers.count):
            rng = procedural_rng(cfg.random_seed, "regional-layer-cost", layer)
            field = gaussian_filter(
                rng.normal(size=self.shape), controls.correlation_length_m / self.step
            )
            field /= max(float(field.std()), 1e-9)
            multiplier = np.exp(controls.route_variation * np.clip(field.ravel(), -2, 2))
            costs.append(self.cost.ravel() * multiplier)
            allowed = (
                np.isfinite(thickness)
                & (
                    thickness
                    >= self.depths[layer] + layers.passage_height_m / 2 + layers.minimum_rock_m
                )
                & (cover >= layers.minimum_rock_m)
            )
            allowed &= along <= self.extents[layer] + self.step
            viable.append(allowed)
            keep = allowed[raw.row] & allowed[raw.col]
            rows.append(raw.row[keep] + layer * n)
            cols.append(raw.col[keep] + layer * n)
            weights.append(
                raw.data[keep] * (multiplier[raw.row[keep]] + multiplier[raw.col[keep]]) / 2
            )
            if controls.outlet_band_width_m:
                local_graph = csr_matrix(
                    (weights[-1], (raw.row[keep], raw.col[keep])), shape=(n, n)
                )
                local_sources = [s for i, s in enumerate(base_sources) if i % layers.count == layer]
                self.layer_goals[layer] = self.select_outlet(
                    local_graph, local_sources, self.extents[layer]
                )

        self.cost_fields = np.asarray(costs).reshape((layers.count, *self.shape))
        self.xy = np.tile(base_xy, (layers.count, 1))
        self.layer_ids = np.repeat(np.arange(layers.count), n)
        # Used only for route avoidance: separate levels must not repel each
        # other as if their plan projections represented a collision.
        self.routing_positions = np.column_stack(
            (self.xy, self.layer_ids * max(4 * self.width, layers.spacing_m))
        )
        self.sources = [cell + (i % layers.count) * n for i, cell in enumerate(base_sources)]
        self.goal = self.layer_goals[-1] + (layers.count - 1) * n
        self.goals = (
            [goal + layer * n for layer, goal in enumerate(self.layer_goals)]
            if layers.preserve_layer_trunks
            else [self.goal]
        )
        run = max(2.5 * float(gaps.max()) / layers.maximum_connection_grade, 8 * self.width)
        first, last = self.independent_length, g.along_extent - run - 2 * self.step
        if last <= first:
            raise GenerationDomainError(
                "Host route is too short for the requested layer spacing and ramp grade"
            )
        count = max(
            2 * layers.count,
            int(math.ceil(g.along_extent * layers.connection_opportunities_per_km / 1000)),
        )
        rng = procedural_rng(cfg.random_seed, "regional-layer-ramps")
        stations = np.linspace(first, last, count)
        stations += rng.uniform(-0.2, 0.2, count) * (last - first) / count
        along = (base_xy - [g.seed_x, g.seed_y]) @ [g.flow_x, g.flow_y]
        starts = np.flatnonzero(np.min(abs(along[:, None] - stations), axis=1) <= 0.55 * self.step)
        proposals = 0
        for layer in range(layers.count - 1):
            report_progress(
                "Layer connections",
                layer,
                layers.count - 1,
                "screening descending ramps against host thickness and grade",
            )
            a = starts[viable[layer][starts]]
            lengths = np.full(len(a), run)
            lateral = np.zeros(len(a))
            if layers.connection_variation:
                ramp_rng = procedural_rng(cfg.random_seed, "regional-layer-ramp-shapes", layer)
                lengths *= 1 + layers.connection_variation * ramp_rng.uniform(-0.15, 0.5, len(a))
                lateral = run * layers.connection_variation * ramp_rng.uniform(-0.35, 0.35, len(a))
            target_xy = (
                base_xy[a]
                + lengths[:, None] * [g.flow_x, g.flow_y]
                + lateral[:, None] * [g.cross_x, g.cross_y]
            )
            inside = (
                (target_xy[:, 0] >= self.x[0])
                & (target_xy[:, 0] <= self.x[-1])
                & (target_xy[:, 1] >= self.y[0])
                & (target_xy[:, 1] <= self.y[-1])
            )
            a, target_xy = a[inside], target_xy[inside]
            ix = np.rint((target_xy[:, 0] - self.x[0]) / np.diff(self.x[:2])[0]).astype(int)
            iy = np.rint((target_xy[:, 1] - self.y[0]) / np.diff(self.y[:2])[0]).astype(int)
            b = iy * len(self.x) + ix
            keep = viable[layer + 1][b]
            a, b = a[keep], b[keep]
            if not len(a):
                continue
            longest = np.linalg.norm(base_xy[b] - base_xy[a], axis=1).max()
            t = np.linspace(0, 1, max(3, int(np.ceil(longest)) + 1))
            points = (
                base_xy[a, None, :] * (1 - t[None, :, None])
                + base_xy[b, None, :] * t[None, :, None]
            )
            samples = self.sample_host(points)
            grid_coords = [
                (points[..., 1] - host.y_coords[0]) / np.diff(host.y_coords[:2])[0],
                (points[..., 0] - host.x_coords[0]) / np.diff(host.x_coords[:2])[0],
            ]
            rock = map_coordinates(host.emplacement_thickness, grid_coords, order=1, mode="nearest")
            depth = self.depths[layer] + gaps[layer] * t * t * (3 - 2 * t)
            z = samples[..., 0] - depth
            distances = np.linalg.norm(base_xy[b] - base_xy[a], axis=1)
            grade = np.diff(z, axis=1) / (distances[:, None] / (len(t) - 1))
            keep = (
                np.isfinite(samples).all(axis=(1, 2))
                & np.isfinite(rock).all(axis=1)
                & (samples[..., 1:] > 0).all(axis=(1, 2))
                & (rock - depth - layers.passage_height_m / 2 >= layers.minimum_rock_m).all(axis=1)
                & (abs(grade) <= 0.95 * layers.maximum_connection_grade).all(axis=1)
                & (grade <= cfg.quality.maximum_uphill_grade).all(axis=1)
            )
            a, b, distances = a[keep], b[keep], distances[keep]
            rows.append(a + layer * n)
            cols.append(b + (layer + 1) * n)
            weights.append(1.8 * distances * (costs[layer][a] + costs[layer + 1][b]) / 2)
            proposals += len(a)
        self.rows, self.cols, self.weights = (
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(weights),
        )
        keep = ~np.isin(self.cols, self.sources)
        self.rows, self.cols, self.weights = self.rows[keep], self.cols[keep], self.weights[keep]
        self.graph = csr_matrix(
            (self.weights, (self.rows, self.cols)), shape=(n * layers.count,) * 2
        )
        if layers.preserve_layer_trunks:
            # Each level keeps its own downstream exit. Within-level cost
            # decreases; a larger layer offset orders every descending ramp.
            # This remains a routing potential, not a hydraulic-head model.
            potentials = []
            same = self.layer_ids[self.rows] == self.layer_ids[self.cols]
            flat_graph = csr_matrix(
                (self.weights[same], (self.rows[same], self.cols[same])),
                shape=self.graph.shape,
            )
            for layer, goal in enumerate(self.goals):
                distance = dijkstra(flat_graph.T.tocsr(), indices=goal)
                potentials.append(distance[layer * n : (layer + 1) * n])
            finite = np.concatenate(potentials)
            offset = float(finite[np.isfinite(finite)].max()) + 1
            self.potential = np.concatenate(
                [p + (layers.count - 1 - layer) * offset for layer, p in enumerate(potentials)]
            )
            keep = (
                np.isfinite(self.potential[self.rows])
                & np.isfinite(self.potential[self.cols])
                & (self.potential[self.rows] > self.potential[self.cols] + 1e-9)
                & ~np.isin(self.rows, self.goals)
            )
            self.rows, self.cols, self.weights = (
                self.rows[keep],
                self.cols[keep],
                self.weights[keep],
            )
            self.graph = csr_matrix((self.weights, (self.rows, self.cols)), shape=self.graph.shape)
            reachable, self.predecessors = dijkstra(
                self.graph.T.tocsr(), indices=self.goal, return_predecessors=True
            )
        else:
            self.potential, self.predecessors = dijkstra(
                self.graph.T.tocsr(), indices=self.goal, return_predecessors=True
            )
            reachable = self.potential
        descending = self.potential[self.rows] > self.potential[self.cols] + 1e-9
        self.rows, self.cols, self.weights = (
            self.rows[descending],
            self.cols[descending],
            self.weights[descending],
        )
        if not np.isfinite(reachable[self.sources]).all():
            raise GenerationDomainError(
                "No viable multi-layer source-to-outlet route; inspect host thickness, extent and connection grade"
            )
        report_progress(
            "Layer connections",
            layers.count - 1,
            layers.count - 1,
            f"{proposals} feasible ramp opportunities; selected routes use only a subset",
        )

    def retained_trunks(self, paths):
        """Fork before a descent to continue the upper passage to its own exit."""
        result: list[list[int]] = []
        same = self.layer_ids[self.rows] == self.layer_ids[self.cols]
        for layer, goal in enumerate(self.goals[:-1]):
            reached = {s for s in self.sources if self.layer_ids[s] == layer}
            edges = [(a, b) for path in paths for a, b in zip(path, path[1:])
                     if self.layer_ids[a] == self.layer_ids[b] == layer]
            while True:
                enlarged = reached | {b for a, b in edges if a in reached}
                if enlarged == reached:
                    break
                reached = enlarged
            starts = sorted(
                {
                    a
                    for path in paths
                    for a, b in zip(path, path[1:])
                    if a in reached and self.layer_ids[b] == layer + 1
                }
            )
            if not starts:
                raise GenerationDomainError(
                    "No descending junction available to retain a layer trunk"
                )
            weights = self.weights * self.feeder_penalties(paths + result)
            graph = csr_matrix(
                (weights[same], (self.rows[same], self.cols[same])), shape=self.graph.shape
            )
            distance, previous = dijkstra(graph.T.tocsr(), indices=goal, return_predecessors=True)
            # Choose the least-cost viable continuation, accounting for the
            # actual swept volume of the existing passages.
            start = min(starts, key=lambda cell: (distance[cell], cell))
            if not np.isfinite(distance[start]):
                raise GenerationDomainError(
                    "No viable retained layer trunk to its downstream outlet"
                )
            path = [start]
            while path[-1] != goal:
                path.append(int(previous[path[-1]]))
            result.append(path)
        return result
