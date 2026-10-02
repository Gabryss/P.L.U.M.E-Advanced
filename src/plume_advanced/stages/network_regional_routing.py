"""Regional least-cost routing and bounded, sustained branch growth.

The coarse directed host graph supplies a potential to the selected termini. New routes
must descend that potential, so split/rejoin cycles never become flow cycles.
The potential is a routing cost, not hydraulic pressure. The immutable host is
sampled again after smoothing. This experimental mode stops at Stage B.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from plume_advanced.procedural import procedural_rng
from plume_advanced.progress import report_progress


@dataclass(frozen=True)
class RegionalGrowthConfig:
    branch_growth: str = "detour"
    planning_step_m: float = 6.0
    correlation_length_m: float = 80.0
    route_variation: float = 0.6
    secondary_scale_weight: float = 0.0
    branch_localization: float = 0.0
    hierarchy_strength: float = 0.0
    width_log_sigma: float = 0.0
    width_correlation_m: float = 10.0
    blind_branch_fraction: float = 0.0
    source_stagger_m: float = 0.0
    source_lateral_jitter_m: float = 0.0
    outlet_count: int = 1
    outlet_band_width_m: float = 0.0
    branches_per_km: float = 8.0
    minimum_branch_length_m: float = 70.0
    maximum_branch_length_m: float = 220.0
    attempts_per_branch: int = 12
    maximum_branches: int = 64
    extra_connections: int = 0
    maximum_grid_cells: int = 250_000

    def __post_init__(self):
        if self.branch_growth not in ("detour", "front"):
            raise ValueError("network.regional.branch_growth must be detour or front")
        for name, value in asdict(self).items():
            if name == "branch_growth":
                continue
            if name == "extra_connections":
                if type(value) is not int or not 0 <= value <= 8:
                    raise ValueError("network.regional.extra_connections must be an integer from 0 to 8")
                continue
            if name in {"outlet_count", "attempts_per_branch", "maximum_branches", "maximum_grid_cells"}:
                if type(value) is not int or value < 1:
                    raise ValueError(f"network.regional.{name} must be a positive integer")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"network.regional.{name} must be finite and nonnegative")
        for name in ("planning_step_m", "correlation_length_m", "minimum_branch_length_m", "width_correlation_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"network.regional.{name} must be positive")
        if self.maximum_branch_length_m < self.minimum_branch_length_m:
            raise ValueError("regional branch lengths must be ordered")
        if self.secondary_scale_weight > 1:
            raise ValueError("network.regional.secondary_scale_weight must be between 0 and 1")
        for name in ("branch_localization", "hierarchy_strength", "blind_branch_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"network.regional.{name} must be between 0 and 1")
        if self.width_log_sigma > 1:
            raise ValueError("network.regional.width_log_sigma must be between 0 and 1")
        if self.outlet_count > 8:
            raise ValueError("network.regional.outlet_count permits at most 8 termini")
        if self.attempts_per_branch > 64 or self.maximum_branches > 256:
            raise ValueError(
                "regional growth permits at most 64 attempts per branch and 256 branches"
            )


class RegionalPlanner:
    """Sparse directed routing graph; all arrays are owned, never host views."""

    connectivity_audit: dict[str, object]

    def __init__(self, generator, host):
        from plume_advanced.stages.network_systems import GenerationDomainError

        cfg, controls = generator.config, generator.config.regional
        if controls.outlet_count > 1 and cfg.layers.enabled:
            raise GenerationDomainError(
                "Multiple regional termini currently require one layer; "
                "use layers.preserve_layer_trunks for separate layer outlets"
            )
        self.config, self.host = cfg, host
        self.geometry = generator._build_flow_geometry(host)
        self.width = 2 * cfg.base_passage_radius
        margin = self.width
        step = controls.planning_step_m
        nx = int(np.floor((np.ptp(host.x_coords) - 2 * margin) / step)) + 1
        ny = int(np.floor((np.ptp(host.y_coords) - 2 * margin) / step)) + 1
        layer_count = cfg.layers.count if cfg.layers.enabled else 1
        if min(nx, ny) < 3 or nx * ny * layer_count > controls.maximum_grid_cells:
            raise GenerationDomainError(
                "Regional planning grid is too small or exceeds network.regional.maximum_grid_cells; adjust host extent or planning_step_m"
            )
        self.x = np.linspace(host.x_coords[0] + margin, host.x_coords[-1] - margin, nx)
        self.y = np.linspace(host.y_coords[0] + margin, host.y_coords[-1] - margin, ny)
        xx, yy = np.meshgrid(self.x, self.y)
        self.xy = np.column_stack((xx.ravel(), yy.ravel()))
        self.layer_ids = np.zeros(len(self.xy), dtype=int)
        self.routing_positions = self.xy
        self.shape = yy.shape
        self.step = max(self.x[1] - self.x[0], self.y[1] - self.y[0])
        coords = [
            (yy - host.y_coords[0]) / (host.y_coords[1] - host.y_coords[0]),
            (xx - host.x_coords[0]) / (host.x_coords[1] - host.x_coords[0]),
        ]
        fields = {
            name: map_coordinates(getattr(host, name), coords, order=1, mode="nearest")
            for name in ("elevation", "growth_cost", "cover_thickness", "roof_competence")
        }
        rng = procedural_rng(cfg.random_seed, "regional-cost-field")
        noise = gaussian_filter(
            rng.normal(size=self.shape), controls.correlation_length_m / self.step
        )
        noise = noise / max(float(noise.std()), 1e-9)
        if controls.secondary_scale_weight:
            fine = gaussian_filter(
                procedural_rng(cfg.random_seed, "regional-secondary-cost").normal(size=self.shape),
                max(1.0, 0.25 * controls.correlation_length_m / self.step),
            )
            fine /= max(float(fine.std()), 1e-9)
            noise = (noise + controls.secondary_scale_weight * fine) / math.sqrt(
                1 + controls.secondary_scale_weight**2
            )
        self.cost = (1 + cfg.growth_cost_weight * np.maximum(fields["growth_cost"], 0)) * np.exp(
            controls.route_variation * np.clip(noise, -2, 2)
        )
        self.cost_fields = self.cost[None, ...]
        viable = np.ones(self.shape, dtype=bool)
        for value in fields.values():
            viable &= np.isfinite(value)
        viable &= (fields["cover_thickness"] > 0) & (fields["roof_competence"] > 0)
        g = self.geometry
        along = (xx - g.seed_x) * g.flow_x + (yy - g.seed_y) * g.flow_y
        viable &= (along >= -self.width) & (along <= g.along_extent + self.width)
        cross = (xx - g.seed_x) * g.cross_x + (yy - g.seed_y) * g.cross_y
        spacing = cfg.systems.source_spacing_widths * self.width
        source_offsets = (np.arange(cfg.systems.count) - (cfg.systems.count - 1) / 2) * spacing
        self.independent_length = (
            cfg.systems.minimum_independent_length_widths * self.width + 2 * self.step
        )
        source_stations = np.zeros(cfg.systems.count)
        if controls.source_stagger_m:
            if controls.source_stagger_m + self.independent_length >= g.along_extent:
                raise GenerationDomainError(
                    "Source stagger leaves no downstream growth region; reduce source_stagger_m"
                )
            source_stations = procedural_rng(cfg.random_seed, "regional-source-layout").uniform(
                0, controls.source_stagger_m, cfg.systems.count
            )
        source_offsets = self.jitter_source_offsets(source_offsets, source_stations)
        lane_radius = np.full(cfg.systems.count, max(0.55 * self.step, spacing / 2 - 0.65 * self.width))
        if controls.source_lateral_jitter_m:
            gaps = np.diff(source_offsets)
            nearest_gap = np.minimum(np.r_[spacing, gaps], np.r_[gaps, spacing])
            lane_radius = nearest_gap / 2 - 0.65 * self.width
        # Reserve separate inlet corridors for the configured persistence
        # length. This is a temporary network constraint, not a host mutation.
        viable &= (along >= self.independent_length + source_stations.max()) | (
            np.any(abs(cross[..., None] - source_offsets) <= lane_radius, axis=-1)
        )
        ids = np.arange(nx * ny).reshape(self.shape)
        rows, cols, weights = [], [], []
        # Intermediate host samples prevent a coarse edge from jumping a narrow
        # forbidden strip or an uphill obstacle between viable endpoint cells.
        report_progress(
            "Regional routing graph", 0, 8, f"{nx * ny:,} cells; {self.step:.1f} m spacing"
        )
        for direction, (dy, dx) in enumerate(
            ((0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1))
        ):
            ys, xs = slice(max(0, -dy), min(ny, ny - dy)), slice(max(0, -dx), min(nx, nx - dx))
            yt, xt = slice(max(0, dy), min(ny, ny + dy)), slice(max(0, dx), min(nx, nx + dx))
            a, b = ids[ys, xs].ravel(), ids[yt, xt].ravel()
            valid = viable[ys, xs].ravel() & viable[yt, xt].ravel()
            if dx and dy:
                # A consistent triangulation is planar: opposite diagonals in
                # a cell cannot silently cross without a shared graph node.
                lower_x = np.minimum(a % nx, b % nx)
                lower_y = np.minimum(a // nx, b // nx)
                valid &= (lower_x + lower_y) % 2 == (0 if dx == dy else 1)
            distance = float(np.hypot(dx * (self.x[1] - self.x[0]), dy * (self.y[1] - self.y[0])))
            points = (
                self.xy[a, None, :]
                + np.linspace(0, 1, max(3, int(np.ceil(distance / 1.5)) + 1))[None, :, None]
                * (self.xy[b] - self.xy[a])[:, None, :]
            )
            sampled = self.sample_host(points)
            valid &= np.all(np.isfinite(sampled), axis=(1, 2))
            valid &= np.all(sampled[:, :, 1:] > 0, axis=(1, 2))
            grades = np.diff(sampled[:, :, 0], axis=1) / (distance / (points.shape[1] - 1))
            valid &= np.all(grades <= cfg.quality.maximum_uphill_grade, axis=1)
            rows.append(a[valid])
            cols.append(b[valid])
            weights.append(
                distance * (self.cost.ravel()[a[valid]] + self.cost.ravel()[b[valid]]) / 2
            )
            report_progress(
                "Regional routing graph",
                direction + 1,
                8,
                "checking substrate and uphill grade between cells",
            )
        self.rows, self.cols, self.weights = (
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(weights),
        )
        self.sources = [
            self.cell(
                np.array([g.seed_x, g.seed_y])
                + source_stations[i] * np.array([g.flow_x, g.flow_y])
                + source_offsets[i]
                * np.array([g.cross_x, g.cross_y])
            )
            for i in range(cfg.systems.count)
        ]
        if len(set(self.sources)) != len(self.sources):
            raise GenerationDomainError(
                "Regional sources resolve to the same planning cell; decrease planning_step_m"
            )
        # Sources inject discharge once; no other route may enter a source.
        keep = ~np.isin(self.cols, self.sources)
        self.rows, self.cols, self.weights = self.rows[keep], self.cols[keep], self.weights[keep]
        self.graph = csr_matrix((self.weights, (self.rows, self.cols)), shape=(nx * ny, nx * ny))
        self.goals = self.select_outlets(self.graph, self.sources, g.along_extent)
        self.goal = self.goals[0]
        if set(self.goals) & set(self.sources):
            raise GenerationDomainError(
                "Regional source and outlet coincide; increase the route extent"
            )
        report_progress(
            "Regional outlet potential", detail="least-cost paths across the two-dimensional host"
        )
        if len(self.goals) == 1:
            self.potential, self.predecessors = dijkstra(
                self.graph.T.tocsr(), indices=self.goal, return_predecessors=True
            )
        else:
            # One scalar potential to the nearest reachable terminus keeps the
            # union of routes acyclic. Independently oriented outlet fields can
            # disagree at encounters and create directed cycles.
            self.potential, self.predecessors, _ = dijkstra(
                self.graph.T.tocsr(), indices=self.goals, return_predecessors=True,
                min_only=True,
            )
        descending = self.potential[self.rows] > self.potential[self.cols] + 1e-9
        self.rows, self.cols, self.weights = (
            self.rows[descending],
            self.cols[descending],
            self.weights[descending],
        )

    def jitter_source_offsets(self, offsets, stations):
        """Bounded seeded proposals; retain inlet order and usable routing lanes.

        Only source placement consumes this stream. Rejected layouts cannot
        perturb the routing-cost, width or branch random fields. Geometry and
        host viability are still checked after snapping and route generation.
        """
        jitter = self.config.regional.source_lateral_jitter_m
        if not jitter:
            return offsets
        g = self.geometry
        minimum_gap = 1.3 * self.width + 1.1 * self.step
        rng = procedural_rng(self.config.random_seed, "regional-source-lateral")
        for _ in range(64):
            proposed = offsets + rng.uniform(-jitter, jitter, len(offsets))
            if np.any(np.diff(proposed) < minimum_gap):
                continue
            xy = (np.array([g.seed_x, g.seed_y])
                  + stations[:, None] * [g.flow_x, g.flow_y]
                  + proposed[:, None] * [g.cross_x, g.cross_y])
            if (np.all((xy[:, 0] >= self.x[0]) & (xy[:, 0] <= self.x[-1]))
                    and np.all((xy[:, 1] >= self.y[0]) & (xy[:, 1] <= self.y[-1]))):
                return proposed
        # Layout exhaustion depends on this candidate's seeded proposal, so it
        # participates in the normal bounded candidate search, not a domain abort.
        raise ValueError(
            "Regional source layout exhausted 64 proposals within host and separation bounds; "
            "reduce source_lateral_jitter_m, increase source spacing, or enlarge the host"
        )

    def select_outlet(self, graph, sources, extent):
        """Choose a common reachable exit in a metric band using routed cost.

        Zero band width keeps the prescribed-axis layout. Each source is solved
        separately to keep temporary distance storage linear in grid size.
        """
        from plume_advanced.stages.network_systems import GenerationDomainError

        g = self.geometry
        origin = np.array([g.seed_x, g.seed_y])
        band = self.config.regional.outlet_band_width_m
        if not band:
            return self.cell(origin + extent * np.array([g.flow_x, g.flow_y]))
        xy = self.xy[:graph.shape[0]]
        along = (xy - origin) @ [g.flow_x, g.flow_y]
        cross = (xy - origin) @ [g.cross_x, g.cross_y]
        candidates = np.flatnonzero(
            (abs(along - extent) <= self.step)
            & (abs(cross) <= band / 2)
            & (np.asarray(graph.getnnz(axis=0)).ravel() > 0)
        )
        if not len(candidates):
            raise GenerationDomainError("No viable cells in the regional outlet band")
        score = np.zeros(len(candidates))
        for i, source in enumerate(sources):
            report_progress("Regional outlet selection", i, len(sources),
                            f"comparing {len(candidates)} host-feasible exit cells")
            score += dijkstra(graph, indices=source)[candidates]
        selected = int(np.argmin(score))
        if not np.isfinite(score[selected]):
            raise GenerationDomainError("No common reachable exit in the regional outlet band")
        report_progress("Regional outlet selection", len(sources), len(sources),
                        "minimum total routed cost from the sources")
        return int(candidates[selected])

    def select_outlets(self, graph, sources, extent):
        """Select separated termini by reduction in source-to-outlet routed cost.

        This is bounded greedy facility selection on host-feasible cells, not a
        prediction of eruption outlets. Distances are solved one source at a
        time; only its terminal-band distances are retained. Count one preserves
        the original algorithm, including its zero-band axial endpoint.
        """
        count = self.config.regional.outlet_count
        if count == 1:
            return [self.select_outlet(graph, sources, extent)]
        g = self.geometry
        origin = np.array([g.seed_x, g.seed_y])
        along = (self.xy - origin) @ [g.flow_x, g.flow_y]
        cross = (self.xy - origin) @ [g.cross_x, g.cross_y]
        separation = max(3 * self.width, 2 * self.step)
        band = self.config.regional.outlet_band_width_m or max(
            float(np.ptp(cross[sources])), 1.5 * (count - 1) * separation
        )
        candidates = np.flatnonzero(
            (abs(along - extent) <= self.step) & (abs(cross) <= band / 2)
            & (np.asarray(graph.getnnz(axis=0)).ravel() > 0)
            & ~np.isin(np.arange(len(self.xy)), sources)
        )
        distances = np.empty((len(sources), len(candidates)))
        for i, source in enumerate(sources):
            report_progress("Regional termini selection", i, len(sources),
                            f"{count} termini; testing {len(candidates)} host cells")
            distances[i] = dijkstra(graph, indices=source)[candidates]
        best = np.full(len(sources), np.inf)
        selected: list[int] = []
        available = np.isfinite(distances).any(axis=0)
        for _ in range(count):
            choices = np.flatnonzero(available)
            if not len(choices):
                raise ValueError(
                    "Regional outlet band cannot supply separated termini; "
                    "increase outlet_band_width_m or reduce outlet_count"
                )
            trial = np.minimum(best[:, None], distances[:, choices])
            missing = (~np.isfinite(trial)).sum(axis=0)
            costs = np.where(np.isfinite(trial), trial, 0).sum(axis=0)
            supply_cost = distances[:, choices].min(axis=0)
            # Cover sources first, then reduce their nearest-terminal cost.
            # If every source prefers an existing outlet, prefer the cheapest
            # supplied alternative, not an arbitrary cell at the band edge.
            winner = int(choices[np.lexsort((choices, supply_cost, costs, missing))[0]])
            selected.append(int(candidates[winner]))
            best = np.minimum(best, distances[:, winner])
            available &= abs(cross[candidates] - cross[candidates[winner]]) >= separation
        if not np.isfinite(best).all():
            raise ValueError("No set of regional termini is reachable from every source")
        report_progress("Regional termini selection", len(sources), len(sources),
                        f"selected {count} separated termini using routed host costs")
        return sorted(selected, key=lambda cell: (float(cross[cell]), cell))

    def cell(self, xy):
        if not (self.x[0] <= xy[0] <= self.x[-1] and self.y[0] <= xy[1] <= self.y[-1]):
            from plume_advanced.stages.network_systems import GenerationDomainError

            raise GenerationDomainError(
                "Regional source or outlet lies outside the usable host; enlarge the host"
            )
        ix, iy = np.argmin(abs(self.x - xy[0])), np.argmin(abs(self.y - xy[1]))
        return int(iy * len(self.x) + ix)

    def sample_host(self, points):
        flat = points.reshape(-1, 2)
        h = self.host
        coords = [
            (flat[:, 1] - h.y_coords[0]) / (h.y_coords[1] - h.y_coords[0]),
            (flat[:, 0] - h.x_coords[0]) / (h.x_coords[1] - h.x_coords[0]),
        ]
        return np.stack(
            [
                map_coordinates(getattr(h, name), coords, order=1, mode="nearest")
                for name in ("elevation", "cover_thickness", "roof_competence")
            ],
            axis=-1,
        ).reshape(*points.shape[:-1], 3)

    def primary_path(self, source):
        if not np.isfinite(self.potential[source]):
            from plume_advanced.stages.network_systems import GenerationDomainError

            raise GenerationDomainError("No viable regional source-to-outlet route in the host")
        path = [source]
        targets = {self.goal} if self.config.layers.enabled else set(self.goals)
        while path[-1] not in targets:
            next_cell = int(self.predecessors[path[-1]])
            if next_cell < 0 or len(path) >= len(self.xy):
                raise ValueError("Broken regional source-to-terminus predecessor chain")
            path.append(next_cell)
        return path

    def terminal_continuations(self, paths):
        """Supply unused termini by genuine forks on the existing host graph.

        Nearest-terminal routing alone can starve every other outlet when one
        downstream corridor is cheaper for all sources. Continue existing
        passages to those termini with bounded, potential-descending routes.
        Incoming occupied cells are excluded so a continuation cannot silently
        rejoin the trunk and then reuse its common endpoint.
        """
        result: list[list[int]] = []
        for goal in self.goals:
            occupied = {cell for path in paths + result for cell in path}
            if goal in occupied:
                continue
            report_progress("Regional terminal continuations", len(result), len(self.goals),
                            "routing sustained downstream forks on the host")
            outgoing: dict[int, list[int]] = {}
            incoming: dict[int, set[int]] = {}
            for path in paths + result:
                for a, b in zip(path, path[1:]):
                    outgoing.setdefault(a, []).append(b)
                    incoming.setdefault(b, set()).add(a)
            junctions = [cell for cell in occupied
                         if len(set(outgoing.get(cell, []))) > 1 or len(incoming.get(cell, set())) > 1]
            distance = cKDTree(self.xy[sorted(occupied)]).query(self.xy)[0]
            penalty = 1 + 4 * np.exp(-distance / self.width)
            keep = ~np.isin(self.cols, sorted(occupied))
            graph = csr_matrix(
                (self.weights[keep] * (penalty[self.rows[keep]] + penalty[self.cols[keep]]) / 2,
                 (self.rows[keep], self.cols[keep])), shape=self.graph.shape,
            )
            costs, previous = dijkstra(graph.T.tocsr(), indices=goal, return_predecessors=True)
            minimum = max(self.config.regional.minimum_branch_length_m, 4 * self.width)
            candidates = []
            g = self.geometry
            stations = (self.xy - [g.seed_x, g.seed_y]) @ [g.flow_x, g.flow_y]
            for start in sorted(outgoing):
                if (not np.isfinite(costs[start]) or start in self.sources
                        or stations[start] < stations[self.sources].max() + self.independent_length
                        or np.linalg.norm(self.xy[goal] - self.xy[start]) < minimum):
                    continue
                # Do not crowd a new fork into an existing merge/split blend.
                if junctions and np.linalg.norm(self.xy[junctions] - self.xy[start], axis=1).min() < 4 * self.width:
                    continue
                delta = self.xy[int(previous[start])] - self.xy[start]
                if not any(
                    delta @ (self.xy[b] - self.xy[start]) >=
                    np.cos(np.radians(self.config.quality.maximum_turn_degrees)) *
                    np.linalg.norm(delta) * np.linalg.norm(self.xy[b] - self.xy[start])
                    for b in outgoing[start]
                ):
                    continue
                candidates.append((float(costs[start]), start))
            # Inspect a bounded number of fork sites before building metric
            # geometry. Full width/curvature checks still run after smoothing.
            for _, start in sorted(candidates)[:64]:
                path = [start]
                while path[-1] != goal:
                    next_cell = int(previous[path[-1]])
                    if next_cell < 0 or len(path) >= len(self.xy):
                        raise ValueError("Broken regional terminal continuation")
                    path.append(next_cell)
                travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.xy[path], axis=0), axis=1))]
                remote = travel > 4 * self.width
                if np.any(distance[path][remote] < 1.1 * self.width):
                    continue
                result.append(path)
                break
            else:
                raise ValueError("No sustained host-feasible continuation to a requested regional terminus")
        report_progress("Regional terminal continuations", len(self.goals), len(self.goals),
                        f"all {len(self.goals)} termini supplied; inspecting geometry next")
        return result

    def detour(self, start, end, occupied, rng):
        distances = cKDTree(self.routing_positions[sorted(occupied)]).query(self.routing_positions)[
            0
        ]
        ds = np.linalg.norm(self.xy - self.xy[start], axis=1)
        de = np.linalg.norm(self.xy - self.xy[end], axis=1)
        # Open only the local junction neighborhoods. The middle must clear
        # existing passages, including branches introduced in previous rounds.
        clearance = self.width * float(rng.uniform(1.2, 2.4))
        allowed = (distances >= clearance) | (ds < 3 * self.width) | (de < 3 * self.width)
        allowed &= (self.potential <= self.potential[start] + 1e-9) & (
            self.potential >= self.potential[end] - 1e-9
        )
        keep = allowed[self.rows] & allowed[self.cols]
        # Prefer a distinct route even within the open junction neighborhoods.
        penalty = 1 + 3 * np.exp(-distances / max(clearance, 1))
        graph = csr_matrix(
            (
                self.weights[keep] * (penalty[self.rows[keep]] + penalty[self.cols[keep]]) / 2,
                (self.rows[keep], self.cols[keep]),
            ),
            shape=self.graph.shape,
        )
        distance, previous = dijkstra(graph, indices=start, return_predecessors=True)
        if not np.isfinite(distance[end]):
            return None
        path = [end]
        while path[-1] != start:
            path.append(int(previous[path[-1]]))
        path.reverse()
        return path
