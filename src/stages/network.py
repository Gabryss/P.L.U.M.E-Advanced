"""Stage B: host-driven braided cave network generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math

import numpy as np

from stages.host_field import HostField

SegmentMetadataValue = str | int | float | bool | None


@dataclass(frozen=True)
class CaveNetworkConfig:
    """Parameters controlling the host-driven braided cave-network generator."""

    random_seed: int | None = None
    source_count: int = 8
    source_band_length: float = 90.0
    source_band_half_width: float = 180.0
    sink_margin: float = 80.0
    trace_max_steps: int = 460
    max_uphill_step: float = 1.2
    forward_alignment_weight: float = 2.2
    downhill_alignment_weight: float = 3.4
    elevation_drop_weight: float = 2.6
    growth_cost_weight: float = 3.2
    roof_weight: float = 1.5
    cover_weight: float = 0.8
    slope_penalty_weight: float = 0.45
    inertia_weight: float = 0.9
    corridor_weight: float = 0.35
    outlet_potential_weight: float = 3.4
    small_trace_count: int = 96
    medium_trace_count: int = 36
    large_trace_count: int = 12
    small_attraction_weight: float = 0.65
    medium_attraction_weight: float = 0.42
    large_attraction_weight: float = 0.20
    small_congestion_threshold: float = 9.0
    medium_congestion_threshold: float = 5.0
    large_congestion_threshold: float = 3.0
    small_congestion_weight: float = 0.12
    medium_congestion_weight: float = 0.32
    large_congestion_weight: float = 0.60
    small_temperature: float = 0.22
    medium_temperature: float = 0.38
    large_temperature: float = 0.55
    small_flux_threshold_quantile: float = 0.58
    medium_flux_threshold_quantile: float = 0.45
    large_flux_threshold_quantile: float = 0.25
    total_flux_threshold_quantile: float = 0.60
    prune_iterations: int = 3
    chamber_flux_quantile: float = 0.82
    base_passage_radius: float = 18.0
    chamber_radius: float = 46.0
    occupancy_smoothing_passes: int = 1
    spur_count: int = 5
    spur_max_steps: int = 24
    spur_lateral_bias: float = 1.3
    spur_congestion_weight: float = 0.85
    channel_count_samples: int = 28
    selected_small_paths: int = 12
    selected_medium_paths: int = 5
    selected_large_paths: int = 2
    selected_spur_paths: int = 3
    maximum_path_overlap: float = 0.72


@dataclass(frozen=True)
class CaveNode:
    """One topological junction in the cave network."""

    node_id: int
    x: float
    y: float
    along_position: float
    lateral_offset: float
    kind: str


@dataclass(frozen=True)
class CavePoint:
    """One sampled point along a cave-network segment."""

    index: int
    x: float
    y: float
    elevation: float
    slope_degrees: float
    cover_thickness: float
    roof_competence: float
    growth_cost: float
    arc_length: float
    width: float


@dataclass(frozen=True)
class CaveSegment:
    """One directed segment between two cave-network nodes."""

    segment_id: int
    start_node_id: int
    end_node_id: int
    kind: str
    z_level: int
    points: tuple[CavePoint, ...]
    metadata: dict[str, SegmentMetadataValue]

    @property
    def total_length(self) -> float:
        return 0.0 if not self.points else self.points[-1].arc_length

    @property
    def mean_width(self) -> float:
        if not self.points:
            return 0.0
        return sum(point.width for point in self.points) / len(self.points)


@dataclass(frozen=True)
class CaveNetwork:
    """Stage-B output for the host-driven braided cave network."""

    config: CaveNetworkConfig
    nodes: tuple[CaveNode, ...]
    segments: tuple[CaveSegment, ...]
    occupancy: np.ndarray
    width_field: np.ndarray
    dominant_route_node_ids: tuple[int, ...]
    slice_along_positions: tuple[float, ...]
    slice_channel_counts: tuple[int, ...]

    def summary(self) -> dict[str, float]:
        """Return scalar summaries for quick inspection."""

        occupied_area = float(self.occupancy.sum())
        segment_lengths = [segment.total_length for segment in self.segments]
        mean_segment_width = (
            sum(segment.mean_width for segment in self.segments) / len(self.segments)
            if self.segments
            else 0.0
        )
        loop_count = float(self._loop_rank())
        terminal_count = float(sum(1 for degree in self._degrees().values() if degree == 1))
        spur_count = float(sum(1 for segment in self.segments if segment.kind == "spur"))

        return {
            "node_count": float(len(self.nodes)),
            "segment_count": float(len(self.segments)),
            "loop_count": loop_count,
            "terminal_count": terminal_count,
            "spur_count": spur_count,
            "occupied_cell_count": occupied_area,
            "total_length": float(sum(segment_lengths)),
            "dominant_route_length": self.dominant_route_length,
            "mean_segment_width": mean_segment_width,
            "max_parallel_channels": float(
                max(self.slice_channel_counts) if self.slice_channel_counts else 0
            ),
        }

    @property
    def dominant_route_length(self) -> float:
        if len(self.dominant_route_node_ids) < 2:
            return 0.0
        node_pairs = set(zip(self.dominant_route_node_ids, self.dominant_route_node_ids[1:]))
        return sum(
            segment.total_length
            for segment in self.segments
            if (segment.start_node_id, segment.end_node_id) in node_pairs
        )

    def _degrees(self) -> dict[int, int]:
        degrees = {node.node_id: 0 for node in self.nodes}
        for segment in self.segments:
            degrees[segment.start_node_id] += 1
            degrees[segment.end_node_id] += 1
        return degrees

    def _loop_rank(self) -> int:
        if not self.nodes:
            return 0

        adjacency = {node.node_id: set() for node in self.nodes}
        for segment in self.segments:
            adjacency[segment.start_node_id].add(segment.end_node_id)
            adjacency[segment.end_node_id].add(segment.start_node_id)

        visited: set[int] = set()
        components = 0
        for node in adjacency:
            if node in visited:
                continue
            components += 1
            stack = [node]
            visited.add(node)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)

        return max(0, len(self.segments) - len(self.nodes) + components)


@dataclass(frozen=True)
class _FlowGeometry:
    flow_x: float
    flow_y: float
    cross_x: float
    cross_y: float
    seed_x: float
    seed_y: float
    along_grid: np.ndarray
    cross_grid: np.ndarray
    along_extent: float
    cell_scale: float


@dataclass(frozen=True)
class _TraceFamily:
    label: str
    count: int
    attraction_weight: float
    congestion_threshold: float
    congestion_weight: float
    temperature: float


@dataclass(frozen=True)
class _BraidZone:
    center_fraction: float
    half_length_fraction: float
    branches: tuple["_ZoneBranch", ...]
    ladder_rungs: tuple[float, ...] = ()
    chamber_radius_scale: float = 1.0


@dataclass(frozen=True)
class _ZoneBranch:
    kind: str
    lateral_offset: float
    start_shift_fraction: float = 0.0
    end_shift_fraction: float = 0.0
    skew: float = 0.0
    wobble: float = 0.0
    phase: float = 0.0
    z_level: int = 0
    merge_shared_cells: bool = True


@dataclass(frozen=True)
class _SelectedPath:
    kind: str
    path: tuple[tuple[int, int], ...]
    z_level: int = 0
    merge_shared_cells: bool = True
    metadata: dict[str, SegmentMetadataValue] | None = None


class CaveNetworkGenerator:
    """Generate a host-driven braided cave network and raster occupancy."""

    FAMILY_LABELS = ("small", "medium", "large", "spur")

    def __init__(self, config: CaveNetworkConfig | None = None) -> None:
        self.config = config or CaveNetworkConfig()

    def generate(self, host_field: HostField) -> CaveNetwork:
        geometry = self._build_flow_geometry(host_field)
        rng = np.random.default_rng(self.config.random_seed)
        source_cells = self._select_source_cells(host_field, geometry)
        support_field = self._build_support_field(host_field, geometry)
        downstream_potential = self._build_downstream_potential(
            host_field=host_field,
            geometry=geometry,
            support_field=support_field,
        )

        backbone_source = min(
            source_cells,
            key=lambda cell: abs(float(geometry.cross_grid[cell])),
        )
        backbone_path = self._trace_backbone_path(
            host_field=host_field,
            geometry=geometry,
            support_field=support_field,
            start_cell=backbone_source,
            downstream_potential=downstream_potential,
        )
        if not backbone_path:
            return CaveNetwork(
                config=self.config,
                nodes=(),
                segments=(),
                occupancy=np.zeros_like(host_field.growth_cost, dtype=bool),
                width_field=np.zeros_like(host_field.growth_cost, dtype=float),
                dominant_route_node_ids=(),
                slice_along_positions=(),
                slice_channel_counts=(),
            )

        selected_paths: list[_SelectedPath] = [
            _SelectedPath(
                kind="backbone",
                path=tuple(self._simplify_path(backbone_path)),
                metadata=self._build_segment_metadata(kind="backbone"),
            )
        ]
        occupied_cells = set(backbone_path)
        backbone_alongs, backbone_crosses = self._build_backbone_profile(backbone_path, geometry)
        braid_zones = self._build_braid_zones(host_field, geometry)
        for zone_index, zone in enumerate(braid_zones):
            zone_paths = self._build_zone_paths(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                backbone_path=backbone_path,
                backbone_alongs=backbone_alongs,
                backbone_crosses=backbone_crosses,
                zone=zone,
                occupied_cells=occupied_cells,
                zone_index=zone_index,
            )
            for selected_path in zone_paths:
                selected_paths.append(selected_path)
                occupied_cells.update(selected_path.path[2:-2])

        _skeleton_mask, total_flux, _family_flux = self._build_representative_fields(
            shape=host_field.growth_cost.shape,
            selected_paths=tuple(selected_paths),
        )

        spur_starts = self._select_spur_start_cells(total_flux, geometry)
        for spur_index, start_cell in enumerate(spur_starts):
            path = self._trace_spur(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                start_cell=start_cell,
                total_flux=total_flux,
                lateral_sign=-1.0 if spur_index % 2 == 0 else 1.0,
                rng=rng,
            )
            if path:
                selected_paths.append(
                    _SelectedPath(
                        kind="spur",
                        path=tuple(self._simplify_path(path)),
                        metadata=self._build_segment_metadata(kind="spur"),
                    )
                )

        skeleton_mask, selected_flux, selected_family_flux = self._build_representative_fields(
            shape=host_field.growth_cost.shape,
            selected_paths=tuple(selected_paths),
        )

        nodes, segments, dominant_route_node_ids = self._extract_graph_from_paths(
            host_field=host_field,
            geometry=geometry,
            selected_paths=tuple(selected_paths),
            total_flux=selected_flux,
        )

        occupancy = np.zeros_like(host_field.growth_cost, dtype=bool)
        width_field = np.zeros_like(host_field.growth_cost, dtype=float)
        for segment in segments:
            self._rasterize_segment(host_field, occupancy, width_field, segment)
        self._paint_structural_chambers(host_field, occupancy, width_field, nodes, segments)
        self._paint_chambers(host_field, occupancy, width_field, total_flux)
        occupancy = self._smooth_occupancy(occupancy)

        slice_along_positions, slice_channel_counts = self._measure_parallel_channels(
            host_field=host_field,
            geometry=geometry,
            mask=skeleton_mask,
        )

        return CaveNetwork(
            config=self.config,
            nodes=tuple(nodes),
            segments=tuple(segments),
            occupancy=occupancy,
            width_field=width_field,
            dominant_route_node_ids=dominant_route_node_ids,
            slice_along_positions=slice_along_positions,
            slice_channel_counts=slice_channel_counts,
        )

    def _build_flow_geometry(self, host_field: HostField) -> _FlowGeometry:
        angle_radians = math.radians(host_field.config.flow_angle_degrees)
        flow_x = math.cos(angle_radians)
        flow_y = math.sin(angle_radians)
        cross_x = math.cos(angle_radians + math.pi / 2.0)
        cross_y = math.sin(angle_radians + math.pi / 2.0)
        seed_x, seed_y = host_field.config.seed_point

        x_grid, y_grid = np.meshgrid(host_field.x_coords, host_field.y_coords)
        along_grid = (x_grid - seed_x) * flow_x + (y_grid - seed_y) * flow_y
        cross_grid = (x_grid - seed_x) * cross_x + (y_grid - seed_y) * cross_y
        along_extent = float(np.max(along_grid)) - self.config.sink_margin
        cell_scale = math.hypot(
            float(host_field.x_coords[1] - host_field.x_coords[0]),
            float(host_field.y_coords[1] - host_field.y_coords[0]),
        )
        return _FlowGeometry(
            flow_x=flow_x,
            flow_y=flow_y,
            cross_x=cross_x,
            cross_y=cross_y,
            seed_x=seed_x,
            seed_y=seed_y,
            along_grid=along_grid,
            cross_grid=cross_grid,
            along_extent=along_extent,
            cell_scale=cell_scale,
        )

    def _build_support_field(self, host_field: HostField, geometry: _FlowGeometry) -> np.ndarray:
        cover_norm = self._normalize_cover_field(host_field)
        slope_norm = np.clip(host_field.slope_degrees / 25.0, 0.0, 1.0)
        corridor_score = np.exp(
            -np.square(
                geometry.cross_grid / max(host_field.config.corridor_width, 1.0)
            )
        )
        support = (
            self.config.growth_cost_weight * (1.0 - host_field.growth_cost)
            + self.config.roof_weight * host_field.roof_competence
            + self.config.cover_weight * cover_norm
            - self.config.slope_penalty_weight * slope_norm
            + self.config.corridor_weight * corridor_score
        )
        return support

    def _select_source_cells(
        self,
        host_field: HostField,
        geometry: _FlowGeometry,
    ) -> tuple[tuple[int, int], ...]:
        support = self._build_support_field(host_field, geometry)
        source_band = (
            (geometry.along_grid >= 0.0)
            & (geometry.along_grid <= self.config.source_band_length)
        )
        cross_band = np.abs(geometry.cross_grid) <= self.config.source_band_half_width
        candidate_mask = source_band & cross_band
        candidate_indices = np.argwhere(candidate_mask)
        if candidate_indices.size == 0:
            source_cell = self._world_to_cell(
                host_field,
                host_field.config.seed_point[0],
                host_field.config.seed_point[1],
            )
            return (source_cell,)

        scored_candidates = []
        for y_index, x_index in candidate_indices:
            downhill_x, downhill_y = host_field.downhill_direction(
                float(host_field.x_coords[x_index]),
                float(host_field.y_coords[y_index]),
                fallback_angle_degrees=host_field.config.flow_angle_degrees,
            )
            flow_alignment = downhill_x * geometry.flow_x + downhill_y * geometry.flow_y
            score = float(support[y_index, x_index]) + 0.35 * flow_alignment
            scored_candidates.append(((int(y_index), int(x_index)), score))
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        selected: list[tuple[int, int]] = []
        lateral_separation = max(2.0 * geometry.cell_scale, 18.0)
        for (y_index, x_index), _score in scored_candidates:
            x_coord = float(host_field.x_coords[x_index])
            y_coord = float(host_field.y_coords[y_index])
            cross_position = self._project_cross(geometry, x_coord, y_coord)
            if any(
                abs(
                    cross_position
                    - self._project_cross(
                        geometry,
                        float(host_field.x_coords[selected_x]),
                        float(host_field.y_coords[selected_y]),
                    )
                )
                < lateral_separation
                for selected_y, selected_x in selected
            ):
                continue
            selected.append((y_index, x_index))
            if len(selected) >= self.config.source_count:
                break

        if not selected:
            selected.append(
                self._world_to_cell(
                    host_field,
                    host_field.config.seed_point[0],
                    host_field.config.seed_point[1],
                )
            )
        return tuple(selected)

    def _trace_downstream(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        downstream_potential: np.ndarray,
        start_cell: tuple[int, int],
        total_flux: np.ndarray,
        family: _TraceFamily,
        rng,
    ) -> list[tuple[int, int]]:
        path = [start_cell]
        previous_step: tuple[float, float] | None = None

        for _ in range(self.config.trace_max_steps):
            current = path[-1]
            current_world = self._cell_to_world(host_field, current)
            current_elevation = float(host_field.elevation[current])
            current_along = float(geometry.along_grid[current])
            current_potential = float(downstream_potential[current])
            if current_along >= geometry.along_extent:
                break
            if not math.isfinite(current_potential):
                break

            downhill_x, downhill_y = host_field.downhill_direction(
                current_world[0],
                current_world[1],
                fallback_angle_degrees=host_field.config.flow_angle_degrees,
            )

            candidates: list[tuple[tuple[int, int], float]] = []
            for next_cell in self._neighbor_cells(host_field, current):
                if next_cell in path[-4:]:
                    continue
                next_world = self._cell_to_world(host_field, next_cell)
                step_x = next_world[0] - current_world[0]
                step_y = next_world[1] - current_world[1]
                step_length = math.hypot(step_x, step_y)
                if math.isclose(step_length, 0.0):
                    continue
                step_unit_x = step_x / step_length
                step_unit_y = step_y / step_length
                flow_alignment = step_unit_x * geometry.flow_x + step_unit_y * geometry.flow_y
                next_along = float(geometry.along_grid[next_cell])
                along_delta = next_along - current_along
                if flow_alignment < -0.05 or along_delta < -0.15 * geometry.cell_scale:
                    continue
                next_potential = float(downstream_potential[next_cell])
                if not math.isfinite(next_potential):
                    continue
                next_elevation = float(host_field.elevation[next_cell])
                uphill = next_elevation - current_elevation
                if uphill > self.config.max_uphill_step:
                    continue
                downhill_gain = max(current_elevation - next_elevation, 0.0)

                downhill_alignment = step_unit_x * downhill_x + step_unit_y * downhill_y
                if downhill_alignment < -0.35:
                    continue
                score = float(support_field[next_cell])
                score += self.config.forward_alignment_weight * flow_alignment
                score += 1.15 * along_delta / max(geometry.cell_scale, 1.0)
                score += 0.45 * next_along / max(geometry.along_extent, 1.0)
                score += self.config.outlet_potential_weight * (
                    (current_potential - next_potential) / max(geometry.cell_scale, 1.0)
                )
                score += self.config.downhill_alignment_weight * downhill_alignment
                score += self.config.elevation_drop_weight * downhill_gain / max(geometry.cell_scale, 1.0)
                score += family.attraction_weight * math.log1p(float(total_flux[next_cell]))
                score -= family.congestion_weight * max(
                    0.0,
                    float(total_flux[next_cell]) - family.congestion_threshold,
                )
                if previous_step is not None:
                    previous_length = math.hypot(previous_step[0], previous_step[1])
                    if previous_length > 0.0:
                        score += self.config.inertia_weight * (
                            (step_x * previous_step[0] + step_y * previous_step[1])
                            / (step_length * previous_length)
                        )

                candidates.append((next_cell, score))

            if not candidates:
                break

            next_cell = self._sample_candidate(candidates, family.temperature, rng)
            next_world = self._cell_to_world(host_field, next_cell)
            previous_step = (
                next_world[0] - current_world[0],
                next_world[1] - current_world[1],
            )
            path.append(next_cell)

        if path and float(geometry.along_grid[path[-1]]) < geometry.along_extent:
            path = self._extend_path_to_sink(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                downstream_potential=downstream_potential,
                path=path,
            )

        return path if len(path) > 2 else []

    def _trace_spur(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        start_cell: tuple[int, int],
        total_flux: np.ndarray,
        lateral_sign: float,
        rng,
    ) -> list[tuple[int, int]]:
        path = [start_cell]
        current_world = self._cell_to_world(host_field, start_cell)
        side_target_x = geometry.cross_x * lateral_sign
        side_target_y = geometry.cross_y * lateral_sign

        for _ in range(self.config.spur_max_steps):
            current = path[-1]
            current_world = self._cell_to_world(host_field, current)
            current_elevation = float(host_field.elevation[current])
            candidates: list[tuple[tuple[int, int], float]] = []
            for next_cell in self._neighbor_cells(host_field, current):
                if next_cell in path[-3:]:
                    continue
                next_world = self._cell_to_world(host_field, next_cell)
                step_x = next_world[0] - current_world[0]
                step_y = next_world[1] - current_world[1]
                step_length = math.hypot(step_x, step_y)
                if math.isclose(step_length, 0.0):
                    continue
                next_elevation = float(host_field.elevation[next_cell])
                uphill = next_elevation - current_elevation
                if uphill > self.config.max_uphill_step:
                    continue
                step_unit_x = step_x / step_length
                step_unit_y = step_y / step_length
                side_alignment = step_unit_x * side_target_x + step_unit_y * side_target_y
                score = float(support_field[next_cell])
                score += self.config.spur_lateral_bias * side_alignment
                score -= self.config.spur_congestion_weight * float(total_flux[next_cell])
                candidates.append((next_cell, score))

            if not candidates:
                break

            next_cell = self._sample_candidate(candidates, 0.45, rng)
            path.append(next_cell)
            if float(total_flux[next_cell]) <= 0.0 and len(path) > 8:
                break

        return path if len(path) > 4 else []

    @staticmethod
    def _deposit_path(path, total_flux: np.ndarray, family_flux: np.ndarray) -> None:
        for cell in path:
            total_flux[cell] += 1.0
            family_flux[cell] += 1.0

    def _build_backbone_profile(
        self,
        path: list[tuple[int, int]],
        geometry: _FlowGeometry,
    ) -> tuple[np.ndarray, np.ndarray]:
        profile = sorted(
            (
                float(geometry.along_grid[cell]),
                float(geometry.cross_grid[cell]),
            )
            for cell in path
        )
        alongs: list[float] = []
        crosses: list[float] = []
        for along, cross in profile:
            if alongs and math.isclose(along, alongs[-1], abs_tol=0.25 * geometry.cell_scale):
                crosses[-1] = 0.5 * (crosses[-1] + cross)
                continue
            alongs.append(along)
            crosses.append(cross)
        return np.array(alongs, dtype=float), np.array(crosses, dtype=float)

    def _trace_backbone_path(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        downstream_potential: np.ndarray,
        start_cell: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [start_cell]
        previous_step: tuple[float, float] | None = None
        max_cross = max(0.55 * host_field.config.corridor_width, 90.0)

        for _ in range(self.config.trace_max_steps + 240):
            current = path[-1]
            current_world = self._cell_to_world(host_field, current)
            current_along = float(geometry.along_grid[current])
            current_potential = float(downstream_potential[current])
            current_elevation = float(host_field.elevation[current])
            if current_along >= geometry.along_extent or not math.isfinite(current_potential):
                break

            downhill_x, downhill_y = host_field.downhill_direction(
                current_world[0],
                current_world[1],
                fallback_angle_degrees=host_field.config.flow_angle_degrees,
            )
            best_candidate: tuple[tuple[int, int], float] | None = None
            for next_cell in self._neighbor_cells(host_field, current):
                if next_cell in path[-6:]:
                    continue
                next_along = float(geometry.along_grid[next_cell])
                if next_along < current_along - 0.2 * geometry.cell_scale:
                    continue
                next_cross = float(geometry.cross_grid[next_cell])
                if abs(next_cross) > max_cross:
                    continue
                next_world = self._cell_to_world(host_field, next_cell)
                step_x = next_world[0] - current_world[0]
                step_y = next_world[1] - current_world[1]
                step_length = math.hypot(step_x, step_y)
                if math.isclose(step_length, 0.0):
                    continue
                step_unit_x = step_x / step_length
                step_unit_y = step_y / step_length
                flow_alignment = step_unit_x * geometry.flow_x + step_unit_y * geometry.flow_y
                downhill_alignment = step_unit_x * downhill_x + step_unit_y * downhill_y
                next_potential = float(downstream_potential[next_cell])
                if not math.isfinite(next_potential):
                    continue
                next_elevation = float(host_field.elevation[next_cell])
                uphill = next_elevation - current_elevation
                if uphill > self.config.max_uphill_step:
                    continue

                score = float(support_field[next_cell])
                score += 1.8 * flow_alignment
                score += 2.2 * downhill_alignment
                score += 4.0 * (current_potential - next_potential) / max(geometry.cell_scale, 1.0)
                score += 0.6 * next_along / max(geometry.along_extent, 1.0)
                score -= 0.85 * abs(next_cross) / max(max_cross, geometry.cell_scale)
                if previous_step is not None:
                    previous_length = math.hypot(previous_step[0], previous_step[1])
                    if previous_length > 0.0:
                        score += 0.9 * (
                            (step_x * previous_step[0] + step_y * previous_step[1])
                            / (step_length * previous_length)
                        )
                if best_candidate is None or score > best_candidate[1]:
                    best_candidate = (next_cell, score)

            if best_candidate is None:
                break
            next_cell = best_candidate[0]
            next_world = self._cell_to_world(host_field, next_cell)
            previous_step = (
                next_world[0] - current_world[0],
                next_world[1] - current_world[1],
            )
            path.append(next_cell)

        if path and float(geometry.along_grid[path[-1]]) < geometry.along_extent:
            path = self._extend_path_to_sink(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                downstream_potential=downstream_potential,
                path=path,
            )
        return path

    def _build_braid_zones(
        self,
        host_field: HostField,
        geometry: _FlowGeometry,
    ) -> tuple[_BraidZone, ...]:
        spread = 0.34 * host_field.config.corridor_width
        return (
            _BraidZone(
                0.16,
                0.065,
                branches=(
                    _ZoneBranch("island_bypass", -0.95 * spread, -0.10, 0.18, skew=-0.45, wobble=0.28, phase=0.7),
                    _ZoneBranch("island_bypass", 0.58 * spread, 0.08, -0.05, skew=0.20, wobble=0.16, phase=2.1),
                ),
                chamber_radius_scale=1.15,
            ),
            _BraidZone(
                0.33,
                0.055,
                branches=(
                    _ZoneBranch("chamber_braid", 0.92 * spread, -0.18, 0.12, skew=0.50, wobble=0.26, phase=1.4),
                    _ZoneBranch("ladder", -0.42 * spread, 0.08, -0.08, skew=-0.10, wobble=0.12, phase=2.7),
                ),
                ladder_rungs=(0.34, 0.68),
                chamber_radius_scale=1.35,
            ),
            _BraidZone(
                0.51,
                0.078,
                branches=(
                    _ZoneBranch("island_bypass", -1.05 * spread, -0.05, 0.10, skew=-0.35, wobble=0.22, phase=0.4),
                    _ZoneBranch("underpass", 0.78 * spread, 0.22, -0.18, skew=0.65, wobble=0.30, phase=1.9, z_level=-1, merge_shared_cells=False),
                    _ZoneBranch("inner_bypass", 0.22 * spread, -0.10, 0.02, skew=0.15, wobble=0.10, phase=2.8),
                ),
                chamber_radius_scale=1.1,
            ),
            _BraidZone(
                0.69,
                0.085,
                branches=(
                    _ZoneBranch("chamber_braid", -1.15 * spread, -0.22, 0.16, skew=-0.55, wobble=0.34, phase=1.2),
                    _ZoneBranch("island_bypass", 0.52 * spread, 0.06, -0.18, skew=0.18, wobble=0.18, phase=2.2),
                    _ZoneBranch("ladder", 0.98 * spread, 0.18, -0.05, skew=0.38, wobble=0.26, phase=0.9),
                ),
                ladder_rungs=(0.28,),
                chamber_radius_scale=1.5,
            ),
            _BraidZone(
                0.86,
                0.055,
                branches=(
                    _ZoneBranch("island_bypass", -0.62 * spread, -0.08, 0.02, skew=-0.15, wobble=0.12, phase=0.3),
                    _ZoneBranch("underpass", 0.96 * spread, 0.20, -0.12, skew=0.55, wobble=0.20, phase=2.5, z_level=1, merge_shared_cells=False),
                ),
                chamber_radius_scale=1.0,
            ),
        )

    def _build_zone_paths(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        backbone_path: list[tuple[int, int]],
        backbone_alongs: np.ndarray,
        backbone_crosses: np.ndarray,
        zone: _BraidZone,
        occupied_cells: set[tuple[int, int]],
        zone_index: int,
    ) -> tuple[_SelectedPath, ...]:
        center_along = zone.center_fraction * geometry.along_extent
        half_length = zone.half_length_fraction * geometry.along_extent
        zone_paths: list[_SelectedPath] = []
        built_branch_paths: list[tuple[_ZoneBranch, tuple[tuple[int, int], ...]]] = []
        for branch in zone.branches:
            start_along = center_along - half_length + branch.start_shift_fraction * half_length
            end_along = center_along + half_length + branch.end_shift_fraction * half_length
            backbone_segment = self._extract_backbone_segment(
                backbone_path=backbone_path,
                geometry=geometry,
                start_along=start_along,
                end_along=end_along,
            )
            if len(backbone_segment) < 8:
                continue
            path = self._build_offset_zone_path(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                backbone_segment=backbone_segment,
                backbone_alongs=backbone_alongs,
                backbone_crosses=backbone_crosses,
                branch=branch,
            )
            if not path:
                continue
            simplified = self._simplify_path(path)
            if len(simplified) < 5:
                continue
            max_cross_delta = max(
                abs(
                    float(geometry.cross_grid[cell])
                    - float(np.interp(float(geometry.along_grid[cell]), backbone_alongs, backbone_crosses))
                    )
                for cell in simplified
            )
            if max_cross_delta < max(0.18 * host_field.config.corridor_width, 18.0):
                continue
            selected = _SelectedPath(
                kind=branch.kind,
                path=tuple(simplified),
                z_level=branch.z_level,
                merge_shared_cells=branch.merge_shared_cells,
                metadata=self._build_segment_metadata(
                    kind=branch.kind,
                    zone_index=zone_index,
                    z_level=branch.z_level,
                ),
            )
            zone_paths.append(selected)
            built_branch_paths.append((branch, selected.path))

        if zone.ladder_rungs and built_branch_paths:
            ladder_paths = self._build_zone_ladders(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                backbone_alongs=backbone_alongs,
                backbone_crosses=backbone_crosses,
                branches=built_branch_paths,
                rungs=zone.ladder_rungs,
                zone_index=zone_index,
            )
            zone_paths.extend(ladder_paths)
        return tuple(zone_paths)

    def _extract_backbone_segment(
        self,
        *,
        backbone_path: list[tuple[int, int]],
        geometry: _FlowGeometry,
        start_along: float,
        end_along: float,
    ) -> list[tuple[int, int]]:
        segment = [
            cell
            for cell in backbone_path
            if start_along <= float(geometry.along_grid[cell]) <= end_along
        ]
        if not segment:
            start_cell = self._cell_on_path_at_along(backbone_path, geometry, start_along)
            end_cell = self._cell_on_path_at_along(backbone_path, geometry, end_along)
            return [start_cell, end_cell]
        first_index = backbone_path.index(segment[0])
        last_index = backbone_path.index(segment[-1])
        return backbone_path[first_index : last_index + 1]

    def _build_offset_zone_path(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        backbone_segment: list[tuple[int, int]],
        backbone_alongs: np.ndarray,
        backbone_crosses: np.ndarray,
        branch: _ZoneBranch,
    ) -> list[tuple[int, int]]:
        if len(backbone_segment) < 2:
            return []
        built_path = [backbone_segment[0]]
        segment_start = float(geometry.along_grid[backbone_segment[0]])
        segment_end = float(geometry.along_grid[backbone_segment[-1]])
        along_span = max(segment_end - segment_start, geometry.cell_scale)

        left_exponent = max(0.35, 1.0 - branch.skew)
        right_exponent = max(0.35, 1.0 + branch.skew)
        peak_t = left_exponent / max(left_exponent + right_exponent, 1e-6)
        peak_raw = max(
            peak_t ** left_exponent * (1.0 - peak_t) ** right_exponent,
            1e-6,
        )

        for index, cell in enumerate(backbone_segment[1:-1], start=1):
            along = float(geometry.along_grid[cell])
            progress = (along - segment_start) / along_span
            clamped_progress = float(np.clip(progress, 0.0, 1.0))
            envelope = (
                clamped_progress ** left_exponent
                * (1.0 - clamped_progress) ** right_exponent
            ) / peak_raw
            wobble = branch.wobble * envelope * math.sin(
                2.0 * math.pi * (1.25 * clamped_progress + branch.phase)
            )
            target_cross = (
                float(np.interp(along, backbone_alongs, backbone_crosses))
                + branch.lateral_offset * envelope
                + 0.35 * host_field.config.corridor_width * wobble
            )
            target_x = geometry.seed_x + geometry.flow_x * along + geometry.cross_x * target_cross
            target_y = geometry.seed_y + geometry.flow_y * along + geometry.cross_y * target_cross
            snapped = self._snap_target_cell(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                target_x=target_x,
                target_y=target_y,
                target_cross=target_cross,
            )
            if snapped == built_path[-1]:
                continue
            built_path.append(snapped)
        if backbone_segment[-1] != built_path[-1]:
            built_path.append(backbone_segment[-1])
        return built_path

    def _build_zone_ladders(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        backbone_alongs: np.ndarray,
        backbone_crosses: np.ndarray,
        branches: list[tuple[_ZoneBranch, tuple[tuple[int, int], ...]]],
        rungs: tuple[float, ...],
        zone_index: int,
    ) -> tuple[_SelectedPath, ...]:
        if len(branches) < 2:
            return ()
        ordered = sorted(
            branches,
            key=lambda item: np.mean([float(geometry.cross_grid[cell]) for cell in item[1]]),
        )
        ladders: list[_SelectedPath] = []
        for rung_fraction in rungs:
            left_branch = ordered[0][1]
            right_branch = ordered[-1][1]
            left_cell = left_branch[min(int(rung_fraction * (len(left_branch) - 1)), len(left_branch) - 1)]
            right_cell = right_branch[min(int(rung_fraction * (len(right_branch) - 1)), len(right_branch) - 1)]
            connector = self._build_connector_path(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                start_cell=left_cell,
                end_cell=right_cell,
                backbone_alongs=backbone_alongs,
                backbone_crosses=backbone_crosses,
            )
            simplified = self._simplify_path(connector)
            if len(simplified) < 3:
                continue
            ladders.append(
                _SelectedPath(
                    kind="ladder",
                    path=tuple(simplified),
                    metadata=self._build_segment_metadata(
                        kind="ladder",
                        zone_index=zone_index,
                    ),
                )
            )
        return tuple(ladders)

    def _build_connector_path(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        start_cell: tuple[int, int],
        end_cell: tuple[int, int],
        backbone_alongs: np.ndarray,
        backbone_crosses: np.ndarray,
    ) -> list[tuple[int, int]]:
        start_world = self._cell_to_world(host_field, start_cell)
        end_world = self._cell_to_world(host_field, end_cell)
        samples = max(
            3,
            int(
                math.ceil(
                    math.hypot(end_world[0] - start_world[0], end_world[1] - start_world[1])
                    / max(geometry.cell_scale, 1.0)
                )
            ),
        )
        path = [start_cell]
        for sample_index in range(1, samples):
            t = sample_index / samples
            x_coord = (1.0 - t) * start_world[0] + t * end_world[0]
            y_coord = (1.0 - t) * start_world[1] + t * end_world[1]
            target_along = (
                (1.0 - t) * float(geometry.along_grid[start_cell])
                + t * float(geometry.along_grid[end_cell])
            )
            target_cross = (
                0.35
                * (
                    float(geometry.cross_grid[start_cell])
                    + float(geometry.cross_grid[end_cell])
                )
                + 0.65 * float(np.interp(target_along, backbone_alongs, backbone_crosses))
            )
            snapped = self._snap_target_cell(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                target_x=x_coord,
                target_y=y_coord,
                target_cross=target_cross,
            )
            if snapped != path[-1]:
                path.append(snapped)
        if path[-1] != end_cell:
            path.append(end_cell)
        return path

    def _snap_target_cell(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        target_x: float,
        target_y: float,
        target_cross: float,
    ) -> tuple[int, int]:
        target_cell = self._world_to_cell(
            host_field,
            min(max(target_x, float(host_field.x_coords[0])), float(host_field.x_coords[-1])),
            min(max(target_y, float(host_field.y_coords[0])), float(host_field.y_coords[-1])),
        )
        best_cell = target_cell
        best_score = -math.inf
        for candidate in self._neighbor_cells(host_field, target_cell) + [target_cell]:
            candidate_cross = float(geometry.cross_grid[candidate])
            candidate_world = self._cell_to_world(host_field, candidate)
            distance_penalty = 0.018 * math.hypot(candidate_world[0] - target_x, candidate_world[1] - target_y)
            cross_penalty = 0.05 * abs(candidate_cross - target_cross)
            score = float(support_field[candidate]) - distance_penalty - cross_penalty
            if score > best_score:
                best_score = score
                best_cell = candidate
        return best_cell

    def _cell_on_path_at_along(
        self,
        path: list[tuple[int, int]],
        geometry: _FlowGeometry,
        target_along: float,
    ) -> tuple[int, int]:
        return min(
            path,
            key=lambda cell: abs(float(geometry.along_grid[cell]) - target_along),
        )

    def _select_sink_cell(
        self,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
    ) -> tuple[int, int]:
        sink_mask = geometry.along_grid >= max(0.92 * geometry.along_extent, geometry.along_extent - 2.0 * geometry.cell_scale)
        sink_candidates = np.argwhere(sink_mask)
        if sink_candidates.size == 0:
            sink_candidates = np.argwhere(geometry.along_grid == np.max(geometry.along_grid))
        return min(
            ((int(y_index), int(x_index)) for y_index, x_index in sink_candidates),
            key=lambda cell: (
                abs(float(geometry.cross_grid[cell])),
                -float(support_field[cell]),
            ),
        )

    def _find_guided_connection(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        start_cell: tuple[int, int],
        end_cell: tuple[int, int],
        backbone_alongs: np.ndarray,
        backbone_crosses: np.ndarray,
        lateral_offset: float,
        occupied_cells: set[tuple[int, int]],
        zone_start_along: float,
        zone_end_along: float,
        zone_half_width: float,
    ) -> list[tuple[int, int]]:
        open_heap: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start_cell)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        best_cost = {start_cell: 0.0}
        allowed_margin = 2.5 * geometry.cell_scale

        while open_heap:
            _priority, current_cost, current = heapq.heappop(open_heap)
            if current == end_cell:
                break
            if current_cost > best_cost.get(current, math.inf):
                continue

            current_along = float(geometry.along_grid[current])
            for next_cell in self._neighbor_cells(host_field, current):
                next_along = float(geometry.along_grid[next_cell])
                if next_along < current_along - 0.35 * geometry.cell_scale:
                    continue
                if next_along < zone_start_along - 2.0 * geometry.cell_scale:
                    continue
                if next_along > zone_end_along + 2.0 * geometry.cell_scale:
                    continue
                target_cross = float(np.interp(next_along, backbone_alongs, backbone_crosses)) + lateral_offset
                cross_delta = abs(float(geometry.cross_grid[next_cell]) - target_cross)
                if cross_delta > zone_half_width + allowed_margin:
                    continue

                on_backbone = next_cell in occupied_cells and next_cell not in {start_cell, end_cell}
                if on_backbone and cross_delta < 0.35 * zone_half_width:
                    continue

                transition = self._transition_cost(
                    host_field=host_field,
                    geometry=geometry,
                    support_field=support_field,
                    current_cell=current,
                    next_cell=next_cell,
                )
                cross_penalty = 0.75 * cross_delta / max(zone_half_width, geometry.cell_scale)
                occupancy_penalty = 0.35 if on_backbone else 0.0
                next_cost = current_cost + transition + cross_penalty + occupancy_penalty
                if next_cost >= best_cost.get(next_cell, math.inf):
                    continue

                best_cost[next_cell] = next_cost
                came_from[next_cell] = current
                heuristic = 0.24 * math.hypot(
                    float(host_field.x_coords[end_cell[1]] - host_field.x_coords[next_cell[1]]),
                    float(host_field.y_coords[end_cell[0]] - host_field.y_coords[next_cell[0]]),
                )
                heapq.heappush(open_heap, (next_cost + heuristic, next_cost, next_cell))

        if end_cell not in came_from:
            return []
        path = [end_cell]
        current = end_cell
        while current != start_cell:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _path_length_cells(
        self,
        path: list[tuple[int, int]],
        host_field: HostField,
    ) -> float:
        if len(path) < 2:
            return 0.0
        length = 0.0
        for current, next_cell in zip(path, path[1:]):
            current_world = self._cell_to_world(host_field, current)
            next_world = self._cell_to_world(host_field, next_cell)
            length += math.hypot(next_world[0] - current_world[0], next_world[1] - current_world[1])
        return length

    def _simplify_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(path) <= 2:
            return list(path)
        simplified = [path[0]]
        for previous, current, next_cell in zip(path, path[1:], path[2:]):
            delta_a = (current[0] - previous[0], current[1] - previous[1])
            delta_b = (next_cell[0] - current[0], next_cell[1] - current[1])
            if delta_a == delta_b:
                continue
            simplified.append(current)
        simplified.append(path[-1])
        return simplified

    def _build_skeleton_mask(
        self,
        total_flux: np.ndarray,
        family_flux: dict[str, np.ndarray],
    ) -> np.ndarray:
        mask = np.zeros_like(total_flux, dtype=bool)
        thresholds = {
            "small": self._quantile_threshold(
                family_flux["small"],
                self.config.small_flux_threshold_quantile,
                minimum=3.0,
            ),
            "medium": self._quantile_threshold(
                family_flux["medium"],
                self.config.medium_flux_threshold_quantile,
                minimum=2.0,
            ),
            "large": self._quantile_threshold(
                family_flux["large"],
                self.config.large_flux_threshold_quantile,
                minimum=1.0,
            ),
            "spur": self._quantile_threshold(family_flux["spur"], 0.45, minimum=1.0),
            "total": self._quantile_threshold(
                total_flux,
                self.config.total_flux_threshold_quantile,
                minimum=4.0,
            ),
        }
        for label in self.FAMILY_LABELS:
            mask |= family_flux[label] >= thresholds[label]
        mask |= total_flux >= thresholds["total"]
        local_max = self._local_maximum(total_flux)
        ridge_mask = total_flux >= (0.92 * local_max)
        ridge_mask |= total_flux >= max(thresholds["total"] * 1.35, 6.0)
        return mask & ridge_mask

    def _select_representative_paths(
        self,
        *,
        traced_paths: list[tuple[str, list[tuple[int, int]]]],
        total_flux: np.ndarray,
        geometry: _FlowGeometry,
    ) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
        selected: list[tuple[str, tuple[tuple[int, int], ...]]] = []
        selected_sets: list[set[tuple[int, int]]] = []
        targets = {
            "small": self.config.selected_small_paths,
            "medium": self.config.selected_medium_paths,
            "large": self.config.selected_large_paths,
            "spur": self.config.selected_spur_paths,
        }
        non_spur_candidates = [
            (path_label, path, self._score_path(path, total_flux, geometry, path_label))
            for path_label, path in traced_paths
            if path_label != "spur" and path
        ]
        if non_spur_candidates:
            best_label, best_path, _score = max(non_spur_candidates, key=lambda item: item[2])
            selected.append((best_label, tuple(best_path)))
            selected_sets.append(set(best_path))
        for label in ("large", "medium", "small", "spur"):
            selected_count = sum(1 for selected_label, _ in selected if selected_label == label)
            for minimum_progress in (0.72, 0.58, 0.42, 0.22, -math.inf):
                candidates = [
                    (path, self._score_path(path, total_flux, geometry, label))
                    for path_label, path in traced_paths
                    if path_label == label
                    and path
                    and (
                        label == "spur"
                        or float(geometry.along_grid[path[-1]]) >= minimum_progress * geometry.along_extent
                    )
                ]
                candidates.sort(key=lambda item: item[1], reverse=True)
                for path, _score in candidates:
                    path_set = set(path)
                    if any(
                        len(path_set & other_set) / max(len(path_set), 1) > self.config.maximum_path_overlap
                        for other_set in selected_sets
                    ):
                        continue
                    selected.append((label, tuple(path)))
                    selected_sets.append(path_set)
                    selected_count += 1
                    if selected_count >= targets[label]:
                        break
                if selected_count >= targets[label]:
                    break
        return tuple(selected)

    @staticmethod
    def _score_path(
        path: list[tuple[int, int]],
        total_flux: np.ndarray,
        geometry: _FlowGeometry,
        label: str,
    ) -> float:
        if len(path) < 2:
            return -math.inf
        start_along = float(geometry.along_grid[path[0]])
        end_along = float(geometry.along_grid[path[-1]])
        progress = end_along - start_along
        unique_cells = len(set(path))
        mean_flux = float(np.mean([total_flux[cell] for cell in path]))
        label_bonus = {"small": 0.0, "medium": 25.0, "large": 50.0, "spur": -20.0}[label]
        return progress + 0.35 * unique_cells + 1.5 * mean_flux + label_bonus

    @staticmethod
    def _family_label_for_kind(kind: str) -> str:
        if kind == "spur":
            return "spur"
        if kind in {"backbone", "chamber_braid"}:
            return "large"
        if kind in {"island_bypass", "underpass"}:
            return "medium"
        return "small"

    @staticmethod
    def _build_segment_metadata(
        *,
        kind: str,
        zone_index: int | None = None,
        z_level: int = 0,
    ) -> dict[str, SegmentMetadataValue]:
        crossing_group_id: str | None = None
        merge_behavior = "merge"
        island_id: str | None = None
        chamber_id: str | None = None

        if kind in {"island_bypass", "inner_bypass"} and zone_index is not None:
            island_id = f"island_zone_{zone_index}"
        if kind in {"chamber_braid", "ladder"} and zone_index is not None:
            chamber_id = f"chamber_zone_{zone_index}"
        if kind == "underpass" and zone_index is not None:
            crossing_group_id = f"crossing_zone_{zone_index}"
            merge_behavior = "cross_under" if z_level < 0 else "cross_over"

        return {
            "crossing_group_id": crossing_group_id,
            "merge_behavior": merge_behavior,
            "island_id": island_id,
            "chamber_id": chamber_id,
            "formation_origin": kind,
        }

    def _build_representative_fields(
        self,
        *,
        shape: tuple[int, int],
        selected_paths: tuple[_SelectedPath, ...],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        mask = np.zeros(shape, dtype=bool)
        flux = np.zeros(shape, dtype=float)
        family_flux = {
            label: np.zeros(shape, dtype=float)
            for label in self.FAMILY_LABELS
        }
        for selected_path in selected_paths:
            family_label = self._family_label_for_kind(selected_path.kind)
            for cell in selected_path.path:
                mask[cell] = True
                flux[cell] += 1.0
                family_flux[family_label][cell] += 1.0
        return mask, flux, family_flux

    def _prune_skeleton_mask(self, mask: np.ndarray) -> np.ndarray:
        current = mask.copy()
        for _ in range(self.config.prune_iterations):
            neighbor_count = self._neighbor_count(current)
            current = np.where(current, neighbor_count >= 2, False)
        return current

    def _build_family_label_grid(self, family_flux: dict[str, np.ndarray]) -> np.ndarray:
        stacked = np.stack(
            [family_flux[label] for label in self.FAMILY_LABELS],
            axis=0,
        )
        dominant = np.argmax(stacked, axis=0)
        support = stacked.max(axis=0)
        dominant = np.where(support > 0.0, dominant, -1)
        return dominant

    def _extract_graph_from_paths(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        selected_paths: tuple[_SelectedPath, ...],
        total_flux: np.ndarray,
    ) -> tuple[list[CaveNode], list[CaveSegment], tuple[int, ...]]:
        if not selected_paths:
            return [], [], ()

        path_use_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        all_path_cells: set[tuple[int, int]] = set()
        for selected_path in selected_paths:
            all_path_cells.update(selected_path.path)
            if not selected_path.merge_shared_cells:
                continue
            for cell in selected_path.path:
                path_use_counts[cell] += 1

        source_cell = min(
            all_path_cells,
            key=lambda cell: (float(geometry.along_grid[cell]), abs(float(geometry.cross_grid[cell]))),
        )
        sink_cell = max(
            all_path_cells,
            key=lambda cell: (float(geometry.along_grid[cell]), -abs(float(geometry.cross_grid[cell]))),
        )

        node_cell_set: set[tuple[int, int]] = {source_cell, sink_cell}
        chamber_cells: set[tuple[int, int]] = set()
        for selected_path in selected_paths:
            node_cell_set.add(selected_path.path[0])
            node_cell_set.add(selected_path.path[-1])
            if selected_path.kind in {"chamber_braid", "ladder"}:
                chamber_cells.update(selected_path.path)
        for cell, count in path_use_counts.items():
            if count > 1:
                node_cell_set.add(cell)

        node_cells: dict[tuple[int, int], int] = {}
        nodes: list[CaveNode] = []
        for cell in sorted(node_cell_set, key=lambda item: float(geometry.along_grid[item])):
            node_id = len(nodes)
            x_coord, y_coord = self._cell_to_world(host_field, cell)
            node_kind = "junction"
            if cell == source_cell:
                node_kind = "entry"
            elif cell == sink_cell:
                node_kind = "exit"
            elif cell in chamber_cells and path_use_counts.get(cell, 0) >= 2:
                node_kind = "chamber"
            elif path_use_counts.get(cell, 0) == 1:
                node_kind = "terminal"
            nodes.append(
                CaveNode(
                    node_id=node_id,
                    x=x_coord,
                    y=y_coord,
                    along_position=float(geometry.along_grid[cell]),
                    lateral_offset=float(geometry.cross_grid[cell]),
                    kind=node_kind,
                )
            )
            node_cells[cell] = node_id

        segments: list[CaveSegment] = []
        seen_signatures: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
        for selected_path in selected_paths:
            path = selected_path.path
            current_cells = [path[0]]
            for cell in path[1:]:
                current_cells.append(cell)
                if cell not in node_cells:
                    continue
                if (
                    not selected_path.merge_shared_cells
                    and cell not in {path[0], path[-1]}
                ):
                    continue
                start_node_id = node_cells[current_cells[0]]
                end_node_id = node_cells[cell]
                if start_node_id != end_node_id and len(current_cells) >= 2:
                    signature_cells = tuple(current_cells)
                    signature = (min(start_node_id, end_node_id), max(start_node_id, end_node_id), signature_cells)
                    reverse_signature = (
                        min(start_node_id, end_node_id),
                        max(start_node_id, end_node_id),
                        tuple(reversed(signature_cells)),
                    )
                    if signature not in seen_signatures and reverse_signature not in seen_signatures:
                        segment = self._build_segment_from_cells(
                            host_field=host_field,
                            path_cells=current_cells,
                            start_node_id=start_node_id,
                            end_node_id=end_node_id,
                            segment_id=len(segments),
                            total_flux=total_flux,
                            kind=selected_path.kind,
                            z_level=selected_path.z_level,
                            metadata=selected_path.metadata or self._build_segment_metadata(
                                kind=selected_path.kind,
                                z_level=selected_path.z_level,
                            ),
                        )
                        if segment is not None:
                            segments.append(segment)
                            seen_signatures.add(signature)
                current_cells = [cell]

        dominant_route_node_ids = self._dominant_route(nodes, segments, total_flux)
        return nodes, segments, dominant_route_node_ids

    def _build_segment_from_cells(
        self,
        *,
        host_field: HostField,
        path_cells: list[tuple[int, int]],
        start_node_id: int,
        end_node_id: int,
        segment_id: int,
        total_flux: np.ndarray,
        kind: str,
        z_level: int,
        metadata: dict[str, SegmentMetadataValue],
    ) -> CaveSegment | None:
        coordinates = [
            self._cell_to_world(host_field, cell)
            for cell in path_cells
        ]
        coordinates = self._deduplicate_coordinates(coordinates)
        if len(coordinates) < 2:
            return None

        points: list[CavePoint] = []
        arc_length = 0.0
        for index, ((x_coord, y_coord), cell) in enumerate(zip(coordinates, path_cells, strict=False)):
            if index > 0:
                previous_x, previous_y = coordinates[index - 1]
                arc_length += math.hypot(x_coord - previous_x, y_coord - previous_y)
            sample = host_field.sample(x_coord, y_coord)
            width = 2.0 * self._local_radius(host_field, sample, float(total_flux[cell]))
            points.append(
                CavePoint(
                    index=index,
                    x=x_coord,
                    y=y_coord,
                    elevation=sample.elevation,
                    slope_degrees=sample.slope_degrees,
                    cover_thickness=sample.cover_thickness,
                    roof_competence=sample.roof_competence,
                    growth_cost=sample.growth_cost,
                    arc_length=arc_length,
                    width=width,
                )
            )

        return CaveSegment(
            segment_id=segment_id,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            kind=kind,
            z_level=z_level,
            points=tuple(points),
            metadata=dict(metadata),
        )

    def _dominant_route(
        self,
        nodes: list[CaveNode],
        segments: list[CaveSegment],
        total_flux: np.ndarray,
    ) -> tuple[int, ...]:
        if not nodes or not segments:
            return ()

        entry = min(nodes, key=lambda node: node.along_position)
        exit_node = max(nodes, key=lambda node: node.along_position)
        adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for segment in segments:
            mean_flux = np.mean([point.width for point in segment.points]) if segment.points else 1.0
            cost = segment.total_length / max(mean_flux, 1.0)
            adjacency[segment.start_node_id].append((segment.end_node_id, cost))
            adjacency[segment.end_node_id].append((segment.start_node_id, cost))

        distances = {node.node_id: math.inf for node in nodes}
        predecessor: dict[int, int] = {}
        distances[entry.node_id] = 0.0
        heap = [(0.0, entry.node_id)]
        while heap:
            current_distance, node_id = heapq.heappop(heap)
            if current_distance > distances[node_id]:
                continue
            if node_id == exit_node.node_id:
                break
            for neighbor_id, edge_cost in adjacency[node_id]:
                next_distance = current_distance + edge_cost
                if next_distance >= distances[neighbor_id]:
                    continue
                distances[neighbor_id] = next_distance
                predecessor[neighbor_id] = node_id
                heapq.heappush(heap, (next_distance, neighbor_id))

        if math.isinf(distances[exit_node.node_id]):
            return ()

        route = [exit_node.node_id]
        current = exit_node.node_id
        while current != entry.node_id:
            current = predecessor[current]
            route.append(current)
        route.reverse()
        return tuple(route)

    def _select_spur_start_cells(
        self,
        total_flux: np.ndarray,
        geometry: _FlowGeometry,
    ) -> tuple[tuple[int, int], ...]:
        occupied_cells = np.argwhere(total_flux > 0.0)
        if occupied_cells.size == 0 or self.config.spur_count <= 0:
            return ()

        candidates: list[tuple[tuple[int, int], float]] = []
        for y_index, x_index in occupied_cells:
            along_position = float(geometry.along_grid[y_index, x_index])
            if along_position < 0.18 * geometry.along_extent or along_position > 0.82 * geometry.along_extent:
                continue
            score = float(total_flux[y_index, x_index])
            candidates.append(((int(y_index), int(x_index)), score))
        candidates.sort(key=lambda item: item[1], reverse=True)

        selected: list[tuple[int, int]] = []
        minimum_separation = 120.0
        for cell, _score in candidates:
            if any(
                math.hypot(cell[1] - other[1], cell[0] - other[0]) < minimum_separation / max(geometry.cell_scale, 1.0)
                for other in selected
            ):
                continue
            selected.append(cell)
            if len(selected) >= self.config.spur_count:
                break
        return tuple(selected)

    def _measure_parallel_channels(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        mask: np.ndarray,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        along_positions = np.linspace(
            0.0,
            geometry.along_extent,
            self.config.channel_count_samples,
            dtype=float,
        )
        band_half_width = 0.7 * geometry.cell_scale
        counts: list[int] = []
        for along_position in along_positions:
            band_cells = np.argwhere(
                mask
                & (np.abs(geometry.along_grid - along_position) <= band_half_width)
            )
            if band_cells.size == 0:
                counts.append(0)
                continue
            lateral_positions = sorted(float(geometry.cross_grid[y_index, x_index]) for y_index, x_index in band_cells)
            channel_count = 1
            for previous, current in zip(lateral_positions, lateral_positions[1:]):
                if current - previous > 1.8 * geometry.cell_scale:
                    channel_count += 1
            counts.append(channel_count)
        return tuple(float(value) for value in along_positions), tuple(int(value) for value in counts)

    def _build_downstream_potential(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
    ) -> np.ndarray:
        height, width = host_field.growth_cost.shape
        potential = np.full((height, width), math.inf, dtype=float)
        sink_band = geometry.along_grid >= geometry.along_extent
        sink_candidates = np.argwhere(sink_band)
        if sink_candidates.size == 0:
            fallback_cell = (height // 2, width - 1)
            sink_candidates = np.array([fallback_cell], dtype=int)

        heap: list[tuple[float, tuple[int, int]]] = []
        for y_index, x_index in sink_candidates:
            cell = (int(y_index), int(x_index))
            potential[cell] = 0.0
            heapq.heappush(heap, (0.0, cell))

        while heap:
            current_potential, cell = heapq.heappop(heap)
            if current_potential > float(potential[cell]):
                continue
            for previous_cell in self._neighbor_cells(host_field, cell):
                transition = self._transition_cost(
                    host_field=host_field,
                    geometry=geometry,
                    support_field=support_field,
                    current_cell=previous_cell,
                    next_cell=cell,
                )
                next_potential = current_potential + transition
                if next_potential >= float(potential[previous_cell]):
                    continue
                potential[previous_cell] = next_potential
                heapq.heappush(heap, (next_potential, previous_cell))

        return potential

    def _rasterize_segment(
        self,
        host_field: HostField,
        occupancy: np.ndarray,
        width_field: np.ndarray,
        segment: CaveSegment,
    ) -> None:
        for point in segment.points:
            self._paint_disk(
                host_field=host_field,
                occupancy=occupancy,
                width_field=width_field,
                x_coord=point.x,
                y_coord=point.y,
                radius=0.5 * point.width,
            )

    def _paint_chambers(
        self,
        host_field: HostField,
        occupancy: np.ndarray,
        width_field: np.ndarray,
        total_flux: np.ndarray,
    ) -> None:
        threshold = self._quantile_threshold(
            total_flux,
            self.config.chamber_flux_quantile,
            minimum=6.0,
        )
        chamber_cells = np.argwhere(total_flux >= threshold)
        if chamber_cells.size == 0:
            return
        for y_index, x_index in chamber_cells[:: max(1, len(chamber_cells) // 10)]:
            x_coord = float(host_field.x_coords[x_index])
            y_coord = float(host_field.y_coords[y_index])
            self._paint_disk(
                host_field=host_field,
                occupancy=occupancy,
                width_field=width_field,
                x_coord=x_coord,
                y_coord=y_coord,
                radius=self.config.chamber_radius,
            )

    def _paint_structural_chambers(
        self,
        host_field: HostField,
        occupancy: np.ndarray,
        width_field: np.ndarray,
        nodes: list[CaveNode],
        segments: list[CaveSegment],
    ) -> None:
        for node in nodes:
            if node.kind != "chamber":
                continue
            self._paint_disk(
                host_field=host_field,
                occupancy=occupancy,
                width_field=width_field,
                x_coord=node.x,
                y_coord=node.y,
                radius=1.18 * self.config.chamber_radius,
            )
        for segment in segments:
            if segment.kind not in {"chamber_braid", "ladder"} or len(segment.points) < 3:
                continue
            midpoint = segment.points[len(segment.points) // 2]
            radius = self.config.chamber_radius * (1.25 if segment.kind == "chamber_braid" else 0.78)
            self._paint_disk(
                host_field=host_field,
                occupancy=occupancy,
                width_field=width_field,
                x_coord=midpoint.x,
                y_coord=midpoint.y,
                radius=radius,
            )

    def _paint_disk(
        self,
        *,
        host_field: HostField,
        occupancy: np.ndarray,
        width_field: np.ndarray,
        x_coord: float,
        y_coord: float,
        radius: float,
    ) -> None:
        x_spacing = float(host_field.x_coords[1] - host_field.x_coords[0])
        y_spacing = float(host_field.y_coords[1] - host_field.y_coords[0])
        x_index = self._coordinate_to_index(host_field.x_coords, x_coord)
        y_index = self._coordinate_to_index(host_field.y_coords, y_coord)
        x_radius = max(1, int(math.ceil(radius / max(x_spacing, 1.0))))
        y_radius = max(1, int(math.ceil(radius / max(y_spacing, 1.0))))

        for sample_y in range(
            max(0, y_index - y_radius),
            min(len(host_field.y_coords), y_index + y_radius + 1),
        ):
            y_world = float(host_field.y_coords[sample_y])
            for sample_x in range(
                max(0, x_index - x_radius),
                min(len(host_field.x_coords), x_index + x_radius + 1),
            ):
                x_world = float(host_field.x_coords[sample_x])
                distance = math.hypot(x_world - x_coord, y_world - y_coord)
                if distance > radius:
                    continue
                occupancy[sample_y, sample_x] = True
                width_field[sample_y, sample_x] = max(width_field[sample_y, sample_x], 2.0 * radius)

    def _smooth_occupancy(self, occupancy: np.ndarray) -> np.ndarray:
        current = occupancy.copy()
        for _ in range(self.config.occupancy_smoothing_passes):
            neighbor_count = self._neighbor_count(current)
            current = np.where(current, neighbor_count >= 2, neighbor_count >= 5)
        return current

    def _local_radius(self, host_field: HostField, sample, flux_value: float) -> float:
        cover_score = max(
            0.0,
            min(
                1.0,
                (sample.cover_thickness - host_field.config.minimum_stable_cover)
                / max(host_field.config.volcanic_layer_thickness, 1.0),
            ),
        )
        radius = self.config.base_passage_radius
        radius *= 0.82 + 0.42 * (1.0 - sample.growth_cost)
        radius *= 0.88 + 0.24 * sample.roof_competence
        radius *= 0.92 + 0.18 * cover_score
        radius *= 1.0 + 0.09 * math.log1p(max(flux_value, 0.0))
        return max(radius, 8.0)

    def _normalize_cover_field(self, host_field: HostField) -> np.ndarray:
        return np.clip(
            (host_field.cover_thickness - host_field.config.minimum_stable_cover)
            / max(host_field.config.volcanic_layer_thickness, 1.0),
            0.0,
            1.0,
        )

    def _transition_cost(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        current_cell: tuple[int, int],
        next_cell: tuple[int, int],
    ) -> float:
        current_world = self._cell_to_world(host_field, current_cell)
        next_world = self._cell_to_world(host_field, next_cell)
        step_x = next_world[0] - current_world[0]
        step_y = next_world[1] - current_world[1]
        step_length = math.hypot(step_x, step_y)
        if math.isclose(step_length, 0.0):
            return math.inf

        step_unit_x = step_x / step_length
        step_unit_y = step_y / step_length
        flow_alignment = step_unit_x * geometry.flow_x + step_unit_y * geometry.flow_y
        downhill_x, downhill_y = host_field.downhill_direction(
            current_world[0],
            current_world[1],
            fallback_angle_degrees=host_field.config.flow_angle_degrees,
        )
        downhill_alignment = step_unit_x * downhill_x + step_unit_y * downhill_y
        current_elevation = float(host_field.elevation[current_cell])
        next_elevation = float(host_field.elevation[next_cell])
        uphill = max(next_elevation - current_elevation, 0.0)
        along_delta = float(geometry.along_grid[next_cell] - geometry.along_grid[current_cell])

        support_cost = max(0.1, 1.45 - float(support_field[next_cell]))
        transition_cost = step_length * support_cost
        transition_cost += step_length * 0.55 * max(0.0, 0.1 - flow_alignment)
        transition_cost += step_length * 0.75 * max(0.0, 0.15 - downhill_alignment)
        transition_cost += step_length * 0.85 * max(0.0, -along_delta / max(geometry.cell_scale, 1.0))
        transition_cost += 7.5 * uphill
        return transition_cost

    def _extend_path_to_sink(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        downstream_potential: np.ndarray,
        path: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        extended = list(path)
        max_extension_steps = max(24, self.config.trace_max_steps // 2)
        for _ in range(max_extension_steps):
            current = extended[-1]
            current_potential = float(downstream_potential[current])
            current_along = float(geometry.along_grid[current])
            if current_along >= geometry.along_extent or not math.isfinite(current_potential):
                break

            best_cell: tuple[int, int] | None = None
            best_cost = math.inf
            for next_cell in self._neighbor_cells(host_field, current):
                if next_cell in extended[-8:]:
                    continue
                next_potential = float(downstream_potential[next_cell])
                if not math.isfinite(next_potential) or next_potential >= current_potential:
                    continue
                current_elevation = float(host_field.elevation[current])
                next_elevation = float(host_field.elevation[next_cell])
                if next_elevation - current_elevation > 1.5 * self.config.max_uphill_step:
                    continue
                transition = self._transition_cost(
                    host_field=host_field,
                    geometry=geometry,
                    support_field=support_field,
                    current_cell=current,
                    next_cell=next_cell,
                )
                if transition < best_cost:
                    best_cost = transition
                    best_cell = next_cell

            if best_cell is None:
                break
            extended.append(best_cell)

        return extended

    @staticmethod
    def _sample_candidate(
        candidates: list[tuple[tuple[int, int], float]],
        temperature: float,
        rng,
    ) -> tuple[int, int]:
        scores = np.array([score for _, score in candidates], dtype=float)
        scaled = (scores - float(scores.max())) / max(temperature, 1e-6)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        index = int(rng.choice(len(candidates), p=probabilities))
        return candidates[index][0]

    @staticmethod
    def _quantile_threshold(values: np.ndarray, quantile: float, minimum: float) -> float:
        positive = values[values > 0.0]
        if positive.size == 0:
            return minimum
        return max(minimum, float(np.quantile(positive, quantile)))

    @staticmethod
    def _neighbor_count(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(int), 1, mode="constant")
        return (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )

    @staticmethod
    def _local_maximum(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, 1, mode="edge")
        maxima = padded[1:-1, 1:-1].copy()
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                view = padded[
                    1 + delta_y : 1 + delta_y + values.shape[0],
                    1 + delta_x : 1 + delta_x + values.shape[1],
                ]
                maxima = np.maximum(maxima, view)
        return maxima

    def _occupied_neighbors(
        self,
        cell: tuple[int, int],
        occupied_cells: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        neighbors = []
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                if delta_y == 0 and delta_x == 0:
                    continue
                neighbor = (cell[0] + delta_y, cell[1] + delta_x)
                if neighbor in occupied_cells:
                    neighbors.append(neighbor)
        return neighbors

    def _neighbor_cells(
        self,
        host_field: HostField,
        cell: tuple[int, int],
    ) -> list[tuple[int, int]]:
        height, width = host_field.elevation.shape
        neighbors = []
        current_y, current_x = cell
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                if delta_y == 0 and delta_x == 0:
                    continue
                next_y = current_y + delta_y
                next_x = current_x + delta_x
                if 0 <= next_y < height and 0 <= next_x < width:
                    neighbors.append((next_y, next_x))
        return neighbors

    @staticmethod
    def _deduplicate_coordinates(
        coordinates: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        deduplicated: list[tuple[float, float]] = []
        for x_coord, y_coord in coordinates:
            if deduplicated and math.isclose(x_coord, deduplicated[-1][0]) and math.isclose(y_coord, deduplicated[-1][1]):
                continue
            deduplicated.append((x_coord, y_coord))
        return deduplicated

    @staticmethod
    def _coordinate_to_index(coords, value: float) -> int:
        if value <= float(coords[0]):
            return 0
        if value >= float(coords[-1]):
            return len(coords) - 1
        spacing = float(coords[1] - coords[0])
        return int(round((value - float(coords[0])) / spacing))

    def _world_to_cell(self, host_field: HostField, x_coord: float, y_coord: float) -> tuple[int, int]:
        return (
            self._coordinate_to_index(host_field.y_coords, y_coord),
            self._coordinate_to_index(host_field.x_coords, x_coord),
        )

    @staticmethod
    def _cell_to_world(host_field: HostField, cell: tuple[int, int]) -> tuple[float, float]:
        y_index, x_index = cell
        return float(host_field.x_coords[x_index]), float(host_field.y_coords[y_index])

    @staticmethod
    def _project_cross(geometry: _FlowGeometry, x_coord: float, y_coord: float) -> float:
        return (
            (x_coord - geometry.seed_x) * geometry.cross_x
            + (y_coord - geometry.seed_y) * geometry.cross_y
        )
