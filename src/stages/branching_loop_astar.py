"""A*-guided reconnecting loop generation around the trunk spine."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np

from stages.branching_models import (
    BranchCandidate,
    BranchPath,
    BranchPoint,
    DOWNSTREAM_RECONNECT_LOOP,
    LOCAL_BYPASS_LOOP,
    LoopPathConfig,
    MergeEvent,
)
from stages.graph import TrunkGraph
from stages.host_field import HostField


@dataclass(frozen=True)
class _LoopSpec:
    min_target_offset_points: int
    max_target_offset_points: int
    lateral_offset: float
    corridor_width: float
    min_clearance: float
    min_enclosed_area: float
    min_detour_factor: float


class LoopAStarGenerator:
    """Generate reconnecting loops with an explicit off-trunk waypoint."""

    def __init__(self, config: LoopPathConfig | None = None) -> None:
        self.config = config or LoopPathConfig()

    def generate(
        self,
        *,
        host_field: HostField,
        trunk_graph: TrunkGraph,
        candidate: BranchCandidate,
        branch_id: int,
        branch_kind: str,
        trunk_distance_field: Any,
    ) -> BranchPath | None:
        spec = self._spec_for_kind(branch_kind)
        target_index = self._select_target_index(
            trunk_graph=trunk_graph,
            source_index=candidate.trunk_index,
            branch_kind=branch_kind,
            spec=spec,
        )
        if target_index is None:
            return None

        source_point = trunk_graph.points[candidate.trunk_index]
        target_point = trunk_graph.points[target_index]
        waypoint_cell = self._select_waypoint_cell(
            host_field=host_field,
            trunk_distance_field=trunk_distance_field,
            source_point=source_point,
            target_point=target_point,
            side=candidate.side,
            spec=spec,
        )
        if waypoint_cell is None:
            return None

        start_cell = self._world_to_cell(host_field, source_point.x, source_point.y)
        goal_cell = self._world_to_cell(host_field, target_point.x, target_point.y)
        waypoint_coord = self._cell_to_world(host_field, waypoint_cell)
        first_leg = self._astar_leg(
            host_field=host_field,
            trunk_distance_field=trunk_distance_field,
            start_cell=start_cell,
            goal_cell=waypoint_cell,
            start_coord=(source_point.x, source_point.y),
            goal_coord=waypoint_coord,
            spec=spec,
        )
        if first_leg is None:
            return None
        second_leg = self._astar_leg(
            host_field=host_field,
            trunk_distance_field=trunk_distance_field,
            start_cell=waypoint_cell,
            goal_cell=goal_cell,
            start_coord=waypoint_coord,
            goal_coord=(target_point.x, target_point.y),
            spec=spec,
        )
        if second_leg is None:
            return None

        cell_path = first_leg + second_leg[1:]
        if len(cell_path) > self.config.max_path_points:
            return None

        coordinates = [(source_point.x, source_point.y)]
        coordinates.extend(
            self._cell_to_world(host_field, cell)
            for cell in cell_path[1:-1]
        )
        coordinates.append((target_point.x, target_point.y))
        coordinates = self._deduplicate_points(coordinates)
        coordinates = self._smooth_path(coordinates)

        if not self._validate_loop(
            coordinates=coordinates,
            trunk_graph=trunk_graph,
            source_index=candidate.trunk_index,
            target_index=target_index,
            spec=spec,
        ):
            return None

        points = self._build_branch_points(
            host_field=host_field,
            branch_id=branch_id,
            coordinates=coordinates,
        )
        if len(points) < 2:
            return None

        return BranchPath(
            branch_id=branch_id,
            branch_kind=branch_kind,
            source_trunk_index=candidate.trunk_index,
            target_trunk_index=target_index,
            points=points,
            merge_event=MergeEvent(
                branch_id=branch_id,
                branch_kind=branch_kind,
                source_trunk_index=candidate.trunk_index,
                target_trunk_index=target_index,
            ),
            termination_reason='merged',
        )

    def _select_target_index(
        self,
        *,
        trunk_graph: TrunkGraph,
        source_index: int,
        branch_kind: str,
        spec: _LoopSpec,
    ) -> int | None:
        min_index = source_index + spec.min_target_offset_points
        max_index = min(
            source_index + spec.max_target_offset_points,
            len(trunk_graph.points) - 1,
        )
        if min_index > max_index:
            return None

        midpoint = 0.5 * (min_index + max_index)
        best_index: int | None = None
        best_score = -math.inf
        for target_index in range(min_index, max_index + 1):
            target_point = trunk_graph.points[target_index]
            point_score = self._score_trunk_point(target_point)
            if branch_kind == LOCAL_BYPASS_LOOP:
                distance_score = 1.0 - abs(target_index - midpoint) / max((max_index - min_index) / 2.0, 1.0)
            else:
                distance_score = (target_index - min_index) / max(max_index - min_index, 1)
            score = 0.72 * point_score + 0.28 * distance_score
            if score > best_score:
                best_score = score
                best_index = target_index
        return best_index

    def _select_waypoint_cell(
        self,
        *,
        host_field: HostField,
        trunk_distance_field: Any,
        source_point,
        target_point,
        side: int,
        spec: _LoopSpec,
    ) -> tuple[int, int] | None:
        trunk_tangent_x, trunk_tangent_y = self._normalize(
            source_point.tangent_x + target_point.tangent_x,
            source_point.tangent_y + target_point.tangent_y,
        )
        normal_x, normal_y = self._rotate_vector(
            trunk_tangent_x,
            trunk_tangent_y,
            side * math.pi / 2.0,
        )
        midpoint_x = 0.5 * (source_point.x + target_point.x)
        midpoint_y = 0.5 * (source_point.y + target_point.y)
        nominal_x = midpoint_x + spec.lateral_offset * normal_x
        nominal_y = midpoint_y + spec.lateral_offset * normal_y
        nominal_cell = self._world_to_cell_clamped(host_field, nominal_x, nominal_y)

        best_cell: tuple[int, int] | None = None
        best_score = -math.inf
        search_radius = self.config.candidate_search_radius_cells
        for y_index in range(max(0, nominal_cell[0] - search_radius), min(len(host_field.y_coords), nominal_cell[0] + search_radius + 1)):
            for x_index in range(max(0, nominal_cell[1] - search_radius), min(len(host_field.x_coords), nominal_cell[1] + search_radius + 1)):
                x_coord, y_coord = self._cell_to_world(host_field, (y_index, x_index))
                if not host_field.contains(x_coord, y_coord, margin=self.config.boundary_margin):
                    continue
                clearance = float(trunk_distance_field[y_index, x_index])
                if clearance < 0.7 * spec.min_clearance:
                    continue
                sample = host_field.sample(x_coord, y_coord)
                climb = max(sample.elevation - max(source_point.elevation, target_point.elevation), 0.0)
                if climb > self.config.max_waypoint_climb:
                    continue
                distance_to_nominal = math.hypot(x_coord - nominal_x, y_coord - nominal_y)
                score = (
                    1.10 * clearance / max(spec.min_clearance, 1.0)
                    + 0.55 * sample.roof_competence
                    - 0.80 * sample.growth_cost
                    - 0.0035 * distance_to_nominal
                    - 0.45 * climb
                )
                if score > best_score:
                    best_score = score
                    best_cell = (y_index, x_index)
        return best_cell

    def _astar_leg(
        self,
        *,
        host_field: HostField,
        trunk_distance_field: Any,
        start_cell: tuple[int, int],
        goal_cell: tuple[int, int],
        start_coord: tuple[float, float],
        goal_coord: tuple[float, float],
        spec: _LoopSpec,
    ) -> list[tuple[int, int]] | None:
        neighbor_steps = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        height, width = host_field.elevation.shape
        g_score = np.full((height, width), np.inf, dtype=float)
        parent_y = np.full((height, width), -1, dtype=int)
        parent_x = np.full((height, width), -1, dtype=int)

        g_score[start_cell] = 0.0
        open_heap: list[tuple[float, float, tuple[int, int]]] = [
            (self._heuristic(start_coord, goal_coord), 0.0, start_cell)
        ]

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current_cost > g_score[current]:
                continue
            if current == goal_cell:
                return self._reconstruct_path(parent_y, parent_x, start_cell, goal_cell)

            current_y, current_x = current
            current_world = self._cell_to_world(host_field, current)
            current_elevation = float(host_field.elevation[current_y, current_x])

            for delta_y, delta_x in neighbor_steps:
                next_y = current_y + delta_y
                next_x = current_x + delta_x
                if not (0 <= next_y < height and 0 <= next_x < width):
                    continue

                next_world = self._cell_to_world(host_field, (next_y, next_x))
                if not host_field.contains(next_world[0], next_world[1], margin=self.config.boundary_margin):
                    continue

                next_elevation = float(host_field.elevation[next_y, next_x])
                uphill = max(0.0, next_elevation - current_elevation)
                if uphill > self.config.max_uphill_step:
                    continue

                distance_to_terminal = min(
                    math.hypot(next_world[0] - start_coord[0], next_world[1] - start_coord[1]),
                    math.hypot(next_world[0] - goal_coord[0], next_world[1] - goal_coord[1]),
                )
                clearance = float(trunk_distance_field[next_y, next_x])
                if (
                    distance_to_terminal > self.config.terminal_relief_distance
                    and clearance < max(self.config.internal_clearance_floor, 0.35 * spec.min_clearance)
                ):
                    continue

                move_distance = math.hypot(
                    next_world[0] - current_world[0],
                    next_world[1] - current_world[1],
                )
                trunk_penalty = 0.0
                if distance_to_terminal > self.config.terminal_relief_distance:
                    trunk_penalty = self.config.trunk_repulsion_weight * max(
                        0.0,
                        1.0 - clearance / max(self.config.trunk_repulsion_distance, 1.0),
                    ) ** 2
                corridor_distance = self._distance_to_segment(next_world, start_coord, goal_coord)
                corridor_penalty = 0.35 * (corridor_distance / max(spec.corridor_width, 1.0)) ** 2
                growth_cost = float(host_field.growth_cost[next_y, next_x])
                step_cost = move_distance * (
                    1.0
                    + self.config.growth_cost_weight * growth_cost
                    + trunk_penalty
                    + corridor_penalty
                ) + self.config.uphill_penalty_weight * uphill
                tentative_cost = current_cost + step_cost
                if tentative_cost >= g_score[next_y, next_x]:
                    continue

                g_score[next_y, next_x] = tentative_cost
                parent_y[next_y, next_x] = current_y
                parent_x[next_y, next_x] = current_x
                heuristic = self._heuristic(next_world, goal_coord)
                heapq.heappush(
                    open_heap,
                    (tentative_cost + heuristic, tentative_cost, (next_y, next_x)),
                )
        return None

    def _build_branch_points(
        self,
        *,
        host_field: HostField,
        branch_id: int,
        coordinates: list[tuple[float, float]],
    ) -> tuple[BranchPoint, ...]:
        points: list[BranchPoint] = []
        arc_length = 0.0
        for index, (x_coord, y_coord) in enumerate(coordinates):
            if index > 0:
                previous_x, previous_y = coordinates[index - 1]
                arc_length += math.hypot(x_coord - previous_x, y_coord - previous_y)
            if index == 0:
                tangent_x, tangent_y = self._normalize(
                    coordinates[1][0] - x_coord,
                    coordinates[1][1] - y_coord,
                )
            elif index == len(coordinates) - 1:
                tangent_x, tangent_y = self._normalize(
                    x_coord - coordinates[index - 1][0],
                    y_coord - coordinates[index - 1][1],
                )
            else:
                tangent_x, tangent_y = self._normalize(
                    coordinates[index + 1][0] - coordinates[index - 1][0],
                    coordinates[index + 1][1] - coordinates[index - 1][1],
                )

            sample = host_field.sample(x_coord, y_coord)
            points.append(
                BranchPoint(
                    index=index,
                    branch_id=branch_id,
                    x=x_coord,
                    y=y_coord,
                    elevation=sample.elevation,
                    slope_degrees=sample.slope_degrees,
                    cover_thickness=sample.cover_thickness,
                    roof_competence=sample.roof_competence,
                    growth_cost=sample.growth_cost,
                    arc_length=arc_length,
                    tangent_x=tangent_x,
                    tangent_y=tangent_y,
                )
            )
        return tuple(points)

    def _validate_loop(
        self,
        *,
        coordinates: list[tuple[float, float]],
        trunk_graph: TrunkGraph,
        source_index: int,
        target_index: int,
        spec: _LoopSpec,
    ) -> bool:
        if len(coordinates) < 3:
            return False

        core_coordinates = coordinates[3:-3] if len(coordinates) > 8 else coordinates[1:-1]
        internal_clearances = [
            self._distance_to_trunk(point, trunk_graph.points)
            for point in core_coordinates
        ]
        if not internal_clearances:
            return False
        if max(internal_clearances) < spec.min_clearance:
            return False
        if min(internal_clearances) < self.config.internal_clearance_floor:
            return False

        branch_length = sum(
            math.hypot(x1 - x0, y1 - y0)
            for (x0, y0), (x1, y1) in zip(coordinates, coordinates[1:])
        )
        direct_distance = math.hypot(
            coordinates[-1][0] - coordinates[0][0],
            coordinates[-1][1] - coordinates[0][1],
        )
        if branch_length < spec.min_detour_factor * max(direct_distance, 1.0):
            return False

        trunk_segment = [
            (point.x, point.y)
            for point in trunk_graph.points[source_index : target_index + 1]
        ]
        polygon = coordinates + list(reversed(trunk_segment))
        area = abs(self._polygon_area(polygon))
        return area >= spec.min_enclosed_area

    def _spec_for_kind(self, branch_kind: str) -> _LoopSpec:
        if branch_kind == LOCAL_BYPASS_LOOP:
            return _LoopSpec(
                min_target_offset_points=self.config.local_bypass_min_target_offset_points,
                max_target_offset_points=self.config.local_bypass_max_target_offset_points,
                lateral_offset=self.config.local_bypass_lateral_offset,
                corridor_width=self.config.local_bypass_corridor_width,
                min_clearance=self.config.local_bypass_min_clearance,
                min_enclosed_area=self.config.local_bypass_min_enclosed_area,
                min_detour_factor=self.config.local_bypass_min_detour_factor,
            )
        return _LoopSpec(
            min_target_offset_points=self.config.downstream_min_target_offset_points,
            max_target_offset_points=self.config.downstream_max_target_offset_points,
            lateral_offset=self.config.downstream_lateral_offset,
            corridor_width=self.config.downstream_corridor_width,
            min_clearance=self.config.downstream_min_clearance,
            min_enclosed_area=self.config.downstream_min_enclosed_area,
            min_detour_factor=self.config.downstream_min_detour_factor,
        )

    @staticmethod
    def _score_trunk_point(point) -> float:
        low_cost_score = 1.0 - point.growth_cost
        roof_score = point.roof_competence
        slope_score = max(0.0, 1.0 - point.slope_degrees / 18.0)
        return 0.45 * low_cost_score + 0.35 * roof_score + 0.20 * slope_score

    @staticmethod
    def _heuristic(start: tuple[float, float], goal: tuple[float, float]) -> float:
        return math.hypot(goal[0] - start[0], goal[1] - start[1])

    @staticmethod
    def _deduplicate_points(coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
        deduplicated: list[tuple[float, float]] = []
        for coordinate in coordinates:
            if deduplicated and math.isclose(coordinate[0], deduplicated[-1][0]) and math.isclose(coordinate[1], deduplicated[-1][1]):
                continue
            deduplicated.append(coordinate)
        return deduplicated

    def _smooth_path(self, coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(coordinates) < 4:
            return coordinates

        smoothed = list(coordinates)
        for _ in range(self.config.smoothing_iterations):
            next_coordinates = [smoothed[0]]
            for index in range(1, len(smoothed) - 1):
                previous_point = smoothed[index - 1]
                current_point = smoothed[index]
                next_point = smoothed[index + 1]
                blend = self.config.smoothing_blend
                next_coordinates.append(
                    (
                        (1.0 - blend) * current_point[0] + 0.5 * blend * (previous_point[0] + next_point[0]),
                        (1.0 - blend) * current_point[1] + 0.5 * blend * (previous_point[1] + next_point[1]),
                    )
                )
            next_coordinates.append(smoothed[-1])
            smoothed = next_coordinates
        return smoothed

    @staticmethod
    def _reconstruct_path(parent_y, parent_x, start_cell: tuple[int, int], goal_cell: tuple[int, int]) -> list[tuple[int, int]]:
        path = [goal_cell]
        current_y, current_x = goal_cell
        while (current_y, current_x) != start_cell:
            previous_y = int(parent_y[current_y, current_x])
            previous_x = int(parent_x[current_y, current_x])
            if previous_y < 0 or previous_x < 0:
                return []
            path.append((previous_y, previous_x))
            current_y, current_x = previous_y, previous_x
        path.reverse()
        return path

    @staticmethod
    def _distance_to_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        segment_x = end[0] - start[0]
        segment_y = end[1] - start[1]
        segment_length_squared = segment_x * segment_x + segment_y * segment_y
        if math.isclose(segment_length_squared, 0.0):
            return math.hypot(point[0] - start[0], point[1] - start[1])
        projection = ((point[0] - start[0]) * segment_x + (point[1] - start[1]) * segment_y) / segment_length_squared
        projection = max(0.0, min(1.0, projection))
        closest_x = start[0] + projection * segment_x
        closest_y = start[1] + projection * segment_y
        return math.hypot(point[0] - closest_x, point[1] - closest_y)

    @staticmethod
    def _distance_to_trunk(point: tuple[float, float], trunk_points) -> float:
        return min(math.hypot(point[0] - trunk_point.x, point[1] - trunk_point.y) for trunk_point in trunk_points)

    @staticmethod
    def _polygon_area(coordinates: list[tuple[float, float]]) -> float:
        area = 0.0
        for (x0, y0), (x1, y1) in zip(coordinates, coordinates[1:] + coordinates[:1]):
            area += x0 * y1 - x1 * y0
        return 0.5 * area

    @staticmethod
    def _rotate_vector(x_value: float, y_value: float, angle_radians: float) -> tuple[float, float]:
        cosine = math.cos(angle_radians)
        sine = math.sin(angle_radians)
        return cosine * x_value - sine * y_value, sine * x_value + cosine * y_value

    @staticmethod
    def _normalize(x_value: float, y_value: float) -> tuple[float, float]:
        length = math.hypot(x_value, y_value)
        if math.isclose(length, 0.0):
            return 0.0, 0.0
        return x_value / length, y_value / length

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

    def _world_to_cell_clamped(self, host_field: HostField, x_coord: float, y_coord: float) -> tuple[int, int]:
        y_index, x_index = self._world_to_cell(host_field, x_coord, y_coord)
        return (
            max(0, min(y_index, len(host_field.y_coords) - 1)),
            max(0, min(x_index, len(host_field.x_coords) - 1)),
        )

    @staticmethod
    def _cell_to_world(host_field: HostField, cell: tuple[int, int]) -> tuple[float, float]:
        y_index, x_index = cell
        return float(host_field.x_coords[x_index]), float(host_field.y_coords[y_index])
