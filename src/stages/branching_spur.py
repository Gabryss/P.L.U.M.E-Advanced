"""Local dead-end spur generation around the trunk spine."""

from __future__ import annotations

import math
from typing import Any

from stages.branching_models import BranchCandidate, BranchPath, BranchPoint, SPUR, SpurBranchConfig
from stages.graph import TrunkGraph
from stages.host_field import HostField


class SpurBranchGenerator:
    """Generate short dead-end side branches without touching the trunk stage."""

    def __init__(self, config: SpurBranchConfig | None = None) -> None:
        self.config = config or SpurBranchConfig()

    def generate(
        self,
        *,
        host_field: HostField,
        trunk_graph: TrunkGraph,
        candidate: BranchCandidate,
        branch_id: int,
        trunk_distance_field: Any,
    ) -> BranchPath | None:
        source_point = trunk_graph.points[candidate.trunk_index]
        launch_x, launch_y = self._rotate_vector(
            source_point.tangent_x,
            source_point.tangent_y,
            math.radians(candidate.side * self.config.branch_angle_degrees),
        )
        tangent_x, tangent_y = self._normalize(launch_x, launch_y)
        points = [
            self._create_branch_point(
                host_field=host_field,
                branch_id=branch_id,
                index=0,
                x_coord=source_point.x,
                y_coord=source_point.y,
                arc_length=0.0,
                tangent_x=tangent_x,
                tangent_y=tangent_y,
            )
        ]

        termination_reason = 'max_steps'
        max_length = self.config.branch_max_steps * self.config.branch_step_length

        for _ in range(self.config.branch_max_steps):
            current_point = points[-1]
            progress = current_point.arc_length / max(max_length, 1.0)
            downhill_x, downhill_y = host_field.downhill_direction(
                current_point.x,
                current_point.y,
                fallback_angle_degrees=host_field.config.flow_angle_degrees,
            )
            normal_x, normal_y = self._rotate_vector(
                tangent_x,
                tangent_y,
                candidate.side * math.pi / 2.0,
            )
            clearance = self._lookup_grid_value(
                host_field=host_field,
                values=trunk_distance_field,
                x_coord=current_point.x,
                y_coord=current_point.y,
            )
            repulsion_scale = max(
                0.0,
                1.0 - clearance / max(self.config.trunk_repulsion_target, 1.0),
            )
            steer_x, steer_y = self._normalize(
                self.config.downhill_weight * downhill_x
                + self.config.launch_weight * max(0.2, 1.0 - progress) * launch_x
                + self.config.lateral_bias_weight * max(0.35, 1.0 - 0.55 * progress) * normal_x
                + self.config.trunk_repulsion_weight * repulsion_scale * normal_x,
                self.config.downhill_weight * downhill_y
                + self.config.launch_weight * max(0.2, 1.0 - progress) * launch_y
                + self.config.lateral_bias_weight * max(0.35, 1.0 - 0.55 * progress) * normal_y
                + self.config.trunk_repulsion_weight * repulsion_scale * normal_y,
            )
            tangent_x, tangent_y = self._normalize(
                (1.0 - self.config.tangent_blend) * tangent_x
                + self.config.tangent_blend * steer_x,
                (1.0 - self.config.tangent_blend) * tangent_y
                + self.config.tangent_blend * steer_y,
            )
            candidate_x = current_point.x + self.config.branch_step_length * tangent_x
            candidate_y = current_point.y + self.config.branch_step_length * tangent_y

            if not host_field.contains(
                candidate_x,
                candidate_y,
                margin=self.config.branch_boundary_margin,
            ):
                termination_reason = 'boundary'
                break

            sample = host_field.sample(candidate_x, candidate_y)
            if sample.elevation > current_point.elevation + self.config.max_uphill_step:
                termination_reason = 'uphill'
                break
            if (
                len(points) - 1 >= self.config.min_steps_before_cost_stop
                and sample.growth_cost >= self.config.stop_growth_cost
            ):
                termination_reason = 'cost'
                break

            next_point = self._create_branch_point(
                host_field=host_field,
                branch_id=branch_id,
                index=len(points),
                x_coord=candidate_x,
                y_coord=candidate_y,
                arc_length=current_point.arc_length + self.config.branch_step_length,
                tangent_x=tangent_x,
                tangent_y=tangent_y,
            )
            points.append(next_point)

        branch_path = BranchPath(
            branch_id=branch_id,
            branch_kind=SPUR,
            source_trunk_index=candidate.trunk_index,
            target_trunk_index=None,
            points=tuple(points),
            merge_event=None,
            termination_reason=termination_reason,
        )
        if len(branch_path.points) < 2 or branch_path.total_length < self.config.min_total_length:
            return None
        return branch_path

    @staticmethod
    def _create_branch_point(
        *,
        host_field: HostField,
        branch_id: int,
        index: int,
        x_coord: float,
        y_coord: float,
        arc_length: float,
        tangent_x: float,
        tangent_y: float,
    ) -> BranchPoint:
        sample = host_field.sample(x_coord, y_coord)
        return BranchPoint(
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

    @staticmethod
    def _coordinate_to_index(coords, value: float) -> int:
        if value <= float(coords[0]):
            return 0
        if value >= float(coords[-1]):
            return len(coords) - 1
        spacing = float(coords[1] - coords[0])
        return int(round((value - float(coords[0])) / spacing))

    def _lookup_grid_value(self, *, host_field: HostField, values, x_coord: float, y_coord: float) -> float:
        x_index = self._coordinate_to_index(host_field.x_coords, x_coord)
        y_index = self._coordinate_to_index(host_field.y_coords, y_coord)
        return float(values[y_index, x_index])

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
