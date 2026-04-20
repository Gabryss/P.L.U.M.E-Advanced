"""Stage B: trunk graph generation over the host field."""

from __future__ import annotations

from dataclasses import dataclass
import math

from stages.host_field import HostField


@dataclass(frozen=True)
class GraphConfig:
    """Parameters controlling the first trunk-growth graph pass."""

    step_length: float = 22.0
    max_steps: int = 135
    boundary_margin: float = 90.0
    tangent_blend: float = 0.22
    downhill_weight: float = 0.56
    flow_bias_weight: float = 0.29
    corridor_pull_weight: float = 0.15
    meander_amplitude_degrees: float = 9.0
    meander_wavelength: float = 430.0
    minimum_flow_component: float = 0.45
    max_uphill_step: float = 1.4


@dataclass(frozen=True)
class CenterlinePoint:
    """One sampled point along the generated trunk centerline."""

    index: int
    x: float
    y: float
    elevation: float
    slope_degrees: float
    cover_thickness: float
    roof_competence: float
    growth_cost: float
    arc_length: float
    tangent_x: float
    tangent_y: float


@dataclass(frozen=True)
class CenterlineEdge:
    """One directed edge in the centerline graph."""

    start_index: int
    end_index: int
    length: float


@dataclass(frozen=True)
class TrunkGraph:
    """Stage-B output for the first trunk graph prototype."""

    config: GraphConfig
    points: tuple[CenterlinePoint, ...]
    edges: tuple[CenterlineEdge, ...]

    @property
    def total_length(self) -> float:
        return 0.0 if not self.points else self.points[-1].arc_length

    def summary(self) -> dict[str, float]:
        """Return scalar summaries for logging and quick inspection."""

        if not self.points:
            return {
                "point_count": 0.0,
                "edge_count": 0.0,
                "total_length": 0.0,
                "elevation_drop": 0.0,
                "mean_slope_deg": 0.0,
                "mean_growth_cost": 0.0,
            }

        start = self.points[0]
        end = self.points[-1]
        point_count = float(len(self.points))
        edge_count = float(len(self.edges))
        mean_slope = sum(point.slope_degrees for point in self.points) / len(self.points)
        mean_growth_cost = sum(point.growth_cost for point in self.points) / len(self.points)

        return {
            "point_count": point_count,
            "edge_count": edge_count,
            "total_length": self.total_length,
            "elevation_drop": start.elevation - end.elevation,
            "mean_slope_deg": mean_slope,
            "mean_growth_cost": mean_growth_cost,
        }

    def profile(self) -> dict[str, list[float]]:
        """Return arrays convenient for plotting the longitudinal profile."""

        return {
            "arc_length": [point.arc_length for point in self.points],
            "elevation": [point.elevation for point in self.points],
            "slope_degrees": [point.slope_degrees for point in self.points],
            "roof_competence": [point.roof_competence for point in self.points],
            "growth_cost": [point.growth_cost for point in self.points],
        }


class TrunkGraphGenerator:
    """Generate a smooth downhill trunk centerline over the host field."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        self.config = config or GraphConfig()

    def generate(self, host_field: HostField) -> TrunkGraph:
        flow_angle = host_field.config.flow_angle_degrees
        tangent_x, tangent_y = self._initial_tangent(host_field, flow_angle)
        current_x, current_y = host_field.config.seed_point

        points = [
            self._create_point(
                host_field=host_field,
                index=0,
                x_coord=current_x,
                y_coord=current_y,
                arc_length=0.0,
                tangent_x=tangent_x,
                tangent_y=tangent_y,
            )
        ]
        edges: list[CenterlineEdge] = []

        for _ in range(self.config.max_steps):
            current_point = points[-1]
            candidate_x, candidate_y, next_tangent = self._propose_next_position(
                host_field=host_field,
                current_point=current_point,
                tangent_x=tangent_x,
                tangent_y=tangent_y,
                flow_angle=flow_angle,
            )

            if candidate_x is None or candidate_y is None or next_tangent is None:
                break

            if not host_field.contains(candidate_x, candidate_y, margin=self.config.boundary_margin):
                break

            candidate_sample = host_field.sample(candidate_x, candidate_y)
            if candidate_sample.elevation > current_point.elevation + self.config.max_uphill_step:
                break

            next_tangent_x, next_tangent_y = next_tangent
            arc_length = current_point.arc_length + self.config.step_length
            next_point = self._create_point(
                host_field=host_field,
                index=len(points),
                x_coord=candidate_x,
                y_coord=candidate_y,
                arc_length=arc_length,
                tangent_x=next_tangent_x,
                tangent_y=next_tangent_y,
            )
            points.append(next_point)
            edges.append(
                CenterlineEdge(
                    start_index=current_point.index,
                    end_index=next_point.index,
                    length=self.config.step_length,
                )
            )

            tangent_x, tangent_y = next_tangent_x, next_tangent_y

        return TrunkGraph(
            config=self.config,
            points=tuple(points),
            edges=tuple(edges),
        )

    def _create_point(
        self,
        *,
        host_field: HostField,
        index: int,
        x_coord: float,
        y_coord: float,
        arc_length: float,
        tangent_x: float,
        tangent_y: float,
    ) -> CenterlinePoint:
        sample = host_field.sample(x_coord, y_coord)
        return CenterlinePoint(
            index=index,
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

    def _initial_tangent(self, host_field: HostField, flow_angle: float) -> tuple[float, float]:
        flow_x, flow_y = self._vector_from_angle(flow_angle)
        downhill_x, downhill_y = host_field.downhill_direction(
            host_field.config.seed_point[0],
            host_field.config.seed_point[1],
        )
        return self._normalize(
            0.65 * flow_x + 0.35 * downhill_x,
            0.65 * flow_y + 0.35 * downhill_y,
        )

    def _propose_next_position(
        self,
        *,
        host_field: HostField,
        current_point: CenterlinePoint,
        tangent_x: float,
        tangent_y: float,
        flow_angle: float,
    ) -> tuple[float | None, float | None, tuple[float, float] | None]:
        downhill_x, downhill_y = host_field.downhill_direction(
            current_point.x,
            current_point.y,
            fallback_angle_degrees=flow_angle,
        )
        flow_x, flow_y = self._vector_from_angle(flow_angle)
        corridor_pull_x, corridor_pull_y = self._corridor_pull_vector(
            x_coord=current_point.x,
            y_coord=current_point.y,
            seed_point=host_field.config.seed_point,
            flow_angle=flow_angle,
            corridor_width=host_field.config.corridor_width,
        )

        meander_phase = math.sin(
            2.0 * math.pi * current_point.arc_length / self.config.meander_wavelength
        )
        meander_angle = math.radians(self.config.meander_amplitude_degrees) * meander_phase
        meander_x, meander_y = self._rotate_vector(downhill_x, downhill_y, meander_angle)

        steer_x, steer_y = self._normalize(
            self.config.downhill_weight * meander_x
            + self.config.flow_bias_weight * flow_x
            + self.config.corridor_pull_weight * corridor_pull_x,
            self.config.downhill_weight * meander_y
            + self.config.flow_bias_weight * flow_y
            + self.config.corridor_pull_weight * corridor_pull_y,
        )

        next_tangent_x, next_tangent_y = self._normalize(
            (1.0 - self.config.tangent_blend) * tangent_x
            + self.config.tangent_blend * steer_x,
            (1.0 - self.config.tangent_blend) * tangent_y
            + self.config.tangent_blend * steer_y,
        )
        next_tangent_x, next_tangent_y = self._enforce_forward_flow(
            tangent_x=next_tangent_x,
            tangent_y=next_tangent_y,
            flow_angle=flow_angle,
        )
        if math.isclose(next_tangent_x, 0.0) and math.isclose(next_tangent_y, 0.0):
            return None, None, None

        candidate_x = current_point.x + self.config.step_length * next_tangent_x
        candidate_y = current_point.y + self.config.step_length * next_tangent_y
        return candidate_x, candidate_y, (next_tangent_x, next_tangent_y)

    def _corridor_pull_vector(
        self,
        *,
        x_coord: float,
        y_coord: float,
        seed_point: tuple[float, float],
        flow_angle: float,
        corridor_width: float,
    ) -> tuple[float, float]:
        relative_x = x_coord - seed_point[0]
        relative_y = y_coord - seed_point[1]
        cross_track = (
            self._project_along_angle(relative_x, relative_y, flow_angle + 90.0)
            / max(corridor_width, 1.0)
        )
        cross_unit_x, cross_unit_y = self._vector_from_angle(flow_angle + 90.0)
        return self._normalize(-cross_track * cross_unit_x, -cross_track * cross_unit_y)

    def _enforce_forward_flow(
        self,
        *,
        tangent_x: float,
        tangent_y: float,
        flow_angle: float,
    ) -> tuple[float, float]:
        flow_x, flow_y = self._vector_from_angle(flow_angle)
        cross_x, cross_y = self._vector_from_angle(flow_angle + 90.0)

        flow_component = tangent_x * flow_x + tangent_y * flow_y
        cross_component = tangent_x * cross_x + tangent_y * cross_y
        constrained_flow = max(flow_component, self.config.minimum_flow_component)

        return self._normalize(
            constrained_flow * flow_x + cross_component * cross_x,
            constrained_flow * flow_y + cross_component * cross_y,
        )

    @staticmethod
    def _project_along_angle(x_value: float, y_value: float, angle_degrees: float) -> float:
        angle_radians = math.radians(angle_degrees)
        return math.cos(angle_radians) * x_value + math.sin(angle_radians) * y_value

    @staticmethod
    def _vector_from_angle(angle_degrees: float) -> tuple[float, float]:
        angle_radians = math.radians(angle_degrees)
        return math.cos(angle_radians), math.sin(angle_radians)

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
