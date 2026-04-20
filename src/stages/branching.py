"""Stage B.1: branch and merge generation around the stable trunk spine."""

from __future__ import annotations

from dataclasses import dataclass
import math

from stages.graph import TrunkGraph
from stages.host_field import HostField


@dataclass(frozen=True)
class BranchMergeConfig:
    """Parameters controlling the branch/merge sub-stage."""

    max_branch_count: int = 2
    junction_margin_points: int = 10
    min_junction_arc_separation: float = 320.0
    merge_target_offset_points: int = 20
    branch_step_length: float = 18.0
    branch_max_steps: int = 26
    branch_angle_degrees: float = 34.0
    tangent_blend: float = 0.24
    downhill_weight: float = 0.36
    launch_weight: float = 0.28
    lateral_bias_weight: float = 0.42
    merge_pull_weight: float = 0.38
    branch_boundary_margin: float = 75.0
    merge_capture_distance: float = 32.0
    min_steps_before_merge: int = 5
    max_uphill_step: float = 1.2


@dataclass(frozen=True)
class BranchCandidate:
    """A scored trunk location selected as a branch junction."""

    trunk_index: int
    target_trunk_index: int
    score: float
    side: int


@dataclass(frozen=True)
class BranchPoint:
    """One sampled point along a generated branch path."""

    index: int
    branch_id: int
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
class MergeEvent:
    """A branch reconnection back into the trunk spine."""

    branch_id: int
    source_trunk_index: int
    target_trunk_index: int


@dataclass(frozen=True)
class BranchPath:
    """One branch generated from the trunk spine."""

    branch_id: int
    source_trunk_index: int
    target_trunk_index: int
    points: tuple[BranchPoint, ...]
    merge_event: MergeEvent | None
    termination_reason: str

    @property
    def total_length(self) -> float:
        return 0.0 if not self.points else self.points[-1].arc_length


@dataclass(frozen=True)
class BranchMergeNetwork:
    """Output of the branch/merge sub-stage around the trunk graph."""

    config: BranchMergeConfig
    trunk_graph: TrunkGraph
    candidates: tuple[BranchCandidate, ...]
    branches: tuple[BranchPath, ...]

    def summary(self) -> dict[str, float]:
        """Return scalar summaries for quick inspection."""

        branch_count = float(len(self.branches))
        merged_count = float(sum(1 for branch in self.branches if branch.merge_event is not None))
        total_length = sum(branch.total_length for branch in self.branches)
        return {
            "candidate_count": float(len(self.candidates)),
            "branch_count": branch_count,
            "merged_branch_count": merged_count,
            "total_branch_length": total_length,
        }


class BranchMergeGenerator:
    """Generate secondary branch paths around an existing trunk graph."""

    def __init__(self, config: BranchMergeConfig | None = None) -> None:
        self.config = config or BranchMergeConfig()

    def generate(self, host_field: HostField, trunk_graph: TrunkGraph) -> BranchMergeNetwork:
        candidates = self._select_candidates(trunk_graph)
        branches = tuple(
            self._grow_branch(
                host_field=host_field,
                trunk_graph=trunk_graph,
                candidate=candidate,
                branch_id=index,
            )
            for index, candidate in enumerate(candidates)
        )
        return BranchMergeNetwork(
            config=self.config,
            trunk_graph=trunk_graph,
            candidates=candidates,
            branches=branches,
        )

    def _select_candidates(self, trunk_graph: TrunkGraph) -> tuple[BranchCandidate, ...]:
        if not trunk_graph.points:
            return ()

        points = trunk_graph.points
        upper_bound = (
            len(points)
            - self.config.junction_margin_points
            - self.config.merge_target_offset_points
        )
        if upper_bound <= self.config.junction_margin_points:
            return ()

        scored_points: list[tuple[int, float]] = []
        for point in points[self.config.junction_margin_points : upper_bound]:
            score = self._score_candidate(point)
            scored_points.append((point.index, score))

        scored_points.sort(key=lambda item: item[1], reverse=True)

        selected: list[BranchCandidate] = []
        for trunk_index, score in scored_points:
            current_point = points[trunk_index]
            if any(
                abs(current_point.arc_length - points[candidate.trunk_index].arc_length)
                < self.config.min_junction_arc_separation
                for candidate in selected
            ):
                continue

            target_index = min(
                trunk_index + self.config.merge_target_offset_points,
                len(points) - self.config.junction_margin_points - 1,
            )
            side = 1 if len(selected) % 2 == 0 else -1
            selected.append(
                BranchCandidate(
                    trunk_index=trunk_index,
                    target_trunk_index=target_index,
                    score=score,
                    side=side,
                )
            )
            if len(selected) >= self.config.max_branch_count:
                break

        selected.sort(key=lambda candidate: candidate.trunk_index)
        return tuple(selected)

    @staticmethod
    def _score_candidate(point) -> float:
        low_cost_score = 1.0 - point.growth_cost
        roof_score = point.roof_competence
        slope_score = max(0.0, 1.0 - point.slope_degrees / 18.0)
        return 0.45 * low_cost_score + 0.35 * roof_score + 0.20 * slope_score

    def _grow_branch(
        self,
        *,
        host_field: HostField,
        trunk_graph: TrunkGraph,
        candidate: BranchCandidate,
        branch_id: int,
    ) -> BranchPath:
        source_point = trunk_graph.points[candidate.trunk_index]
        target_point = trunk_graph.points[candidate.target_trunk_index]

        launch_x, launch_y = self._rotate_vector(
            source_point.tangent_x,
            source_point.tangent_y,
            math.radians(candidate.side * self.config.branch_angle_degrees),
        )
        tangent_x, tangent_y = self._normalize(launch_x, launch_y)
        side_normal_x, side_normal_y = self._rotate_vector(
            source_point.tangent_x,
            source_point.tangent_y,
            candidate.side * math.pi / 2.0,
        )

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

        merge_event: MergeEvent | None = None
        termination_reason = "max_steps"

        for _ in range(self.config.branch_max_steps):
            current_point = points[-1]
            progress = current_point.arc_length / max(
                self.config.branch_max_steps * self.config.branch_step_length,
                1.0,
            )
            downhill_x, downhill_y = host_field.downhill_direction(
                current_point.x,
                current_point.y,
                fallback_angle_degrees=host_field.config.flow_angle_degrees,
            )
            target_pull_x, target_pull_y = self._normalize(
                target_point.x - current_point.x,
                target_point.y - current_point.y,
            )

            lateral_weight = self.config.lateral_bias_weight * max(0.0, 1.0 - progress)
            merge_weight = self.config.merge_pull_weight * progress
            launch_weight = self.config.launch_weight * max(0.0, 1.0 - progress)

            steer_x, steer_y = self._normalize(
                self.config.downhill_weight * downhill_x
                + launch_weight * launch_x
                + lateral_weight * side_normal_x
                + merge_weight * target_pull_x,
                self.config.downhill_weight * downhill_y
                + launch_weight * launch_y
                + lateral_weight * side_normal_y
                + merge_weight * target_pull_y,
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
                termination_reason = "boundary"
                break

            candidate_sample = host_field.sample(candidate_x, candidate_y)
            if candidate_sample.elevation > current_point.elevation + self.config.max_uphill_step:
                termination_reason = "uphill"
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

            if (
                len(points) - 1 >= self.config.min_steps_before_merge
                and self._distance(candidate_x, candidate_y, target_point.x, target_point.y)
                <= self.config.merge_capture_distance
            ):
                merge_event = MergeEvent(
                    branch_id=branch_id,
                    source_trunk_index=candidate.trunk_index,
                    target_trunk_index=candidate.target_trunk_index,
                )
                termination_reason = "merged"
                break

        return BranchPath(
            branch_id=branch_id,
            source_trunk_index=candidate.trunk_index,
            target_trunk_index=candidate.target_trunk_index,
            points=tuple(points),
            merge_event=merge_event,
            termination_reason=termination_reason,
        )

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
    def _rotate_vector(x_value: float, y_value: float, angle_radians: float) -> tuple[float, float]:
        cosine = math.cos(angle_radians)
        sine = math.sin(angle_radians)
        return cosine * x_value - sine * y_value, sine * x_value + cosine * y_value

    @staticmethod
    def _distance(x0: float, y0: float, x1: float, y1: float) -> float:
        return math.hypot(x1 - x0, y1 - y0)

    @staticmethod
    def _normalize(x_value: float, y_value: float) -> tuple[float, float]:
        length = math.hypot(x_value, y_value)
        if math.isclose(length, 0.0):
            return 0.0, 0.0
        return x_value / length, y_value / length
