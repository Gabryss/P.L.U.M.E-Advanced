"""Stage B.1: mixed branch and merge generation around the stable trunk spine."""

from __future__ import annotations

import math

import numpy as np

from stages.branching_loop_astar import LoopAStarGenerator
from stages.branching_models import (
    BRANCH_KINDS,
    DOWNSTREAM_RECONNECT_LOOP,
    LOCAL_BYPASS_LOOP,
    SPUR,
    BranchCandidate,
    BranchMergeConfig,
    BranchMergeNetwork,
    BranchPath,
)
from stages.branching_spur import SpurBranchGenerator
from stages.graph import TrunkGraph
from stages.host_field import HostField


class BranchMergeGenerator:
    """Generate a loop-heavy mix of reconnecting loops and dead-end spurs."""

    def __init__(self, config: BranchMergeConfig | None = None) -> None:
        self.config = config or BranchMergeConfig()
        self.loop_generator = LoopAStarGenerator(self.config.loop)
        self.spur_generator = SpurBranchGenerator(self.config.spur)

    def generate(self, host_field: HostField, trunk_graph: TrunkGraph) -> BranchMergeNetwork:
        trunk_distance_field = self._build_trunk_distance_field(host_field, trunk_graph)
        candidates = self._select_candidates(trunk_graph)
        branch_plan = self._build_branch_kind_plan()

        branches: list[BranchPath] = []
        for candidate in candidates:
            if len(branches) >= self.config.max_branch_count:
                break

            planned_kind = branch_plan[min(len(branches), len(branch_plan) - 1)]
            local_needed, downstream_needed = self._remaining_loop_requirements(branches)
            for branch_kind in self._attempt_order(
                planned_kind,
                local_needed=local_needed,
                downstream_needed=downstream_needed,
            ):
                branch = self._generate_branch(
                    branch_kind=branch_kind,
                    host_field=host_field,
                    trunk_graph=trunk_graph,
                    candidate=candidate,
                    branch_id=len(branches),
                    trunk_distance_field=trunk_distance_field,
                )
                if branch is None or self._conflicts_with_existing(branch, branches, trunk_graph):
                    continue
                branches.append(branch)
                break

        return BranchMergeNetwork(
            config=self.config,
            trunk_graph=trunk_graph,
            candidates=candidates,
            branches=tuple(branches),
        )

    def _generate_branch(
        self,
        *,
        branch_kind: str,
        host_field: HostField,
        trunk_graph: TrunkGraph,
        candidate: BranchCandidate,
        branch_id: int,
        trunk_distance_field,
    ) -> BranchPath | None:
        if branch_kind == SPUR:
            return self.spur_generator.generate(
                host_field=host_field,
                trunk_graph=trunk_graph,
                candidate=candidate,
                branch_id=branch_id,
                trunk_distance_field=trunk_distance_field,
            )
        return self.loop_generator.generate(
            host_field=host_field,
            trunk_graph=trunk_graph,
            candidate=candidate,
            branch_id=branch_id,
            branch_kind=branch_kind,
            trunk_distance_field=trunk_distance_field,
        )

    def _select_candidates(self, trunk_graph: TrunkGraph) -> tuple[BranchCandidate, ...]:
        if not trunk_graph.points:
            return ()

        points = trunk_graph.points
        upper_bound = (
            len(points)
            - self.config.junction_margin_points
            - self.config.loop.downstream_max_target_offset_points
        )
        if upper_bound <= self.config.junction_margin_points:
            return ()

        scored_points: list[tuple[int, float]] = []
        for point in points[self.config.junction_margin_points : upper_bound]:
            scored_points.append((point.index, self._score_candidate(point)))
        scored_points.sort(key=lambda item: item[1], reverse=True)

        selected: list[BranchCandidate] = []
        max_candidates = max(self.config.max_branch_count * self.config.candidate_pool_multiplier, 1)
        for trunk_index, score in scored_points:
            current_point = points[trunk_index]
            if any(
                abs(current_point.arc_length - points[candidate.trunk_index].arc_length)
                < self.config.min_junction_arc_separation
                for candidate in selected
            ):
                continue
            side = 1 if len(selected) % 2 == 0 else -1
            selected.append(BranchCandidate(trunk_index=trunk_index, score=score, side=side))
            if len(selected) >= max_candidates:
                break

        selected.sort(key=lambda candidate: candidate.trunk_index)
        return tuple(selected)

    @staticmethod
    def _score_candidate(point) -> float:
        low_cost_score = 1.0 - point.growth_cost
        roof_score = point.roof_competence
        slope_score = max(0.0, 1.0 - point.slope_degrees / 18.0)
        return 0.45 * low_cost_score + 0.35 * roof_score + 0.20 * slope_score

    def _build_branch_kind_plan(self) -> tuple[str, ...]:
        if self.config.max_branch_count <= 0:
            return ()

        weights = {
            LOCAL_BYPASS_LOOP: self.config.local_bypass_weight,
            DOWNSTREAM_RECONNECT_LOOP: self.config.downstream_reconnect_weight,
            SPUR: self.config.spur_weight,
        }
        minimums = {
            LOCAL_BYPASS_LOOP: self.config.minimum_local_bypass_count,
            DOWNSTREAM_RECONNECT_LOOP: self.config.minimum_downstream_loop_count,
            SPUR: 0,
        }
        total_weight = sum(max(weight, 0.0) for weight in weights.values()) or 1.0
        raw_counts = {
            kind: self.config.max_branch_count * max(weight, 0.0) / total_weight
            for kind, weight in weights.items()
        }
        counts = {kind: int(math.floor(raw_counts[kind])) for kind in BRANCH_KINDS}
        remainders = {kind: raw_counts[kind] - counts[kind] for kind in BRANCH_KINDS}

        for kind, minimum in minimums.items():
            counts[kind] = max(counts[kind], minimum)

        while sum(counts.values()) > self.config.max_branch_count:
            removable = [
                kind
                for kind in BRANCH_KINDS
                if counts[kind] > minimums.get(kind, 0)
            ]
            if not removable:
                break
            removable.sort(key=lambda kind: (weights[kind], counts[kind]), reverse=False)
            counts[removable[0]] -= 1

        while sum(counts.values()) < self.config.max_branch_count:
            addable = sorted(
                BRANCH_KINDS,
                key=lambda kind: (remainders[kind], weights[kind]),
                reverse=True,
            )
            counts[addable[0]] += 1

        plan: list[str] = []
        rotation = [LOCAL_BYPASS_LOOP, DOWNSTREAM_RECONNECT_LOOP, SPUR]
        remaining = counts.copy()
        while len(plan) < self.config.max_branch_count:
            progressed = False
            for branch_kind in rotation:
                if remaining[branch_kind] <= 0:
                    continue
                plan.append(branch_kind)
                remaining[branch_kind] -= 1
                progressed = True
                if len(plan) >= self.config.max_branch_count:
                    break
            if not progressed:
                break
        return tuple(plan)

    def _remaining_loop_requirements(self, branches: list[BranchPath]) -> tuple[int, int]:
        local_count = sum(1 for branch in branches if branch.branch_kind == LOCAL_BYPASS_LOOP)
        downstream_count = sum(
            1 for branch in branches if branch.branch_kind == DOWNSTREAM_RECONNECT_LOOP
        )
        return (
            max(0, self.config.minimum_local_bypass_count - local_count),
            max(0, self.config.minimum_downstream_loop_count - downstream_count),
        )

    @staticmethod
    def _attempt_order(
        planned_kind: str,
        *,
        local_needed: int,
        downstream_needed: int,
    ) -> tuple[str, ...]:
        order: list[str] = []
        required_loops: list[str] = []
        if local_needed > 0:
            required_loops.append(LOCAL_BYPASS_LOOP)
        if downstream_needed > 0:
            required_loops.append(DOWNSTREAM_RECONNECT_LOOP)

        branch_sequence = [planned_kind, *required_loops, LOCAL_BYPASS_LOOP, DOWNSTREAM_RECONNECT_LOOP]
        for branch_kind in branch_sequence:
            if required_loops and branch_kind == SPUR:
                continue
            if branch_kind not in order:
                order.append(branch_kind)

        if not required_loops:
            if planned_kind == SPUR:
                order.insert(0, SPUR)
            else:
                order.append(SPUR)

        return tuple(order)

    def _conflicts_with_existing(
        self,
        branch: BranchPath,
        existing_branches: list[BranchPath],
        trunk_graph: TrunkGraph,
    ) -> bool:
        source_arc = trunk_graph.points[branch.source_trunk_index].arc_length
        target_arc = (
            trunk_graph.points[branch.target_trunk_index].arc_length
            if branch.target_trunk_index is not None
            else None
        )
        for existing in existing_branches:
            existing_source_arc = trunk_graph.points[existing.source_trunk_index].arc_length
            if abs(source_arc - existing_source_arc) < 0.7 * self.config.min_junction_arc_separation:
                return True
            if target_arc is None or existing.target_trunk_index is None:
                continue
            existing_target_arc = trunk_graph.points[existing.target_trunk_index].arc_length
            if abs(target_arc - existing_target_arc) < 0.4 * self.config.min_junction_arc_separation:
                return True
        return False

    @staticmethod
    def _build_trunk_distance_field(host_field: HostField, trunk_graph: TrunkGraph):
        x_grid, y_grid = np.meshgrid(host_field.x_coords, host_field.y_coords)
        distance_field = np.full_like(x_grid, np.inf, dtype=float)
        for point in trunk_graph.points:
            distance_field = np.minimum(
                distance_field,
                np.hypot(x_grid - point.x, y_grid - point.y),
            )
        return distance_field
