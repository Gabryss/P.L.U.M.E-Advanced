"""Shared data models for the stage-B.1 mixed branching sub-stage."""

from __future__ import annotations

from dataclasses import dataclass, field

from stages.graph import TrunkGraph

LOCAL_BYPASS_LOOP = "local_bypass_loop"
DOWNSTREAM_RECONNECT_LOOP = "downstream_reconnect_loop"
SPUR = "spur"
LOOP_KINDS = (LOCAL_BYPASS_LOOP, DOWNSTREAM_RECONNECT_LOOP)
BRANCH_KINDS = (LOCAL_BYPASS_LOOP, DOWNSTREAM_RECONNECT_LOOP, SPUR)


@dataclass(frozen=True)
class LoopPathConfig:
    """Parameters controlling the reconnecting loop generator."""

    boundary_margin: float = 75.0
    candidate_search_radius_cells: int = 8
    terminal_relief_distance: float = 90.0
    growth_cost_weight: float = 1.9
    uphill_penalty_weight: float = 7.5
    trunk_repulsion_distance: float = 120.0
    trunk_repulsion_weight: float = 2.4
    max_uphill_step: float = 2.0
    max_waypoint_climb: float = 4.0
    max_path_points: int = 420
    smoothing_iterations: int = 4
    smoothing_blend: float = 0.42
    internal_clearance_floor: float = 28.0
    local_bypass_min_target_offset_points: int = 12
    local_bypass_max_target_offset_points: int = 24
    local_bypass_lateral_offset: float = 170.0
    local_bypass_corridor_width: float = 120.0
    local_bypass_min_clearance: float = 95.0
    local_bypass_min_enclosed_area: float = 12000.0
    local_bypass_min_detour_factor: float = 1.12
    downstream_min_target_offset_points: int = 26
    downstream_max_target_offset_points: int = 48
    downstream_lateral_offset: float = 300.0
    downstream_corridor_width: float = 170.0
    downstream_min_clearance: float = 145.0
    downstream_min_enclosed_area: float = 32000.0
    downstream_min_detour_factor: float = 1.20


@dataclass(frozen=True)
class SpurBranchConfig:
    """Parameters controlling local dead-end spur growth."""

    branch_step_length: float = 18.0
    branch_max_steps: int = 20
    branch_angle_degrees: float = 34.0
    tangent_blend: float = 0.24
    downhill_weight: float = 0.40
    launch_weight: float = 0.30
    lateral_bias_weight: float = 0.52
    trunk_repulsion_target: float = 82.0
    trunk_repulsion_weight: float = 0.22
    stop_growth_cost: float = 0.72
    min_steps_before_cost_stop: int = 5
    min_total_length: float = 90.0
    branch_boundary_margin: float = 75.0
    max_uphill_step: float = 1.2


@dataclass(frozen=True)
class BranchMergeConfig:
    """Parameters controlling the mixed branch / merge sub-stage."""

    max_branch_count: int = 4
    candidate_pool_multiplier: int = 3
    junction_margin_points: int = 8
    min_junction_arc_separation: float = 240.0
    minimum_local_bypass_count: int = 1
    minimum_downstream_loop_count: int = 1
    local_bypass_weight: float = 0.45
    downstream_reconnect_weight: float = 0.30
    spur_weight: float = 0.25
    loop: LoopPathConfig = field(default_factory=LoopPathConfig)
    spur: SpurBranchConfig = field(default_factory=SpurBranchConfig)


@dataclass(frozen=True)
class BranchCandidate:
    """A scored trunk location selected as a branch junction."""

    trunk_index: int
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
    branch_kind: str
    source_trunk_index: int
    target_trunk_index: int


@dataclass(frozen=True)
class BranchPath:
    """One branch generated from the trunk spine."""

    branch_id: int
    branch_kind: str
    source_trunk_index: int
    target_trunk_index: int | None
    points: tuple[BranchPoint, ...]
    merge_event: MergeEvent | None
    termination_reason: str

    @property
    def total_length(self) -> float:
        return 0.0 if not self.points else self.points[-1].arc_length

    @property
    def is_loop(self) -> bool:
        return self.branch_kind in LOOP_KINDS


@dataclass(frozen=True)
class BranchMergeNetwork:
    """Output of the branch / merge sub-stage around the trunk graph."""

    config: BranchMergeConfig
    trunk_graph: TrunkGraph
    candidates: tuple[BranchCandidate, ...]
    branches: tuple[BranchPath, ...]

    def summary(self) -> dict[str, float]:
        """Return scalar summaries for quick inspection."""

        branch_count = float(len(self.branches))
        merged_count = float(sum(1 for branch in self.branches if branch.merge_event is not None))
        total_length = sum(branch.total_length for branch in self.branches)
        local_bypass_count = float(
            sum(1 for branch in self.branches if branch.branch_kind == LOCAL_BYPASS_LOOP)
        )
        downstream_count = float(
            sum(1 for branch in self.branches if branch.branch_kind == DOWNSTREAM_RECONNECT_LOOP)
        )
        spur_count = float(sum(1 for branch in self.branches if branch.branch_kind == SPUR))

        return {
            "candidate_count": float(len(self.candidates)),
            "branch_count": branch_count,
            "loop_count": local_bypass_count + downstream_count,
            "merged_branch_count": merged_count,
            "local_bypass_count": local_bypass_count,
            "downstream_reconnect_count": downstream_count,
            "spur_count": spur_count,
            "total_branch_length": total_length,
        }
