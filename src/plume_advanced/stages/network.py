"""Stage B: host-driven procedural lava-tube network generation."""

from __future__ import annotations

import heapq
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

from plume_advanced.procedural import procedural_rng
from plume_advanced.stages.host_field import HostField

SegmentMetadataValue = str | int | float | bool | None


@dataclass(frozen=True)
class BraidGrammarConfig:
    """Seed-sampled controls for network split/merge motifs."""

    zone_count: tuple[int, int] = (4, 6)
    center_fraction: tuple[float, float] = (0.14, 0.88)
    min_center_spacing: float = 0.11
    half_length_fraction: tuple[float, float] = (0.045, 0.09)
    branches_per_zone: tuple[int, int] = (2, 4)
    lateral_offset_scale: tuple[float, float] = (0.35, 1.25)
    start_shift_fraction: tuple[float, float] = (-0.24, 0.24)
    end_shift_fraction: tuple[float, float] = (-0.24, 0.24)
    skew: tuple[float, float] = (-0.65, 0.65)
    wobble: tuple[float, float] = (0.08, 0.36)
    underpass_probability: float = 0.22
    ladder_probability: float = 0.48
    ladder_rung_count: tuple[int, int] = (1, 2)
    chamber_radius_scale: tuple[float, float] = (1.0, 1.55)


@dataclass(frozen=True)
class LobeGrowthConfig:
    """Controls for process-based stochastic lava-lobe propagation."""

    path_count: tuple[int, int] = (8, 12)
    maximum_steps: tuple[int, int] = (42, 96)
    minimum_persistence_steps: int = 8
    minimum_anchor_spacing_fraction: float = 0.055
    terrain_perturbation_m: float = 2.4
    perturbation_correlation_cells: float = 3.5
    candidate_temperature: float = 0.38
    inertia_weight: float = 1.35
    perturbed_slope_weight: float = 2.10
    downstream_potential_weight: float = 3.20
    initial_divergence_weight: float = 2.40
    channel_avoidance_weight: float = 1.80
    channel_reuse_weight: float = 3.60
    exposed_cooling_multiplier: float = 5.0
    retirement_temperature_k: float = 1_060.0
    branch_flux_fraction: tuple[float, float] = (0.18, 0.58)
    retired_path_fraction: float = 0.24
    # Event-driven secondary growth. Breakouts are launched where finite
    # parent discharge encounters poor carrying capacity, weak confinement,
    # strong curvature, or a seeded obstruction. Accepted lobes modify a
    # temporary emplacement surface before later fronts are traced.
    breakout_capacity_weight: float = 2.40
    breakout_confinement_weight: float = 1.65
    breakout_curvature_weight: float = 1.10
    breakout_blockage_weight: float = 1.35
    minimum_viable_flux_fraction: float = 0.075
    coalescence_flux_return_fraction: float = 0.72
    deposition_feedback_m: float = 0.85
    deposition_spread_cells: float = 1.75
    # Low-frequency planform controls for the arterial route.  These are
    # explicit heuristic controls (rather than hidden density effects) so a
    # seeded network can be made sinuous while retaining host-field steering.
    backbone_curvature_fraction: float = 0.20
    backbone_curvature_wavelength_fraction: float = 0.38
    backbone_curvature_secondary_fraction: float = 0.35


@dataclass(frozen=True)
class EmplacementHistoryConfig:
    """Controls for the staged construction and preservation of tube routes."""

    phase_count: tuple[int, int] = (3, 5)
    active_phase_span: tuple[int, int] = (1, 3)
    stacked_lobe_fraction: float = 0.34
    maximum_absolute_level: int = 2
    chamber_formation_probability: float = 0.28
    vertical_capture_chamber_probability: float = 0.62
    roof_failure_probability: float = 0.16
    # Finite discharge available to each eruptive phase, expressed relative
    # to ``network.source_flux``.  Later phases inherit only established
    # passages and deposited relief from earlier pulses.
    phase_flux_budget_fraction: float = 1.0
    reoccupation_probability: float = 0.24
    breakout_probability: float = 0.58
    retirement_flux_threshold: float = 0.10
    drained_pool_enabled: bool = True
    drained_pool_probability: float = 0.96
    drained_pool_count: tuple[int, int] = (1, 3)
    drained_pool_min_spacing_m: float = 180.0
    drained_pool_flux_quantile: float = 0.72
    drained_pool_flux_weight: float = 1.0
    drained_pool_grade_weight: float = 1.0
    drained_pool_slope_break_weight: float = 0.8
    drained_pool_length_m: tuple[float, float] = (35.0, 90.0)
    drained_pool_width_ratio: tuple[float, float] = (2.0, 4.5)
    drained_pool_max_width_m: float = 28.0
    drained_pool_depth_m: tuple[float, float] = (5.0, 12.0)


@dataclass(frozen=True)
class CaveNetworkConfig:
    """Parameters controlling the host-driven lava-tube network generator."""

    random_seed: int | None = None
    growth_model: str = "hybrid_lobe"
    network_density: float = 1.0
    emplacement_backend: str = "internal"
    flowy_executable: str | None = None
    flowy_timeout_s: float = 120.0
    flowy_output_path: str | None = None
    downflow_ensemble_size: int = 8
    # Independent opportunity controls; ``network_density`` remains the
    # monotonic endmember selector while these knobs describe process rates.
    lobe_launch_rate: float = 1.0
    loop_probability: float = 1.0
    capture_probability: float = 1.0
    chamber_gain: float = 1.0
    braid_grammar: BraidGrammarConfig = BraidGrammarConfig()
    lobe_growth: LobeGrowthConfig = LobeGrowthConfig()
    emplacement_history: EmplacementHistoryConfig = EmplacementHistoryConfig()
    body_spatial_scale: float = 1.0
    target_route_length_m: float = 5_000.0
    source_count: int = 8
    source_flux: float = 1.0
    source_temperature_k: float = 1_450.0
    cooling_k_per_m: float = 0.025
    nominal_flow_speed_m_s: float = 0.35
    source_band_length: float = 90.0
    source_band_half_width: float = 180.0
    sink_margin: float = 80.0
    trace_max_steps: int = 460
    max_uphill_step: float = 1.2
    growth_cost_weight: float = 3.2
    corridor_weight: float = 0.35
    chamber_flux_quantile: float = 0.82
    base_passage_radius: float = 4.6
    minimum_passage_radius: float = 3.2
    maximum_passage_radius: float = 6.0
    chamber_radius: float = 46.0
    chamber_radius_fraction: float = 0.70
    minimum_branch_offset_widths: float = 1.25
    paint_flux_chambers: bool = False
    occupancy_smoothing_passes: int = 1
    spur_count: int = 5
    spur_max_steps: int = 24
    spur_lateral_bias: float = 1.3
    spur_congestion_weight: float = 0.85
    channel_count_samples: int = 28


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
    flux: float = 0.0
    temperature_k: float = 0.0
    age_s: float = 0.0


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

    @property
    def mean_flux(self) -> float:
        if not self.points:
            return 0.0
        return sum(point.flux for point in self.points) / len(self.points)


@dataclass(frozen=True)
class CaveJunction:
    """One higher-level morphological transition region in the cave network."""

    junction_id: int
    kind: str
    node_ids: tuple[int, ...]
    segment_ids: tuple[int, ...]
    center_x: float
    center_y: float
    along_position: float
    blend_length: float
    split_style: str
    merge_style: str
    capacity_bias: float
    metadata: dict[str, SegmentMetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class CaveNetwork:
    """Stage-B output for the host-driven lava-tube network."""

    config: CaveNetworkConfig
    nodes: tuple[CaveNode, ...]
    segments: tuple[CaveSegment, ...]
    junctions: tuple[CaveJunction, ...]
    occupancy: np.ndarray
    width_field: np.ndarray
    dominant_route_node_ids: tuple[int, ...]
    slice_along_positions: tuple[float, ...]
    slice_channel_counts: tuple[int, ...]
    slice_visible_channel_counts: tuple[int, ...]
    backend_provenance: dict[str, SegmentMetadataValue] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        """Return scalar summaries for quick inspection."""

        occupied_area = float(self.occupancy.sum())
        segment_lengths = [segment.total_length for segment in self.segments]
        point_widths = [point.width for segment in self.segments for point in segment.points]
        mean_segment_width = float(np.mean(point_widths)) if point_widths else 0.0
        min_segment_width = float(np.min(point_widths)) if point_widths else 0.0
        max_segment_width = float(np.max(point_widths)) if point_widths else 0.0
        point_fluxes = [point.flux for segment in self.segments for point in segment.points]
        point_temperatures = [
            point.temperature_k
            for segment in self.segments
            for point in segment.points
            if point.temperature_k > 0.0
        ]
        loop_count = float(self._loop_rank())
        terminal_count = float(sum(1 for degree in self._degrees().values() if degree == 1))
        spur_count = float(
            sum(
                1
                for segment in self.segments
                if segment.kind in {"spur", "abandoned_lobe", "stalled_lobe"}
            )
        )
        lobe_path_ids = {
            str(segment.metadata["lobe_path_id"])
            for segment in self.segments
            if segment.metadata.get("lobe_path_id") is not None
        }
        primary_branch_kinds = {
            "anastomosis",
            "chamber_braid",
            "distributary",
            "inner_bypass",
            "island_bypass",
            "underpass",
        }
        branch_persistence = [
            segment.total_length / max(segment.mean_width, 1e-6)
            for segment in self.segments
            if segment.kind in primary_branch_kinds and segment.points
        ]
        weighted_sinuosity_numerator = 0.0
        uphill_distance = 0.0
        sustained_uphill_steps = 0
        for segment in self.segments:
            if len(segment.points) < 2:
                continue
            first, last = segment.points[0], segment.points[-1]
            chord = math.hypot(last.x - first.x, last.y - first.y)
            weighted_sinuosity_numerator += segment.total_length * (
                segment.total_length / max(chord, 1e-9)
            )
            previous_uphill = False
            for current, following in zip(segment.points, segment.points[1:]):
                step_length = math.hypot(
                    following.x - current.x,
                    following.y - current.y,
                )
                uphill = following.elevation > current.elevation
                if uphill:
                    uphill_distance += step_length
                if uphill and previous_uphill:
                    sustained_uphill_steps += 1
                previous_uphill = uphill
        z_levels = {segment.z_level for segment in self.segments}
        emplacement_phases = []
        breakout_records: dict[str, dict[str, SegmentMetadataValue]] = {}
        for segment in self.segments:
            phase_value = segment.metadata.get("emplacement_phase_count", 1)
            emplacement_phases.append(
                int(phase_value) if isinstance(phase_value, (int, float)) else 1
            )
            path_id = segment.metadata.get("lobe_path_id")
            if path_id is not None and segment.metadata.get("branching_process") is not None:
                breakout_records.setdefault(str(path_id), segment.metadata)
        breakout_scores = [
            float(metadata.get("breakout_score", 0.0)) for metadata in breakout_records.values()
        ]
        breakout_fractions = [
            float(metadata.get("branch_flux_fraction", 0.0))
            for metadata in breakout_records.values()
        ]
        allocated_flux = sum(
            float(metadata.get("initial_flux", 0.0)) for metadata in breakout_records.values()
        )
        returned_flux = sum(
            float(metadata.get("coalescence_returned_flux", 0.0))
            for metadata in breakout_records.values()
        )
        phase_utilization: dict[int, float] = {}
        for metadata in breakout_records.values():
            phase = int(metadata.get("birth_phase", 0) or 0)
            budget = float(metadata.get("phase_flux_budget", 0.0) or 0.0)
            allocated = float(metadata.get("phase_flux_allocated", 0.0) or 0.0)
            if budget > 0.0:
                phase_utilization[phase] = max(
                    phase_utilization.get(phase, 0.0),
                    allocated / budget,
                )
        pool_records = {
            str(segment.metadata.get("pool_id", segment.segment_id)): segment.metadata
            for segment in self.segments
            if segment.metadata.get("chamber_type") == "drained_lava_pool"
        }
        pool_widths = [float(item.get("pool_width_m", 0.0)) for item in pool_records.values()]
        pool_aspects = [float(item.get("pool_aspect_ratio", 0.0)) for item in pool_records.values()]
        pool_outlet_ratios = [
            float(item.get("pool_outlet_ratio", 0.0)) for item in pool_records.values()
        ]

        return {
            "node_count": float(len(self.nodes)),
            "segment_count": float(len(self.segments)),
            "entry_count": float(sum(node.kind == "entry" for node in self.nodes)),
            "junction_count": float(len(self.junctions)),
            "loop_count": loop_count,
            "network_density": self.config.network_density,
            "lobe_path_count": float(len(lobe_path_ids)),
            "breakout_event_count": float(len(breakout_records)),
            "mean_breakout_score": float(np.mean(breakout_scores)) if breakout_scores else 0.0,
            "mean_branch_flux_fraction": float(np.mean(breakout_fractions))
            if breakout_fractions
            else 0.0,
            "coalescence_flux_return_ratio": returned_flux / max(allocated_flux, 1e-9),
            "reoccupied_path_count": float(
                sum(
                    bool(metadata.get("reoccupied", False))
                    for metadata in breakout_records.values()
                )
            ),
            "piracy_event_count": float(
                sum(bool(metadata.get("pirated", False)) for metadata in breakout_records.values())
            ),
            "flux_starved_retired_count": float(
                sum(
                    metadata.get("formation_state") == "flux_starved_retired"
                    for metadata in breakout_records.values()
                )
            ),
            "mean_phase_budget_utilization": (
                float(np.mean(tuple(phase_utilization.values()))) if phase_utilization else 0.0
            ),
            "max_phase_budget_utilization": max(phase_utilization.values(), default=0.0),
            "anastomosis_count": float(
                sum(segment.kind == "anastomosis" for segment in self.segments)
            ),
            "emplacement_phase_count": float(max(emplacement_phases, default=1)),
            "vertical_level_count": float(len(z_levels)),
            "stacked_segment_count": float(sum(segment.z_level != 0 for segment in self.segments)),
            "vertical_capture_count": float(
                sum(
                    bool(segment.metadata.get("vertical_capture", False))
                    for segment in self.segments
                )
            ),
            "process_chamber_count": float(
                sum(
                    bool(segment.metadata.get("chamber_forming", False))
                    for segment in self.segments
                )
            ),
            "drained_lava_pool_count": float(len(pool_records)),
            "drained_lava_pool_mean_width_m": float(np.mean(pool_widths)) if pool_widths else 0.0,
            "drained_lava_pool_mean_aspect_ratio": float(np.mean(pool_aspects))
            if pool_aspects
            else 0.0,
            "drained_lava_pool_mean_outlet_ratio": float(np.mean(pool_outlet_ratios))
            if pool_outlet_ratios
            else 0.0,
            "skylight_prone_segment_count": float(
                sum(
                    segment.metadata.get("roof_state") == "skylight_prone"
                    for segment in self.segments
                )
            ),
            "retired_lobe_count": float(
                sum(segment.kind in {"abandoned_lobe", "stalled_lobe"} for segment in self.segments)
            ),
            "terminal_count": terminal_count,
            "spur_count": spur_count,
            "occupied_cell_count": occupied_area,
            "total_length": float(sum(segment_lengths)),
            "dominant_route_length": self.dominant_route_length,
            "mean_segment_width": mean_segment_width,
            "min_segment_width": min_segment_width,
            "max_segment_width": max_segment_width,
            "maximum_flux": float(max(point_fluxes, default=0.0)),
            "mean_temperature_k": float(np.mean(point_temperatures)) if point_temperatures else 0.0,
            "max_flow_conservation_error": self.max_flow_conservation_error(),
            "max_parallel_channels": float(
                max(self.slice_channel_counts) if self.slice_channel_counts else 0
            ),
            "max_visible_parallel_channels": float(
                max(self.slice_visible_channel_counts) if self.slice_visible_channel_counts else 0
            ),
            "primary_branch_count": float(len(branch_persistence)),
            "mean_branch_persistence_widths": float(
                np.mean(branch_persistence) if branch_persistence else 0.0
            ),
            "short_branch_fraction": float(
                np.mean(np.asarray(branch_persistence) < 3.0) if branch_persistence else 0.0
            ),
            "length_weighted_mean_sinuosity": float(
                weighted_sinuosity_numerator / max(sum(segment_lengths), 1e-9)
            )
            if segment_lengths
            else 1.0,
            "uphill_distance_m": float(uphill_distance),
            "sustained_uphill_step_count": float(sustained_uphill_steps),
        }

    def max_flow_conservation_error(self) -> float:
        """Return the largest normalized split/merge flux imbalance."""

        incoming: defaultdict[int, float] = defaultdict(float)
        outgoing: defaultdict[int, float] = defaultdict(float)
        for segment in self.segments:
            if not segment.points:
                continue
            flux = max(segment.mean_flux, 0.0)
            outgoing[segment.start_node_id] += flux
            incoming[segment.end_node_id] += flux
        errors: list[float] = []
        for node in self.nodes:
            if node.kind in {"entry", "exit", "terminal", "spur_terminal"}:
                continue
            if incoming[node.node_id] <= 0.0 or outgoing[node.node_id] <= 0.0:
                continue
            denominator = max(incoming[node.node_id], outgoing[node.node_id], 1e-9)
            errors.append(abs(incoming[node.node_id] - outgoing[node.node_id]) / denominator)
        return float(max(errors, default=0.0))

    @property
    def dominant_route_length(self) -> float:
        if len(self.dominant_route_node_ids) < 2:
            return 0.0
        total = 0.0
        for start_node_id, end_node_id in zip(
            self.dominant_route_node_ids,
            self.dominant_route_node_ids[1:],
        ):
            candidates = [
                segment
                for segment in self.segments
                if segment.start_node_id == start_node_id and segment.end_node_id == end_node_id
            ]
            if candidates:
                total += max(candidates, key=lambda segment: segment.mean_flux).total_length
        return total

    def _degrees(self) -> dict[int, int]:
        degrees = {node.node_id: 0 for node in self.nodes}
        for segment in self.segments:
            degrees[segment.start_node_id] += 1
            degrees[segment.end_node_id] += 1
        return degrees

    def _loop_rank(self) -> int:
        if not self.nodes:
            return 0

        adjacency: dict[int, set[int]] = {node.node_id: set() for node in self.nodes}
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


def export_network_report(
    cave_network: CaveNetwork,
    output_path: str | Path,
) -> Path:
    """Write topology and visibility diagnostics for regression review."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    segments_by_kind: dict[str, list[CaveSegment]] = defaultdict(list)
    for segment in cave_network.segments:
        segments_by_kind[segment.kind].append(segment)

    kind_summary: dict[str, dict[str, float]] = {}
    for kind, segments in sorted(segments_by_kind.items()):
        persistence = [
            segment.total_length / max(segment.mean_width, 1e-6)
            for segment in segments
            if segment.points
        ]
        kind_summary[kind] = {
            "count": float(len(segments)),
            "total_length_m": float(sum(segment.total_length for segment in segments)),
            "mean_persistence_widths": float(np.mean(persistence) if persistence else 0.0),
        }

    breakout_paths: dict[str, dict[str, SegmentMetadataValue]] = {}
    for segment in cave_network.segments:
        path_id = segment.metadata.get("lobe_path_id")
        if path_id is None or segment.metadata.get("branching_process") is None:
            continue
        breakout_paths.setdefault(str(path_id), segment.metadata)
    breakout_events = [
        {
            "path_id": path_id,
            **{
                name: metadata.get(name)
                for name in (
                    "birth_phase",
                    "death_phase",
                    "branching_process",
                    "breakout_trigger",
                    "breakout_score",
                    "breakout_capacity_deficit",
                    "breakout_confinement_loss",
                    "breakout_curvature",
                    "breakout_blockage",
                    "parent_flux_before_split",
                    "parent_flux_after_split",
                    "branch_flux_fraction",
                    "initial_flux",
                    "coalescence_returned_flux",
                    "deposition_feedback_m",
                    "termination",
                    "formation_state",
                    "event_type",
                    "new_path",
                    "reoccupied",
                    "pirated",
                    "reoccupation_mode",
                    "reoccupation_target_lobe_id",
                    "branch_order",
                    "loop_mechanism",
                    "phase_flux_budget",
                    "phase_flux_allocated",
                    "requested_flux",
                    "chamber_type",
                    "process_cause",
                    "pool_length_m",
                    "pool_width_m",
                    "pool_depth_m",
                    "pool_aspect_ratio",
                    "pool_outlet_ratio",
                    "pool_inlet_count",
                    "pool_outlet_count",
                    "pool_site_score",
                )
            },
        }
        for path_id, metadata in sorted(breakout_paths.items())
    ]

    report = {
        "schema": "plume.cave-network-diagnostics.v1",
        "summary": cave_network.summary(),
        "emplacement_backend": cave_network.backend_provenance,
        "body_spatial_scale": cave_network.config.body_spatial_scale,
        "target_route_length_m": cave_network.config.target_route_length_m,
        "minimum_branch_offset_widths": (cave_network.config.minimum_branch_offset_widths),
        "segment_kinds": kind_summary,
        "breakout_events": breakout_events,
        "visibility_profile": [
            {
                "along_position_m": along,
                "skeleton_channels": skeleton,
                "visible_channels": visible,
            }
            for along, skeleton, visible in zip(
                cave_network.slice_along_positions,
                cave_network.slice_channel_counts,
                cave_network.slice_visible_channel_counts,
            )
        ],
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


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


@dataclass(frozen=True)
class _LobeTrace:
    path: tuple[tuple[int, int], ...]
    merged: bool
    maximum_lateral_separation: float
    final_temperature_k: float
    initial_flux: float


@dataclass(frozen=True)
class _BreakoutSite:
    """One process-scored opportunity for a secondary tube breakout."""

    cell: tuple[int, int]
    backbone_index: int
    score: float
    trigger: str
    capacity_deficit: float
    confinement_loss: float
    curvature: float
    blockage: float


@dataclass(frozen=True)
class EmplacementProposal:
    """Backend-neutral route evidence consumed by PLUME graph construction."""

    paths: tuple[tuple[tuple[float, float], ...], ...]
    backend: str
    version: str
    provenance: dict[str, SegmentMetadataValue]


class DownflowReferenceBackend:
    """Reference perturbed-DEM steepest-descent path ensemble.

    This is a transparent PLUME implementation of the published stochastic
    perturbed-DEM idea, not a claim to ship or call an official DOWNFLOW
    library.  It returns route evidence only; PLUME owns graph semantics.
    """

    name = "downflow_reference"
    version = "reference-1"

    def propose(
        self,
        host_field: HostField,
        geometry: _FlowGeometry,
        *,
        start_cell: tuple[int, int],
        seed: int | None,
        steps: int,
        uphill_limit: float,
        ensemble_size: int = 8,
    ) -> EmplacementProposal:
        ensemble: list[list[tuple[int, int]]] = []
        corridor_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for member in range(max(1, ensemble_size)):
            rng = procedural_rng(seed, "downflow-reference", member)
            perturbation = CaveNetworkGenerator._correlated_terrain_perturbation(
                host_field.elevation.shape,
                amplitude_m=max(0.5, 0.03 * float(np.ptp(host_field.elevation))),
                correlation_cells=5.0,
                rng=rng,
            )
            path = [start_cell]
            for _ in range(max(1, steps)):
                current = path[-1]
                if float(geometry.along_grid[current]) >= geometry.along_extent:
                    break
                candidates = []
                current_elevation = float(host_field.elevation[current])
                y_index, x_index = current
                ny, nx = host_field.elevation.shape
                neighbors = [
                    (yy, xx)
                    for yy in range(max(0, y_index - 1), min(ny, y_index + 2))
                    for xx in range(max(0, x_index - 1), min(nx, x_index + 2))
                    if (yy, xx) != current
                ]
                for candidate in neighbors:
                    if candidate in path[-8:]:
                        continue
                    along_delta = float(
                        geometry.along_grid[candidate] - geometry.along_grid[current]
                    )
                    if along_delta < -0.25 * geometry.cell_scale:
                        continue
                    uphill = float(host_field.elevation[candidate]) - current_elevation
                    if uphill > uphill_limit:
                        continue
                    score = float(host_field.elevation[candidate] + perturbation[candidate])
                    score -= 0.08 * along_delta
                    score += 1e-6 * float(rng.random())
                    candidates.append((score, candidate))
                if not candidates:
                    break
                path.append(min(candidates, key=lambda item: item[0])[1])
            ensemble.append(path)
            for cell in set(path):
                corridor_counts[cell] += 1
        ranked = sorted(corridor_counts.items(), key=lambda item: (-item[1], item[0]))
        persistence_floor = max(1, int(math.ceil(0.25 * max(1, ensemble_size))))
        persistent_cells = {cell for cell, count in ranked if count >= persistence_floor}
        scored_paths = []
        for path in ensemble:
            score = sum(corridor_counts[cell] for cell in path if cell in persistent_cells)
            scored_paths.append((score, tuple(path)))
        scored_paths.sort(key=lambda item: (-item[0], item[1]))
        selected_paths = tuple(
            tuple(CaveNetworkGenerator._cell_to_world(host_field, cell) for cell in path)
            for _score, path in scored_paths[: min(3, len(scored_paths))]
            if len(path) >= 3
        )
        coordinates = selected_paths or ((),)
        return EmplacementProposal(
            paths=coordinates,
            backend=self.name,
            version=self.version,
            provenance={
                "algorithm": "stochastic_perturbed_dem_steepest_descent",
                "official_library": False,
                "seed_namespace": "downflow-reference",
                "ensemble_size": ensemble_size,
                "persistent_corridor_cells": len(persistent_cells),
            },
        )


class FlowyBackend:
    """Strict adapter for an explicitly configured flowy-code/flowy executable."""

    name = "flowy"
    version = "external"

    def __init__(
        self, executable: str, *, timeout_s: float = 120.0, output_path: str | None = None
    ):
        self.executable = executable
        self.timeout_s = timeout_s
        self.output_path = output_path

    def propose(
        self,
        host_field: HostField,
        geometry: _FlowGeometry,
        *,
        start_cell: tuple[int, int],
        seed: int | None,
        steps: int,
        uphill_limit: float,
    ) -> EmplacementProposal:
        del geometry, steps, uphill_limit
        run_dir = Path(tempfile.mkdtemp(prefix="plume-flowy-"))
        output_dir = Path(self.output_path) if self.output_path else run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        dem_path = run_dir / "terrain.asc"
        input_path = run_dir / "input.toml"
        elevation = np.asarray(host_field.elevation, dtype=float)
        x0, y0 = float(host_field.x_coords[0]), float(host_field.y_coords[0])
        cellsize = float(
            np.mean([host_field.config.grid.spacing_x, host_field.config.grid.spacing_y])
        )
        with dem_path.open("w", encoding="utf-8") as stream:
            stream.write(
                f"ncols {elevation.shape[1]}\nnrows {elevation.shape[0]}\n"
                f"xllcorner {x0}\nyllcorner {y0}\ncellsize {cellsize}\nNODATA_value -9999\n"
            )
            np.savetxt(stream, elevation[::-1], fmt="%.8g")
        input_path.write_text(
            "\n".join(
                [
                    'run_name = "plume_backend"',
                    "write_lobes_csv = true",
                    f'source = "{dem_path}"',
                    "vent_flag = 0",
                    f"x_vent = [{float(host_field.x_coords[start_cell[1]])}]",
                    f"y_vent = [{float(host_field.y_coords[start_cell[0]])}]",
                    "east_to_vent = 1000.0",
                    "west_to_vent = 1000.0",
                    "south_to_vent = 1000.0",
                    "north_to_vent = 1000.0",
                    "hazard_flag = 1",
                    "masking_threshold = 0.97",
                    "n_flows = 1",
                    "min_n_lobes = 64",
                    "max_n_lobes = 64",
                    "volume_flag = 1",
                    "total_volume = 100000.0",
                    "fixed_dimension_flag = 1",
                    "lobe_area = 100.0",
                    "thickness_ratio = 1.0",
                    "topo_mod_flag = 0",
                    "thickening_parameter = 0.2",
                    "lobe_exponent = 0.1",
                    "max_slope_prob = 0.995",
                    "inertial_exponent = 0.125",
                    "rng_seed = " + str(seed if seed is not None else 0),
                    "[Advanced]",
                    "restart_files = []",
                    "n_init = 1",
                    "n_check_loop = 0",
                    "start_from_dist_flag = 0",
                    "dist_fact = 1.0",
                    "npoints = 20",
                    "aspect_ratio_coeff = 2.0",
                    "max_aspect_ratio = 2.5",
                    'shape_name = ""',
                    "saveraster_flag = 1",
                    "saveshape_flag = 0",
                    "plot_lobes_flag = 0",
                    "plot_flow_flag = 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            self.executable,
            str(input_path),
            "-a",
            str(dem_path),
            "-n",
            "plume_backend",
            "-o",
            str(output_dir),
        ]
        if Path(self.executable).suffix == ".py":
            command = [sys.executable, *command]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Selected flowy backend executable is unavailable: {self.executable}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"flowy backend timed out after {self.timeout_s}s") from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"flowy backend failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        version = self.version
        try:
            version_result = subprocess.run(
                [self.executable, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(self.timeout_s, 10.0),
            )
            version_text = (version_result.stdout or version_result.stderr).strip()
            tokens = [
                token.strip("vV")
                for token in version_text.split()
                if any(char.isdigit() for char in token)
            ]
            if tokens:
                version = tokens[-1]
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            thickness_files = sorted(output_dir.glob("*_thickness_full.asc"))
            if not thickness_files:
                raise ValueError(f"flowy output missing *_thickness_full.asc in {output_dir}")
            thickness = np.loadtxt(thickness_files[0], skiprows=6)
            if thickness.ndim != 2 or not np.isfinite(thickness).any():
                raise ValueError("flowy thickness raster is empty or malformed")
            csv_files = sorted(output_dir.glob("lobes_*.csv"))
            paths = (
                self._paths_from_lobes_csv(csv_files[0])
                if csv_files
                else self._paths_from_thickness(thickness, host_field)
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"flowy backend output parsing failed in {output_dir}") from exc
        if not paths:
            raise ValueError("flowy output contained no usable paths")
        return EmplacementProposal(
            paths=tuple(paths),
            backend=self.name,
            version=version,
            provenance={
                "executable": self.executable,
                "output_schema": "*_thickness_full.asc + lobes_*.csv",
                "stdout": completed.stdout.strip()[:500],
            },
        )

    @staticmethod
    def _paths_from_lobes_csv(path: Path) -> list[tuple[tuple[float, float], ...]]:
        import csv

        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {"centerx", "centery", "idx_parent"} <= set(rows[0]):
            return []
        children: defaultdict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            parent = int(float(row.get("idx_parent", -1)))
            if parent >= 0:
                children[parent].append(index)
        roots = [
            index for index, row in enumerate(rows) if int(float(row.get("idx_parent", -1))) < 0
        ]
        if not roots:
            return []
        root = max(roots, key=lambda index: float(rows[index].get("n_descendents", 0.0)))
        chain = [root]
        while children.get(chain[-1]):
            chain.append(
                max(
                    children[chain[-1]],
                    key=lambda index: float(rows[index].get("n_descendents", 0.0)),
                )
            )
        points = [(float(rows[index]["centerx"]), float(rows[index]["centery"])) for index in chain]
        return [tuple(points)] if len(points) >= 2 else []

    @staticmethod
    def _paths_from_thickness(
        thickness: np.ndarray,
        host_field: HostField,
    ) -> list[tuple[tuple[float, float], ...]]:
        indices = np.argwhere(thickness > max(float(np.nanmax(thickness)) * 0.35, 1e-12))
        if len(indices) < 2:
            return []
        points = [
            (
                float(host_field.x_coords[min(index[1], len(host_field.x_coords) - 1)]),
                float(host_field.y_coords[max(0, len(host_field.y_coords) - 1 - index[0])]),
            )
            for index in indices[:: max(1, len(indices) // 256)]
        ]
        return [tuple(points)] if len(points) >= 2 else []


class CaveNetworkGenerator:
    """Generate a host-driven lava-tube network and raster occupancy."""

    FAMILY_LABELS = ("small", "medium", "large", "spur")

    def __init__(self, config: CaveNetworkConfig | None = None) -> None:
        self.config = config or CaveNetworkConfig()

    def generate(self, host_field: HostField) -> CaveNetwork:
        geometry = self._build_flow_geometry(host_field)
        # Match Stage A's baseline semantics: an unspecified seed produces a
        # stable canonical network, while configured seeds select variations.
        rng = procedural_rng(self.config.random_seed, "network-grammar")
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
        # Density zero is a deliberately coherent low-complexity endmember:
        # retain only the arterial route (rather than silently retaining the
        # configured multi-vent feeder fan).
        if self.config.network_density <= 0.0:
            source_cells = (backbone_source,)
        backend_provenance: dict[str, SegmentMetadataValue] = {
            "backend": "internal",
            "version": "builtin",
            "provenance": "plume_hybrid_lobe",
        }
        proposal_path_count = 1
        backbone_perturbation = None
        if self.config.growth_model == "hybrid_lobe":
            backbone_perturbation = self._correlated_terrain_perturbation(
                host_field.elevation.shape,
                amplitude_m=self.config.lobe_growth.terrain_perturbation_m,
                correlation_cells=self.config.lobe_growth.perturbation_correlation_cells,
                rng=procedural_rng(self.config.random_seed, "backbone-downflow"),
            )
        if self.config.emplacement_backend == "internal":
            backbone_path = self._trace_backbone_path(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                start_cell=backbone_source,
                downstream_potential=downstream_potential,
                terrain_perturbation=backbone_perturbation,
            )
        else:
            proposal = self._emplacement_proposal(
                host_field=host_field,
                geometry=geometry,
                start_cell=backbone_source,
            )
            proposal_path_count = len(proposal.paths)
            backbone_path = self._proposal_path_to_cells(host_field, proposal.paths[0])
            if len(backbone_path) < 3:
                raise ValueError(
                    f"Selected emplacement backend {proposal.backend!r} returned an unusable backbone path"
                )
            downstream_progress = float(
                geometry.along_grid[backbone_path[-1]] - geometry.along_grid[backbone_path[0]]
            )
            # A proposal is a backbone prior, not merely a local vent lobe.
            # Require it to explain a material part of the configured domain
            # before PLUME is allowed to extend branches from it.
            minimum_progress = max(2.0 * geometry.cell_scale, 0.25 * geometry.along_extent)
            if downstream_progress < minimum_progress:
                raise ValueError(
                    f"Selected emplacement backend {proposal.backend!r} produced only "
                    f"{downstream_progress:.3f} m of downstream progress; at least "
                    f"{minimum_progress:.3f} m is required"
                )
            backend_provenance = {
                "backend": proposal.backend,
                "version": proposal.version,
                **proposal.provenance,
            }
        if not backbone_path:
            return CaveNetwork(
                config=self.config,
                nodes=(),
                segments=(),
                junctions=(),
                occupancy=np.zeros_like(host_field.growth_cost, dtype=bool),
                width_field=np.zeros_like(host_field.growth_cost, dtype=float),
                dominant_route_node_ids=(),
                slice_along_positions=(),
                slice_channel_counts=(),
                slice_visible_channel_counts=(),
            )

        backend_provenance.update(
            {
                "proposal_path_count": proposal_path_count,
                "proposal_cell_count": len(backbone_path),
                "proposal_downstream_progress_m": float(
                    geometry.along_grid[backbone_path[-1]] - geometry.along_grid[backbone_path[0]]
                ),
                "secondary_growth_model": (
                    "flux_breakout_avulsion"
                    if self.config.growth_model == "hybrid_lobe"
                    else "legacy_braid"
                ),
            }
        )

        emplacement_phase_count = self._sample_int_range(
            procedural_rng(self.config.random_seed, "emplacement-phase-count"),
            self.config.emplacement_history.phase_count,
        )
        selected_paths: list[_SelectedPath] = [
            _SelectedPath(
                kind="backbone",
                path=tuple(self._simplify_path(backbone_path)),
                metadata=self._build_emplacement_metadata(
                    kind="backbone",
                    phase_count=emplacement_phase_count,
                    birth_phase=0,
                    death_phase=emplacement_phase_count - 1,
                    formation_state="persistent_arterial",
                ),
            )
        ]
        selected_paths[0].metadata.update(backend_provenance)
        occupied_cells = set(backbone_path)
        backbone_alongs, backbone_crosses = self._build_backbone_profile(backbone_path, geometry)
        for source_cell in source_cells:
            if source_cell == backbone_source:
                continue
            source_along = float(geometry.along_grid[source_cell])
            join_along = min(
                geometry.along_extent,
                max(
                    source_along + self.config.source_band_length,
                    0.08 * geometry.along_extent,
                ),
            )
            join_cell = self._cell_on_path_at_along(
                backbone_path,
                geometry,
                join_along,
            )
            feeder = self._simplify_path(
                self._build_connector_path(
                    host_field=host_field,
                    geometry=geometry,
                    support_field=support_field,
                    start_cell=source_cell,
                    end_cell=join_cell,
                    backbone_alongs=backbone_alongs,
                    backbone_crosses=backbone_crosses,
                )
            )
            if len(feeder) < 2:
                continue
            selected_paths.append(
                _SelectedPath(
                    kind="source_feeder",
                    path=tuple(feeder),
                    metadata=self._build_emplacement_metadata(
                        kind="source_feeder",
                        phase_count=emplacement_phase_count,
                        birth_phase=0,
                        death_phase=emplacement_phase_count - 1,
                        formation_state="persistent_feeder",
                    ),
                )
            )
            occupied_cells.update(feeder[:-1])
        if self.config.growth_model == "hybrid_lobe":
            selected_paths.extend(
                self._build_lobe_growth_paths(
                    host_field=host_field,
                    geometry=geometry,
                    support_field=support_field,
                    downstream_potential=downstream_potential,
                    backbone_path=backbone_path,
                    backbone_alongs=backbone_alongs,
                    backbone_crosses=backbone_crosses,
                    initial_paths=tuple(selected_paths),
                    phase_count=emplacement_phase_count,
                )
            )
        else:
            braid_zones = self._build_braid_zones(host_field, geometry, rng)
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

            _, connectable_flux, _ = self._build_representative_fields(
                shape=host_field.growth_cost.shape,
                selected_paths=tuple(
                    selected_path
                    for selected_path in selected_paths
                    if selected_path.merge_shared_cells
                ),
            )
            spur_starts = self._select_spur_start_cells(connectable_flux, geometry)
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
                            merge_shared_cells=False,
                            metadata=self._build_segment_metadata(kind="spur"),
                        )
                    )

        selected_paths = self._annotate_drained_pool_chambers(
            host_field=host_field,
            geometry=geometry,
            selected_paths=tuple(selected_paths),
            rng=procedural_rng(self.config.random_seed, "drained-lava-pools"),
        )
        skeleton_mask, selected_flux, _ = self._build_representative_fields(
            shape=host_field.growth_cost.shape,
            selected_paths=tuple(selected_paths),
        )

        nodes, segments, _dominant_route_node_ids = self._extract_graph_from_paths(
            host_field=host_field,
            geometry=geometry,
            selected_paths=tuple(selected_paths),
            total_flux=selected_flux,
        )
        segments = self._smooth_graph_routes(host_field, segments)
        segments = self._orient_segments_for_flow(nodes, segments)
        segments = self._repair_source_reachability(nodes, segments)
        segments = self._assign_conserved_flow(nodes, segments)
        segments = self._annotate_emplacement_flux_history(segments)
        dominant_route_node_ids = self._dominant_route(nodes, segments)
        self._validate_generated_graph(nodes, segments, dominant_route_node_ids)
        junctions = self._build_junctions(nodes, segments)

        occupancy = np.zeros_like(host_field.growth_cost, dtype=bool)
        width_field = np.zeros_like(host_field.growth_cost, dtype=float)
        for segment in segments:
            self._rasterize_segment(host_field, occupancy, width_field, segment)
        self._paint_structural_chambers(host_field, occupancy, width_field, nodes, segments)
        if self.config.paint_flux_chambers:
            self._paint_chambers(host_field, occupancy, width_field, total_flux)
        occupancy = self._smooth_occupancy(occupancy)
        # Never let coarse host-grid morphology filtering erase a valid
        # centerline. Geometry is generated from metric section profiles, but
        # Stage-B occupancy must still remain a faithful topology diagnostic.
        occupancy |= skeleton_mask

        slice_along_positions, slice_channel_counts = self._measure_parallel_channels(
            geometry=geometry,
            segments=segments,
            include_passage_width=False,
        )
        _, slice_visible_channel_counts = self._measure_parallel_channels(
            geometry=geometry,
            segments=segments,
            include_passage_width=True,
        )

        return CaveNetwork(
            config=self.config,
            nodes=tuple(nodes),
            segments=tuple(segments),
            junctions=tuple(junctions),
            occupancy=occupancy,
            width_field=width_field,
            dominant_route_node_ids=dominant_route_node_ids,
            slice_along_positions=slice_along_positions,
            slice_channel_counts=slice_channel_counts,
            slice_visible_channel_counts=slice_visible_channel_counts,
            backend_provenance=backend_provenance,
        )

    def _emplacement_proposal(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        start_cell: tuple[int, int],
    ) -> EmplacementProposal:
        backend = self.config.emplacement_backend
        if backend == "downflow_reference":
            return DownflowReferenceBackend().propose(
                host_field,
                geometry,
                start_cell=start_cell,
                seed=self.config.random_seed,
                steps=self.config.trace_max_steps + 240,
                uphill_limit=self.config.max_uphill_step,
                ensemble_size=self.config.downflow_ensemble_size,
            )
        if backend == "flowy":
            if not self.config.flowy_executable:
                raise ValueError("emplacement_backend='flowy' requires network.flowy_executable")
            return FlowyBackend(
                self.config.flowy_executable,
                timeout_s=self.config.flowy_timeout_s,
                output_path=self.config.flowy_output_path,
            ).propose(
                host_field,
                geometry,
                start_cell=start_cell,
                seed=self.config.random_seed,
                steps=self.config.trace_max_steps,
                uphill_limit=self.config.max_uphill_step,
            )
        raise ValueError(f"Unknown emplacement backend: {backend!r}")

    def _proposal_path_to_cells(
        self,
        host_field: HostField,
        path: tuple[tuple[float, float], ...],
    ) -> list[tuple[int, int]]:
        """Rasterize proposal points while erasing chronological loops.

        External emplacement models may revisit an older lobe or raster cell.
        A repeated cell cannot be admitted into PLUME's directed backbone: it
        would create a semantic flow cycle during flux propagation.  Loop
        erasure preserves the prefix leading into the revisit and discards the
        intervening closed walk deterministically.
        """

        cells: list[tuple[int, int]] = []
        positions: dict[tuple[int, int], int] = {}
        for x_coord, y_coord in path:
            cell = self._world_to_cell(host_field, x_coord, y_coord)
            if cells and cell == cells[-1]:
                continue
            previous = positions.get(cell)
            if previous is not None:
                for removed in cells[previous + 1 :]:
                    positions.pop(removed, None)
                del cells[previous + 1 :]
                continue
            positions[cell] = len(cells)
            cells.append(cell)
        return cells

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
        available_extent = float(np.max(along_grid)) - self.config.sink_margin
        along_extent = min(available_extent, self.config.target_route_length_m)
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
        corridor_score = np.exp(
            -np.square(geometry.cross_grid / max(host_field.config.corridor_width, 1.0))
        )
        support = (
            self.config.growth_cost_weight * (1.0 - host_field.routing_cost)
            + self.config.corridor_weight * corridor_score
        )
        return support

    def _annotate_drained_pool_chambers(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        selected_paths: tuple[_SelectedPath, ...],
        rng: np.random.Generator,
    ) -> tuple[_SelectedPath, ...]:
        """Select sparse, process-caused drained lava pools on existing routes."""
        history = self.config.emplacement_history
        if not history.drained_pool_enabled or not selected_paths:
            return selected_paths
        if float(rng.random()) > history.drained_pool_probability:
            return selected_paths
        _mask, route_flux, _families = self._build_representative_fields(
            shape=host_field.elevation.shape,
            selected_paths=selected_paths,
        )
        positive_flux = route_flux[route_flux > 0.0]
        flux_reference = (
            float(np.quantile(positive_flux, history.drained_pool_flux_quantile))
            if positive_flux.size
            else 0.0
        )
        candidates: list[tuple[float, int, int, tuple[int, int], str, float, float]] = []
        for path_index, selected_path in enumerate(selected_paths):
            if (
                selected_path.kind in {"source_feeder", "spur", "underpass"}
                or len(selected_path.path) < 7
            ):
                continue
            cells = selected_path.path
            coalesced = (
                bool((selected_path.metadata or {}).get("coalesced", False))
                or selected_path.kind == "anastomosis"
            )
            stride = max(1, (len(cells) - 6) // 24)
            for center_index in range(3, len(cells) - 3, stride):
                center = cells[center_index]
                before = cells[center_index - 3]
                after = cells[center_index + 3]
                distance = max(
                    math.hypot(
                        float(host_field.x_coords[after[1]] - host_field.x_coords[before[1]]),
                        float(host_field.y_coords[after[0]] - host_field.y_coords[before[0]]),
                    ),
                    geometry.cell_scale,
                )
                grade = (
                    abs(float(host_field.elevation[after] - host_field.elevation[before]))
                    / distance
                )
                slope_before = float(host_field.slope_degrees[before])
                slope_after = float(host_field.slope_degrees[after])
                slope_break = abs(slope_after - slope_before) / 18.0
                local_flux = float(route_flux[center])
                flux_score = local_flux / max(flux_reference, 1e-9)
                capacity_loss = 1.0 - float(host_field.flow_capacity[center])
                low_grade_score = float(np.exp(-grade / 0.08))
                progress = center_index / max(len(cells) - 1, 1)
                coalescence_approach = progress if coalesced and progress >= 0.55 else 0.0
                if not (coalescence_approach > 0.0 or slope_break >= 0.08 or capacity_loss >= 0.35):
                    continue
                cause = (
                    "coalescence"
                    if coalescence_approach >= 0.72
                    else "slope_break"
                    if slope_break >= 0.08
                    else "capacity_loss"
                )
                score = (
                    history.drained_pool_flux_weight * flux_score
                    + history.drained_pool_grade_weight * low_grade_score
                    + history.drained_pool_slope_break_weight * slope_break
                    + 0.7 * capacity_loss
                    + 0.8 * coalescence_approach
                )
                candidates.append(
                    (
                        score,
                        path_index,
                        center_index,
                        center,
                        cause,
                        local_flux,
                        low_grade_score,
                    )
                )
        if not candidates:
            return selected_paths
        candidates.sort(key=lambda item: (-item[0], item[1]))
        target_count = int(
            rng.integers(history.drained_pool_count[0], history.drained_pool_count[1] + 1)
        )
        chosen: list[tuple[float, int, int, tuple[int, int], str, float, float]] = []
        chosen_alongs: list[float] = []
        chosen_paths: set[int] = set()
        for candidate in candidates:
            if len(chosen) >= target_count:
                break
            if candidate[1] in chosen_paths:
                continue
            along = float(geometry.along_grid[candidate[3]])
            if any(
                abs(along - previous) < history.drained_pool_min_spacing_m
                for previous in chosen_alongs
            ):
                continue
            chosen.append(candidate)
            chosen_alongs.append(along)
            chosen_paths.add(candidate[1])
        if not chosen:
            return selected_paths
        chosen_by_path = {item[1]: item for item in chosen}
        annotated: list[_SelectedPath] = []
        for path_index, selected_path in enumerate(selected_paths):
            candidate = chosen_by_path.get(path_index)
            if candidate is None:
                annotated.append(selected_path)
                continue
            (
                _score,
                _index,
                center_index,
                center,
                cause,
                local_flux,
                low_grade_score,
            ) = candidate
            # Use the configured local passage envelope as the reference so
            # pool widening remains an auditable 2--5x morphological change,
            # independent of raster flux units.
            center_x, center_y = self._cell_to_world(host_field, center)
            host_sample = host_field.sample(center_x, center_y)
            base_width = 2.0 * self._local_radius(
                host_field,
                host_sample,
                float(route_flux[center]),
            )
            length_m = float(rng.uniform(*history.drained_pool_length_m))
            requested_width_ratio = float(rng.uniform(*history.drained_pool_width_ratio))
            pool_width = min(
                base_width * requested_width_ratio,
                history.drained_pool_max_width_m,
            )
            width_ratio = pool_width / max(base_width, 1e-9)
            depth_m = float(rng.uniform(*history.drained_pool_depth_m))
            metadata = dict(selected_path.metadata or {})
            metadata["pool_parent_chamber_id"] = metadata.get("chamber_id")
            metadata.update(
                {
                    "chamber_type": "drained_lava_pool",
                    "pool_id": f"drained_pool_{path_index}",
                    "chamber_id": f"drained_pool_{path_index}",
                    "process_cause": cause,
                    "pool_length_m": length_m,
                    "pool_width_m": pool_width,
                    "pool_depth_m": depth_m,
                    "pool_aspect_ratio": length_m / max(pool_width, 1e-9),
                    "pool_outlet_ratio": width_ratio,
                    "pool_inlet_count": 2 if cause == "coalescence" else 1,
                    "pool_outlet_count": 1,
                    "pool_outlet_width_m": base_width,
                    "pool_site_score": float(_score),
                    "pool_flux": local_flux,
                    "pool_low_grade_score": low_grade_score,
                    "pool_center_along_m": float(geometry.along_grid[center]),
                    "pool_center_x_m": center_x,
                    "pool_center_y_m": center_y,
                    "pool_center_row": int(center[0]),
                    "pool_center_column": int(center[1]),
                    "pool_center_path_index": center_index,
                }
            )
            annotated.append(replace(selected_path, metadata=metadata))
        return tuple(annotated)

    @staticmethod
    def _drained_pool_center_cell(
        selected_path: _SelectedPath,
    ) -> tuple[int, int] | None:
        """Return the validated room centre recorded during process selection."""
        if not selected_path.path:
            return None
        metadata = selected_path.metadata or {}
        if metadata.get("chamber_type") != "drained_lava_pool":
            return None
        row = metadata.get("pool_center_row")
        column = metadata.get("pool_center_column")
        if isinstance(row, (int, float)) and isinstance(column, (int, float)):
            recorded_cell = (int(row), int(column))
            if recorded_cell in selected_path.path:
                return recorded_cell
        raw_index = metadata.get("pool_center_path_index", len(selected_path.path) // 2)
        pool_index = (
            int(raw_index) if isinstance(raw_index, (int, float)) else len(selected_path.path) // 2
        )
        return selected_path.path[int(np.clip(pool_index, 0, len(selected_path.path) - 1))]

    @classmethod
    def _metadata_for_path_section(
        cls,
        selected_path: _SelectedPath,
        path_cells: list[tuple[int, int]],
    ) -> dict[str, SegmentMetadataValue]:
        """Keep room semantics only on segments incident to the room centre."""
        metadata = dict(
            selected_path.metadata
            or cls._build_segment_metadata(
                kind=selected_path.kind,
                z_level=selected_path.z_level,
            )
        )
        pool_center = cls._drained_pool_center_cell(selected_path)
        if pool_center is None or pool_center in {path_cells[0], path_cells[-1]}:
            return metadata
        parent_chamber_id = metadata.get("pool_parent_chamber_id")
        for key in tuple(metadata):
            if key.startswith("pool_") or key in {
                "chamber_type",
                "chamber_id",
                "process_cause",
            }:
                metadata.pop(key, None)
        metadata["chamber_id"] = parent_chamber_id
        return metadata

    def _select_source_cells(
        self,
        host_field: HostField,
        geometry: _FlowGeometry,
    ) -> tuple[tuple[int, int], ...]:
        support = self._build_support_field(host_field, geometry)
        source_band = (geometry.along_grid >= 0.0) & (
            geometry.along_grid <= self.config.source_band_length
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

    def _build_lobe_growth_paths(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        downstream_potential: np.ndarray,
        backbone_path: list[tuple[int, int]],
        backbone_alongs: np.ndarray,
        backbone_crosses: np.ndarray,
        initial_paths: tuple[_SelectedPath, ...],
        phase_count: int,
    ) -> tuple[_SelectedPath, ...]:
        """Grow persistent distributaries from seeded, terrain-led lava lobes.

        This is a compact hybrid of DOWNFLOW-style perturbed topography and
        MrLavaLoba-style active fronts. Each trace retains momentum, initially
        avoids the supplying channel, and later prefers an energetically cheap
        downstream coalescence. Branches can instead cool and retire as blind
        lobes. The graph is extracted from the resulting path history.
        """

        controls = self.config.lobe_growth
        rng = procedural_rng(self.config.random_seed, "lobe-growth")
        path_count = int(
            round(
                self._sample_int_range(rng, controls.path_count)
                * self.config.network_density
                * self.config.lobe_launch_rate
            )
        )
        breakout_sites = self._select_breakout_sites(
            host_field=host_field,
            geometry=geometry,
            support_field=support_field,
            backbone_path=backbone_path,
            count=path_count,
            rng=rng,
        )
        if not breakout_sites:
            return ()

        existing_cells = set(backbone_path)
        for selected_path in initial_paths:
            existing_cells.update(selected_path.path)

        retired_count = int(round(len(breakout_sites) * controls.retired_path_fraction))
        retired_indices = set(
            int(index)
            for index in rng.choice(
                len(breakout_sites),
                size=min(retired_count, len(breakout_sites)),
                replace=False,
            )
        )
        stacked_candidates = tuple(range(max(1, int(math.ceil(0.65 * len(breakout_sites))))))
        stacked_count = min(
            len(stacked_candidates),
            int(round(len(breakout_sites) * self.config.emplacement_history.stacked_lobe_fraction)),
        )
        stacked_indices = set(
            int(index)
            for index in rng.choice(
                stacked_candidates,
                size=stacked_count,
                replace=False,
            )
        )
        side_counts = {-1: 0, 1: 0}
        # Each emplacement pulse begins with a finite arterial discharge. A
        # breakout removes flux from the downstream parent and a successful
        # coalescence returns only the configured surviving fraction.
        phase_flux = np.full(
            (max(phase_count, 1), len(backbone_path)),
            self.config.source_flux,
            dtype=float,
        )
        phase_budget = max(
            float(self.config.source_flux)
            * float(self.config.emplacement_history.phase_flux_budget_fraction),
            0.0,
        )
        phase_allocated = np.zeros(max(phase_count, 1), dtype=float)
        # Reoccupation targets are prior secondary lobe centerlines, not the
        # arterial backbone or source feeders (which remain valid merge
        # destinations but are not counted as passage reuse).
        reusable_cells = {
            cell
            for selected_path in initial_paths
            if selected_path.kind not in {"backbone", "source_feeder"}
            for cell in selected_path.path
        }
        reusable_path_ids = {
            cell: str((selected_path.metadata or {}).get("lobe_path_id"))
            for selected_path in initial_paths
            if selected_path.kind not in {"backbone", "source_feeder"}
            for cell in selected_path.path
            if (selected_path.metadata or {}).get("lobe_path_id") is not None
        }
        reusable_path_orders = {
            cell: int((selected_path.metadata or {}).get("branch_order", 0))
            for selected_path in initial_paths
            if selected_path.kind not in {"backbone", "source_feeder"}
            for cell in selected_path.path
        }
        established_cells = set(existing_cells)
        emplacement_surface = np.array(host_field.elevation, dtype=float, copy=True)
        result: list[_SelectedPath] = []
        # Assign phases deterministically first, then process the event queue
        # in chronological order.  This makes relief and passage reuse from an
        # earlier pulse available to every later pulse, independent of the
        # spatial ranking of breakout sites.
        phase_assignments: dict[int, int] = {}
        for branch_index in range(len(breakout_sites)):
            phase_rng = procedural_rng(
                self.config.random_seed,
                "lobe-emplacement-history",
                branch_index,
            )
            if len(breakout_sites) <= 1:
                phase_assignments[branch_index] = 0
            else:
                phase_position = branch_index / (len(breakout_sites) - 1)
                phase_value = int(round(phase_position * (phase_count - 1)))
                phase_value += int(phase_rng.choice((-1, 0, 1), p=(0.16, 0.68, 0.16)))
                phase_assignments[branch_index] = int(np.clip(phase_value, 0, phase_count - 1))
        ordered_indices = sorted(
            range(len(breakout_sites)),
            key=lambda index: (phase_assignments[index], index),
        )
        for branch_index in ordered_indices:
            breakout = breakout_sites[branch_index]
            anchor = breakout.cell
            branch_rng = procedural_rng(
                self.config.random_seed,
                "lobe-front",
                branch_index,
                anchor[0],
                anchor[1],
            )
            if side_counts[-1] == side_counts[1]:
                lateral_sign = -1 if branch_rng.random() < 0.5 else 1
            else:
                lateral_sign = min(side_counts, key=lambda sign: side_counts[sign])
            side_counts[lateral_sign] += 1
            phase_rng = procedural_rng(
                self.config.random_seed,
                "lobe-emplacement-history",
                branch_index,
            )
            birth_phase = phase_assignments[branch_index]
            active_span = self._sample_int_range(
                phase_rng,
                self.config.emplacement_history.active_phase_span,
            )
            death_phase = min(phase_count - 1, birth_phase + active_span - 1)
            z_level = self._emplacement_z_level(
                birth_phase=birth_phase,
                phase_count=phase_count,
                stacked=branch_index in stacked_indices,
                rng=phase_rng,
            )
            allow_capture = True
            if z_level != 0 and self.config.capture_probability < 1.0:
                allow_capture = float(phase_rng.random()) <= self.config.capture_probability
            allow_loop = True
            if self.config.loop_probability < 1.0:
                allow_loop = float(branch_rng.random()) <= self.config.loop_probability
            permit_merge = (
                branch_index not in retired_indices
                and allow_loop
                and (z_level == 0 or allow_capture)
            )
            anchor_index = int(np.clip(breakout.backbone_index, 0, len(backbone_path) - 1))
            parent_flux_before = float(phase_flux[birth_phase, anchor_index])
            requested_fraction = self._sample_float_range(
                branch_rng,
                controls.branch_flux_fraction,
            )
            maximum_breakout_score = max(
                controls.breakout_capacity_weight
                + controls.breakout_confinement_weight
                + controls.breakout_curvature_weight
                + controls.breakout_blockage_weight,
                1e-9,
            )
            severity = float(np.clip(breakout.score / maximum_breakout_score, 0.0, 1.0))
            split_fraction = float(
                np.clip(requested_fraction * (0.72 + 0.38 * severity), 0.0, 0.82)
            )
            requested_flux = parent_flux_before * split_fraction
            phase_remaining = max(phase_budget - phase_allocated[birth_phase], 0.0)
            initial_flux = min(requested_flux, phase_remaining)
            minimum_viable_flux = self.config.source_flux * max(
                controls.minimum_viable_flux_fraction,
                self.config.emplacement_history.retirement_flux_threshold,
            )
            if initial_flux <= 1e-9:
                continue
            viable_flux = initial_flux >= minimum_viable_flux
            # Reoccupation is only available to a later phase when a nearby
            # established passage lies in a compatible downstream corridor.
            # The front still starts at the physical breakout anchor; no path
            # teleportation is introduced.
            anchor_along = float(geometry.along_grid[anchor])
            nearby_established = [
                cell
                for cell in reusable_cells
                if 0.45 * self.config.lobe_growth.minimum_persistence_steps * geometry.cell_scale
                <= float(geometry.along_grid[cell]) - anchor_along
                <= 10.0 * geometry.cell_scale
                and math.hypot(cell[0] - anchor[0], cell[1] - anchor[1]) <= 12.0
            ]
            anchor_world = self._cell_to_world(host_field, anchor)
            downhill_x, downhill_y = host_field.downhill_direction(
                anchor_world[0],
                anchor_world[1],
                fallback_angle_degrees=math.degrees(math.atan2(geometry.flow_y, geometry.flow_x)),
            )
            downstream_alignment = downhill_x * geometry.flow_x + downhill_y * geometry.flow_y
            reoccupied = bool(
                birth_phase > 0
                and nearby_established
                and downstream_alignment >= -0.15
                and phase_rng.random() < self.config.emplacement_history.reoccupation_probability
            )
            reoccupation_target = (
                min(
                    nearby_established,
                    key=lambda cell: math.hypot(cell[0] - anchor[0], cell[1] - anchor[1]),
                )
                if reoccupied
                else None
            )
            reoccupation_target_lobe_id = (
                reusable_path_ids.get(reoccupation_target)
                if reoccupation_target is not None
                else None
            )
            branch_order = (
                reusable_path_orders.get(reoccupation_target, 0) + 1
                if reoccupation_target is not None
                else 1
            )
            reuse_prefix = (
                self._build_connector_path(
                    host_field=host_field,
                    geometry=geometry,
                    support_field=support_field,
                    start_cell=anchor,
                    end_cell=reoccupation_target,
                    backbone_alongs=backbone_alongs,
                    backbone_crosses=backbone_crosses,
                )
                if reoccupation_target is not None
                else []
            )
            trace_start_cell = reoccupation_target or anchor
            permit_merge = viable_flux and (
                permit_merge or (reoccupied and (z_level == 0 or allow_capture))
            )
            initial_direction = self._breakout_initial_direction(
                host_field=host_field,
                geometry=geometry,
                backbone_path=backbone_path,
                backbone_index=anchor_index,
                lateral_sign=float(lateral_sign),
            )
            trace = self._trace_lobe_front(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                downstream_potential=downstream_potential,
                backbone_alongs=backbone_alongs,
                backbone_crosses=backbone_crosses,
                start_cell=trace_start_cell,
                existing_cells=existing_cells,
                lateral_sign=float(lateral_sign),
                permit_merge=permit_merge,
                merge_target_cells=(
                    {reoccupation_target} if reoccupation_target is not None else None
                ),
                initial_flux=initial_flux,
                initial_direction=initial_direction,
                emplacement_surface=emplacement_surface,
                rng=branch_rng,
            )
            if len(trace.path) < 3:
                lateral_sign *= -1
                retry_rng = procedural_rng(
                    self.config.random_seed,
                    "lobe-front-retry",
                    branch_index,
                    anchor[0],
                    anchor[1],
                )
                trace = self._trace_lobe_front(
                    host_field=host_field,
                    geometry=geometry,
                    support_field=support_field,
                    downstream_potential=downstream_potential,
                    backbone_alongs=backbone_alongs,
                    backbone_crosses=backbone_crosses,
                    start_cell=trace_start_cell,
                    existing_cells=existing_cells,
                    lateral_sign=float(lateral_sign),
                    permit_merge=permit_merge,
                    merge_target_cells=(
                        {reoccupation_target} if reoccupation_target is not None else None
                    ),
                    initial_flux=initial_flux,
                    initial_direction=self._breakout_initial_direction(
                        host_field=host_field,
                        geometry=geometry,
                        backbone_path=backbone_path,
                        backbone_index=anchor_index,
                        lateral_sign=float(lateral_sign),
                    ),
                    emplacement_surface=emplacement_surface,
                    rng=retry_rng,
                )
            if reuse_prefix:
                trace = replace(
                    trace,
                    path=tuple(reuse_prefix[:-1]) + trace.path,
                )
            if not trace.merged:
                trace = self._trim_stranded_lobe_to_standoff(
                    trace=trace,
                    existing_cells=existing_cells,
                    shape=host_field.elevation.shape,
                    cell_scale=geometry.cell_scale,
                )
            simplified = self._simplify_path(list(trace.path))
            if len(simplified) < 3:
                continue

            phase_allocated[birth_phase] += trace.initial_flux
            parent_flux_after = max(parent_flux_before - trace.initial_flux, 0.0)
            phase_flux[birth_phase, anchor_index:] = np.maximum(
                phase_flux[birth_phase, anchor_index:] - trace.initial_flux,
                0.0,
            )
            returned_flux = 0.0
            if trace.merged:
                returned_flux = trace.initial_flux * controls.coalescence_flux_return_fraction
                merge_along = float(geometry.along_grid[trace.path[-1]])
                merge_index = int(
                    np.argmin(
                        np.abs(
                            np.asarray(
                                [geometry.along_grid[cell] for cell in backbone_path],
                                dtype=float,
                            )
                            - merge_along
                        )
                    )
                )
                phase_flux[birth_phase, merge_index:] += returned_flux
            deposited_relief = self._deposit_lobe_feedback(
                emplacement_surface=emplacement_surface,
                path=trace.path,
                flux_fraction=trace.initial_flux / max(self.config.source_flux, 1e-9),
            )

            pirated = bool(
                reoccupied
                and not trace.merged
                and phase_rng.random() < self.config.emplacement_history.breakout_probability
            )
            if not viable_flux:
                kind = "abandoned_lobe"
            elif trace.merged:
                kind = "anastomosis"
            elif trace.final_temperature_k <= controls.retirement_temperature_k:
                kind = "abandoned_lobe"
            else:
                kind = "stalled_lobe"
            event_type = (
                "retired"
                if not viable_flux or kind == "abandoned_lobe"
                else "pirated"
                if pirated
                else "reoccupation"
                if reoccupied
                else "coalesced"
                if trace.merged
                else "stalled"
                if kind == "stalled_lobe"
                else "new_breakout"
            )
            vertical_capture = trace.merged and z_level != 0
            loop_mechanism = None
            if trace.merged:
                if vertical_capture:
                    loop_mechanism = "vertical_capture"
                elif reoccupied:
                    loop_mechanism = "passage_reoccupation"
                elif breakout.trigger == "seeded_blockage":
                    loop_mechanism = "obstacle_bypass"
                elif breakout.trigger == "capacity_overflow":
                    loop_mechanism = "overflow_anastomosis"
                elif breakout.trigger == "bend_overflow":
                    loop_mechanism = "bend_bypass"
                else:
                    loop_mechanism = "lateral_avulsion"
            chamber_probability = (
                self.config.emplacement_history.vertical_capture_chamber_probability
                if vertical_capture
                else self.config.emplacement_history.chamber_formation_probability
            )
            normalized_flux = float(
                np.clip(
                    trace.initial_flux / max(self.config.source_flux, 1e-9),
                    0.0,
                    1.0,
                )
            )
            chamber_probability *= (0.65 + 0.55 * normalized_flux) * max(
                self.config.chamber_gain, 0.0
            )
            chamber_forming = bool(
                trace.merged and phase_rng.random() < min(chamber_probability, 1.0)
            )
            formation_state = (
                "flux_starved_retired"
                if not viable_flux
                else "vertically_captured"
                if vertical_capture
                else "coalesced"
                if trace.merged
                else "pirated_breakout"
                if pirated
                else "reoccupied_passage"
                if reoccupied
                else "thermally_abandoned"
                if trace.final_temperature_k <= controls.retirement_temperature_k
                else "stranded"
            )
            regime = self._local_emplacement_regime(
                host_field=host_field,
                geometry=geometry,
                cell=anchor,
            )
            roof_state = self._sample_roof_state(
                host_field=host_field,
                path=trace.path,
                rng=phase_rng,
            )
            metadata = self._build_emplacement_metadata(
                kind=kind,
                phase_count=phase_count,
                birth_phase=birth_phase,
                death_phase=death_phase,
                formation_state=formation_state,
                zone_index=branch_index,
                z_level=z_level,
                chamber_forming=chamber_forming,
                chamber_radius_scale=0.88 + 0.30 * normalized_flux,
            )
            metadata.update(
                {
                    "growth_model": "hybrid_lobe",
                    "lobe_path_id": f"lobe_{branch_index}",
                    "termination": "coalesced" if trace.merged else "cooled_or_stranded",
                    "initial_flux": trace.initial_flux,
                    "requested_flux": requested_flux,
                    "phase_flux_budget": phase_budget,
                    "phase_flux_allocated": phase_allocated[birth_phase],
                    "parent_flux_replenished": returned_flux,
                    "reoccupied": reoccupied,
                    "reoccupied_path": reoccupied,
                    "new_path": not reoccupied,
                    "pirated": pirated,
                    "event_type": event_type,
                    "loop_mechanism": loop_mechanism,
                    "reoccupation_mode": (
                        "piracy_breakout"
                        if pirated
                        else "reactivate"
                        if reoccupied
                        else "new_breakout"
                    ),
                    "downstream_alignment": downstream_alignment,
                    "reoccupation_target_cell": (
                        list(reoccupation_target) if reoccupation_target is not None else None
                    ),
                    "reoccupation_target_lobe_id": reoccupation_target_lobe_id,
                    "branch_order": branch_order,
                    "coalesced": bool(trace.merged),
                    "stalled": bool(not trace.merged and kind == "stalled_lobe"),
                    "retired": bool(kind == "abandoned_lobe"),
                    "branching_process": "flux_breakout_avulsion",
                    "breakout_trigger": breakout.trigger,
                    "breakout_score": breakout.score,
                    "breakout_capacity_deficit": breakout.capacity_deficit,
                    "breakout_confinement_loss": breakout.confinement_loss,
                    "breakout_curvature": breakout.curvature,
                    "breakout_blockage": breakout.blockage,
                    "parent_flux_before_split": parent_flux_before,
                    "parent_flux_after_split": parent_flux_after,
                    "branch_flux_fraction": split_fraction,
                    "coalescence_returned_flux": returned_flux,
                    "deposition_feedback_m": deposited_relief,
                    "final_temperature_k": trace.final_temperature_k,
                    "maximum_lateral_separation_m": trace.maximum_lateral_separation,
                    "vertical_capture": vertical_capture,
                    "emplacement_regime": regime,
                    "roof_state": roof_state,
                }
            )
            if vertical_capture:
                metadata["merge_behavior"] = "vertical_capture"
            selected = _SelectedPath(
                kind=kind,
                path=tuple(simplified),
                z_level=z_level,
                metadata=metadata,
            )
            result.append(selected)
            existing_cells.update(trace.path)
            established_cells.update(trace.path)
            reusable_cells.update(trace.path)
            reusable_path_ids.update({cell: f"lobe_{branch_index}" for cell in trace.path})
            reusable_path_orders.update({cell: branch_order for cell in trace.path})
        return tuple(result)

    def _trim_stranded_lobe_to_standoff(
        self,
        *,
        trace: _LobeTrace,
        existing_cells: set[tuple[int, int]],
        shape: tuple[int, int],
        cell_scale: float,
    ) -> _LobeTrace:
        """Prevent an unmerged tube from grazing an older passage shell.

        A centerline can remain in a distinct raster cell yet curve back to
        within one passage diameter after spline smoothing. Such a route is
        neither a geological coalescence nor a distinct lobe and can create a
        tangential voxel contact. Preserve its divergent portion and retire it
        before the first unresolved return.
        """

        if trace.merged or len(trace.path) < 4 or not existing_cells:
            return trace
        existing_mask = np.zeros(shape, dtype=bool)
        for cell in existing_cells:
            existing_mask[cell] = True
        distance_m = distance_transform_edt(~existing_mask) * max(cell_scale, 1e-9)
        standoff_m = max(
            2.0 * self.config.base_passage_radius * self.config.minimum_branch_offset_widths,
            1.35 * cell_scale,
        )
        departure_index = next(
            (
                index
                for index, cell in enumerate(trace.path[1:], start=1)
                if float(distance_m[cell]) >= standoff_m
            ),
            None,
        )
        if departure_index is None:
            return replace(trace, path=trace.path[:2])
        contact_index = next(
            (
                index
                for index, cell in enumerate(
                    trace.path[departure_index + 1 :],
                    start=departure_index + 1,
                )
                if float(distance_m[cell]) < standoff_m
            ),
            None,
        )
        if contact_index is None:
            return trace
        return replace(trace, path=trace.path[:contact_index])

    def _emplacement_z_level(
        self,
        *,
        birth_phase: int,
        phase_count: int,
        stacked: bool,
        rng: np.random.Generator,
    ) -> int:
        """Place earlier preserved routes above the younger arterial tube."""

        maximum_level = self.config.emplacement_history.maximum_absolute_level
        if not stacked or maximum_level <= 0:
            return 0
        midpoint = 0.5 * max(phase_count - 1, 1)
        distance_from_middle = abs(birth_phase - midpoint) / max(midpoint, 1.0)
        magnitude = 1
        if maximum_level >= 2 and distance_from_middle > 0.70 and rng.random() < 0.35:
            magnitude = min(2, maximum_level)
        return magnitude

    @staticmethod
    def _local_emplacement_regime(
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        cell: tuple[int, int],
    ) -> str:
        """Classify the terrain process responsible for a local breakout."""

        slope = float(host_field.slope_degrees[cell])
        capacity = float(host_field.flow_capacity[cell])
        lateral_fraction = abs(float(geometry.cross_grid[cell])) / max(
            host_field.config.corridor_width,
            1.0,
        )
        if slope >= 12.0:
            return "erosional_steep"
        if slope <= 4.5 and capacity >= 0.55:
            return "inflating_distal"
        if lateral_fraction >= 0.55:
            return "unconfined_margin"
        return "confined_arterial"

    def _sample_roof_state(
        self,
        *,
        host_field: HostField,
        path: tuple[tuple[int, int], ...],
        rng: np.random.Generator,
    ) -> str:
        """Preserve whether a route roofed over, partly failed, or stayed open."""

        if not path:
            return "intact_tube"
        competence = float(np.mean([host_field.roof_competence[cell] for cell in path]))
        cover = float(np.mean([host_field.cover_thickness[cell] for cell in path]))
        cover_scale = max(host_field.config.volcanic_layer_thickness, 1.0)
        weakness = float(
            np.clip(0.62 * (1.0 - competence) + 0.38 * (1.0 - cover / cover_scale), 0.0, 1.0)
        )
        failure_probability = self.config.emplacement_history.roof_failure_probability
        draw = float(rng.random())
        if draw < 0.35 * failure_probability * weakness:
            return "open_channel"
        if draw < failure_probability * (0.45 + weakness):
            return "skylight_prone"
        if draw < failure_probability * (0.90 + 1.35 * weakness):
            return "partial_roof"
        return "intact_tube"

    def _select_breakout_sites(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        backbone_path: list[tuple[int, int]],
        count: int,
        rng: np.random.Generator,
    ) -> tuple[_BreakoutSite, ...]:
        """Select separated overflow, avulsion, and obstruction events.

        Unlike a generic anchor sampler, this evaluates why the supplying
        arterial tube would fail locally. The four process terms remain
        descriptive proxies rather than a full rheological simulation, but
        they make branch birth conditional on the host and parent route.
        """

        if count <= 0 or len(backbone_path) < 5:
            return ()
        lower = int(round(0.10 * (len(backbone_path) - 1)))
        upper = int(round(0.90 * (len(backbone_path) - 1)))
        unique_candidates: list[tuple[int, tuple[int, int]]] = []
        seen: set[tuple[int, int]] = set()
        for index in range(lower, upper + 1):
            cell = backbone_path[index]
            if cell in seen:
                continue
            seen.add(cell)
            unique_candidates.append((index, cell))
        blockage_field = self._correlated_terrain_perturbation(
            host_field.elevation.shape,
            amplitude_m=1.0,
            correlation_cells=max(
                1.0,
                0.65 * self.config.lobe_growth.perturbation_correlation_cells,
            ),
            rng=rng,
        )
        blockage_min = float(np.min(blockage_field))
        blockage_span = max(float(np.max(blockage_field)) - blockage_min, 1e-9)
        blockage_field = (blockage_field - blockage_min) / blockage_span
        candidates = [
            self._breakout_site_metrics(
                host_field=host_field,
                geometry=geometry,
                support_field=support_field,
                backbone_path=backbone_path,
                backbone_index=index,
                blockage=float(blockage_field[cell]),
            )
            for index, cell in unique_candidates
        ]
        # Local bank relief is often nearly constant at DEM resolution. Rank
        # each driver along the supplying route so the dominant label denotes
        # an anomalous process opportunity rather than the largest raw unit.
        driver_names = (
            "capacity_deficit",
            "confinement_loss",
            "curvature",
            "blockage",
        )
        ranked_drivers: dict[str, np.ndarray] = {}
        for name in driver_names:
            values = np.asarray([getattr(site, name) for site in candidates], dtype=float)
            order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
            ranked_drivers[name] = order.astype(float) / max(len(values) - 1, 1)
        controls = self.config.lobe_growth
        calibrated: list[_BreakoutSite] = []
        for index, site in enumerate(candidates):
            components = {
                "capacity_overflow": controls.breakout_capacity_weight
                * float(ranked_drivers["capacity_deficit"][index]),
                "margin_avulsion": controls.breakout_confinement_weight
                * float(ranked_drivers["confinement_loss"][index]),
                "bend_overflow": controls.breakout_curvature_weight
                * float(ranked_drivers["curvature"][index]),
                "seeded_blockage": controls.breakout_blockage_weight
                * float(ranked_drivers["blockage"][index]),
            }
            calibrated.append(
                replace(
                    site,
                    score=float(sum(components.values()) + 0.18 * float(support_field[site.cell])),
                    trigger=max(components, key=components.get),
                )
            )
        density_scale = math.sqrt(max(self.config.network_density, 0.05))
        minimum_spacing = (
            self.config.lobe_growth.minimum_anchor_spacing_fraction
            * geometry.along_extent
            / density_scale
        )
        sites: list[_BreakoutSite] = []
        while calibrated and len(sites) < count:
            scores = np.asarray([site.score for site in calibrated], dtype=float)
            probabilities = np.exp(scores - float(scores.max()))
            probabilities /= float(probabilities.sum())
            chosen_index = int(rng.choice(len(calibrated), p=probabilities))
            chosen = calibrated[chosen_index]
            sites.append(chosen)
            chosen_along = float(geometry.along_grid[chosen.cell])
            calibrated = [
                site
                for site in calibrated
                if abs(float(geometry.along_grid[site.cell]) - chosen_along) >= minimum_spacing
            ]
        return tuple(sorted(sites, key=lambda site: float(geometry.along_grid[site.cell])))

    def _breakout_site_metrics(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        backbone_path: list[tuple[int, int]],
        backbone_index: int,
        blockage: float,
    ) -> _BreakoutSite:
        cell = backbone_path[backbone_index]
        controls = self.config.lobe_growth
        slope_fraction = float(np.clip(host_field.slope_degrees[cell] / 18.0, 0.0, 1.0))
        capacity_deficit = float(
            np.clip(
                0.72 * (1.0 - float(host_field.flow_capacity[cell]))
                + 0.28 * (1.0 - slope_fraction),
                0.0,
                1.0,
            )
        )

        center_elevation = float(host_field.elevation[cell])
        lateral_relief: list[float] = []
        for neighbor in self._neighbor_cells(host_field, cell):
            dx = float(host_field.x_coords[neighbor[1]] - host_field.x_coords[cell[1]])
            dy = float(host_field.y_coords[neighbor[0]] - host_field.y_coords[cell[0]])
            length = max(math.hypot(dx, dy), 1e-9)
            lateral_alignment = abs(
                (dx / length) * geometry.cross_x + (dy / length) * geometry.cross_y
            )
            if lateral_alignment < 0.55:
                continue
            lateral_relief.append(
                max(float(host_field.elevation[neighbor]) - center_elevation, 0.0)
            )
        bank_relief = float(np.mean(lateral_relief)) if lateral_relief else 0.0
        confinement_loss = float(
            1.0
            - np.clip(
                bank_relief / max(0.85 * self.config.base_passage_radius, 1.0),
                0.0,
                1.0,
            )
        )

        previous_cell = backbone_path[max(0, backbone_index - 2)]
        following_cell = backbone_path[min(len(backbone_path) - 1, backbone_index + 2)]
        previous_world = self._cell_to_world(host_field, previous_cell)
        current_world = self._cell_to_world(host_field, cell)
        following_world = self._cell_to_world(host_field, following_cell)
        incoming = np.asarray(
            (current_world[0] - previous_world[0], current_world[1] - previous_world[1]),
            dtype=float,
        )
        outgoing = np.asarray(
            (following_world[0] - current_world[0], following_world[1] - current_world[1]),
            dtype=float,
        )
        incoming /= max(float(np.linalg.norm(incoming)), 1e-9)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1e-9)
        curvature = float(np.clip(0.5 * (1.0 - np.dot(incoming, outgoing)), 0.0, 1.0))

        components = {
            "capacity_overflow": controls.breakout_capacity_weight * capacity_deficit,
            "margin_avulsion": controls.breakout_confinement_weight * confinement_loss,
            "bend_overflow": controls.breakout_curvature_weight * curvature,
            "seeded_blockage": controls.breakout_blockage_weight * blockage,
        }
        trigger = max(components, key=components.get)
        score = float(sum(components.values()) + 0.18 * float(support_field[cell]))
        return _BreakoutSite(
            cell=cell,
            backbone_index=backbone_index,
            score=score,
            trigger=trigger,
            capacity_deficit=capacity_deficit,
            confinement_loss=confinement_loss,
            curvature=curvature,
            blockage=float(blockage),
        )

    def _breakout_initial_direction(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        backbone_path: list[tuple[int, int]],
        backbone_index: int,
        lateral_sign: float,
    ) -> np.ndarray:
        """Combine the parent tangent with a side-wall breakout impulse."""

        previous_cell = backbone_path[max(0, backbone_index - 2)]
        following_cell = backbone_path[min(len(backbone_path) - 1, backbone_index + 2)]
        previous_world = self._cell_to_world(host_field, previous_cell)
        following_world = self._cell_to_world(host_field, following_cell)
        parent_tangent = np.asarray(
            (
                following_world[1] - previous_world[1],
                following_world[0] - previous_world[0],
            ),
            dtype=float,
        )
        parent_tangent /= max(float(np.linalg.norm(parent_tangent)), 1e-9)
        direction = parent_tangent + 0.70 * lateral_sign * np.asarray(
            (geometry.cross_y, geometry.cross_x),
            dtype=float,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        return direction

    def _deposit_lobe_feedback(
        self,
        *,
        emplacement_surface: np.ndarray,
        path: tuple[tuple[int, int], ...],
        flux_fraction: float,
    ) -> float:
        """Inflate the temporary surface so later fronts can avulse naturally."""

        amplitude = self.config.lobe_growth.deposition_feedback_m * float(
            np.clip(flux_fraction, 0.0, 1.0) ** 0.55
        )
        if amplitude <= 0.0 or not path:
            return 0.0
        stamp = np.zeros_like(emplacement_surface, dtype=float)
        for cell in path:
            stamp[cell] = 1.0
        spread = gaussian_filter(
            stamp,
            sigma=self.config.lobe_growth.deposition_spread_cells,
            mode="constant",
        )
        maximum = float(np.max(spread))
        if maximum <= 0.0:
            return 0.0
        emplacement_surface += amplitude * spread / maximum
        return amplitude

    def _trace_lobe_front(
        self,
        *,
        host_field: HostField,
        geometry: _FlowGeometry,
        support_field: np.ndarray,
        downstream_potential: np.ndarray,
        backbone_alongs: np.ndarray,
        backbone_crosses: np.ndarray,
        start_cell: tuple[int, int],
        existing_cells: set[tuple[int, int]],
        lateral_sign: float,
        permit_merge: bool,
        merge_target_cells: set[tuple[int, int]] | None = None,
        initial_flux: float,
        initial_direction: np.ndarray,
        emplacement_surface: np.ndarray,
        rng: np.random.Generator,
    ) -> _LobeTrace:
        controls = self.config.lobe_growth
        perturbed_elevation = emplacement_surface + self._correlated_terrain_perturbation(
            host_field.elevation.shape,
            amplitude_m=controls.terrain_perturbation_m,
            correlation_cells=controls.perturbation_correlation_cells,
            rng=rng,
        )

        start_along = float(geometry.along_grid[start_cell])
        minimum_steps = max(2, controls.minimum_persistence_steps)
        minimum_merge_along = start_along + 0.55 * minimum_steps * geometry.cell_scale
        eligible_merge_mask = np.zeros_like(host_field.elevation, dtype=bool)
        if permit_merge:
            merge_cells = merge_target_cells if merge_target_cells else existing_cells
            for cell in merge_cells:
                if float(geometry.along_grid[cell]) >= minimum_merge_along:
                    eligible_merge_mask[cell] = True
        has_merge_targets = bool(np.any(eligible_merge_mask))
        if has_merge_targets:
            channel_distance = distance_transform_edt(~eligible_merge_mask)
        else:
            channel_distance = np.full_like(host_field.elevation, math.inf, dtype=float)

        path = [start_cell]
        maximum_steps = self._sample_int_range(rng, controls.maximum_steps)
        previous_step = np.asarray(initial_direction, dtype=float)
        previous_step /= max(float(np.linalg.norm(previous_step)), 1e-9)
        temperature = max(
            controls.retirement_temperature_k,
            self.config.source_temperature_k - self.config.cooling_k_per_m * max(start_along, 0.0),
        )
        maximum_separation = 0.0
        merged = False
        uphill_streak = 0

        for step_index in range(maximum_steps):
            current = path[-1]
            current_world = self._cell_to_world(host_field, current)
            current_along = float(geometry.along_grid[current])
            current_cross = float(geometry.cross_grid[current])
            current_potential = float(downstream_potential[current])
            divergence_fraction = max(
                0.0,
                1.0 - step_index / max(1.7 * minimum_steps, 1.0),
            )
            candidates: list[tuple[tuple[int, int], float]] = []
            for next_cell in self._neighbor_cells(host_field, current):
                if next_cell in path:
                    continue
                next_along = float(geometry.along_grid[next_cell])
                along_delta = next_along - current_along
                if along_delta < -0.18 * geometry.cell_scale:
                    continue
                if abs(float(geometry.cross_grid[next_cell])) > max(
                    0.72 * host_field.config.corridor_width,
                    6.0 * geometry.cell_scale,
                ):
                    continue
                on_existing = next_cell in existing_cells
                eligible_merge = bool(eligible_merge_mask[next_cell])
                if on_existing and not (
                    permit_merge and step_index + 1 >= minimum_steps and eligible_merge
                ):
                    continue

                next_world = self._cell_to_world(host_field, next_cell)
                step_vector = np.asarray(
                    (
                        next_world[1] - current_world[1],
                        next_world[0] - current_world[0],
                    ),
                    dtype=float,
                )
                step_length = max(float(np.linalg.norm(step_vector)), 1e-9)
                step_direction = step_vector / step_length
                actual_uphill = float(emplacement_surface[next_cell] - emplacement_surface[current])
                hydraulic_head = (
                    self.config.max_uphill_step
                    + 0.65
                    * self.config.base_passage_radius
                    * (initial_flux / max(self.config.source_flux, 1e-9)) ** 0.25
                )
                if actual_uphill > hydraulic_head:
                    continue
                if actual_uphill > 0.0 and uphill_streak >= 1 and not eligible_merge:
                    continue
                perturbed_drop = float(
                    perturbed_elevation[current] - perturbed_elevation[next_cell]
                )
                next_potential = float(downstream_potential[next_cell])
                if not math.isfinite(next_potential):
                    continue
                potential_gain = (current_potential - next_potential) / max(
                    geometry.cell_scale,
                    1.0,
                )
                inertia = float(np.dot(previous_step, step_direction))
                cross_delta = (float(geometry.cross_grid[next_cell]) - current_cross) / max(
                    geometry.cell_scale, 1.0
                )
                distance_gain = (
                    float(channel_distance[current] - channel_distance[next_cell])
                    if has_merge_targets
                    else 0.0
                )
                proximity = (
                    math.exp(-0.5 * float(channel_distance[next_cell]))
                    if has_merge_targets
                    else 0.0
                )
                score = float(support_field[next_cell])
                score += controls.inertia_weight * inertia
                score += controls.perturbed_slope_weight * np.clip(
                    perturbed_drop / max(0.25 * geometry.cell_scale, 1.0),
                    -1.5,
                    1.5,
                )
                score += controls.downstream_potential_weight * np.clip(
                    potential_gain,
                    -1.5,
                    1.5,
                )
                score += (
                    controls.initial_divergence_weight
                    * divergence_fraction
                    * lateral_sign
                    * cross_delta
                )
                score -= controls.channel_avoidance_weight * divergence_fraction * proximity
                score += controls.channel_reuse_weight * (1.0 - divergence_fraction) * distance_gain
                score += 8.0 if eligible_merge and step_index + 1 >= minimum_steps else 0.0
                candidates.append((next_cell, float(score)))

            if not candidates:
                break
            next_cell = self._sample_candidate(
                candidates,
                controls.candidate_temperature,
                rng,
            )
            next_world = self._cell_to_world(host_field, next_cell)
            step_vector = np.asarray(
                (
                    next_world[1] - current_world[1],
                    next_world[0] - current_world[0],
                ),
                dtype=float,
            )
            step_length = float(np.linalg.norm(step_vector))
            previous_step = step_vector / max(step_length, 1e-9)
            path.append(next_cell)
            selected_uphill = float(emplacement_surface[next_cell]) - float(
                emplacement_surface[current]
            )
            uphill_streak = uphill_streak + 1 if selected_uphill > 0.0 else 0

            next_along = float(geometry.along_grid[next_cell])
            reference_cross = float(np.interp(next_along, backbone_alongs, backbone_crosses))
            maximum_separation = max(
                maximum_separation,
                abs(float(geometry.cross_grid[next_cell]) - reference_cross),
            )
            temperature = max(
                273.15,
                temperature
                - self.config.cooling_k_per_m * controls.exposed_cooling_multiplier * step_length,
            )
            if bool(eligible_merge_mask[next_cell]) and len(path) > minimum_steps:
                merged = True
                break
            if temperature <= controls.retirement_temperature_k and len(path) > minimum_steps:
                break
            if next_along >= geometry.along_extent:
                break

        return _LobeTrace(
            path=tuple(path),
            merged=merged,
            maximum_lateral_separation=maximum_separation,
            final_temperature_k=temperature,
            initial_flux=initial_flux,
        )

    @staticmethod
    def _correlated_terrain_perturbation(
        shape: tuple[int, int],
        *,
        amplitude_m: float,
        correlation_cells: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a smooth zero-mean DOWNFLOW-style elevation perturbation."""

        if amplitude_m <= 0.0:
            return np.zeros(shape, dtype=float)
        correlated = gaussian_filter(
            rng.standard_normal(shape),
            sigma=max(correlation_cells, 0.25),
            mode="reflect",
        )
        correlated -= float(np.mean(correlated))
        scale = max(float(np.std(correlated)), 1e-9)
        return amplitude_m * correlated / scale

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
        has_left_existing_network = False
        uphill_streak = 0

        for _ in range(self.config.spur_max_steps):
            current = path[-1]
            current_world = self._cell_to_world(host_field, current)
            current_elevation = float(host_field.elevation[current])
            candidates: list[tuple[tuple[int, int], float]] = []
            for next_cell in self._neighbor_cells(host_field, current):
                if next_cell in path[-3:]:
                    continue
                # Spurs are terminal distributaries. Once one leaves the
                # existing network, do not let it reconnect downstream and
                # silently become a cyclic bypass.
                if next_cell != start_cell and float(total_flux[next_cell]) > 0.0:
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
                if uphill > 0.0 and uphill_streak >= 1:
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
            selected_uphill = float(host_field.elevation[next_cell]) - current_elevation
            uphill_streak = uphill_streak + 1 if selected_uphill > 0.0 else 0
            if float(total_flux[next_cell]) <= 0.0:
                has_left_existing_network = True
            if has_left_existing_network and len(path) > 8:
                break

        return path if len(path) > 4 else []

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
        terrain_perturbation: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        path = [start_cell]
        previous_step: tuple[float, float] | None = None
        uphill_streak = 0
        max_cross = max(0.55 * host_field.config.corridor_width, 90.0)
        curvature = self.config.lobe_growth
        curvature_rng = procedural_rng(self.config.random_seed, "backbone-curvature")
        curvature_phase = float(curvature_rng.uniform(0.0, 2.0 * math.pi))
        secondary_phase = float(curvature_rng.uniform(0.0, 2.0 * math.pi))

        for _ in range(self.config.trace_max_steps + 240):
            current = path[-1]
            current_world = self._cell_to_world(host_field, current)
            current_along = float(geometry.along_grid[current])
            current_potential = float(downstream_potential[current])
            current_elevation = float(host_field.elevation[current])
            current_perturbed_elevation = current_elevation + (
                float(terrain_perturbation[current]) if terrain_perturbation is not None else 0.0
            )
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
                # A single-cell reversal is a natural bend-scale feature, but
                # repeated positive grades produce an implausible climbing
                # arterial.  Permit one local reversal and then require a
                # downhill step (unless the route has reached its sink).
                if uphill > 0.0 and uphill_streak >= 1:
                    continue
                next_perturbed_elevation = next_elevation + (
                    float(terrain_perturbation[next_cell])
                    if terrain_perturbation is not None
                    else 0.0
                )

                score = float(support_field[next_cell])
                score += 1.8 * flow_alignment
                score += 2.2 * downhill_alignment
                score += 4.0 * (current_potential - next_potential) / max(geometry.cell_scale, 1.0)
                score += 0.6 * next_along / max(geometry.along_extent, 1.0)
                score -= 0.85 * abs(next_cross) / max(max_cross, geometry.cell_scale)
                progress = next_along / max(geometry.along_extent, 1.0)
                wavelength = max(
                    curvature.backbone_curvature_wavelength_fraction,
                    0.08,
                )
                target_cross = (
                    curvature.backbone_curvature_fraction
                    * max_cross
                    * math.sin(2.0 * math.pi * progress / wavelength + curvature_phase)
                )
                target_cross += (
                    curvature.backbone_curvature_fraction
                    * max_cross
                    * curvature.backbone_curvature_secondary_fraction
                    * math.sin(math.pi * progress / max(1.4 * wavelength, 0.12) + secondary_phase)
                )
                target_cross = float(np.clip(target_cross, -0.88 * max_cross, 0.88 * max_cross))
                # Steer toward a smooth, seeded lateral target while keeping
                # the host downhill and support terms authoritative.
                score += (
                    1.75
                    * (
                        abs(target_cross - float(geometry.cross_grid[current]))
                        - abs(target_cross - next_cross)
                    )
                    / max(geometry.cell_scale, 1.0)
                )
                score += 1.35 * np.clip(
                    (current_perturbed_elevation - next_perturbed_elevation)
                    / max(0.25 * geometry.cell_scale, 1.0),
                    -1.5,
                    1.5,
                )
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
            selected_uphill = float(host_field.elevation[next_cell]) - current_elevation
            uphill_streak = uphill_streak + 1 if selected_uphill > 0.0 else 0
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
        rng,
    ) -> tuple[_BraidZone, ...]:
        grammar = self.config.braid_grammar
        spread = 0.34 * host_field.config.corridor_width
        zone_count = self._sample_int_range(rng, grammar.zone_count)
        center_min, center_max = grammar.center_fraction
        raw_centers = sorted(
            float(rng.uniform(center_min, center_max)) for _ in range(zone_count * 4)
        )
        centers: list[float] = []
        for center in raw_centers:
            if all(abs(center - existing) >= grammar.min_center_spacing for existing in centers):
                centers.append(center)
            if len(centers) >= zone_count:
                break
        while len(centers) < zone_count:
            centers.append(float(rng.uniform(center_min, center_max)))
        centers.sort()

        zones: list[_BraidZone] = []
        for center in centers:
            branch_count = self._sample_int_range(rng, grammar.branches_per_zone)
            branch_count = max(2, branch_count)
            branches: list[_ZoneBranch] = []
            signs = [-1.0, 1.0]
            for branch_index in range(branch_count):
                sign = signs[branch_index % 2]
                if branch_index >= 2:
                    sign *= -1.0 if rng.random() < 0.5 else 1.0
                kind = self._sample_branch_kind(rng)
                z_level = 0
                merge_shared_cells = True
                if rng.random() < grammar.underpass_probability:
                    kind = "underpass"
                    z_level = -1 if rng.random() < 0.5 else 1
                    merge_shared_cells = False
                lateral_scale = self._sample_float_range(rng, grammar.lateral_offset_scale)
                sampled_offset = lateral_scale * spread
                minimum_offset = max(
                    self.config.minimum_branch_offset_widths
                    * 2.0
                    * self.config.base_passage_radius,
                    0.19 * host_field.config.corridor_width,
                )
                branches.append(
                    _ZoneBranch(
                        kind=kind,
                        lateral_offset=sign * max(sampled_offset, minimum_offset),
                        start_shift_fraction=self._sample_float_range(
                            rng, grammar.start_shift_fraction
                        ),
                        end_shift_fraction=self._sample_float_range(
                            rng, grammar.end_shift_fraction
                        ),
                        skew=self._sample_float_range(rng, grammar.skew),
                        wobble=self._sample_float_range(rng, grammar.wobble),
                        phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                        z_level=z_level,
                        merge_shared_cells=merge_shared_cells,
                    )
                )

            ladder_rungs: tuple[float, ...] = ()
            if branch_count >= 2 and rng.random() < grammar.ladder_probability:
                rung_count = self._sample_int_range(rng, grammar.ladder_rung_count)
                ladder_rungs = tuple(
                    sorted(float(rng.uniform(0.25, 0.75)) for _ in range(rung_count))
                )

            zones.append(
                _BraidZone(
                    center_fraction=center,
                    half_length_fraction=self._sample_float_range(
                        rng, grammar.half_length_fraction
                    ),
                    branches=tuple(branches),
                    ladder_rungs=ladder_rungs,
                    chamber_radius_scale=self._sample_float_range(
                        rng, grammar.chamber_radius_scale
                    ),
                )
            )

        return tuple(zones)

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
                    - float(
                        np.interp(
                            float(geometry.along_grid[cell]), backbone_alongs, backbone_crosses
                        )
                    )
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
                    chamber_radius_scale=zone.chamber_radius_scale,
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
                chamber_radius_scale=zone.chamber_radius_scale,
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

        for index, cell in enumerate(backbone_segment[1:-1], start=1):
            along = float(geometry.along_grid[cell])
            progress = (along - segment_start) / along_span
            clamped_progress = float(np.clip(progress, 0.0, 1.0))
            envelope = self._natural_split_envelope(
                clamped_progress,
                branch.skew,
                branch.phase,
            )
            meander = (
                branch.wobble
                * envelope
                * (
                    0.72 * math.sin(2.0 * math.pi * 0.58 * clamped_progress + branch.phase)
                    + 0.28 * math.sin(2.0 * math.pi * 1.31 * clamped_progress + 0.5 * branch.phase)
                )
            )
            target_cross = (
                float(np.interp(along, backbone_alongs, backbone_crosses))
                + branch.lateral_offset * envelope
                + 0.18 * host_field.config.corridor_width * meander
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

    @staticmethod
    def _natural_split_envelope(
        progress: float,
        skew: float,
        phase: float,
    ) -> float:
        """Return an asymmetric, rounded offset for a split/merge path."""

        t = float(np.clip(progress, 0.0, 1.0))
        if t <= 0.0 or t >= 1.0:
            return 0.0
        warped = float(
            np.clip(
                t
                + 0.10 * skew * math.sin(math.pi * t)
                + 0.025 * math.sin(phase) * math.sin(2.0 * math.pi * t),
                0.0,
                1.0,
            )
        )
        exponent = 1.15 + 0.25 * abs(skew)
        rounded = max(math.sin(math.pi * warped), 0.0) ** exponent
        breathing = 1.0 + 0.035 * rounded * math.sin(2.0 * math.pi * 0.55 * t + phase)
        return max(0.0, rounded * breathing)

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
        chamber_radius_scale: float,
    ) -> tuple[_SelectedPath, ...]:
        # Grade-separated branches deliberately do not form graph junctions at
        # interior XY crossings. Attaching a ladder to one would therefore
        # create a floating two-node component rather than a physical passage.
        connectable_branches = [item for item in branches if item[0].merge_shared_cells]
        if len(connectable_branches) < 2:
            return ()
        ordered = sorted(
            connectable_branches,
            key=lambda item: np.mean([float(geometry.cross_grid[cell]) for cell in item[1]]),
        )
        ladders: list[_SelectedPath] = []
        for rung_index, rung_fraction in enumerate(rungs):
            pair_index = (zone_index + rung_index) % (len(ordered) - 1)
            left_branch = ordered[pair_index][1]
            right_branch = ordered[pair_index + 1][1]
            left_alongs = [float(geometry.along_grid[cell]) for cell in left_branch]
            right_alongs = [float(geometry.along_grid[cell]) for cell in right_branch]
            overlap_start = max(min(left_alongs), min(right_alongs))
            overlap_end = min(max(left_alongs), max(right_alongs))
            overlap_span = overlap_end - overlap_start
            if overlap_span <= 2.0 * geometry.cell_scale:
                continue
            center_along = overlap_start + rung_fraction * overlap_span
            cross_gap = abs(
                float(np.mean([geometry.cross_grid[cell] for cell in right_branch]))
                - float(np.mean([geometry.cross_grid[cell] for cell in left_branch]))
            )
            along_skew = min(
                0.38 * overlap_span,
                max(3.5 * geometry.cell_scale, 0.55 * cross_gap),
            )
            skew_sign = -1.0 if (zone_index + rung_index) % 2 else 1.0
            left_target = center_along - 0.5 * skew_sign * along_skew
            right_target = center_along + 0.5 * skew_sign * along_skew
            left_cell = min(
                left_branch,
                key=lambda cell: abs(float(geometry.along_grid[cell]) - left_target),
            )
            right_cell = min(
                right_branch,
                key=lambda cell: abs(float(geometry.along_grid[cell]) - right_target),
            )
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
            metadata = self._build_segment_metadata(
                kind="ladder",
                zone_index=zone_index,
                chamber_radius_scale=chamber_radius_scale,
            )
            metadata["connection_style"] = "oblique_anastomosis"
            ladders.append(
                _SelectedPath(
                    kind="ladder",
                    path=tuple(simplified),
                    metadata=metadata,
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
            target_along = (1.0 - t) * float(geometry.along_grid[start_cell]) + t * float(
                geometry.along_grid[end_cell]
            )
            target_cross = 0.35 * (
                float(geometry.cross_grid[start_cell]) + float(geometry.cross_grid[end_cell])
            ) + 0.65 * float(np.interp(target_along, backbone_alongs, backbone_crosses))
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
            distance_penalty = 0.018 * math.hypot(
                candidate_world[0] - target_x, candidate_world[1] - target_y
            )
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

    @staticmethod
    def _family_label_for_kind(kind: str) -> str:
        if kind in {"spur", "abandoned_lobe", "stalled_lobe"}:
            return "spur"
        if kind in {"backbone", "chamber_braid", "anastomosis"}:
            return "large"
        if kind in {"island_bypass", "underpass", "distributary"}:
            return "medium"
        return "small"

    @staticmethod
    def _build_segment_metadata(
        *,
        kind: str,
        zone_index: int | None = None,
        z_level: int = 0,
        chamber_radius_scale: float = 1.0,
    ) -> dict[str, SegmentMetadataValue]:
        crossing_group_id: str | None = None
        merge_behavior = "merge"
        island_id: str | None = None
        chamber_id: str | None = None

        if kind in {"island_bypass", "inner_bypass"} and zone_index is not None:
            island_id = f"island_zone_{zone_index}"
        if kind == "distributary" and zone_index is not None:
            island_id = f"lobe_path_{zone_index}"
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
            "chamber_radius_scale": chamber_radius_scale,
            "formation_origin": kind,
        }

    @classmethod
    def _build_emplacement_metadata(
        cls,
        *,
        kind: str,
        phase_count: int,
        birth_phase: int,
        death_phase: int,
        formation_state: str,
        zone_index: int | None = None,
        z_level: int = 0,
        chamber_forming: bool = False,
        chamber_radius_scale: float = 1.0,
    ) -> dict[str, SegmentMetadataValue]:
        metadata = cls._build_segment_metadata(
            kind=kind,
            zone_index=zone_index,
            z_level=z_level,
            chamber_radius_scale=chamber_radius_scale,
        )
        if chamber_forming and zone_index is not None:
            metadata["chamber_id"] = f"coalescence_lobe_{zone_index}"
        metadata.update(
            {
                "emplacement_phase_count": phase_count,
                "birth_phase": birth_phase,
                "death_phase": death_phase,
                "active_phase_count": death_phase - birth_phase + 1,
                "formation_state": formation_state,
                "branch_order": 0 if kind in {"backbone", "source_feeder"} else 1,
                "chamber_forming": chamber_forming,
                "vertical_capture": False,
                "emplacement_regime": "confined_arterial",
                "roof_state": "intact_tube",
            }
        )
        return metadata

    def _build_representative_fields(
        self,
        *,
        shape: tuple[int, int],
        selected_paths: tuple[_SelectedPath, ...],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        mask = np.zeros(shape, dtype=bool)
        flux = np.zeros(shape, dtype=float)
        family_flux = {label: np.zeros(shape, dtype=float) for label in self.FAMILY_LABELS}
        for selected_path in selected_paths:
            family_label = self._family_label_for_kind(selected_path.kind)
            for cell in selected_path.path:
                mask[cell] = True
                flux[cell] += 1.0
                family_flux[family_label][cell] += 1.0
        return mask, flux, family_flux

    @staticmethod
    def _restore_attachment_cells(paths: tuple[_SelectedPath, ...]) -> tuple[_SelectedPath, ...]:
        """Restore junctions removed independently from simplified polylines.

        A feeder endpoint on a straight trunk must split the trunk even if
        that cell was discarded as collinear. Interior grade-separated paths
        remain unsplit; only their explicit endpoints attach to other routes.
        """
        candidates = sorted({cell for path in paths for cell in path.path})
        result = []
        for path in paths:
            if not path.merge_shared_cells:
                result.append(path)
                continue
            restored = [path.path[0]]
            for first, last in zip(path.path, path.path[1:]):
                dy, dx = last[0] - first[0], last[1] - first[1]
                length_squared = dy * dy + dx * dx
                attachments = []
                for cell in candidates:
                    cy, cx = cell[0] - first[0], cell[1] - first[1]
                    along = cy * dy + cx * dx
                    if cy * dx == cx * dy and 0 < along < length_squared:
                        attachments.append((along, cell))
                restored.extend(cell for _, cell in sorted(attachments))
                restored.append(last)
            result.append(replace(path, path=tuple(restored)))
        return tuple(result)

    @staticmethod
    def _smooth_graph_routes(host: HostField, segments: list[CaveSegment]) -> list[CaveSegment]:
        """Fit flowing routes after graph extraction, retaining exact nodes.

        Smoothing operates in metres and is bounded by passage width. Cubic
        connection regions share a parent direction instead of leaving the
        40--140 degree corners produced by snapped host-grid paths.
        """
        incident: dict[int, list[tuple[CaveSegment, np.ndarray]]] = defaultdict(list)
        for segment in segments:
            xy = np.asarray([(point.x, point.y) for point in segment.points])
            for node, direction in (
                (segment.start_node_id, xy[1] - xy[0]),
                (segment.end_node_id, xy[-1] - xy[-2]),
            ):
                direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
                incident[node].append((segment, direction))
        node_directions = {
            node: max(
                values,
                key=lambda item: (
                    item[0].kind == "backbone",
                    item[0].mean_width,
                    item[0].total_length,
                ),
            )[1]
            for node, values in incident.items()
        }
        result = []
        for segment in segments:
            raw = np.asarray([(point.x, point.y) for point in segment.points])
            arc = np.asarray([point.arc_length for point in segment.points])
            length = float(arc[-1])
            if length <= 1e-6:
                result.append(segment)
                continue
            spacing = max(0.5, min(3.0, 0.3 * segment.mean_width))
            distances = np.linspace(0.0, length, max(5, int(math.ceil(length / spacing)) + 1))
            linear = np.column_stack([np.interp(distances, arc, raw[:, axis]) for axis in range(2)])
            sigma_m = min(2.0 * segment.mean_width, 0.15 * length)
            smooth = gaussian_filter1d(
                linear, sigma_m / (distances[1] - distances[0]), axis=0, mode="nearest"
            )
            delta = smooth - linear
            delta *= np.minimum(
                1.0, segment.mean_width / np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
            )[:, None]
            smooth = linear + delta
            smooth[0], smooth[-1] = raw[0], raw[-1]
            curve = CubicSpline(distances, smooth, axis=0)
            coords = curve(distances)
            reach = min(4.0 * segment.mean_width, 0.45 * length)
            for index, node in ((0, segment.start_node_id), (-1, segment.end_node_id)):
                direction = node_directions[node].copy()
                original = raw[1] - raw[0] if index == 0 else raw[-1] - raw[-2]
                if np.dot(direction, original) < 0:
                    direction = -direction
                if index == 0:
                    transition = CubicHermiteSpline(
                        [0.0, reach], [raw[0], curve(reach)], [direction, curve(reach, 1)], axis=0
                    )
                    mask = distances <= reach
                else:
                    transition = CubicHermiteSpline(
                        [length - reach, length],
                        [curve(length - reach), raw[-1]],
                        [curve(length - reach, 1), direction],
                        axis=0,
                    )
                    mask = distances >= length - reach
                coords[mask] = transition(distances[mask])
            coords[0], coords[-1] = raw[0], raw[-1]
            # Cubic interpolation may overshoot an edge endpoint by a few
            # millimetres even though every routed cell is inside the host.
            # Keep the fitted centreline within the field sampled below.
            if hasattr(host, "x_coords") and hasattr(host, "y_coords"):
                coords[:, 0] = np.clip(coords[:, 0], host.x_coords[0], host.x_coords[-1])
                coords[:, 1] = np.clip(coords[:, 1], host.y_coords[0], host.y_coords[-1])
            new_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
            widths = np.interp(distances, arc, [point.width for point in segment.points])
            points = []
            for index, (xy, distance, width) in enumerate(zip(coords, new_arc, widths)):
                substrate = host.sample(float(xy[0]), float(xy[1]))
                points.append(
                    CavePoint(
                        index=index,
                        x=float(xy[0]),
                        y=float(xy[1]),
                        elevation=substrate.elevation,
                        slope_degrees=substrate.slope_degrees,
                        cover_thickness=substrate.cover_thickness,
                        roof_competence=substrate.roof_competence,
                        growth_cost=substrate.growth_cost,
                        arc_length=float(distance),
                        width=float(width),
                    )
                )
            result.append(replace(segment, points=tuple(points)))
        return result

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

        selected_paths = self._restore_attachment_cells(selected_paths)

        path_use_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        all_path_cells: set[tuple[int, int]] = set()
        for selected_path in selected_paths:
            all_path_cells.update(selected_path.path)
            if not selected_path.merge_shared_cells:
                if selected_path.path:
                    path_use_counts[selected_path.path[0]] += 1
                    path_use_counts[selected_path.path[-1]] += 1
                continue
            for cell in selected_path.path:
                path_use_counts[cell] += 1

        source_cells = {
            selected_path.path[0]
            for selected_path in selected_paths
            if selected_path.kind in {"backbone", "source_feeder"} and selected_path.path
        }
        sink_cell = max(
            all_path_cells,
            key=lambda cell: (
                float(geometry.along_grid[cell]),
                -abs(float(geometry.cross_grid[cell])),
            ),
        )

        node_cell_set: set[tuple[int, int]] = set(source_cells) | {sink_cell}
        chamber_cells: set[tuple[int, int]] = set()
        for selected_path in selected_paths:
            node_cell_set.add(selected_path.path[0])
            node_cell_set.add(selected_path.path[-1])
            pool_center = self._drained_pool_center_cell(selected_path)
            if pool_center is not None:
                node_cell_set.add(pool_center)
                chamber_cells.add(pool_center)
            if selected_path.kind in {"chamber_braid", "ladder"} or bool(
                (selected_path.metadata or {}).get("chamber_forming", False)
            ):
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
            if cell in source_cells:
                node_kind = "entry"
            elif cell == sink_cell:
                node_kind = "exit"
            elif cell in chamber_cells and (
                path_use_counts.get(cell, 0) >= 2
                or any(
                    self._drained_pool_center_cell(selected_path) == cell
                    for selected_path in selected_paths
                )
            ):
                node_kind = "chamber"
            elif path_use_counts.get(cell, 0) == 1:
                node_kind = (
                    "spur_terminal"
                    if any(
                        selected_path.kind in {"spur", "abandoned_lobe"}
                        and cell in {selected_path.path[0], selected_path.path[-1]}
                        for selected_path in selected_paths
                    )
                    else "terminal"
                )
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
                if not selected_path.merge_shared_cells and cell not in {path[0], path[-1]}:
                    continue
                start_node_id = node_cells[current_cells[0]]
                end_node_id = node_cells[cell]
                if start_node_id != end_node_id and len(current_cells) >= 2:
                    signature_cells = tuple(current_cells)
                    signature = (
                        min(start_node_id, end_node_id),
                        max(start_node_id, end_node_id),
                        signature_cells,
                    )
                    reverse_signature = (
                        min(start_node_id, end_node_id),
                        max(start_node_id, end_node_id),
                        tuple(reversed(signature_cells)),
                    )
                    if (
                        signature not in seen_signatures
                        and reverse_signature not in seen_signatures
                    ):
                        segment = self._build_segment_from_cells(
                            host_field=host_field,
                            path_cells=current_cells,
                            start_node_id=start_node_id,
                            end_node_id=end_node_id,
                            segment_id=len(segments),
                            total_flux=total_flux,
                            kind=selected_path.kind,
                            z_level=selected_path.z_level,
                            metadata=self._metadata_for_path_section(
                                selected_path,
                                current_cells,
                            ),
                        )
                        if segment is not None:
                            segments.append(segment)
                            seen_signatures.add(signature)
                current_cells = [cell]

        return nodes, segments, ()

    @staticmethod
    def _reverse_segment(segment: CaveSegment) -> CaveSegment:
        total_length = segment.total_length
        reversed_points = tuple(
            replace(
                point,
                index=index,
                arc_length=total_length - point.arc_length,
            )
            for index, point in enumerate(reversed(segment.points))
        )
        return replace(
            segment,
            start_node_id=segment.end_node_id,
            end_node_id=segment.start_node_id,
            points=reversed_points,
        )

    @staticmethod
    def _orient_segments_for_flow(
        nodes: list[CaveNode],
        segments: list[CaveSegment],
    ) -> list[CaveSegment]:
        """Make stored start/end and point order agree with physical flow.

        Segments leaving an entry and segments entering a terminal retain
        their semantic construction direction, even when they bend slightly
        upstream. Every interior edge follows a strict global flow ordering.
        The only exceptions therefore leave graph roots or enter graph leaves,
        which preserves physical feeder/spur meaning while guaranteeing that
        cross-family intersections cannot create directed cycles.
        """

        node_lookup = {node.node_id: node for node in nodes}
        undirected_degree = {node.node_id: 0 for node in nodes}
        for segment in segments:
            undirected_degree[segment.start_node_id] += 1
            undirected_degree[segment.end_node_id] += 1
        oriented: list[CaveSegment] = []
        for segment in segments:
            start = node_lookup[segment.start_node_id]
            end = node_lookup[segment.end_node_id]
            starts_at_entry = start.kind == "entry"
            ends_at_terminal = (
                end.kind in {"terminal", "spur_terminal"} and undirected_degree[end.node_id] == 1
            )
            if starts_at_entry or ends_at_terminal:
                oriented.append(segment)
                continue
            if (start.along_position, start.node_id) <= (end.along_position, end.node_id):
                oriented.append(segment)
                continue
            oriented.append(CaveNetworkGenerator._reverse_segment(segment))
        return oriented

    @classmethod
    def _repair_source_reachability(
        cls,
        nodes: list[CaveNode],
        segments: list[CaveSegment],
    ) -> list[CaveSegment]:
        """Orient local extrema into the source-reachable directed graph.

        Intersections between independently sampled path families can produce
        a tiny along-flow local minimum: all of its incident edges point away
        even though the undirected graph is connected. Reverse one acyclic cut
        edge at a time so every emitted node receives source flow.
        """

        entries = {node.node_id for node in nodes if node.kind == "entry"}
        if not entries:
            return segments

        repaired = list(segments)
        for _ in range(len(nodes)):
            outgoing: defaultdict[int, list[int]] = defaultdict(list)
            for segment in repaired:
                outgoing[segment.start_node_id].append(segment.end_node_id)
            reachable = set(entries)
            pending = list(entries)
            while pending:
                for downstream in outgoing.get(pending.pop(), []):
                    if downstream not in reachable:
                        reachable.add(downstream)
                        pending.append(downstream)
            if len(reachable) == len(nodes):
                # Source reachability alone is insufficient when a local
                # equal-along tie points an entry into a dead-end branch.
                # Repair a boundary edge from the exit-reachable region into
                # the blocked component, preserving an acyclic orientation.
                exits = {node.node_id for node in nodes if node.kind == "exit"}
                can_reach_exit = set(exits)
                reverse_adjacency: defaultdict[int, list[int]] = defaultdict(list)
                for segment in repaired:
                    reverse_adjacency[segment.end_node_id].append(segment.start_node_id)
                pending_exit = list(exits)
                while pending_exit:
                    for upstream in reverse_adjacency[pending_exit.pop()]:
                        if upstream not in can_reach_exit:
                            can_reach_exit.add(upstream)
                            pending_exit.append(upstream)
                blocked_entries = entries - can_reach_exit
                if not blocked_entries:
                    return repaired
                candidates = sorted(
                    (
                        (index, segment)
                        for index, segment in enumerate(repaired)
                        if segment.start_node_id in can_reach_exit
                        and segment.end_node_id not in can_reach_exit
                    ),
                    key=lambda item: (item[1].total_length, item[1].segment_id),
                )
                for index, segment in candidates:
                    candidate = list(repaired)
                    candidate[index] = cls._reverse_segment(segment)
                    try:
                        cls._topological_node_ids(nodes, candidate)
                    except ValueError:
                        continue
                    repaired = candidate
                    break
                else:
                    return repaired
                continue

            candidates = sorted(
                (
                    (index, segment)
                    for index, segment in enumerate(repaired)
                    if segment.start_node_id not in reachable and segment.end_node_id in reachable
                ),
                key=lambda item: (item[1].total_length, item[1].segment_id),
            )
            for index, segment in candidates:
                candidate = list(repaired)
                candidate[index] = cls._reverse_segment(segment)
                try:
                    cls._topological_node_ids(nodes, candidate)
                except ValueError:
                    continue
                repaired = candidate
                break
            else:
                return repaired
        return repaired

    @staticmethod
    def _topological_node_ids(
        nodes: list[CaveNode],
        segments: list[CaveSegment],
    ) -> tuple[int, ...]:
        """Return deterministic directed-graph order, rejecting flow cycles."""

        node_lookup = {node.node_id: node for node in nodes}
        indegree = {node.node_id: 0 for node in nodes}
        outgoing: defaultdict[int, list[int]] = defaultdict(list)
        for segment in segments:
            if segment.start_node_id not in indegree or segment.end_node_id not in indegree:
                raise ValueError(f"Segment {segment.segment_id} references an unknown node")
            indegree[segment.end_node_id] += 1
            outgoing[segment.start_node_id].append(segment.end_node_id)

        heap = [
            (node_lookup[node_id].along_position, node_id)
            for node_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(heap)
        ordered: list[int] = []
        while heap:
            _along, node_id = heapq.heappop(heap)
            ordered.append(node_id)
            for downstream in outgoing.get(node_id, []):
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    heapq.heappush(
                        heap,
                        (node_lookup[downstream].along_position, downstream),
                    )
        if len(ordered) != len(nodes):
            cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
            raise ValueError(f"Cave network contains a directed flow cycle at nodes {cyclic}")
        return tuple(ordered)

    def _assign_conserved_flow(
        self,
        nodes: list[CaveNode],
        segments: list[CaveSegment],
    ) -> list[CaveSegment]:
        """Propagate source flux through the directed graph and annotate points."""

        if not nodes or not segments:
            return segments
        outgoing: defaultdict[int, list[int]] = defaultdict(list)
        segment_lookup = {segment.segment_id: segment for segment in segments}
        for segment in segments:
            outgoing[segment.start_node_id].append(segment.segment_id)

        available_flux: defaultdict[int, float] = defaultdict(float)
        temperature_energy: defaultdict[int, float] = defaultdict(float)
        age_flux: defaultdict[int, float] = defaultdict(float)
        entry_nodes = [node for node in nodes if node.kind == "entry"]
        entry_strengths: list[float] = []
        for node in entry_nodes:
            initial_capacity = sum(
                max(segment_lookup[segment_id].mean_width, 1e-6) ** 2
                for segment_id in outgoing.get(node.node_id, [])
            )
            source_rng = procedural_rng(
                self.config.random_seed,
                "source-strength",
                node.node_id,
            )
            seeded_variation = math.exp(float(source_rng.normal(0.0, 0.18)))
            entry_strengths.append(max(initial_capacity, 1e-6) * seeded_variation)
        strength_sum = sum(entry_strengths)
        for node, strength in zip(entry_nodes, entry_strengths, strict=True):
            # ``source_flux`` remains the mean inlet discharge, so the total
            # supply is stable while individual fissure-fed entries vary with
            # their local carrying capacity and named seed.
            inlet_flux = (
                self.config.source_flux * len(entry_nodes) * strength / max(strength_sum, 1e-9)
            )
            available_flux[node.node_id] += inlet_flux
            temperature_energy[node.node_id] += inlet_flux * self.config.source_temperature_k

        flux_by_segment: dict[int, float] = {}
        temperature_by_segment: dict[int, float] = {}
        age_by_segment: dict[int, float] = {}
        node_lookup = {node.node_id: node for node in nodes}
        for node_id in self._topological_node_ids(nodes, segments):
            node = node_lookup[node_id]
            segment_ids = outgoing.get(node_id, [])
            node_flux = available_flux[node_id]
            if not segment_ids or node_flux <= 0.0:
                continue
            node_temperature = temperature_energy[node.node_id] / node_flux
            node_age = age_flux[node.node_id] / node_flux
            weights = np.asarray(
                [
                    max(segment_lookup[segment_id].mean_width, 1e-6) ** 2
                    for segment_id in segment_ids
                ],
                dtype=float,
            )
            weights /= float(np.sum(weights))
            for segment_id, weight in zip(segment_ids, weights, strict=True):
                segment = segment_lookup[segment_id]
                segment_flux = node_flux * float(weight)
                travel_time = segment.total_length / max(self.config.nominal_flow_speed_m_s, 1e-6)
                cooled_temperature = max(
                    273.15,
                    node_temperature - self.config.cooling_k_per_m * segment.total_length,
                )
                downstream = segment.end_node_id
                flux_by_segment[segment_id] = segment_flux
                temperature_by_segment[segment_id] = node_temperature
                age_by_segment[segment_id] = node_age
                available_flux[downstream] += segment_flux
                temperature_energy[downstream] += segment_flux * cooled_temperature
                age_flux[downstream] += segment_flux * (node_age + travel_time)

        resolved: list[CaveSegment] = []
        for segment in segments:
            flux = flux_by_segment.get(segment.segment_id, 0.0)
            start_temperature = temperature_by_segment.get(
                segment.segment_id,
                self.config.source_temperature_k,
            )
            start_age = age_by_segment.get(segment.segment_id, 0.0)
            flow_scale = float(
                np.clip(
                    (max(flux, 1e-9) / max(self.config.source_flux, 1e-9)) ** 0.20,
                    0.68,
                    1.45,
                )
            )
            points = tuple(
                replace(
                    point,
                    width=float(
                        np.clip(
                            point.width * flow_scale,
                            2.0 * self.config.minimum_passage_radius,
                            2.0 * self.config.maximum_passage_radius,
                        )
                    ),
                    flux=flux,
                    temperature_k=max(
                        273.15,
                        start_temperature - self.config.cooling_k_per_m * point.arc_length,
                    ),
                    age_s=(
                        start_age + point.arc_length / max(self.config.nominal_flow_speed_m_s, 1e-6)
                    ),
                )
                for point in segment.points
            )
            resolved.append(replace(segment, points=points))
        return resolved

    @staticmethod
    def _annotate_emplacement_flux_history(
        segments: list[CaveSegment],
    ) -> list[CaveSegment]:
        """Summarize staged activity without corrupting final graph conservation.

        Point flux remains the conserved reference discharge used by downstream
        geometry. These scalar history fields describe how concentrated that
        discharge was while each route was active during construction.
        """

        resolved: list[CaveSegment] = []
        for segment in segments:
            metadata = dict(segment.metadata)
            phase_value = metadata.get("emplacement_phase_count", 1)
            birth_value = metadata.get("birth_phase", 0)
            death_value = metadata.get("death_phase", 0)
            phase_count = max(
                int(phase_value) if isinstance(phase_value, (int, float)) else 1,
                1,
            )
            birth_phase = int(birth_value) if isinstance(birth_value, (int, float)) else 0
            death_phase = (
                int(death_value) if isinstance(death_value, (int, float)) else phase_count - 1
            )
            active_count = max(death_phase - birth_phase + 1, 1)
            duty_cycle = active_count / phase_count
            peak_flux = segment.mean_flux * (1.0 + 0.22 * (1.0 - duty_cycle))
            uphill_distance = 0.0
            sustained_uphill_steps = 0
            previous_uphill = False
            for first, second in zip(segment.points, segment.points[1:]):
                step_length = math.hypot(second.x - first.x, second.y - first.y)
                uphill = second.elevation > first.elevation
                if uphill:
                    uphill_distance += step_length
                if uphill and previous_uphill:
                    sustained_uphill_steps += 1
                previous_uphill = uphill
            process_grade = bool(metadata.get("vertical_capture", False)) or (
                metadata.get("formation_origin", segment.kind)
                in {"anastomosis", "underpass", "chamber_braid", "ladder"}
            )
            grade_profile = (
                f"process_uphill_{metadata.get('formation_origin', segment.kind)}"
                if sustained_uphill_steps > 0 and process_grade
                else "uphill_unresolved"
                if sustained_uphill_steps > 0
                else "downhill_with_local_reversals"
                if uphill_distance > 0.0
                else "downhill"
            )
            metadata.update(
                {
                    "active_phase_count": active_count,
                    "emplacement_duty_cycle": duty_cycle,
                    "phase_weighted_flux": segment.mean_flux * duty_cycle,
                    "peak_formation_flux": peak_flux,
                    "peak_flux_phase": birth_phase,
                    "uphill_distance_m": uphill_distance,
                    "sustained_uphill_step_count": sustained_uphill_steps,
                    "grade_profile": grade_profile,
                }
            )
            resolved.append(replace(segment, metadata=metadata))
        return resolved

    def _build_junctions(
        self,
        nodes: list[CaveNode],
        segments: list[CaveSegment],
    ) -> list[CaveJunction]:
        if not nodes or not segments:
            return []

        adjacency: dict[int, list[CaveSegment]] = defaultdict(list)
        for segment in segments:
            adjacency[segment.start_node_id].append(segment)
            adjacency[segment.end_node_id].append(segment)

        candidate_node_ids = {
            node.node_id
            for node in nodes
            if node.kind in {"junction", "chamber"} or len(adjacency[node.node_id]) >= 3
        }
        if not candidate_node_ids:
            return []

        node_lookup = {node.node_id: node for node in nodes}
        visited: set[int] = set()
        clusters: list[set[int]] = []
        passage_width = float(np.median([segment.mean_width for segment in segments]))
        max_along_gap = 3.0 * passage_width
        max_distance = 4.0 * passage_width
        for node_id in sorted(
            candidate_node_ids, key=lambda item: node_lookup[item].along_position
        ):
            if node_id in visited:
                continue
            cluster = {node_id}
            queue = [node_id]
            visited.add(node_id)
            while queue:
                current_id = queue.pop()
                current = node_lookup[current_id]
                for neighbor_id in candidate_node_ids:
                    if neighbor_id in visited:
                        continue
                    neighbor = node_lookup[neighbor_id]
                    along_gap = abs(neighbor.along_position - current.along_position)
                    distance = math.hypot(neighbor.x - current.x, neighbor.y - current.y)
                    shared_segment = any(
                        segment.start_node_id in {current_id, neighbor_id}
                        and segment.end_node_id in {current_id, neighbor_id}
                        for segment in segments
                    )
                    cluster_span = max(
                        math.hypot(
                            neighbor.x - node_lookup[member].x, neighbor.y - node_lookup[member].y
                        )
                        for member in cluster
                    )
                    if (
                        cluster_span <= max_distance
                        and along_gap <= max_along_gap
                        and (distance <= max_distance or shared_segment)
                    ):
                        visited.add(neighbor_id)
                        cluster.add(neighbor_id)
                        queue.append(neighbor_id)
            clusters.append(cluster)

        junctions: list[CaveJunction] = []
        for cluster in clusters:
            cluster_nodes = [node_lookup[node_id] for node_id in sorted(cluster)]
            segment_ids = sorted(
                {segment.segment_id for node_id in cluster for segment in adjacency[node_id]}
            )
            cluster_segments = [segments[segment_id] for segment_id in segment_ids]
            if any(segment.kind == "underpass" for segment in cluster_segments):
                kind = "crossing"
                split_style = "constant_envelope_then_divide"
                merge_style = "constant_envelope_then_divide"
                capacity_bias = 0.92
            elif any(node.kind == "chamber" for node in cluster_nodes) or any(
                segment.kind == "chamber_braid"
                or bool(segment.metadata.get("chamber_forming", False))
                for segment in cluster_segments
            ):
                kind = "chamber"
                split_style = "pre_widen_then_split"
                merge_style = "pre_widen_then_split"
                capacity_bias = 1.18
            elif any(
                segment.kind in {"island_bypass", "distributary", "anastomosis"}
                for segment in cluster_segments
            ):
                kind = "split_merge"
                split_style = "constant_envelope_then_divide"
                merge_style = "constant_envelope_then_divide"
                capacity_bias = 0.98
            else:
                kind = "junction"
                split_style = "constant_envelope_then_divide"
                merge_style = "constant_envelope_then_divide"
                capacity_bias = 1.0

            segment_widths = [segment.mean_width for segment in cluster_segments if segment.points]
            mean_width = float(np.mean(segment_widths)) if segment_widths else 24.0
            blend_length = 3.0 * mean_width
            pool_metadata: dict[str, SegmentMetadataValue] = {}
            for segment in cluster_segments:
                metadata = segment.metadata
                if metadata.get("chamber_type") != "drained_lava_pool":
                    continue
                pool_x = metadata.get("pool_center_x_m")
                pool_y = metadata.get("pool_center_y_m")
                if not isinstance(pool_x, (int, float)) or not isinstance(pool_y, (int, float)):
                    continue
                if not any(
                    math.hypot(node.x - float(pool_x), node.y - float(pool_y)) <= 1e-6
                    for node in cluster_nodes
                ):
                    continue
                pool_metadata = {
                    key: metadata[key]
                    for key in (
                        "pool_id",
                        "chamber_id",
                        "chamber_type",
                        "process_cause",
                        "pool_length_m",
                        "pool_width_m",
                        "pool_depth_m",
                        "pool_aspect_ratio",
                        "pool_outlet_ratio",
                        "pool_outlet_width_m",
                        "pool_inlet_count",
                        "pool_outlet_count",
                        "pool_center_along_m",
                        "pool_center_x_m",
                        "pool_center_y_m",
                    )
                    if key in metadata
                }
                break
            junctions.append(
                CaveJunction(
                    junction_id=len(junctions),
                    kind=kind,
                    node_ids=tuple(node.node_id for node in cluster_nodes),
                    segment_ids=tuple(segment_ids),
                    center_x=float(np.mean([node.x for node in cluster_nodes])),
                    center_y=float(np.mean([node.y for node in cluster_nodes])),
                    along_position=float(np.mean([node.along_position for node in cluster_nodes])),
                    blend_length=blend_length,
                    split_style=split_style,
                    merge_style=merge_style,
                    capacity_bias=capacity_bias,
                    metadata=pool_metadata,
                )
            )
        return junctions

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
        coordinates = [self._cell_to_world(host_field, cell) for cell in path_cells]
        coordinates = self._deduplicate_coordinates(coordinates)
        if len(coordinates) < 2:
            return None

        points: list[CavePoint] = []
        arc_length = 0.0
        for index, ((x_coord, y_coord), cell) in enumerate(
            zip(coordinates, path_cells, strict=False)
        ):
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
    ) -> tuple[int, ...]:
        if not nodes or not segments:
            return ()

        entries = [node for node in nodes if node.kind == "entry"]
        if not entries:
            entries = [min(nodes, key=lambda node: node.along_position)]
        exits = [node for node in nodes if node.kind == "exit"]
        exit_node = max(exits or nodes, key=lambda node: node.along_position)
        outgoing: defaultdict[int, list[CaveSegment]] = defaultdict(list)
        for segment in segments:
            outgoing[segment.start_node_id].append(segment)

        scores = {node.node_id: -math.inf for node in nodes}
        predecessor: dict[int, int] = {}
        entry_ids = {entry.node_id for entry in entries}
        for entry in entries:
            scores[entry.node_id] = 0.0
        for node_id in self._topological_node_ids(nodes, segments):
            if not math.isfinite(scores[node_id]):
                continue
            for segment in outgoing.get(node_id, []):
                # Integrated transported flux rewards both sustained flow and
                # route persistence without allowing a short late feeder to
                # masquerade as the main tube.
                next_score = scores[node_id] + segment.mean_flux * segment.total_length
                neighbor_id = segment.end_node_id
                if next_score <= scores[neighbor_id]:
                    continue
                scores[neighbor_id] = next_score
                predecessor[neighbor_id] = node_id

        if not math.isfinite(scores[exit_node.node_id]):
            return ()

        route = [exit_node.node_id]
        current = exit_node.node_id
        while current not in entry_ids:
            current = predecessor[current]
            route.append(current)
        route.reverse()
        return tuple(route)

    def _validate_generated_graph(
        self,
        nodes: list[CaveNode],
        segments: list[CaveSegment],
        dominant_route_node_ids: tuple[int, ...],
    ) -> None:
        """Reject graph/state defects before Stage B can be consumed."""

        node_ids = {node.node_id for node in nodes}
        entries = {node.node_id for node in nodes if node.kind == "entry"}
        exits = {node.node_id for node in nodes if node.kind == "exit"}
        if not entries:
            raise ValueError("Cave network has no entry nodes")
        if len(exits) != 1:
            raise ValueError(f"Cave network must have exactly one exit node, found {len(exits)}")

        outgoing: defaultdict[int, list[CaveSegment]] = defaultdict(list)
        reverse: defaultdict[int, set[int]] = defaultdict(set)
        for segment in segments:
            if segment.start_node_id not in node_ids or segment.end_node_id not in node_ids:
                raise ValueError(f"Segment {segment.segment_id} references an unknown node")
            if segment.start_node_id == segment.end_node_id:
                raise ValueError(f"Segment {segment.segment_id} is a self-loop")
            if len(segment.points) < 2:
                raise ValueError(f"Segment {segment.segment_id} has fewer than two points")
            arcs = np.asarray([point.arc_length for point in segment.points], dtype=float)
            if not np.isfinite(arcs).all() or np.any(np.diff(arcs) <= 0.0):
                raise ValueError(f"Segment {segment.segment_id} has invalid arc-length ordering")
            if not math.isclose(float(arcs[0]), 0.0, abs_tol=1e-8):
                raise ValueError(f"Segment {segment.segment_id} does not start at zero arc length")
            if segment.mean_flux <= 0.0:
                raise ValueError(f"Segment {segment.segment_id} has no source-reachable lava flux")
            temperatures = np.asarray(
                [point.temperature_k for point in segment.points],
                dtype=float,
            )
            ages = np.asarray([point.age_s for point in segment.points], dtype=float)
            if (
                not np.isfinite(temperatures).all()
                or not np.isfinite(ages).all()
                or np.any(np.diff(temperatures) > 1e-8)
                or np.any(np.diff(ages) < -1e-8)
            ):
                raise ValueError(f"Segment {segment.segment_id} has inconsistent thermal state")
            outgoing[segment.start_node_id].append(segment)
            reverse[segment.end_node_id].add(segment.start_node_id)

        reachable = set(entries)
        pending = list(entries)
        while pending:
            for segment in outgoing.get(pending.pop(), []):
                if segment.end_node_id not in reachable:
                    reachable.add(segment.end_node_id)
                    pending.append(segment.end_node_id)
        if reachable != node_ids:
            missing = sorted(node_ids - reachable)
            raise ValueError(f"Cave network contains source-unreachable nodes {missing}")

        exit_id = next(iter(exits))
        can_reach_exit = {exit_id}
        pending = [exit_id]
        while pending:
            for upstream in reverse.get(pending.pop(), set()):
                if upstream not in can_reach_exit:
                    can_reach_exit.add(upstream)
                    pending.append(upstream)
        blocked_entries = sorted(entries - can_reach_exit)
        if blocked_entries:
            raise ValueError(f"Cave network entries cannot reach the exit: {blocked_entries}")

        if (
            len(dominant_route_node_ids) < 2
            or dominant_route_node_ids[0] not in entries
            or dominant_route_node_ids[-1] != exit_id
        ):
            raise ValueError("Cave network has no valid directed dominant route")
        route_edges = {(segment.start_node_id, segment.end_node_id) for segment in segments}
        if any(
            pair not in route_edges
            for pair in zip(dominant_route_node_ids, dominant_route_node_ids[1:])
        ):
            raise ValueError("Dominant route references a missing directed segment")

        network = CaveNetwork(
            config=self.config,
            nodes=tuple(nodes),
            segments=tuple(segments),
            junctions=(),
            occupancy=np.empty((0, 0), dtype=bool),
            width_field=np.empty((0, 0), dtype=float),
            dominant_route_node_ids=dominant_route_node_ids,
            slice_along_positions=(),
            slice_channel_counts=(),
            slice_visible_channel_counts=(),
        )
        if network.max_flow_conservation_error() > 1e-8:
            raise ValueError("Cave network violates split/merge flux conservation")

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
            if (
                along_position < 0.18 * geometry.along_extent
                or along_position > 0.82 * geometry.along_extent
            ):
                continue
            score = float(total_flux[y_index, x_index])
            candidates.append(((int(y_index), int(x_index)), score))
        candidates.sort(key=lambda item: item[1], reverse=True)

        selected: list[tuple[int, int]] = []
        minimum_separation = 120.0
        for cell, _score in candidates:
            if any(
                math.hypot(cell[1] - other[1], cell[0] - other[0])
                < minimum_separation / max(geometry.cell_scale, 1.0)
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
        geometry: _FlowGeometry,
        segments: list[CaveSegment],
        include_passage_width: bool,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        along_positions = np.linspace(
            0.0,
            geometry.along_extent,
            self.config.channel_count_samples,
            dtype=float,
        )
        counts: list[int] = []
        for along_position in along_positions:
            crossings: list[tuple[float, float]] = []
            for segment in segments:
                for first, second in zip(segment.points, segment.points[1:]):
                    first_along = self._project_along(geometry, first.x, first.y)
                    second_along = self._project_along(geometry, second.x, second.y)
                    along_delta = second_along - first_along
                    if math.isclose(along_delta, 0.0, abs_tol=1e-9):
                        if not math.isclose(
                            along_position,
                            first_along,
                            abs_tol=0.25 * geometry.cell_scale,
                        ):
                            continue
                        interpolation = 0.5
                    else:
                        interpolation = (along_position - first_along) / along_delta
                        if interpolation < 0.0 or interpolation > 1.0:
                            continue
                    x_coord = first.x + interpolation * (second.x - first.x)
                    y_coord = first.y + interpolation * (second.y - first.y)
                    cross = self._project_cross(geometry, x_coord, y_coord)
                    width = first.width + interpolation * (second.width - first.width)
                    crossings.append((cross, max(width, 0.0)))

            if not crossings:
                counts.append(0)
                continue

            if include_passage_width:
                intervals = sorted(
                    (cross - 0.5 * width, cross + 0.5 * width) for cross, width in crossings
                )
                channel_count = 0
                current_end = -math.inf
                for start, end in intervals:
                    if start > current_end + 1e-6:
                        channel_count += 1
                        current_end = end
                    else:
                        current_end = max(current_end, end)
            else:
                lateral_positions = sorted(cross for cross, _width in crossings)
                separation = max(0.25 * geometry.cell_scale, 1e-6)
                channel_count = 1
                previous = lateral_positions[0]
                for current in lateral_positions[1:]:
                    if current - previous > separation:
                        channel_count += 1
                    previous = current
            counts.append(channel_count)
        return tuple(float(value) for value in along_positions), tuple(
            int(value) for value in counts
        )

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
        representative_radius = self.config.chamber_radius * self.config.chamber_radius_fraction
        painted_pool_nodes: set[int] = set()
        for segment in segments:
            metadata = segment.metadata
            if metadata.get("chamber_type") != "drained_lava_pool" or len(segment.points) < 2:
                continue
            pool_node_id = next(
                (
                    node_id
                    for node_id in (segment.start_node_id, segment.end_node_id)
                    if nodes[node_id].kind == "chamber"
                ),
                None,
            )
            if pool_node_id is None or pool_node_id in painted_pool_nodes:
                continue
            pool_node = nodes[pool_node_id]
            if pool_node_id == segment.start_node_id:
                direction = np.asarray(
                    (
                        segment.points[1].x - segment.points[0].x,
                        segment.points[1].y - segment.points[0].y,
                    ),
                    dtype=float,
                )
            else:
                direction = np.asarray(
                    (
                        segment.points[-1].x - segment.points[-2].x,
                        segment.points[-1].y - segment.points[-2].y,
                    ),
                    dtype=float,
                )
            length = float(metadata.get("pool_length_m", 0.0) or 0.0)
            width = float(metadata.get("pool_width_m", 0.0) or 0.0)
            self._paint_ellipse(
                host_field=host_field,
                occupancy=occupancy,
                width_field=width_field,
                x_coord=pool_node.x,
                y_coord=pool_node.y,
                direction=direction,
                length=max(length, width),
                width=max(width, 2.0 * segment.mean_width),
            )
            painted_pool_nodes.add(pool_node_id)
        for node in nodes:
            if node.kind != "chamber":
                continue
            incident_scales: list[float] = []
            for segment in segments:
                if node.node_id not in {
                    segment.start_node_id,
                    segment.end_node_id,
                }:
                    continue
                scale_value = segment.metadata.get("chamber_radius_scale", 1.0)
                if isinstance(scale_value, (int, float)):
                    incident_scales.append(float(scale_value))
            self._paint_disk(
                host_field=host_field,
                occupancy=occupancy,
                width_field=width_field,
                x_coord=node.x,
                y_coord=node.y,
                radius=representative_radius * max(incident_scales, default=1.0),
            )
        for segment in segments:
            is_process_chamber = bool(segment.metadata.get("chamber_forming", False))
            is_drained_pool = segment.metadata.get("chamber_type") == "drained_lava_pool"
            if (segment.kind not in {"chamber_braid", "ladder"} and not is_process_chamber) or len(
                segment.points
            ) < 3:
                continue
            if is_drained_pool:
                continue
            midpoint = segment.points[len(segment.points) // 2]
            scale_value = segment.metadata.get("chamber_radius_scale", 1.0)
            chamber_scale = float(scale_value) if isinstance(scale_value, (int, float)) else 1.0
            radius = (
                representative_radius
                * (0.90 if segment.kind == "chamber_braid" or is_process_chamber else 0.62)
                * chamber_scale
            )
            self._paint_disk(
                host_field=host_field,
                occupancy=occupancy,
                width_field=width_field,
                x_coord=midpoint.x,
                y_coord=midpoint.y,
                radius=radius,
            )

    def _paint_ellipse(
        self,
        *,
        host_field: HostField,
        occupancy: np.ndarray,
        width_field: np.ndarray,
        x_coord: float,
        y_coord: float,
        direction: np.ndarray,
        length: float,
        width: float,
    ) -> None:
        """Paint one flow-aligned, horizontally broad room footprint."""
        norm = float(np.linalg.norm(direction))
        along = direction / max(norm, 1e-9)
        across = np.asarray((-along[1], along[0]), dtype=float)
        x_grid, y_grid = np.meshgrid(host_field.x_coords, host_field.y_coords)
        delta_x = x_grid - x_coord
        delta_y = y_grid - y_coord
        along_distance = delta_x * along[0] + delta_y * along[1]
        across_distance = delta_x * across[0] + delta_y * across[1]
        normalized_radius = np.square(along_distance / max(0.5 * length, 1e-9)) + np.square(
            across_distance / max(0.5 * width, 1e-9)
        )
        mask = normalized_radius <= 1.0
        occupancy[mask] = True
        width_field[mask] = np.maximum(width_field[mask], width)

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
        return float(
            np.clip(
                radius,
                self.config.minimum_passage_radius,
                self.config.maximum_passage_radius,
            )
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
        transition_cost += (
            step_length * 0.85 * max(0.0, -along_delta / max(geometry.cell_scale, 1.0))
        )
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
    def _sample_branch_kind(rng) -> str:
        kinds = ("island_bypass", "chamber_braid", "inner_bypass")
        probabilities = (0.48, 0.34, 0.18)
        return str(rng.choice(kinds, p=probabilities))

    @staticmethod
    def _sample_float_range(rng, value_range: tuple[float, float]) -> float:
        minimum, maximum = value_range
        if minimum > maximum:
            raise ValueError(f"Invalid range with min > max: {value_range!r}")
        return float(rng.uniform(minimum, maximum))

    @staticmethod
    def _sample_int_range(rng, value_range: tuple[int, int]) -> int:
        minimum, maximum = value_range
        if minimum > maximum:
            raise ValueError(f"Invalid range with min > max: {value_range!r}")
        return int(rng.integers(minimum, maximum + 1))

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
            if (
                deduplicated
                and math.isclose(x_coord, deduplicated[-1][0])
                and math.isclose(y_coord, deduplicated[-1][1])
            ):
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

    def _world_to_cell(
        self, host_field: HostField, x_coord: float, y_coord: float
    ) -> tuple[int, int]:
        return (
            self._coordinate_to_index(host_field.y_coords, y_coord),
            self._coordinate_to_index(host_field.x_coords, x_coord),
        )

    @staticmethod
    def _cell_to_world(host_field: HostField, cell: tuple[int, int]) -> tuple[float, float]:
        y_index, x_index = cell
        return float(host_field.x_coords[x_index]), float(host_field.y_coords[y_index])

    @staticmethod
    def _project_along(geometry: _FlowGeometry, x_coord: float, y_coord: float) -> float:
        return (x_coord - geometry.seed_x) * geometry.flow_x + (
            y_coord - geometry.seed_y
        ) * geometry.flow_y

    @staticmethod
    def _project_cross(geometry: _FlowGeometry, x_coord: float, y_coord: float) -> float:
        return (x_coord - geometry.seed_x) * geometry.cross_x + (
            y_coord - geometry.seed_y
        ) * geometry.cross_y
