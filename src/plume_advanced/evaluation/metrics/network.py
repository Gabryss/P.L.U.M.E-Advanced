"""Topology, state-consistency, and host-exposure metrics for Stage B."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from plume_advanced.stages.host_field import HostField
from plume_advanced.stages.network import CaveNetwork, CaveSegment

PRIMARY_BRANCH_KINDS = {
    "anastomosis",
    "chamber_braid",
    "distributary",
    "inner_bypass",
    "island_bypass",
    "underpass",
}


def network_metrics(
    network: CaveNetwork,
    host_field: HostField | None = None,
    *,
    conservation_tolerance: float = 1e-6,
) -> dict[str, Any]:
    degrees = {node.node_id: 0 for node in network.nodes}
    incoming: defaultdict[int, list[CaveSegment]] = defaultdict(list)
    outgoing: defaultdict[int, list[CaveSegment]] = defaultdict(list)
    for segment in network.segments:
        degrees[segment.start_node_id] += 1
        degrees[segment.end_node_id] += 1
        outgoing[segment.start_node_id].append(segment)
        incoming[segment.end_node_id].append(segment)
    components = _component_count(network)
    persistence = [
        segment.total_length / max(segment.mean_width, 1e-9)
        for segment in network.segments
        if segment.kind in PRIMARY_BRANCH_KINDS and segment.points
    ]
    sinuosities = np.asarray([_sinuosity(segment) for segment in network.segments], dtype=float)
    lengths = np.asarray([segment.total_length for segment in network.segments], dtype=float)
    state = _state_consistency(network, incoming, outgoing, conservation_tolerance)
    split_nodes = [node_id for node_id in degrees if len(outgoing[node_id]) > 1]
    merge_nodes = [node_id for node_id in degrees if len(incoming[node_id]) > 1]
    entry_ids = {node.node_id for node in network.nodes if node.kind == "entry"}
    exit_ids = {node.node_id for node in network.nodes if node.kind == "exit"}
    source_reachable = set(entry_ids)
    pending = list(entry_ids)
    while pending:
        for segment in outgoing[pending.pop()]:
            if segment.end_node_id not in source_reachable:
                source_reachable.add(segment.end_node_id)
                pending.append(segment.end_node_id)
    can_reach_exit = set(exit_ids)
    pending = list(exit_ids)
    while pending:
        node_id = pending.pop()
        for segment in incoming[node_id]:
            if segment.start_node_id not in can_reach_exit:
                can_reach_exit.add(segment.start_node_id)
                pending.append(segment.start_node_id)
    island_ids = {
        str(segment.metadata.get("island_id"))
        for segment in network.segments
        if segment.metadata.get("island_id") is not None
    }
    lobe_path_ids = {
        str(segment.metadata.get("lobe_path_id"))
        for segment in network.segments
        if segment.metadata.get("lobe_path_id") is not None
    }
    roof_states = Counter(
        str(segment.metadata.get("roof_state", "unspecified"))
        for segment in network.segments
    )
    phase_counts = [
        int(value)
        for segment in network.segments
        if isinstance(
            (value := segment.metadata.get("emplacement_phase_count")),
            (int, float),
        )
    ]
    report: dict[str, Any] = {
        "node_count": len(network.nodes),
        "edge_count": len(network.segments),
        "network_density": network.config.network_density,
        "lobe_path_count": len(lobe_path_ids),
        "anastomosis_count": sum(
            segment.kind == "anastomosis" for segment in network.segments
        ),
        "emplacement_phase_count": max(phase_counts, default=1),
        "stacked_segment_count": sum(
            segment.z_level != 0 for segment in network.segments
        ),
        "vertical_capture_count": sum(
            bool(segment.metadata.get("vertical_capture", False))
            for segment in network.segments
        ),
        "process_chamber_segment_count": sum(
            bool(segment.metadata.get("chamber_forming", False))
            for segment in network.segments
        ),
        "roof_state_histogram": dict(sorted(roof_states.items())),
        "retired_lobe_count": sum(
            segment.kind in {"abandoned_lobe", "stalled_lobe"}
            for segment in network.segments
        ),
        "total_centerline_length_m": float(np.sum(lengths)),
        "main_route_length_m": float(network.dominant_route_length),
        "source_count": sum(node.kind == "entry" for node in network.nodes),
        "terminal_count": sum(degree == 1 for degree in degrees.values()),
        "chamber_count": sum(junction.kind == "chamber" for junction in network.junctions),
        "connected_component_count": components,
        "source_unreachable_node_count": len(set(degrees) - source_reachable),
        "entries_without_exit_path_count": len(entry_ids - can_reach_exit),
        "zero_flux_segment_count": sum(segment.mean_flux <= 0.0 for segment in network.segments),
        "cyclomatic_number": max(0, len(network.segments) - len(network.nodes) + components),
        "split_junction_count": len(split_nodes),
        "merge_junction_count": len(merge_nodes),
        "split_merge_region_count": sum(
            junction.kind in {"braid", "split_merge", "chamber"}
            or ("split" in junction.kind and "merge" in junction.kind)
            for junction in network.junctions
        ),
        "node_degree_histogram": {
            str(degree): count for degree, count in sorted(Counter(degrees.values()).items())
        },
        "mean_branch_factor": float(np.mean([len(outgoing[node]) for node in split_nodes]))
        if split_nodes
        else 0.0,
        "max_branch_factor": max((len(outgoing[node]) for node in split_nodes), default=0),
        "independent_braid_or_island_count": len(island_ids | lobe_path_ids),
        "underpass_count": sum(segment.kind == "underpass" for segment in network.segments),
        "distinct_z_level_count": len({segment.z_level for segment in network.segments}),
        "vertically_overlapping_xy_passage_count": _vertical_overlap_count(network),
        "length_weighted_mean_sinuosity": float(np.average(sinuosities, weights=lengths))
        if lengths.size and float(np.sum(lengths)) > 0.0
        else 1.0,
        "sinuosity_p95": float(np.percentile(sinuosities, 95.0)) if sinuosities.size else 1.0,
        "branch_persistence_length_median_m": float(
            np.median(
                [
                    segment.total_length
                    for segment in network.segments
                    if segment.kind in PRIMARY_BRANCH_KINDS
                ]
            )
        )
        if persistence
        else 0.0,
        "branch_persistence_diameters_median": float(np.median(persistence))
        if persistence
        else 0.0,
        "vertical_separation_at_underpasses_m": _underpass_separations(network),
        **state,
    }
    if host_field is not None:
        report["host_exposure"] = host_exposure(network, host_field)
    return report


def host_exposure(network: CaveNetwork, host_field: HostField) -> dict[str, float]:
    names = ("slope", "cover", "fracture", "capacity", "stability", "total")
    totals = {name: 0.0 for name in names}
    raw_totals = {
        name: 0.0
        for name in (
            "slope_degrees",
            "cover_thickness",
            "fracture_intensity",
            "flow_capacity",
            "roof_stability",
        )
    }
    total_length = 0.0
    fields = {
        "slope": host_field.routing_slope_penalty,
        "cover": host_field.routing_cover_penalty,
        "fracture": host_field.routing_fracture_penalty,
        "capacity": host_field.routing_capacity_penalty,
        "stability": host_field.routing_stability_penalty,
        "total": host_field.routing_cost,
    }
    raw_fields = {
        "slope_degrees": host_field.slope_degrees,
        "cover_thickness": host_field.cover_thickness,
        "fracture_intensity": host_field.fracture_intensity,
        "flow_capacity": host_field.flow_capacity,
        "roof_stability": host_field.roof_stability,
    }
    for segment in network.segments:
        for first, second in zip(segment.points, segment.points[1:]):
            weight = max(second.arc_length - first.arc_length, 0.0)
            if weight <= 0.0:
                continue
            x_coord = 0.5 * (first.x + second.x)
            y_coord = 0.5 * (first.y + second.y)
            if not host_field.contains(x_coord, y_coord):
                continue
            for name, values in fields.items():
                totals[name] += weight * host_field._bilinear_sample(values, x_coord, y_coord)
            for name, values in raw_fields.items():
                raw_totals[name] += weight * host_field._bilinear_sample(values, x_coord, y_coord)
            total_length += weight
    denominator = max(total_length, 1e-12)
    return {
        **{f"weighted_{name}_penalty": value / denominator for name, value in totals.items()},
        **{f"weighted_raw_{name}": value / denominator for name, value in raw_totals.items()},
        "sampled_length_m": total_length,
    }


def symmetric_centerline_distance(
    first: CaveNetwork,
    second: CaveNetwork,
    *,
    spacing_m: float = 5.0,
) -> dict[str, float]:
    first_points = _sample_centerlines(first, spacing_m)
    second_points = _sample_centerlines(second, spacing_m)
    if not first_points.size or not second_points.size:
        return {"mean_m": float("nan"), "p95_m": float("nan")}
    first_distance = cKDTree(first_points).query(second_points, workers=1)[0]
    second_distance = cKDTree(second_points).query(first_points, workers=1)[0]
    distances = np.concatenate((first_distance, second_distance))
    return {
        "mean_m": float(np.mean(distances)),
        "p95_m": float(np.percentile(distances, 95.0)),
    }


def _state_consistency(
    network: CaveNetwork,
    incoming: dict[int, list[CaveSegment]],
    outgoing: dict[int, list[CaveSegment]],
    tolerance: float,
) -> dict[str, Any]:
    absolute: list[float] = []
    relative: list[float] = []
    exceeded = 0
    for node in network.nodes:
        if node.kind in {"entry", "exit", "terminal", "spur_terminal"}:
            continue
        if not incoming[node.node_id] or not outgoing[node.node_id]:
            continue
        in_flux = sum(max(segment.mean_flux, 0.0) for segment in incoming[node.node_id])
        out_flux = sum(max(segment.mean_flux, 0.0) for segment in outgoing[node.node_id])
        residual = abs(in_flux - out_flux)
        normalized = residual / max(in_flux, out_flux, 1e-12)
        absolute.append(residual)
        relative.append(normalized)
        exceeded += normalized > tolerance
    temperature_violations = 0
    age_violations = 0
    for segment in network.segments:
        temperatures = np.asarray([point.temperature_k for point in segment.points], dtype=float)
        ages = np.asarray([point.age_s for point in segment.points], dtype=float)
        if temperatures.size > 1:
            temperature_violations += int(np.count_nonzero(np.diff(temperatures) > tolerance))
            age_violations += int(np.count_nonzero(np.diff(ages) < -tolerance))
    return {
        "max_absolute_flux_conservation_residual": max(absolute, default=0.0),
        "max_relative_flux_conservation_residual": max(relative, default=0.0),
        "split_merge_nodes_exceeding_tolerance": exceeded,
        "temperature_monotonicity_violation_count": temperature_violations,
        "lava_age_monotonicity_violation_count": age_violations,
    }


def _component_count(network: CaveNetwork) -> int:
    adjacency: dict[int, set[int]] = {node.node_id: set() for node in network.nodes}
    for segment in network.segments:
        adjacency[segment.start_node_id].add(segment.end_node_id)
        adjacency[segment.end_node_id].add(segment.start_node_id)
    remaining = set(adjacency)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            neighbors = adjacency[stack.pop()] & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
    return count


def _sinuosity(segment: CaveSegment) -> float:
    if len(segment.points) < 2:
        return 1.0
    first, last = segment.points[0], segment.points[-1]
    # Stage-B arc length is planar; use the matching planar endpoint chord.
    # Mixing it with surface elevation could produce an impossible value < 1.
    chord = math.hypot(last.x - first.x, last.y - first.y)
    return segment.total_length / max(chord, 1e-9)


def _vertical_overlap_count(network: CaveNetwork) -> int:
    count = 0
    for index, first in enumerate(network.segments):
        if not first.points:
            continue
        first_xy = np.asarray([(point.x, point.y) for point in first.points])
        first_tree = cKDTree(first_xy)
        for second in network.segments[index + 1 :]:
            if first.z_level == second.z_level or not second.points:
                continue
            second_xy = np.asarray([(point.x, point.y) for point in second.points])
            distance = float(np.min(first_tree.query(second_xy, workers=1)[0]))
            if distance <= 0.5 * (first.mean_width + second.mean_width):
                count += 1
    return count


def _underpass_separations(network: CaveNetwork) -> list[float]:
    separations: list[float] = []
    for segment in network.segments:
        if segment.kind != "underpass" or not segment.points:
            continue
        value = segment.metadata.get("vertical_separation_m")
        if isinstance(value, (int, float)):
            separations.append(float(value))
        else:
            separations.append(float(abs(segment.z_level) * segment.mean_width))
    return separations


def _sample_centerlines(network: CaveNetwork, spacing_m: float) -> np.ndarray:
    sampled: list[tuple[float, float, float]] = []
    for segment in network.segments:
        if not segment.points:
            continue
        count = max(1, int(np.ceil(segment.total_length / spacing_m)))
        arcs = np.linspace(0.0, segment.total_length, count + 1)
        point_arcs = np.asarray([point.arc_length for point in segment.points], dtype=float)
        for arc in arcs:
            sampled.append(
                (
                    float(np.interp(arc, point_arcs, [point.x for point in segment.points])),
                    float(np.interp(arc, point_arcs, [point.y for point in segment.points])),
                    float(
                        np.interp(arc, point_arcs, [point.elevation for point in segment.points])
                    ),
                )
            )
    return np.asarray(sampled, dtype=float)


__all__ = ["host_exposure", "network_metrics", "symmetric_centerline_distance"]
