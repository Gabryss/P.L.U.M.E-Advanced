"""Geometry-only diagnostics for trunk and branch networks.

These routines consume Stage-B dataclasses, plain dictionaries, or lightweight
fixtures. They never mutate a generator or infer a pass/fail decision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from ._common import arc_lengths, as_xy, elevations, finite, point_sequence, summary, value


def _paths(network: Any) -> list[tuple[str, Any]]:
    """Normalize a TrunkGraph, BranchMergeNetwork, or path iterable."""

    trunk = value(network, "trunk_graph", "trunk", default=None)
    branches = value(network, "branches", default=None)
    if trunk is not None:
        result = [("trunk", trunk)]
        for branch in branches or ():
            result.append((str(value(branch, "branch_kind", "kind", default="branch")), branch))
        return result
    if value(network, "points", "samples", default=None) is not None:
        return [(str(value(network, "branch_kind", "kind", default="trunk")), network)]
    segments = value(network, "segments", default=None)
    if segments is not None:
        return [(str(value(segment, "kind", default="segment")), segment) for segment in segments]
    if isinstance(network, Mapping):
        candidate = network.get("paths", network.get("branches", ()))
    else:
        candidate = network or ()
    result = []
    for item in candidate:
        result.append((str(value(item, "branch_kind", "kind", default="branch")), item))
    return result


def _path_record(kind: str, path: Any) -> dict[str, Any]:
    points = point_sequence(path)
    arcs = arc_lengths(points)
    coordinates = [as_xy(point) for point in points]
    geometric_length = sum(
        math.dist(first, second)
        for first, second in zip(coordinates, coordinates[1:])
        if first is not None and second is not None
    )
    length = geometric_length if geometric_length > 0 else (arcs[-1] if arcs else 0.0)
    displacement = (
        math.dist(coordinates[0], coordinates[-1])
        if len(coordinates) > 1 and coordinates[0] and coordinates[-1]
        else 0.0
    )
    # A closed path has no meaningful chord. Returning a finite value keeps
    # JSON standards-compliant while making closure apparent to callers.
    sinuosity = length / displacement if displacement > 1.0e-12 else 0.0
    return {
        "kind": kind,
        "branch_id": value(path, "branch_id", default=None),
        "source_trunk_index": value(path, "source_trunk_index", default=None),
        "target_trunk_index": value(path, "target_trunk_index", default=None),
        "point_count": len(points),
        "length": length,
        "chord_length": displacement,
        "sinuosity": sinuosity,
        "is_closed": bool(len(coordinates) > 1 and displacement <= 1.0e-12),
        "uphill": sustained_uphill_diagnostics(path),
    }


def network_sinuosity_statistics(network_or_paths: Any) -> dict[str, Any]:
    """Return per-kind and overall sinuosity quantiles.

    Values are computed from sampled XY coordinates, making this independent
    of a generator's nominal step length. Path ordering does not affect output.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for kind, path in _paths(network_or_paths):
        record = _path_record(kind, path)
        # Closed paths are represented separately rather than as infinity.
        if record["sinuosity"] > 0.0:
            grouped[kind].append(float(record["sinuosity"]))
    by_kind = {kind: summary(grouped[kind]) for kind in sorted(grouped)}
    return {
        "by_kind": by_kind,
        "all": summary(value_ for values in grouped.values() for value_ in values),
    }


def sustained_uphill_diagnostics(path_or_points: Any, *, elevation_tolerance: float = 1.0e-9) -> dict[str, Any]:
    """Measure uphill runs using supplied or XY-derived arc length.

    A run is a maximal sequence of positive elevation changes. The report is
    deliberately descriptive: no threshold is encoded as a quality verdict.
    """

    points = point_sequence(path_or_points)
    arcs = arc_lengths(points)
    z_values = elevations(points)
    segments: list[tuple[float, float]] = []
    for index in range(1, len(points)):
        first, second = z_values[index - 1], z_values[index]
        ds = max(arcs[index] - arcs[index - 1], 0.0)
        if first is None or second is None or ds <= 0:
            continue
        segments.append((ds, second - first))

    uphill_length = 0.0
    uphill_rise = 0.0
    max_grade = 0.0
    runs: list[dict[str, float]] = []
    current_length = 0.0
    current_rise = 0.0
    for ds, dz in segments:
        if dz > elevation_tolerance:
            uphill_length += ds
            uphill_rise += dz
            max_grade = max(max_grade, dz / ds)
            current_length += ds
            current_rise += dz
        elif current_length > 0.0:
            runs.append({"length": current_length, "rise": current_rise})
            current_length, current_rise = 0.0, 0.0
    if current_length > 0.0:
        runs.append({"length": current_length, "rise": current_rise})

    total_length = arcs[-1] if arcs else 0.0
    return {
        "segment_count": len(segments),
        "total_length": total_length,
        "uphill_length": uphill_length,
        "uphill_fraction": uphill_length / total_length if total_length > 0 else 0.0,
        "uphill_rise": uphill_rise,
        "max_uphill_grade": max_grade,
        "sustained_run_count": len(runs),
        "max_sustained_uphill_length": max((run["length"] for run in runs), default=0.0),
        "runs": runs,
    }


def _topology(records: list[dict[str, Any]], network: Any) -> dict[str, Any]:
    branches = [record for record in records if record["kind"] != "trunk"]
    merged = sum(1 for record in branches if record["target_trunk_index"] is not None)
    loops = sum(1 for record in branches if record["is_closed"] or record["target_trunk_index"] is not None)
    node_count = sum(record["point_count"] for record in records)
    edge_count = sum(max(record["point_count"] - 1, 0) for record in records)
    kind_counts: dict[str, int] = defaultdict(int)
    for record in records:
        kind_counts[record["kind"]] += 1
    denominator = max(len(branches), 1)
    generator_summary = value(network, "summary", default=None) if network is not None else None
    if callable(generator_summary):
        generator_summary = generator_summary()
    return {
        "path_count": len(records),
        "branch_count": len(branches),
        "node_count": node_count,
        "edge_count": edge_count,
        "edge_to_node_ratio": edge_count / node_count if node_count else 0.0,
        "edges_per_branch": edge_count / denominator,
        "merged_branch_count": merged,
        "merge_fraction": merged / denominator,
        "loop_count": loops,
        "loop_fraction": loops / denominator,
        "kind_counts": dict(sorted(kind_counts.items())),
        "kind_fractions": {
            kind: count / max(len(records), 1) for kind, count in sorted(kind_counts.items())
        },
        "generator_summary": generator_summary,
    }


def network_diagnostics(network: Any) -> dict[str, Any]:
    """Build the complete machine-readable network diagnostics report."""

    records = [_path_record(kind, path) for kind, path in _paths(network)]
    # Stable sort makes reports invariant to branch container ordering.
    records.sort(key=lambda item: (item["kind"], item["branch_id"] is None, str(item["branch_id"])))
    grouped_uphill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_uphill[record["kind"]].append(record["uphill"])
    return {
        "schema_version": "1.0",
        "paths": records,
        "sinuosity": network_sinuosity_statistics(network),
        "uphill": {"by_kind": {kind: diagnostics_summary(grouped_uphill[kind]) for kind in sorted(grouped_uphill)}},
        "topology": _topology(records, network),
    }


def diagnostics_summary(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    reports = list(reports)
    fields = ("uphill_length", "uphill_fraction", "uphill_rise", "max_uphill_grade", "max_sustained_uphill_length")
    return {field: summary(float(report[field]) for report in reports) for field in fields}


compute_network_metrics = network_diagnostics
evaluate_network = network_diagnostics
