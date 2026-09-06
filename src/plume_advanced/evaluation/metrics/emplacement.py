"""Diagnostics for staged emplacement history metadata.

The generator has evolved from a single-history representation toward richer
phase and lobe records.  This module deliberately treats those fields as
optional: old ``CaveNetwork`` objects still produce a complete report with
``available`` flags and zero-safe values rather than fabricated history.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from plume_advanced.stages.network import CaveNetwork, CaveSegment

_PHASE_KEYS = ("emplacement_phase_count", "phase_count", "phases")
_BIRTH_KEYS = ("birth_phase", "emplacement_birth_phase", "phase_birth")
_DEATH_KEYS = ("death_phase", "emplacement_death_phase", "phase_death")
_STATE_KEYS = ("formation_state", "termination", "outcome", "status")
_PATH_KEYS = ("lobe_path_id", "path_id", "route_id", "branch_id")
_NEW_KEYS = ("new_path", "is_new_path", "path_is_new", "new_route")
_REUSE_KEYS = (
    "reoccupied_path",
    "is_reoccupied",
    "reoccupied",
    "reused_path",
    "path_reuse",
    "path_reoccupation_count",
    "reoccupied_fraction",
)
_ORDER_KEYS = ("branch_order", "branch_depth", "hierarchy_level", "generation", "order")
_LOOP_KEYS = ("loop_mechanism", "loop_type", "loop_kind", "loop_reason", "reconnect_mechanism")
_CROSSING_KEYS = ("crossing_group_id", "crossing_id", "cross_group")
_COALESCE_KEYS = ("coalescence_id", "merge_id", "coalescence_group_id", "chamber_id")


def _first(metadata: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return default


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "new", "reoccupied", "reused"}:
            return True
        if normalized in {"false", "no", "n", "0", "old", "original"}:
            return False
    return None


def _path_id(segment: CaveSegment, index: int) -> str:
    value = _first(segment.metadata, _PATH_KEYS)
    return str(value) if value is not None else f"segment_{segment.segment_id}_{index}"


def _phase_count(network: CaveNetwork, metadata: Iterable[dict[str, Any]]) -> tuple[int, bool]:
    values = [_number(_first(item, _PHASE_KEYS)) for item in metadata]
    finite = [int(round(value)) for value in values if value is not None and value >= 1.0]
    if finite:
        return max(finite), True
    # A configured phase range is not evidence that a legacy object carries
    # per-path history.  Keep the unavailable baseline to one neutral phase
    # instead of projecting an invented multi-phase timeline.
    return 1, False


def _phase_bounds(metadata: dict[str, Any], phase_count: int) -> tuple[int, int]:
    birth_value = _number(_first(metadata, _BIRTH_KEYS, 0.0))
    death_value = _number(_first(metadata, _DEATH_KEYS, float(phase_count - 1)))
    birth = max(0, min(phase_count - 1, int(round(birth_value if birth_value is not None else 0.0))))
    death = max(birth, min(phase_count - 1, int(round(death_value if death_value is not None else phase_count - 1))))
    return birth, death


def _path_records(network: CaveNetwork, phase_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(network.segments):
        metadata = dict(segment.metadata or {})
        key = _path_id(segment, index)
        record = grouped.setdefault(
            key,
            {
                "path_id": key,
                "metadata": metadata,
                "segments": [],
                "length_m": 0.0,
                "mean_flux": [],
            },
        )
        record["segments"].append(segment)
        record["length_m"] += max(float(segment.total_length), 0.0)
        record["mean_flux"].append(float(segment.mean_flux))
        # Segment metadata should agree within a path; retain the first
        # record deterministically, but fill missing values from later pieces.
        for key_name, value in metadata.items():
            if record["metadata"].get(key_name) is None and value is not None:
                record["metadata"][key_name] = value
    for record in grouped.values():
        metadata = record["metadata"]
        birth, death = _phase_bounds(metadata, phase_count)
        record["birth_phase"] = birth
        record["death_phase"] = death
        record["mean_flux"] = float(np.mean(record["mean_flux"])) if record["mean_flux"] else 0.0
    return [grouped[key] for key in sorted(grouped)]


def _outcome(metadata: dict[str, Any], kind: str, death: int, phase_count: int) -> str:
    state = str(_first(metadata, _STATE_KEYS, "")).strip().lower()
    if kind == "abandoned_lobe" or state in {"abandoned", "thermally_abandoned", "retired"}:
        return "retired"
    if kind == "stalled_lobe" or state in {"stalled", "stranded", "cooled_or_stranded"}:
        return "stalled"
    returned = _number(metadata.get("coalescence_returned_flux", 0.0)) or 0.0
    if state in {"coalesced", "vertically_captured", "merged", "coalescence"} or returned > 0.0:
        return "coalesced"
    if state in {"persistent", "persistent_arterial", "persistent_feeder", "survived", "active"}:
        return "survived"
    if death < phase_count - 1:
        return "retired"
    return "survived"


def _explicit_bool(metadata: dict[str, Any], keys: Iterable[str]) -> bool | None:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        parsed = _truth(value)
        if parsed is not None:
            return parsed
        numeric = _number(value)
        if numeric is not None and key.endswith("fraction"):
            return numeric > 0.0
    return None


def _phase_activity(records: list[dict[str, Any]], phase_count: int) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for phase in range(phase_count):
        active = [record for record in records if record["birth_phase"] <= phase <= record["death_phase"]]
        allocated = sum(
            (_number(record["metadata"].get("initial_flux")) or 0.0)
            for record in records
            if record["birth_phase"] == phase
        )
        returned = sum(
            (_number(record["metadata"].get("coalescence_returned_flux")) or 0.0)
            for record in records
            if record["birth_phase"] == phase
        )
        parent_before = sum(
            (_number(record["metadata"].get("parent_flux_before_split")) or 0.0)
            for record in records
            if record["birth_phase"] == phase
        )
        active_flux = sum(max(float(record["mean_flux"]), 0.0) for record in active)
        phases.append(
            {
                "phase": phase,
                "active_path_count": len(active),
                "active_segment_count": sum(len(record["segments"]) for record in active),
                "active_length_m": float(sum(record["length_m"] for record in active)),
                "active_mean_flux": float(active_flux / len(active)) if active else 0.0,
                "allocated_flux": float(allocated),
                "returned_flux": float(returned),
                "net_allocated_flux": float(max(allocated - returned, 0.0)),
                "budget_utilization": (
                    float(allocated / parent_before) if parent_before > 0.0 else None
                ),
            }
        )
    return phases


def _longitudinal_diagnostics(network: CaveNetwork) -> dict[str, Any]:
    reversals = 0
    steps = 0
    uphill_steps = 0
    grades: list[float] = []
    slopes: list[float] = []
    invalid_slopes = 0
    previous_sign: int | None = None
    for segment in network.segments:
        for point in segment.points:
            slope = _number(point.slope_degrees)
            if slope is None or slope < 0.0 or slope > 90.0:
                invalid_slopes += 1
            else:
                slopes.append(slope)
        for first, second in zip(segment.points, segment.points[1:]):
            ds = math.hypot(second.x - first.x, second.y - first.y)
            if ds <= 1e-9:
                ds = max(float(second.arc_length - first.arc_length), 0.0)
            if ds <= 1e-9:
                continue
            dz = float(second.elevation - first.elevation)
            sign = 1 if dz > 1e-9 else -1 if dz < -1e-9 else 0
            if sign:
                if previous_sign is not None and sign != previous_sign:
                    reversals += 1
                previous_sign = sign
            steps += 1
            uphill_steps += sign > 0
            grades.append(dz / ds)
    available = steps > 0 or bool(slopes)
    return {
        "available": available,
        "longitudinal_step_count": steps,
        "elevation_reversal_count": reversals,
        "elevation_reversal_fraction": reversals / max(steps - 1, 1) if steps else None,
        "uphill_step_fraction": uphill_steps / steps if steps else None,
        "grade_q95": float(np.percentile(np.abs(grades), 95.0)) if grades else None,
        "grade_max": float(np.max(np.abs(grades))) if grades else None,
        "slope_degrees_median": float(np.median(slopes)) if slopes else None,
        "slope_degrees_q95": float(np.percentile(slopes, 95.0)) if slopes else None,
        "invalid_slope_point_count": invalid_slopes,
        "slope_plausibility": bool(available and invalid_slopes == 0),
    }


def emplacement_metrics(network: CaveNetwork) -> dict[str, Any]:
    """Return tolerant phase, lobe, flux, and longitudinal diagnostics."""

    metadata = [dict(segment.metadata or {}) for segment in network.segments]
    phase_count, phase_metadata_available = _phase_count(network, metadata)
    records = _path_records(network, phase_count)
    history_available = any(
        any(key in item for key in (_PHASE_KEYS + _BIRTH_KEYS + _DEATH_KEYS + _STATE_KEYS))
        for item in metadata
    )

    outcomes = Counter()
    outcome_available = False
    new_count = 0
    reoccupied_count = 0
    explicit_path_classification = False
    for record in records:
        item = record["metadata"]
        kinds = {segment.kind for segment in record["segments"]}
        outcome = _outcome(item, next(iter(sorted(kinds)), ""), record["death_phase"], phase_count)
        if any(key in item for key in _STATE_KEYS) or kinds & {"abandoned_lobe", "stalled_lobe", "anastomosis"}:
            outcome_available = True
            outcomes[outcome] += 1
        is_new = _explicit_bool(item, _NEW_KEYS)
        is_reused = _explicit_bool(item, _REUSE_KEYS)
        if is_new is not None or is_reused is not None:
            explicit_path_classification = True
            if is_reused is True:
                reoccupied_count += 1
            elif is_new is True:
                new_count += 1

    total_paths = len(records)
    path_shares = {
        "available": explicit_path_classification,
        "new_path_count": new_count,
        "reoccupied_path_count": reoccupied_count,
        "classified_path_count": new_count + reoccupied_count,
        "new_path_share": new_count / max(new_count + reoccupied_count, 1) if explicit_path_classification else None,
        "reoccupied_path_share": reoccupied_count / max(new_count + reoccupied_count, 1) if explicit_path_classification else None,
        "unclassified_path_count": total_paths - new_count - reoccupied_count,
    }

    branch_orders: Counter[str] = Counter()
    branch_order_available = False
    hierarchy: dict[str, dict[str, float | int]] = {}
    for record in records:
        value = _number(_first(record["metadata"], _ORDER_KEYS))
        if value is None:
            continue
        branch_order_available = True
        key = str(int(round(value)))
        branch_orders[key] += 1
        bucket = hierarchy.setdefault(key, {"path_count": 0, "length_m": 0.0, "flux": 0.0})
        bucket["path_count"] += 1
        bucket["length_m"] += float(record["length_m"])
        bucket["flux"] += float(record["mean_flux"])

    dominant_pairs = set(zip(network.dominant_route_node_ids, network.dominant_route_node_ids[1:]))
    trunk_segments = [
        segment
        for segment in network.segments
        if (segment.start_node_id, segment.end_node_id) in dominant_pairs
        or (segment.end_node_id, segment.start_node_id) in dominant_pairs
    ]
    lengths = [max(float(segment.total_length), 0.0) for segment in network.segments]
    fluxes = [max(float(segment.mean_flux), 0.0) for segment in network.segments]
    total_length = sum(lengths)
    total_flux = sum(fluxes)
    trunk_length = sum(max(float(segment.total_length), 0.0) for segment in trunk_segments)
    trunk_flux = sum(max(float(segment.mean_flux), 0.0) for segment in trunk_segments)
    flux_hhi = sum((flux / total_flux) ** 2 for flux in fluxes) if total_flux > 0.0 else None

    loop_counts: Counter[str] = Counter()
    for item in metadata:
        loop_type = _first(item, _LOOP_KEYS)
        if loop_type is not None:
            loop_counts[str(loop_type)] += 1
    crossing_groups: defaultdict[str, list[CaveSegment]] = defaultdict(list)
    coalescence_groups: defaultdict[str, list[CaveSegment]] = defaultdict(list)
    for segment in network.segments:
        crossing = _first(segment.metadata, _CROSSING_KEYS)
        if crossing is not None:
            crossing_groups[str(crossing)].append(segment)
        coalescence = _first(segment.metadata, _COALESCE_KEYS)
        if coalescence is not None:
            coalescence_groups[str(coalescence)].append(segment)
    multi_level = sum(len({segment.z_level for segment in group}) > 1 for group in crossing_groups.values())
    # Flux budget fields are path-level records and may be repeated on every
    # segment belonging to the same lobe.  Use de-duplicated path records so
    # split paths do not inflate allocation or return totals.
    returned_flux = sum(
        (_number(record["metadata"].get("coalescence_returned_flux")) or 0.0)
        for record in records
    )
    initial_flux = sum(
        (_number(record["metadata"].get("initial_flux")) or 0.0)
        for record in records
    )

    return {
        "available": bool(history_available),
        "phase_count": phase_count,
        "phase_metadata_available": phase_metadata_available,
        "phase_activity_available": phase_metadata_available,
        "phase_activity": _phase_activity(records, phase_count),
        "flux_budget": {
            "available": bool(initial_flux or returned_flux),
            "initial_flux": float(initial_flux),
            "returned_flux": float(returned_flux),
            "net_flux": float(max(initial_flux - returned_flux, 0.0)),
            "coalescence_return_ratio": float(returned_flux / initial_flux) if initial_flux > 0.0 else None,
        },
        "path_classification": path_shares,
        "outcomes": {
            "available": outcome_available,
            "path_count": total_paths,
            "survived_count": int(outcomes["survived"]),
            "stalled_count": int(outcomes["stalled"]),
            "retired_count": int(outcomes["retired"]),
            "coalesced_count": int(outcomes["coalesced"]),
            "counts": dict(sorted(outcomes.items())),
        },
        "branch_hierarchy": {
            "available": branch_order_available,
            "order_histogram": dict(sorted(branch_orders.items(), key=lambda item: int(item[0]))),
            "max_order": max((int(key) for key in branch_orders), default=None),
            "by_order": {key: hierarchy[key] for key in sorted(hierarchy, key=int)},
        },
        "trunk_dominance": {
            "available": bool(network.dominant_route_node_ids),
            "trunk_segment_count": len(trunk_segments),
            "trunk_length_share": trunk_length / total_length if total_length > 0.0 else None,
            "trunk_flux_share": trunk_flux / total_flux if total_flux > 0.0 else None,
            "flux_concentration_hhi": flux_hhi,
        },
        "loop_diagnostics": {
            "available": bool(loop_counts),
            "counts": dict(sorted(loop_counts.items())),
        },
        "crossing_coalescence": {
            "crossing_group_count": len(crossing_groups),
            "crossing_segment_count": sum(len(group) for group in crossing_groups.values()),
            "crossing_multi_level_group_count": multi_level,
            "coalescence_group_count": len(coalescence_groups),
            "coalescence_event_count": int(outcomes["coalesced"]),
            "returned_flux": float(returned_flux),
            "sanity_available": bool(crossing_groups or coalescence_groups),
            "finite_nonnegative_return_flux": returned_flux >= 0.0 and math.isfinite(returned_flux),
        },
        "longitudinal": _longitudinal_diagnostics(network),
    }


__all__ = ["emplacement_metrics"]
