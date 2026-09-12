"""Deterministic morphology screening, not a geological certification.

Lengths are normalized by passage width where possible. The limits below are
explicit engineering heuristics: they reject construction artifacts but do not
assert that a particular natural cave cannot contain a sharp bend or a pit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from plume_advanced.stages.host_field import HostField
    from plume_advanced.stages.network import CaveNetwork
    from plume_advanced.stages.section_field import SectionField

QUALITY_VERSION = "plume.network-quality.v1"
BLIND_KINDS = {"spur", "abandoned_lobe", "stalled_lobe"}


@dataclass(frozen=True)
class NetworkQualityConfig:
    enabled: bool = True
    max_attempts: int = 8
    repair_passes: int = 3
    maximum_turn_degrees: float = 55.0
    minimum_bend_radius_widths: float = 0.65
    maximum_sinuosity: float = 3.0
    maximum_width_cap_fraction: float = 0.55
    maximum_width_gradient: float = 0.6
    maximum_wiggle_degrees: float = 12.0
    terminal_width_ratio: float = 0.45
    maximum_similar_loop_pairs: int = 2
    minimum_route_extent_fraction: float = 0.35
    maximum_section_grade: float = 0.5
    maximum_uphill_run_widths: float = 8.0
    maximum_uphill_grade: float = 0.08
    minimum_crossing_clearance_m: float = 1.0
    minimum_passage_separation_widths: float = 0.70
    maximum_transverse_run_widths: float = 4.0
    minimum_downstream_alignment: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("network.quality.enabled must be a boolean")
        for name in ("max_attempts", "repair_passes", "maximum_similar_loop_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"network.quality.{name} must be a nonnegative integer")
        if not 1 <= self.max_attempts <= 64 or self.repair_passes > 8:
            raise ValueError("network.quality requires 1..64 attempts and 0..8 repair passes")
        integer_names = {"enabled", "max_attempts", "repair_passes", "maximum_similar_loop_pairs"}
        for name, value in asdict(self).items():
            if name not in integer_names and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"network.quality.{name} must be finite and positive")
        for name in (
            "maximum_width_cap_fraction",
            "terminal_width_ratio",
            "minimum_route_extent_fraction",
            "minimum_downstream_alignment",
        ):
            if getattr(self, name) >= 1:
                raise ValueError(f"network.quality.{name} must be below one")
        if self.maximum_turn_degrees >= 180:
            raise ValueError("network.quality.maximum_turn_degrees must be below 180")


class NetworkQualityError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        failed = report.get("checks", [])
        if report.get("attempts"):
            failed = report["attempts"][-1].get("checks", [])
        names = [c["name"] for c in failed if not c["passed"] and c["severity"] == "error"]
        details = [c["value"] for c in failed if c["name"] == "candidate_generation"]
        super().__init__(
            "Network morphology rejected before meshing: "
            + ", ".join(names)
            + ("; " + "; ".join(details) if details else "")
        )


def write_quality_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def shape_hash(network: CaveNetwork) -> str:
    """Exact numeric geometry fingerprint, independent of Python hash/order."""
    digest = hashlib.sha256()
    for segment in sorted(network.segments, key=lambda s: s.segment_id):
        digest.update(
            json.dumps(
                [
                    segment.segment_id,
                    segment.start_node_id,
                    segment.end_node_id,
                    segment.kind,
                    segment.z_level,
                ]
            ).encode()
        )
        digest.update(
            np.asarray(
                [[p.x, p.y, p.width, p.elevation, p.flux] for p in segment.points], dtype="<f8"
            ).tobytes()
        )
    return digest.hexdigest()


def _turn_metrics(xy: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.diff(xy, axis=0)
    length = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(length[:, None], 1e-12)
    angle = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))
    # Discrete curvature uses actual chord distances, not station indices.
    radius = (length[:-1] + length[1:]) / np.maximum(2 * angle, 1e-12)
    return np.degrees(angle), radius / np.maximum(width[1:-1], 1e-9)


def _crossings(chains: dict, network: CaveNetwork) -> list[dict]:
    """Exact projected line intersections, with a deterministic spatial broad phase.

    Exempt only the local neighborhood of a shared graph node. Grade-separated
    crossings are retained for actual floor/roof clearance screening in Stage C.
    """
    pieces, centers, radii = [], [], []
    for segment in sorted(network.segments, key=lambda s: s.segment_id):
        coords, widths = chains[segment.segment_id]
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            pieces.append((segment, i, a, b, max(widths[i : i + 2])))
            centers.append((a[:2] + b[:2]) * 0.5)
            radii.append(np.linalg.norm(b[:2] - a[:2]) * 0.5)
    if not pieces:
        return []
    pairs = cKDTree(centers).query_pairs(2 * max(radii) + 1e-6, output_type="ndarray")
    found: dict[tuple[int, int], list[dict[str, Any]]] = {}
    node_xy = {n.node_id: np.array([n.x, n.y]) for n in network.nodes}
    for first, second in sorted(map(tuple, pairs.tolist())):
        s, i, a, b, w = pieces[first]
        t, j, c, d, v = pieces[second]
        if s.segment_id == t.segment_id and abs(i - j) <= 1:
            continue
        u, z, q = b[:2] - a[:2], d[:2] - c[:2], c[:2] - a[:2]

        def cross(x, y):
            return x[0] * y[1] - x[1] * y[0]

        determinant = cross(u, z)
        if abs(determinant) < 1e-10:
            continue
        alpha, beta = cross(q, z) / determinant, cross(q, u) / determinant
        if not (0 <= alpha <= 1 and 0 <= beta <= 1):
            continue
        xy = a[:2] + alpha * u
        shared = {s.start_node_id, s.end_node_id} & {t.start_node_id, t.end_node_id}
        if s.segment_id != t.segment_id and any(
            np.linalg.norm(xy - node_xy[node]) <= 4 * max(w, v) for node in shared
        ):
            continue
        key = (s.segment_id, t.segment_id)
        found.setdefault(key, []).append(
            {
                "segments": list(key),
                "xy_m": xy.tolist(),
                "first_piece": i,
                "second_piece": j,
                "first_fraction": float(alpha),
                "second_fraction": float(beta),
                "same_level": s.z_level == t.z_level,
            }
        )
    return [hit for key in sorted(found) for hit in found[key]]


def assess_network(
    network: CaveNetwork, host: HostField | None = None, sections: SectionField | None = None
) -> dict:
    controls = network.config.quality
    checks = []

    def check(name, passed, value, limit=None, segments=(), severity="error"):
        checks.append(
            dict(
                name=name,
                passed=bool(passed),
                severity=severity,
                value=value,
                limit=limit,
                segment_ids=sorted(set(map(int, segments))),
            )
        )

    from plume_advanced.stages.network import CaveNetworkGenerator

    try:
        CaveNetworkGenerator(network.config)._validate_generated_graph(
            list(network.nodes), list(network.segments), network.dominant_route_node_ids
        )
        check("directed_connectivity_and_flow", True, "connected, source reachable, conserved")
    except ValueError as error:
        check("directed_connectivity_and_flow", False, str(error))

    nodes = {n.node_id: n for n in network.nodes}
    check(
        "unique_identifiers",
        len(nodes) == len(network.nodes)
        and len({s.segment_id for s in network.segments}) == len(network.segments),
        len(network.segments),
    )
    indegree = {node: 0 for node in nodes}
    outgoing_edges: dict[int, list[int]] = {node: [] for node in nodes}
    for s in network.segments:
        if s.start_node_id in nodes and s.end_node_id in nodes:
            outgoing_edges[s.start_node_id].append(s.end_node_id)
            indegree[s.end_node_id] += 1
    pending = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        for target in sorted(outgoing_edges[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    check("acyclic_flow_direction", visited == len(nodes), len(nodes) - visited, 0)
    from plume_advanced.stages.network_systems import assess_systems
    assess_systems(network, check)
    from plume_advanced.stages.network_gallery_growth import assess_gallery_history
    assess_gallery_history(network, check)
    chains = {}
    invalid, endpoint, outside, turns, tight, sinuous, gradients, terminals = ([] for _ in range(8))
    total_length = cap_length = 0.0
    max_turn, min_radius, max_gradient = 0.0, float("inf"), 0.0
    wiggles, maximum_wiggle = [], 0.0
    outgoing = {s.start_node_id for s in network.segments}
    degrees = network._degrees()
    flow_direction = np.array([0.0, 1.0])
    if len(network.dominant_route_node_ids) > 1:
        first = nodes.get(network.dominant_route_node_ids[0])
        last = nodes.get(network.dominant_route_node_ids[-1])
        if first and last:
            flow_direction = np.array([last.x - first.x, last.y - first.y], dtype=float)
            flow_direction /= max(float(np.linalg.norm(flow_direction)), 1e-9)
    transverse, maximum_transverse = [], 0.0
    loop_shapes = []
    for s in network.segments:
        xy = np.array([[p.x, p.y] for p in s.points], dtype=float)
        widths = np.array([p.width for p in s.points])
        if (
            len(xy) < 2
            or not np.isfinite(xy).all()
            or not np.isfinite(widths).all()
            or np.any(widths <= 0)
        ):
            invalid.append(s.segment_id)
            continue
        chains[s.segment_id] = (xy, widths)
        lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if s.kind == "backbone":
            alignment = (np.diff(xy, axis=0) @ flow_direction) / np.maximum(lengths, 1e-9)
            run = 0.0
            for distance, width, downstream in zip(
                lengths, 0.5 * (widths[:-1] + widths[1:]), alignment
            ):
                run = (
                    run + float(distance / width)
                    if downstream < controls.minimum_downstream_alignment
                    else 0.0
                )
                maximum_transverse = max(maximum_transverse, run)
                if run > controls.maximum_transverse_run_widths:
                    transverse.append(s.segment_id)
        if np.any(lengths < 1e-7):
            invalid.append(s.segment_id)
        if sum(lengths) > 8 * widths.mean():
            from scipy.ndimage import gaussian_filter1d

            arc = np.r_[0, np.cumsum(lengths)]
            step = max(0.25, float(np.median(widths)) / 4)
            positions = np.linspace(0, arc[-1], int(np.ceil(arc[-1] / step)) + 1)
            uniform = np.column_stack([np.interp(positions, arc, xy[:, k]) for k in range(2)])
            d = np.diff(uniform, axis=0)
            heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            residual = heading - gaussian_filter1d(heading, 4, mode="nearest")
            wiggle = float(np.degrees(np.sqrt(np.mean(residual**2))))
            maximum_wiggle = max(maximum_wiggle, wiggle)
            if wiggle > controls.maximum_wiggle_degrees:
                wiggles.append(s.segment_id)
        for node_id, p in ((s.start_node_id, xy[0]), (s.end_node_id, xy[-1])):
            n = nodes.get(node_id)
            if n is None or np.linalg.norm(p - np.array([n.x, n.y])) > 1e-5:
                endpoint.append(s.segment_id)
        if host is not None and (
            np.any(xy[:, 0] < host.x_coords[0] - 1e-6)
            or np.any(xy[:, 0] > host.x_coords[-1] + 1e-6)
            or np.any(xy[:, 1] < host.y_coords[0] - 1e-6)
            or np.any(xy[:, 1] > host.y_coords[-1] + 1e-6)
        ):
            outside.append(s.segment_id)
        angle, radius = _turn_metrics(xy, widths)
        if len(angle):
            max_turn = max(max_turn, float(max(angle)))
            min_radius = min(min_radius, float(min(radius)))
            if max(angle) > controls.maximum_turn_degrees:
                turns.append(s.segment_id)
            if min(radius) < controls.minimum_bend_radius_widths:
                tight.append(s.segment_id)
        chord = np.linalg.norm(xy[-1] - xy[0])
        if sum(lengths) / max(chord, widths.mean()) > controls.maximum_sinuosity:
            sinuous.append(s.segment_id)
        grad = np.abs(np.diff(widths)) / np.maximum(lengths, 1e-9)
        max_gradient = max(max_gradient, float(max(grad)))
        if max(grad) > controls.maximum_width_gradient:
            gradients.append(s.segment_id)
        total_length += sum(lengths)
        cap_length += sum(
            lengths[
                np.minimum(widths[:-1], widths[1:]) >= 1.999 * network.config.maximum_passage_radius
            ]
        )
        if s.kind in BLIND_KINDS and s.end_node_id not in outgoing and degrees[s.end_node_id] == 1:
            if widths[-1] / max(widths) > controls.terminal_width_ratio + 1e-6:
                terminals.append(s.segment_id)
        system_loop_arm = (
            network.config.systems.count > 1
            and sum(t.start_node_id == s.start_node_id for t in network.segments) > 1
            and sum(t.end_node_id == s.end_node_id for t in network.segments) > 1
        )
        if (s.kind == "anastomosis" or system_loop_arm or s.metadata.get("island_id")) and chord > 4 * widths.mean():
            arc = np.r_[0, np.cumsum(lengths)]
            sampled = np.column_stack(
                [np.interp(np.linspace(0, arc[-1], 32), arc, xy[:, k]) for k in range(2)]
            )
            direction = (xy[-1] - xy[0]) / chord
            offset = sampled - xy[0]
            normalized = (
                np.column_stack(
                    [offset @ direction, np.abs(offset @ np.array([-direction[1], direction[0]]))]
                )
                / chord
            )
            # A near-straight fragment is not a loop motif. Likewise, two
            # graph fragments belonging to the same lobe are not independent
            # repetitions. Only compare substantial, almost identical bows.
            if np.max(normalized[:, 1]) > 0.1:
                loop_shapes.append(
                    (
                        s.segment_id,
                        arc[-1],
                        normalized,
                        s.metadata.get("island_id") or s.metadata.get("lobe_path_id", f"segment-{s.segment_id}"),
                    )
                )

    check("finite_positive_geometry", not invalid, len(invalid), 0, invalid)
    check("graph_endpoint_agreement", not endpoint, len(set(endpoint)), 0, endpoint)
    check("host_bounds", not outside, len(outside), 0, outside)
    check("local_turn_angle", not turns, max_turn, controls.maximum_turn_degrees, turns)
    check(
        "bend_radius_relative_to_width",
        not tight,
        min_radius if np.isfinite(min_radius) else None,
        controls.minimum_bend_radius_widths,
        tight,
    )
    check("segment_sinuosity", not sinuous, len(sinuous), controls.maximum_sinuosity, sinuous)
    check(
        "arterial_transverse_runs",
        not transverse,
        maximum_transverse,
        controls.maximum_transverse_run_widths,
        transverse,
    )
    check("width_gradient", not gradients, max_gradient, controls.maximum_width_gradient, gradients)
    check(
        "small_scale_heading_wiggles",
        not wiggles,
        maximum_wiggle,
        controls.maximum_wiggle_degrees,
        wiggles,
    )
    fraction = float(cap_length / max(total_length, 1e-9))
    check(
        "width_cap_saturation",
        fraction <= controls.maximum_width_cap_fraction,
        fraction,
        controls.maximum_width_cap_fraction,
    )
    check("blind_branch_taper", not terminals, len(terminals), 0, terminals)
    similar = []
    for i, (sid, length, shape, lobe) in enumerate(loop_shapes):
        for tid, other_length, other_shape, other_lobe in loop_shapes[i + 1 :]:
            if (
                lobe != other_lobe
                and abs(np.log(length / other_length)) < 0.05
                and np.sqrt(np.mean((shape - other_shape) ** 2)) < 0.015
            ):
                similar.append((sid, tid))
    check(
        "repeated_loop_shapes",
        len(similar) <= controls.maximum_similar_loop_pairs,
        len(similar),
        controls.maximum_similar_loop_pairs,
        [sid for pair in similar for sid in pair],
    )
    extent = 0.0
    if len(network.dominant_route_node_ids) > 1:
        start, end = (
            nodes.get(i)
            for i in (network.dominant_route_node_ids[0], network.dominant_route_node_ids[-1])
        )
        if start and end:
            extent = float(np.hypot(end.x - start.x, end.y - start.y))
    target_extent = network.config.target_route_length_m
    if host is not None:
        target_extent = min(target_extent, float(np.hypot(np.ptp(host.x_coords), np.ptp(host.y_coords))))
    check(
        "minimum_route_extent",
        extent >= controls.minimum_route_extent_fraction * target_extent,
        extent,
        controls.minimum_route_extent_fraction * target_extent,
    )
    if checks[0]["passed"] and not invalid and len(chains) == len(network.segments):
        crossings = _crossings(chains, network)
        unmodeled = [h for h in crossings if h["same_level"]]
        check(
            "unmodeled_plan_crossings",
            not unmodeled,
            len(unmodeled),
            0,
            [sid for h in unmodeled for sid in h["segments"]],
        )

    if sections is not None:
        checks.extend(assess_sections(network, sections)["checks"])
    from plume_advanced.stages.network_topology import assess_topology
    assess_topology(network, sections, check)
    from plume_advanced.stages.network_interconnected import assess_interconnected
    assess_interconnected(network, host, sections, check)
    return {
        "schema": QUALITY_VERSION,
        "accepted": all(c["passed"] for c in checks if c["severity"] == "error"),
        "scope": "network_and_sections" if sections is not None else "network_only",
        "checks": checks,
        "shape_sha256": shape_hash(network),
        "thresholds": asdict(controls),
    }


def assess_sections(network: CaveNetwork, sections: SectionField) -> dict:
    """Screen the actual post-blend profiles, including their 3D crossings."""
    controls = network.config.quality
    checks, bad, steep, uphill, turns, invalid_frames, clearances = [], [], [], [], [], [], []
    chains, fields = {}, {f.segment_id: f for f in sections.segment_fields}
    maximum_grade = 0.0
    proximity_samples = []
    for sid, f in sorted(fields.items()):
        samples = f.samples
        xyz = np.array([[s.x, s.y, s.z] for s in samples])
        width = np.array([s.tube_width for s in samples])
        if (
            len(xyz) < 2
            or not np.isfinite(xyz).all()
            or not np.isfinite(width).all()
            or np.any(width <= 0)
        ):
            bad.append(sid)
            continue
        chains[sid] = (xyz, width)
        delta = np.diff(xyz, axis=0)
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        # Very close stations at joins are assessed by net displacement over
        # a width-scale window below, not a noisy dz / sub-mm chord quotient.
        arc = np.r_[0, np.cumsum(horizontal)]
        if arc[-1] > 0:
            dense_arc = np.linspace(
                0, arc[-1], max(2, int(np.ceil(arc[-1] / max(0.5, np.median(width) / 3))) + 1)
            )
            for distance in dense_arc:
                proximity_samples.append(
                    (
                        sid,
                        distance,
                        np.interp(distance, arc, xyz[:, 0]),
                        np.interp(distance, arc, xyz[:, 1]),
                        np.interp(distance, arc, width),
                        np.interp(distance, arc, [s.floor_world_z for s in samples]),
                        np.interp(distance, arc, [s.roof_world_z for s in samples]),
                    )
                )
        step = max(0.5, float(np.median(width)) * 0.5)
        targets = np.arange(0, max(arc[-1] - step, 0), step)
        grades = (
            np.interp(targets + step, arc, xyz[:, 2]) - np.interp(targets, arc, xyz[:, 2])
        ) / step
        if len(grades):
            maximum_grade = max(maximum_grade, float(max(abs(grades))))
            if max(abs(grades)) > controls.maximum_section_grade:
                steep.append(sid)
            run = 0.0
            for g in grades:
                run = run + step if g > controls.maximum_uphill_grade else 0.0
                if run > controls.maximum_uphill_run_widths * np.median(width):
                    uphill.append(sid)
        angle, _ = _turn_metrics(xyz[:, :2], width)
        if len(angle) and max(angle) > controls.maximum_turn_degrees:
            turns.append(sid)
        for s in samples:
            profile = np.asarray(s.profile_points)
            frame = np.array([s.tangent, s.normal, s.binormal])
            if (
                not np.isfinite(profile).all()
                or not np.isfinite([s.floor_world_z, s.roof_world_z, s.tube_height]).all()
                or s.tube_height <= 0
                or len(profile) < 3
                or profile.ndim != 2
                or profile.shape[1] != 2
                or np.ptp(profile[:, 0]) <= 0
                or np.ptp(profile[:, 1]) <= 0
                or s.roof_world_z <= s.floor_world_z
            ):
                bad.append(sid)
            if not np.isfinite(frame).all() or not np.allclose(
                frame @ frame.T, np.eye(3), atol=1e-4
            ):
                invalid_frames.append(sid)
    if set(chains) == {s.segment_id for s in network.segments}:
        for hit in _crossings(chains, network):
            intervals = []
            for label, sid in zip(("first", "second"), hit["segments"]):
                ss = fields[sid].samples
                i, u = hit[label + "_piece"], hit[label + "_fraction"]
                intervals.append(
                    [
                        (1 - u) * getattr(ss[i], name) + u * getattr(ss[i + 1], name)
                        for name in ("floor_world_z", "roof_world_z")
                    ]
                )
            a, b = intervals
            gap = max(a[0] - b[1], b[0] - a[1])
            if gap < controls.minimum_crossing_clearance_m:
                clearances.extend(hit["segments"])
    else:
        bad.extend(s.segment_id for s in network.segments if s.segment_id not in chains)
    proximity = []
    if proximity_samples and network.nodes:
        dense = np.asarray(proximity_samples)
        # Exclude local junction envelopes here; exact crossing checks above
        # still enforce connectivity there. This checks nonlocal wall overlaps.
        node_distance = cKDTree([[n.x, n.y] for n in network.nodes]).query(dense[:, 2:4])[0]
        dense = dense[node_distance > 4 * dense[:, 4]]
        if len(dense):
            pairs = cKDTree(dense[:, 2:4]).query_pairs(
                controls.minimum_passage_separation_widths * float(max(dense[:, 4])),
                output_type="ndarray",
            )
            for i, j in sorted(map(tuple, pairs.tolist())):
                a, b = dense[i], dense[j]
                if a[0] == b[0] and abs(a[1] - b[1]) < 4 * max(a[4], b[4]):
                    continue
                distance = np.linalg.norm(a[2:4] - b[2:4])
                if (
                    distance < controls.minimum_passage_separation_widths * 0.5 * (a[4] + b[4])
                    and max(a[5] - b[6], b[5] - a[6]) < controls.minimum_crossing_clearance_m
                ):
                    proximity.extend([int(a[0]), int(b[0])])
    connections: dict[int, list[tuple[int, float]]] = {}
    for segment in network.segments:
        if segment.segment_id not in fields or not fields[segment.segment_id].samples:
            continue
        samples = fields[segment.segment_id].samples
        for node, sample in (
            (segment.start_node_id, samples[0]),
            (segment.end_node_id, samples[-1]),
        ):
            connections.setdefault(node, []).append((segment.segment_id, sample.floor_world_z))
    floor_mismatch = [
        sid
        for group in connections.values()
        if len(group) > 1 and np.ptp([floor for _, floor in group]) > 0.01
        for sid, _ in group
    ]
    for name, ids, limit, value in (
        ("section_profiles_finite_positive", bad, 0, len(set(bad))),
        ("section_frames_orthonormal", invalid_frames, 0, len(set(invalid_frames))),
        ("section_bend_continuity", turns, controls.maximum_turn_degrees, len(set(turns))),
        ("section_grade", steep, controls.maximum_section_grade, maximum_grade),
        ("sustained_uphill_sections", uphill, controls.maximum_uphill_run_widths, len(set(uphill))),
        (
            "crossing_roof_floor_clearance",
            clearances,
            controls.minimum_crossing_clearance_m,
            len(set(clearances)),
        ),
        (
            "nonlocal_passage_overlap",
            proximity,
            controls.minimum_passage_separation_widths,
            len(set(proximity)),
        ),
        ("junction_floor_continuity", floor_mismatch, 0.01, len(set(floor_mismatch))),
    ):
        checks.append(
            dict(
                name=name,
                passed=not ids,
                severity="error",
                value=value,
                limit=limit,
                segment_ids=sorted(set(map(int, ids))),
            )
        )
    return {
        "schema": QUALITY_VERSION,
        "accepted": all(c["passed"] for c in checks),
        "checks": checks,
    }
