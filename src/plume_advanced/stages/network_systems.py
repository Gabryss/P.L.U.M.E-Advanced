"""Host-conditioned growth of interacting arterial systems.

This is a procedural routing model, not a time-dependent lava solver. Ordered
fronts have independently seeded, correlated routing preferences. Capture and
release use different thresholds and a minimum residence distance. A shared
front emits one graph edge, regardless of how many systems occupy it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, map_coordinates

from plume_advanced.procedural import derive_subseed, procedural_rng


@dataclass(frozen=True)
class NetworkSystemsConfig:
    count: int = 1
    source_spacing_widths: float = 12.0
    lateral_variation_widths: float = 18.0
    correlation_length_widths: float = 70.0
    merge_distance_widths: float = 3.0
    split_distance_widths: float = 8.0
    minimum_shared_length_widths: float = 25.0
    minimum_independent_length_widths: float = 30.0
    split_confirmation_widths: float = 8.0
    interaction_spacing_widths: float = 8.0
    require_merge: bool = True
    require_split: bool = True

    def __post_init__(self):
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or not 1 <= self.count <= 8
        ):
            raise ValueError("network.systems.count must be an integer in [1, 8]")
        for name, value in asdict(self).items():
            if name in {"require_merge", "require_split"}:
                if not isinstance(value, bool):
                    raise ValueError(f"network.systems.{name} must be a boolean")
            elif name != "count" and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"network.systems.{name} must be finite and positive")
        if self.split_distance_widths <= self.merge_distance_widths:
            raise ValueError(
                "network.systems.split_distance_widths must exceed merge_distance_widths"
            )
        if self.source_spacing_widths <= self.merge_distance_widths:
            raise ValueError(
                "network.systems.source_spacing_widths must exceed merge_distance_widths"
            )
        if min(self.minimum_shared_length_widths, self.minimum_independent_length_widths) < 3:
            raise ValueError("network.systems persistence lengths must be at least three widths")


def _cross_bounds(host, geometry, along, margin):
    """Intersect flow-aligned stations with the actual rectangular host."""
    lower, upper = np.full_like(along, -np.inf), np.full_like(along, np.inf)
    for coords, origin, flow, cross in (
        (host.x_coords, geometry.seed_x, geometry.flow_x, geometry.cross_x),
        (host.y_coords, geometry.seed_y, geometry.flow_y, geometry.cross_y),
    ):
        base = origin + along * flow
        if abs(cross) < 1e-10:
            if np.any(base < coords[0] + margin) or np.any(base > coords[-1] - margin):
                raise ValueError(
                    "Interacting systems do not fit inside the host along the flow direction"
                )
            continue
        a, b = (coords[0] + margin - base) / cross, (coords[-1] - margin - base) / cross
        lower, upper = np.maximum(lower, np.minimum(a, b)), np.minimum(upper, np.maximum(a, b))
    if np.any(lower >= upper):
        raise ValueError("Host is too narrow for interacting systems")
    return lower, upper


def preferred_tracks(generator, host, geometry, *, corridor=None, preserve_identity=False):
    """Independent smooth proposals, biased by the shared host routing field.

    Lateral order is preserved when preferences cross; systems can coalesce
    but cannot pass through each other without an explicit connection.
    """
    config, controls = generator.config, generator.config.systems
    width = 2 * config.base_passage_radius
    step = max(1.0, min(width, geometry.cell_scale))
    along = np.linspace(0, geometry.along_extent, max(8, int(geometry.along_extent / step) + 1))
    lower, upper = _cross_bounds(host, geometry, along, 2 * width)
    spacing = controls.source_spacing_widths * width
    sources = (np.arange(controls.count) - (controls.count - 1) / 2) * spacing
    if sources[0] < lower[0] or sources[-1] > upper[0]:
        raise ValueError(
            "Source spacing/count does not fit inside the host; widen the host or reduce spacing"
        )
    correlation = controls.correlation_length_widths * width
    knots = np.linspace(0, along[-1], max(4, int(np.ceil(along[-1] / correlation)) + 1))
    track_rows = []
    for system_id, source in enumerate(sources):
        rng = procedural_rng(config.random_seed, "network-system", system_id)
        noise = rng.normal(0, controls.lateral_variation_widths * width, len(knots))
        noise[0] = 0
        target = source + CubicSpline(knots, noise, bc_type="natural")(along)
        if corridor is not None:
            # Independent preferences share a host corridor, not a prescribed
            # confluence. Their convergence and divergence determine events.
            target += corridor - source * (1 - np.exp(-along / (1.5 * correlation)))
        target = np.clip(target, lower, upper)
        # Sample substrate around each proposal. A soft minimum avoids jumps
        # between nearly equivalent cells while retaining the host's influence.
        candidates = np.clip(
            target[:, None] + np.linspace(-4 * width, 4 * width, 17), lower[:, None], upper[:, None]
        )
        x = geometry.seed_x + along[:, None] * geometry.flow_x + candidates * geometry.cross_x
        y = geometry.seed_y + along[:, None] * geometry.flow_y + candidates * geometry.cross_y
        ix = (x - host.x_coords[0]) / (host.x_coords[1] - host.x_coords[0])
        iy = (y - host.y_coords[0]) / (host.y_coords[1] - host.y_coords[0])
        cost = map_coordinates(host.growth_cost, [iy, ix], order=1, mode="nearest")
        cost = (
            config.growth_cost_weight * cost + ((candidates - target[:, None]) / (2 * width)) ** 2
        )
        weights = np.exp(-(cost - cost.min(axis=1, keepdims=True)))
        preferred = (candidates * weights).sum(axis=1) / weights.sum(axis=1)
        preferred = gaussian_filter1d(preferred, max(1, 0.10 * correlation / step), mode="nearest")
        fade = np.clip(along / max(correlation * 0.5, 1), 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        preferred = source * (1 - fade) + preferred * fade
        track_rows.append(np.clip(preferred, lower, upper))
    tracks = np.asarray(track_rows)
    if preserve_identity:
        # Project crossing preferences onto lateral order without swapping
        # source IDs or replacing one system's random stream with another's.
        enforce_source_order(tracks)
    else:
        tracks = np.sort(tracks, axis=0)
    return along, tracks, width


def enforce_source_order(tracks: np.ndarray) -> None:
    """Project each station onto lateral source order without swapping identities."""
    for index in range(tracks.shape[1]):
        blocks: list[tuple[float, int]] = []
        for value in tracks[:, index]:
            blocks.append((float(value), 1))
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                right, left = blocks.pop(), blocks.pop()
                count = left[1] + right[1]
                blocks.append(((left[0] * left[1] + right[0] * right[1]) / count, count))
        tracks[:, index] = [mean for mean, count in blocks for _ in range(count)]


def plan_interactions(along, tracks, width, controls, *, events=None, local_spacing=False, connection_check=None):
    """Return explicit directed nodes and disjoint shared-passage records.

    Each record is (start node, end node, first station, last station,
    participating system IDs). Only neighboring fronts may join. A single
    event can be either a merge or a split; four-way contact is not implicit.
    """
    nodes = [(float(along[0]), float(tracks[i, 0]), "entry") for i in range(len(tracks))]
    active: dict[tuple[int, ...], tuple[int, int]] = {(i,): (i, 0) for i in range(len(tracks))}
    records = []
    last_event = -float("inf")

    def finish(group, index, node_id):
        start, first = active.pop(group)
        records.append((start, node_id, first, index, group))

    for index in range(1, len(along) - 1):
        position = along[index]
        if (
            (not local_spacing and position - last_event < controls.interaction_spacing_widths * width)
            or along[-1] - position
            < max(controls.minimum_independent_length_widths, controls.minimum_shared_length_widths)
            * width
        ):
            continue
        groups = sorted(active)
        split = None
        for group in groups:
            first = active[group][1]
            if (
                len(group) < 2
                or position - along[first] < controls.minimum_shared_length_widths * width
                or (local_spacing and position-along[first] < controls.interaction_spacing_widths*width)
            ):
                continue
            gaps = np.diff(tracks[list(group), index])
            boundary = int(np.argmax(gaps))
            confirmation = max(
                first,
                int(np.searchsorted(along, position - controls.split_confirmation_widths * width)),
            )
            stable_gap = (
                tracks[group[boundary + 1], confirmation : index + 1]
                - tracks[group[boundary], confirmation : index + 1]
            )
            if position - along[first] >= controls.split_confirmation_widths * width and np.all(
                stable_gap >= controls.split_distance_widths * width
            ):
                if connection_check is not None and not connection_check("split", group[:boundary+1], group[boundary+1:], index):
                    continue
                split = group, boundary + 1
                break
        if split is not None:
            group, boundary = split
            node_id = len(nodes)
            nodes.append((float(position), float(np.mean(tracks[list(group), index])), "junction"))
            finish(group, index, node_id)
            active[group[:boundary]] = (node_id, index)
            active[group[boundary:]] = (node_id, index)
            if events is not None:
                events.append(
                    dict(
                        kind="split",
                        node_id=node_id,
                        station_m=float(position),
                        before=[list(group)],
                        after=[list(group[:boundary]), list(group[boundary:])],
                        preference_gap_m=float(
                            tracks[group[boundary], index] - tracks[group[boundary - 1], index]
                        ),
                    )
                )
            last_event = position
            continue
        for left, right in zip(groups, groups[1:]):
            old_enough = all(
                position - along[active[g][1]]
                >= width * max(
                    controls.minimum_shared_length_widths
                    if len(g) > 1
                    else controls.minimum_independent_length_widths,
                    controls.interaction_spacing_widths if local_spacing else 0,
                )
                for g in (left, right)
            )
            # Distance between occupied fronts, rather than the extremes of a
            # wide shared preference bundle, determines physical capture.
            distance = np.mean(tracks[list(right), index]) - np.mean(tracks[list(left), index])
            if old_enough and distance <= controls.merge_distance_widths * width:
                if connection_check is not None and not connection_check("merge", left, right, index):
                    continue
                group = left + right
                node_id = len(nodes)
                nodes.append(
                    (float(position), float(np.mean(tracks[list(group), index])), "junction")
                )
                finish(left, index, node_id)
                finish(right, index, node_id)
                active[group] = (node_id, index)
                if events is not None:
                    events.append(
                        dict(
                            kind="merge",
                            node_id=node_id,
                            station_m=float(position),
                            before=[list(left), list(right)],
                            after=[list(group)],
                            preference_gap_m=float(distance),
                        )
                    )
                last_event = position
                break
    for group in sorted(active):
        node_id = len(nodes)
        nodes.append((float(along[-1]), float(np.mean(tracks[list(group), -1])), "exit"))
        finish(group, len(along) - 1, node_id)
    return nodes, records


def generate_system_network(generator, host):
    from plume_advanced.stages.network import CaveNode, CavePoint, CaveSegment

    config = generator.config
    if config.emplacement_backend != "internal":
        raise ValueError(
            "Multiple systems currently require internal emplacement and hybrid_lobe growth"
        )
    geometry = generator._build_flow_geometry(host)
    along, tracks, width = preferred_tracks(generator, host, geometry)
    planned_nodes, records = plan_interactions(along, tracks, width, config.systems)

    def world(a: float, c: float) -> tuple[float, float]:
        return (
            geometry.seed_x + a * geometry.flow_x + c * geometry.cross_x,
            geometry.seed_y + a * geometry.flow_y + c * geometry.cross_y,
        )

    nodes = [CaveNode(i, *world(a, c), a, c, kind) for i, (a, c, kind) in enumerate(planned_nodes)]
    segments = []
    for sid, (start, end, first, last, group) in enumerate(records):
        a = along[first : last + 1]
        c = tracks[list(group), first : last + 1].mean(axis=0).copy()
        # Restore shared endpoints with smooth, bounded-distance transitions.
        reach = min(0.4 * (a[-1] - a[0]), 12 * width)
        for delta, distance in (
            (nodes[start].lateral_offset - c[0], a - a[0]),
            (nodes[end].lateral_offset - c[-1], a[-1] - a),
        ):
            t = np.clip(distance / max(reach, 1e-9), 0, 1)
            c += delta * (1 - t * t * (3 - 2 * t))
        xy = np.column_stack(world(a, c))
        arcs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        points = []
        for i, ((x, y), arc) in enumerate(zip(xy, arcs)):
            sample = host.sample(float(x), float(y))
            points.append(
                CavePoint(
                    i,
                    float(x),
                    float(y),
                    sample.elevation,
                    sample.slope_degrees,
                    sample.cover_thickness,
                    sample.roof_competence,
                    sample.growth_cost,
                    float(arc),
                    width,
                )
            )
        metadata = generator._build_emplacement_metadata(
            kind="backbone",
            phase_count=1,
            birth_phase=0,
            death_phase=0,
            formation_state="shared_arterial" if len(group) > 1 else "system_arterial",
        )
        metadata.update(
            system_ids=list(group),
            shared_system_count=len(group),
            network_process="interacting_systems_v1",
        )
        if nodes[start].kind == "entry":
            metadata["source_system_id"] = group[0]
            metadata["system_seed"] = derive_subseed(config.random_seed, "network-system", group[0])
        segments.append(CaveSegment(sid, start, end, "backbone", 0, tuple(points), metadata))
    skeleton = np.zeros_like(host.growth_cost, dtype=bool)
    for segment in segments:
        for point in segment.points:
            skeleton[generator._world_to_cell(host, point.x, point.y)] = True
    return generator._finish_network(
        host,
        geometry,
        nodes,
        segments,
        backend_provenance={
            "backend": "internal",
            "version": "interacting_systems_v1",
            "system_count": config.systems.count,
            "system_seeds": [
                derive_subseed(config.random_seed, "network-system", i)
                for i in range(config.systems.count)
            ],
        },
        skeleton_mask=skeleton,
        total_flux=np.zeros_like(host.growth_cost),
    )


def annotate_source_lineage(segments, topological_ids):
    """Track transported source identities, including both arms after a split."""
    from dataclasses import replace

    incoming: defaultdict[int, set[int]] = defaultdict(set)
    outgoing = defaultdict(list)
    for segment in segments:
        outgoing[segment.start_node_id].append(segment)
    resolved = {}
    for node_id in topological_ids:
        for segment in outgoing[node_id]:
            contributors = incoming[node_id].copy()
            source = segment.metadata.get("source_system_id")
            if source is not None:
                contributors.add(int(source))
            incoming[segment.end_node_id].update(contributors)
            resolved[segment.segment_id] = replace(
                segment,
                metadata=dict(segment.metadata, contributing_system_ids=sorted(contributors)),
            )
    return [resolved[s.segment_id] for s in segments]


def _front_segments(network):
    return [
        s
        for s in network.segments
        if not (
            network.config.topology.generation_mode == "independent_growth"
            and s.metadata.get("topology_role") == "side_branch"
        )
    ]


def system_summary(network):
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for segment in _front_segments(network):
        incoming[segment.end_node_id].append(segment)
        outgoing[segment.start_node_id].append(segment)
    return {
        "system_count": network.config.systems.count,
        "system_merge_count": sum(len(incoming[n.node_id]) > 1 for n in network.nodes),
        "system_split_count": sum(len(outgoing[n.node_id]) > 1 for n in network.nodes),
        "exit_count": sum(n.kind == "exit" for n in network.nodes),
        "shared_passage_length_m": sum(
            s.total_length
            for s in network.segments
            if isinstance(s.metadata.get("system_ids"), list) and len(s.metadata["system_ids"]) > 1
        ),
    }


def assess_systems(network, check):
    """Check front identity, source lineage and persistence after every repair."""
    controls = network.config.systems
    if controls.count == 1 or (
        network.config.topology.style == "trunk_dominated"
        and network.config.topology.generation_mode != "independent_growth"
    ):
        return
    from plume_advanced.stages.network import CaveNetworkGenerator

    incoming, outgoing = defaultdict(list), defaultdict(list)
    identity, lineage, persistence = [], [], []
    sources: list[int | None] = []
    membership = {}
    run_lengths: defaultdict[Any, float] = defaultdict(float)
    for segment in _front_segments(network):
        run_lengths[segment.metadata.get("front_run_id", segment.segment_id)] += (
            segment.total_length
        )
    for segment in _front_segments(network):
        incoming[segment.end_node_id].append(segment)
        outgoing[segment.start_node_id].append(segment)
        ids = segment.metadata.get("system_ids", [])
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or not 0 <= i < controls.count for i in ids)
            or ids != sorted(set(ids))
        ):
            identity.append(segment.segment_id)
            ids = []
        membership[segment.segment_id] = ids
        minimum = (
            controls.minimum_shared_length_widths
            if len(ids) > 1
            else controls.minimum_independent_length_widths
        )
        if (
            run_lengths[segment.metadata.get("front_run_id", segment.segment_id)] + 1e-6
            < minimum * 2 * network.config.base_passage_radius
        ):
            persistence.append(segment.segment_id)
    for node in network.nodes:
        before, after = incoming[node.node_id], outgoing[node.node_id]
        in_ids = [i for s in before for i in membership[s.segment_id]]
        out_ids = [i for s in after for i in membership[s.segment_id]]
        if not before and not after:
            continue  # A blind breakout terminal is outside the arterial front graph.
        if node.kind == "entry":
            sources.extend(s.metadata.get("source_system_id") for s in after)
            if before or len(after) != 1 or len(out_ids) != 1:
                identity.extend(s.segment_id for s in before + after)
            elif after[0].metadata.get("source_system_id") != out_ids[0]:
                identity.extend(s.segment_id for s in after)
        elif node.kind == "exit":
            if after or len(before) != 1:
                identity.extend(s.segment_id for s in before + after)
        elif (
            sorted(in_ids) != sorted(out_ids)
            or len(in_ids) != len(set(in_ids))
            or not before
            or not after
            or (len(before) > 1 and len(after) > 1)
        ):
            identity.extend(s.segment_id for s in before + after)
    check(
        "system_sources",
        all(type(i) is int for i in sources)
        and sorted(i for i in sources if isinstance(i, int)) == list(range(controls.count)),
        len(sources),
        controls.count,
    )
    check("system_identity_at_connections", not identity, len(identity), 0, identity)
    try:
        order = CaveNetworkGenerator._topological_node_ids(
            list(network.nodes), list(network.segments)
        )
        expected = annotate_source_lineage(network.segments, order)
        lineage = [
            s.segment_id
            for s, wanted in zip(network.segments, expected)
            if s.metadata.get("contributing_system_ids")
            != wanted.metadata["contributing_system_ids"]
        ]
    except (ValueError, KeyError, TypeError):
        lineage = [s.segment_id for s in network.segments]
    check("system_source_lineage", not lineage, len(lineage), 0, lineage)
    check("system_passage_persistence", not persistence, len(persistence), 0, persistence)
    stats = system_summary(network)
    check(
        "system_merge_opportunity",
        not controls.require_merge or stats["system_merge_count"] > 0,
        stats["system_merge_count"],
        1 if controls.require_merge else 0,
    )
    check(
        "system_split_opportunity",
        not controls.require_split or stats["system_split_count"] > 0,
        stats["system_split_count"],
        1 if controls.require_split else 0,
    )
    inlet = sum(
        s.mean_flux for n in network.nodes if n.kind == "entry" for s in outgoing[n.node_id]
    )
    all_outgoing = {s.start_node_id for s in network.segments}
    outlet = sum(s.mean_flux for s in network.segments if s.end_node_id not in all_outgoing)
    expected = network.config.source_flux * controls.count
    error = max(abs(inlet - expected), abs(outlet - expected)) / max(expected, 1e-9)
    check("system_total_discharge", error <= 1e-8, error, 1e-8)
