"""Independent arterial growth constrained by a dominant-gallery morphology.

The interaction planner decides connections from independent host-biased routes.
Chronological pulses grow finite-budget blind breakouts and can reuse them later.
This remains a procedural formation model, not a thermofluid simulation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from plume_advanced.procedural import derive_subseed, procedural_rng
from plume_advanced.stages.network_systems import plan_interactions, preferred_tracks

if TYPE_CHECKING:
    from plume_advanced.stages.network import CavePoint, CaveSegment


def independent_preferences(generator, host, geometry):
    cfg = generator.config
    if cfg.topology.style == "interconnected":
        from plume_advanced.stages.network_interconnected import route_preferences
        return route_preferences(generator, host, geometry)
    common = copy(generator)
    common.config = replace(
        cfg,
        random_seed=derive_subseed(cfg.random_seed, "gallery-corridor"),
        systems=replace(
            cfg.systems,
            count=1,
            lateral_variation_widths=cfg.topology.lateral_variation_widths,
            correlation_length_widths=cfg.topology.correlation_length_widths,
        ),
    )
    _, corridor, _ = preferred_tracks(common, host, geometry)
    return preferred_tracks(generator, host, geometry, corridor=corridor[0], preserve_identity=True)


def _points(host, xy, widths):
    from plume_advanced.stages.network import CavePoint

    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    result = []
    for index, (p, distance, width) in enumerate(zip(xy, arc, widths)):
        sample = host.sample(*map(float, p))
        result.append(
            CavePoint(
                index,
                float(p[0]),
                float(p[1]),
                sample.elevation,
                sample.slope_degrees,
                sample.cover_thickness,
                sample.roof_competence,
                sample.growth_cost,
                float(distance),
                float(width),
            )
        )
    return tuple(result)


def _split_segment(nodes, segments, sid, index, *, node_kind="junction", metadata=None):
    from plume_advanced.stages.network import CaveNode

    segment = next(s for s in segments if s.segment_id == sid)
    p = segment.points[index]
    node_id = max(n.node_id for n in nodes) + 1
    nodes.append(CaveNode(node_id, p.x, p.y, 0.0, 0.0, node_kind))
    next_id = max(s.segment_id for s in segments) + 1
    first_meta = dict(segment.metadata, **(metadata or {}))
    last_meta = dict(first_meta)
    last_meta.pop("source_system_id", None)
    last_meta.pop("system_seed", None)
    first = replace(
        segment, end_node_id=node_id, points=segment.points[: index + 1], metadata=first_meta
    )
    last = replace(
        segment,
        segment_id=next_id,
        start_node_id=node_id,
        metadata=last_meta,
        points=tuple(
            replace(q, index=i, arc_length=q.arc_length - p.arc_length)
            for i, q in enumerate(segment.points[index:])
        ),
    )
    segments[:] = [s for s in segments if s.segment_id != sid] + [first, last]
    return node_id, next_id + 1


def _mark_islands(segments):
    groups = defaultdict(list)
    for s in segments:
        groups[s.start_node_id, s.end_node_id].append(s.segment_id)
    islands = {
        sid: f"growth_island_{a}_{b}"
        for (a, b), ids in sorted(groups.items())
        if len(ids) == 2
        for sid in ids
    }
    return [
        replace(
            s,
            kind="island_bypass",
            metadata=dict(s.metadata, topology_role="island_arm", island_id=islands[s.segment_id]),
        )
        if s.segment_id in islands
        else s
        for s in segments
    ]


def generate_gallery_growth(generator, host):
    from plume_advanced.stages.network import CaveNode, CaveSegment

    cfg = generator.config
    if cfg.systems.count < 2 or (
        cfg.emplacement_backend != "internal"
    ):
        raise ValueError("Independent gallery growth requires multiple internal systems")
    if cfg.emplacement_history.stacked_lobe_fraction:
        raise ValueError("Independent gallery growth currently supports one layer")
    geometry = generator._build_flow_geometry(host)
    along, tracks, width = independent_preferences(generator, host, geometry)
    interactions: list[dict[str, Any]] = []
    interconnected = cfg.topology.style == "interconnected"
    connection_check = None
    if interconnected:
        from plume_advanced.stages.network_interconnected import connection_guard
        connection_check = connection_guard(generator, host, geometry, along, tracks, width)
    planned, records = plan_interactions(
        along, tracks, width, cfg.systems, events=interactions,
        local_spacing=interconnected, connection_check=connection_check,
        simple_splits=not interconnected,
    )

    def world(a, c):
        return np.column_stack(
            (
                geometry.seed_x + a * geometry.flow_x + c * geometry.cross_x,
                geometry.seed_y + a * geometry.flow_y + c * geometry.cross_y,
            )
        )

    nodes = []
    for i, (a, c, kind) in enumerate(planned):
        position = world(np.array([a]), np.array([c]))[0]
        nodes.append(CaveNode(i, float(position[0]), float(position[1]), a, c, kind))
    phases = generator._sample_int_range(
        procedural_rng(cfg.random_seed, "emplacement-phase-count"),
        cfg.emplacement_history.phase_count,
    )
    segments = []
    for sid, (start, end, first, last, group) in enumerate(records):
        a = along[first : last + 1]
        c = tracks[list(group), first : last + 1].mean(axis=0).copy()
        # Keep local gallery confluences inside the junction neighborhood;
        # a twelve-width blend can fuse distinct arms far before their node.
        reach = min(0.4 * (a[-1] - a[0]), (12 if interconnected else 4) * width)
        for delta, distance in (
            (nodes[start].lateral_offset - c[0], a - a[0]),
            (nodes[end].lateral_offset - c[-1], a[-1] - a),
        ):
            u = np.clip(distance / max(reach, 1e-9), 0, 1)
            c += delta * (1 - u * u * (3 - 2 * u))
        rng = procedural_rng(cfg.random_seed, "gallery-width", sid)
        phase = float(rng.uniform(0, 2 * np.pi))
        scale = 1.0 if len(group) == cfg.systems.count else 0.72
        widths = np.clip(
            width
            * scale
            * (
                1
                + cfg.topology.width_variation
                * (
                    0.72 * np.sin(a / (9 * width) * 2 * np.pi + phase)
                    + 0.28 * np.sin(a / (4.3 * width) * 2 * np.pi - 0.7 * phase)
                )
            ),
            2 * cfg.minimum_passage_radius,
            1.9 * cfg.maximum_passage_radius,
        )
        metadata = generator._build_emplacement_metadata(
            kind="backbone",
            phase_count=phases,
            birth_phase=0,
            death_phase=phases - 1,
            formation_state="independent_arterial",
        )
        metadata.update(
            system_ids=list(group),
            shared_system_count=len(group),
            front_run_id=sid,
            front_start_station_m=float(a[0]),
            front_end_station_m=float(a[-1]),
            topology_role="feeder" if nodes[start].kind == "entry" else "trunk",
            topology_style=cfg.topology.style,
            network_process="interconnected_systems_v1" if interconnected else "independent_gallery_v1",
            active_phases=list(range(phases)),
            path_id=f"front_{sid}",
            phase_flux_budget=cfg.source_flux
            * cfg.systems.count
            * cfg.emplacement_history.phase_flux_budget_fraction,
        )
        if nodes[start].kind == "entry":
            metadata.update(
                source_system_id=group[0],
                system_seed=derive_subseed(cfg.random_seed, "network-system", group[0]),
            )
        segments.append(
            CaveSegment(
                sid, start, end, "backbone", 0, _points(host, world(a, c), widths), metadata
            )
        )
    segments = _mark_islands(segments)
    # Screen actual grown front topology before doing more expensive history
    # and section work. No synthetic island is inserted to meet the target.
    count = len({s.metadata["island_id"] for s in segments if s.metadata.get("island_id")})
    if not interconnected and not cfg.topology.island_count[0] <= count <= cfg.topology.island_count[1]:
        raise ValueError(
            f"Independent routes produced {count} local island splits outside the requested range"
        )
    segments, history = grow_phase_history(generator, host, nodes, segments, phases)
    segments = generator._assign_conserved_flow(nodes, segments)
    segments = refresh_phase_discharge(generator, nodes, segments)
    segments = generator._annotate_emplacement_flux_history(segments)
    segments = add_pool_history(generator, host, geometry, nodes, segments)
    # Record actual station coordinates for newly inserted branch/pool nodes.
    nodes = [
        replace(
            n,
            along_position=(n.x - geometry.seed_x) * geometry.flow_x
            + (n.y - geometry.seed_y) * geometry.flow_y,
            lateral_offset=(n.x - geometry.seed_x) * geometry.cross_x
            + (n.y - geometry.seed_y) * geometry.cross_y,
        )
        for n in nodes
    ]
    skeleton = np.zeros_like(host.growth_cost, dtype=bool)
    for s in segments:
        for p in s.points:
            skeleton[generator._world_to_cell(host, p.x, p.y)] = True
    return generator._finish_network(
        host,
        geometry,
        nodes,
        segments,
        backend_provenance=dict(
            backend="internal",
            version="interconnected_systems_v1" if interconnected else "independent_gallery_v1",
            flow_direction=[geometry.flow_x, geometry.flow_y],
            interaction_spacing_scope="participating_fronts" if interconnected else "global",
            system_count=cfg.systems.count,
            system_seeds=[
                derive_subseed(cfg.random_seed, "network-system", i)
                for i in range(cfg.systems.count)
            ],
            interaction_events=interactions,
            phase_events=history,
            implemented_features=[
                "independent_host_routing",
                "capture_release",
                "phase_discharge",
                "blind_breakouts",
                "cooling_retirement",
                "passage_reoccupation",
                "drained_pools",
            ],
            unsupported_features=[
                "stacked_levels",
                "vertical_capture",
                "deposition_feedback",
                "legacy_cell_lobe_grammar",
                "full_thermofluid_solver",
            ],
        ),
        skeleton_mask=skeleton,
        total_flux=np.zeros_like(host.growth_cost),
    )


def grow_phase_history(generator, host, nodes, segments, phases):
    cfg, history = generator.config, generator.config.emplacement_history
    rng = procedural_rng(cfg.random_seed, "gallery-phase-events")
    target = int(
        rng.integers(cfg.topology.side_branch_count[0], cfg.topology.side_branch_count[1] + 1)
    )
    events = []
    width = 2 * cfg.base_passage_radius
    sites: list[tuple[float, float]] = []
    for phase in range(phases):
        # Old passages must actually have become inactive before reopening.
        for index, s in enumerate(segments):
            if s.metadata.get("topology_role") != "side_branch":
                continue
            active = s.metadata["active_phases"]
            if phase == max(active) + 1:
                events.append(dict(kind="inactive", phase=phase, segment_id=s.segment_id))
            if phase > max(active) + 1 and rng.random() < history.reoccupation_probability:
                span = generator._sample_int_range(rng, history.active_phase_span)
                active = sorted(set(active + list(range(phase, min(phases, phase + span)))))
                segments[index] = replace(
                    s,
                    metadata=dict(
                        s.metadata,
                        active_phases=active,
                        reoccupied=True,
                        reoccupation_phase=phase,
                        death_phase=max(active),
                    ),
                )
                events.append(dict(kind="reoccupation", phase=phase, segment_id=s.segment_id))
        if len(sites) >= target or rng.random() > history.breakout_probability:
            continue
        segments = generator._assign_conserved_flow(nodes, segments)
        candidates = []
        for s in segments:
            if (
                s.metadata.get("topology_role") != "trunk"
                or len(s.metadata.get("system_ids", [])) < 2
            ):
                continue
            for index, p in enumerate(s.points):
                if min(p.arc_length, s.total_length - p.arc_length) < 2 * width:
                    continue
                if any(math.hypot(p.x - x, p.y - y) < 4 * width for x, y in sites):
                    continue
                local = host.sample(p.x, p.y)
                score = cfg.lobe_growth.breakout_capacity_weight * (1 - local.flow_capacity)
                score += cfg.lobe_growth.breakout_confinement_weight * (1 - local.roof_competence)
                candidates.append((score + float(rng.uniform(0, 0.1)), s.segment_id, index))
        if not candidates:
            continue
        proposal = None
        for score, sid, index in sorted(candidates, reverse=True)[:16]:
            parent = next(s for s in segments if s.segment_id == sid)
            supply = parent.mean_flux
            allocated = min(
                supply * float(rng.uniform(*cfg.lobe_growth.branch_flux_fraction)),
                cfg.source_flux * cfg.systems.count * history.phase_flux_budget_fraction,
            )
            if allocated < cfg.source_flux * max(
                cfg.lobe_growth.minimum_viable_flux_fraction, history.retirement_flux_threshold
            ):
                events.append(dict(kind="flux_starved_breakout", phase=phase, parent_segment_id=sid))
                continue
            points, reason = trace_blind_breakout(generator, host, parent, index, rng, allocated)
            if (
                len(points) < 4
                or points[-1].arc_length < cfg.topology.side_branch_length_widths[0] * width * 0.8
                or not breakout_has_clearance(points, segments, sid)
                or any(not (host.x_coords[0] <= p.x <= host.x_coords[-1]
                            and host.y_coords[0] <= p.y <= host.y_coords[-1]) for p in points)
            ):
                continue
            proposal = points, reason
            break
        if proposal is None:
            continue
        points, reason = proposal
        start, new_sid = _split_segment(nodes, segments, sid, index)
        from plume_advanced.stages.network import CaveNode, CaveSegment

        end = max(n.node_id for n in nodes) + 1
        tip = points[-1]
        nodes.append(CaveNode(end, tip.x, tip.y, 0.0, 0.0, "spur_terminal"))
        span = generator._sample_int_range(rng, history.active_phase_span)
        active = list(range(phase, min(phases, phase + span)))
        metadata = generator._build_emplacement_metadata(
            kind="spur",
            phase_count=phases,
            birth_phase=phase,
            death_phase=max(active),
            formation_state="retired_breakout",
        )
        metadata.update(
            topology_style=cfg.topology.style,
            topology_role="side_branch",
            network_process="interconnected_systems_v1" if cfg.topology.style == "interconnected" else "independent_gallery_v1",
            active_phases=active,
            quality_terminal_taper=True,
            initial_flux=allocated,
            lobe_path_id=f"gallery_breakout_{new_sid}",
            branching_process="finite_budget_metric_breakout",
            breakout_trigger="host_capacity_and_confinement",
            breakout_score=float(score),
            branch_flux_fraction=allocated / max(supply, 1e-9),
            parent_flux_before_split=supply,
            phase_flux_allocated=allocated,
            retirement_reason=reason,
            path_id=f"breakout_{new_sid}",
            phase_flux_budget=cfg.source_flux
            * cfg.systems.count
            * history.phase_flux_budget_fraction,
            parent_segment_id=sid,
            system_ids=[],
            reoccupied=False,
        )
        segments.append(CaveSegment(new_sid, start, end, "spur", 0, points, metadata))
        sites.append((points[0].x, points[0].y))
        events.append(
            dict(
                kind="breakout",
                phase=phase,
                segment_id=new_sid,
                path_id=f"breakout_{new_sid}",
                phase_flux_budget=cfg.source_flux
                * cfg.systems.count
                * history.phase_flux_budget_fraction,
                parent_segment_id=sid,
                allocated_flux=allocated,
                phase_budget=cfg.source_flux
                * cfg.systems.count
                * history.phase_flux_budget_fraction,
                retirement_reason=reason,
            )
        )
    return segments, events


def breakout_has_clearance(
    points: Sequence[CavePoint], segments: Sequence[CaveSegment], parent_id: int
) -> bool:
    """Conservatively exclude blind branches that cut through another passage.

    The connected parent is handled by the final curvature/crossing checks.
    Other routes use point-to-chord distances and half a branch sampling step
    of extra clearance, covering the intervals between sampled branch points.
    """
    if len(points) < 2:
        return False
    starts, ends, radii = [], [], []
    for segment in segments:
        if segment.segment_id == parent_id:
            continue
        for a, b in zip(segment.points, segment.points[1:]):
            starts.append((a.x, a.y))
            ends.append((b.x, b.y))
            radii.append(.5 * max(a.width, b.width))
    if not starts:
        return True
    start = np.asarray(starts)
    delta = np.asarray(ends) - start
    radius = np.asarray(radii)
    length_squared = np.sum(delta * delta, axis=1)
    branch = np.array([(p.x, p.y) for p in points])
    margin = .5 * np.linalg.norm(np.diff(branch, axis=0), axis=1).max(initial=0)
    for point, xy in zip(points, branch):
        fraction = np.clip(np.sum((xy - start) * delta, axis=1)
                           / np.maximum(length_squared, 1e-12), 0, 1)
        distances = np.linalg.norm(xy - start - fraction[:, None] * delta, axis=1)
        if np.any(distances < radius + .5 * point.width + margin):
            return False
    return True


def trace_blind_breakout(generator, host, parent, index, rng, flux):
    cfg = generator.config
    p = parent.points[index]
    tangent = np.array(
        [
            parent.points[index + 1].x - parent.points[index - 1].x,
            parent.points[index + 1].y - parent.points[index - 1].y,
        ]
    )
    tangent /= np.linalg.norm(tangent)
    cross = np.array([-tangent[1], tangent[0]])
    sign = float(rng.choice([-1, 1]))
    base = 2 * cfg.base_passage_radius
    reach = float(rng.uniform(*cfg.topology.side_branch_length_widths)) * base
    step = max(0.4, 0.15 * base)
    limit = generator._sample_int_range(rng, cfg.lobe_growth.maximum_steps)
    path_xy = [np.array([p.x, p.y])]
    direction = tangent.copy()
    distance = 0.0
    reason = "pulse_extent"
    temperature = p.temperature_k
    for _ in range(limit):
        if distance >= reach:
            break
        angle = sign * (0.15 + 0.7 * min(distance / (1.2 * base), 1.0))
        preferred = tangent * math.cos(angle) + cross * math.sin(angle)
        offsets = np.linspace(-0.16, 0.16, 7)
        options = np.array(
            [
                preferred * math.cos(o) + np.array([-preferred[1], preferred[0]]) * math.sin(o)
                for o in offsets
            ]
        )
        locations = path_xy[-1] + step * options
        costs = np.array([host.sample(*q).growth_cost for q in locations]) * cfg.growth_cost_weight
        costs += cfg.lobe_growth.inertia_weight * (1 - options @ direction)
        weights = np.exp(-(costs - costs.min()) / max(cfg.lobe_growth.candidate_temperature, 0.01))
        chosen = int(rng.choice(len(options), p=weights / weights.sum()))
        direction = 0.8 * direction + 0.2 * options[chosen]
        direction /= np.linalg.norm(direction)
        path_xy.append(path_xy[-1] + step * direction)
        distance += step
        temperature -= cfg.cooling_k_per_m * cfg.lobe_growth.exposed_cooling_multiplier * step
        if temperature <= cfg.lobe_growth.retirement_temperature_k:
            reason = "cooled_below_retirement_temperature"
            break
    xy = np.asarray(path_xy)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    widths = np.full(
        len(xy),
        max(
            2 * cfg.minimum_passage_radius,
            0.55 * base * float(np.clip((flux / cfg.source_flux) ** 0.5, 0.65, 1.15)),
        ),
    )
    u = np.clip((arc / max(arc[-1], 1e-9) - 0.55) / 0.45, 0, 1)
    widths *= 1 - 0.65 * u * u * (3 - 2 * u)
    result = _points(host, xy, widths)
    return tuple(
        replace(
            q,
            temperature_k=max(
                273.15,
                p.temperature_k
                - cfg.cooling_k_per_m * cfg.lobe_growth.exposed_cooling_multiplier * q.arc_length,
            ),
            flux=flux,
        )
        for q in result
    ), reason


def refresh_phase_discharge(generator, nodes, segments):
    """Route each pulse on its active graph, with a finite shared breakout budget.

    Reference point flux describes the union of preserved passages. Phase flux
    is a separate conserved ledger: dormant branches receive zero, and active
    blind branches draw only their requested allocation from upstream supply.
    """
    phase_count = max(int(s.metadata.get("emplacement_phase_count", 1)) for s in segments)
    flows = {s.segment_id: [0.0] * phase_count for s in segments}
    order = generator._topological_node_ids(nodes, segments)
    entry_ids = {n.node_id for n in nodes if n.kind == "entry"}
    source_supply: defaultdict[int, float] = defaultdict(float)
    for s in segments:
        if s.start_node_id in entry_ids:
            source_supply[s.start_node_id] += s.mean_flux
    for phase in range(phase_count):
        outgoing = defaultdict(list)
        available = defaultdict(float, source_supply)
        budget = (
            generator.config.source_flux
            * generator.config.systems.count
            * generator.config.emplacement_history.phase_flux_budget_fraction
        )
        for s in segments:
            if phase in s.metadata["active_phases"]:
                outgoing[s.start_node_id].append(s)
        for node_id in order:
            active = outgoing[node_id]
            branches = sorted(
                (s for s in active if s.metadata.get("topology_role") == "side_branch"),
                key=lambda s: s.segment_id,
            )
            fronts = [s for s in active if s.metadata.get("topology_role") != "side_branch"]
            remaining = available[node_id]
            for s in branches:
                allocation = min(
                    float(s.metadata["initial_flux"]),
                    budget,
                    remaining * 0.6 if fronts else remaining,
                )
                flows[s.segment_id][phase] = allocation
                available[s.end_node_id] += allocation
                budget -= allocation
                remaining -= allocation
            capacity = sum(s.mean_width**2 for s in fronts)
            for s in fronts:
                allocation = remaining * s.mean_width**2 / max(capacity, 1e-12)
                flows[s.segment_id][phase] = allocation
                available[s.end_node_id] += allocation
    return [
        replace(s, metadata=dict(s.metadata, phase_fluxes=flows[s.segment_id])) for s in segments
    ]


def assess_gallery_history(network, check):
    """Validate chronological activity, conserved pulses, and interaction evidence."""
    if network.config.topology.generation_mode != "independent_growth":
        return
    cfg = network.config
    phases = max(int(s.metadata.get("emplacement_phase_count", 1)) for s in network.segments)
    chronology, ledger, source_seeds = [], [], []
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for s in network.segments:
        incoming[s.end_node_id].append(s)
        outgoing[s.start_node_id].append(s)
        active = s.metadata.get("active_phases", [])
        flows = s.metadata.get("phase_fluxes", [])
        if (
            not active
            or any(type(i) is not int or not 0 <= i < phases for i in active)
            or active != sorted(set(active))
            or active[0] != s.metadata.get("birth_phase")
            or active[-1] != s.metadata.get("death_phase")
        ):
            chronology.append(s.segment_id)
        gaps = any(b > a + 1 for a, b in zip(active, active[1:]))
        if bool(s.metadata.get("reoccupied", False)) != gaps:
            chronology.append(s.segment_id)
        if (
            len(flows) != phases
            or not np.isfinite(flows).all()
            or min(flows, default=-1) < 0
            or any(f != 0 for i, f in enumerate(flows) if i not in active)
        ):
            ledger.append(s.segment_id)
        source = s.metadata.get("source_system_id")
        if source is not None and s.metadata.get("system_seed") != derive_subseed(
            cfg.random_seed, "network-system", source
        ):
            source_seeds.append(s.segment_id)
    check("growth_source_streams", not source_seeds, len(source_seeds), 0, source_seeds)
    check("phase_chronology", not chronology, len(chronology), 0, chronology)
    check("phase_activity_ledger", not ledger, len(ledger), 0, ledger)
    max_error, max_budget = 0.0, 0.0
    expected = cfg.source_flux * cfg.systems.count
    if not ledger:
        for phase in range(phases):
            supplied = drained = allocated = 0.0
            for node in network.nodes:
                before, after = incoming[node.node_id], outgoing[node.node_id]
                inflow = sum(s.metadata["phase_fluxes"][phase] for s in before)
                outflow = sum(s.metadata["phase_fluxes"][phase] for s in after)
                if before and after:
                    max_error = max(max_error, abs(inflow - outflow))
                elif not before:
                    supplied += outflow
                else:
                    drained += inflow
            allocated = sum(
                s.metadata["phase_fluxes"][phase]
                for s in network.segments
                if s.metadata.get("topology_role") == "side_branch"
            )
            max_budget = max(max_budget, allocated)
            max_error = max(max_error, abs(supplied - expected), abs(drained - expected))
    check("phase_discharge_conservation", not ledger and max_error <= 1e-8, max_error, 1e-8)
    budget = expected * cfg.emplacement_history.phase_flux_budget_fraction
    check("phase_breakout_budget", not ledger and max_budget <= budget + 1e-8, max_budget, budget)
    event_errors = []
    events = network.backend_provenance.get("interaction_events", [])
    front_edges = [s for s in network.segments if s.metadata.get("topology_role") != "side_branch"]
    nodes = {n.node_id: n for n in network.nodes}
    previous = -float("inf")
    previous_by_source: defaultdict[int, float] = defaultdict(lambda: -float("inf"))
    for event in events:
        node_id = event["node_id"]
        before = sorted(s.metadata["system_ids"] for s in front_edges if s.end_node_id == node_id)
        after = sorted(s.metadata["system_ids"] for s in front_edges if s.start_node_id == node_id)
        distance = event["preference_gap_m"]
        members = {i for group in event["before"] for i in group}
        if cfg.topology.style == "interconnected":
            previous = max(previous_by_source[i] for i in members)
        threshold = (
            (
                cfg.systems.merge_distance_widths
                if event["kind"] == "merge"
                else cfg.systems.split_distance_widths
            )
            * 2
            * cfg.base_passage_radius
        )
        valid = (
            distance <= threshold + 1e-8
            if event["kind"] == "merge"
            else distance >= threshold - 1e-8
        )
        if (
            before != sorted(event["before"])
            or after != sorted(event["after"])
            or node_id not in nodes
            or abs(nodes[node_id].along_position - event["station_m"]) > 1e-6
            or event["station_m"] - previous
            < cfg.systems.interaction_spacing_widths * 2 * cfg.base_passage_radius - 1e-8
            or not valid
        ):
            event_errors.append(node_id)
        previous = event["station_m"]
        for i in members:
            previous_by_source[i] = event["station_m"]
    graph_events = {
        n.node_id
        for n in network.nodes
        if sum(s.end_node_id == n.node_id for s in front_edges) > 1
        or sum(s.start_node_id == n.node_id for s in front_edges) > 1
    }
    check(
        "growth_interaction_evidence",
        not event_errors and graph_events == {e["node_id"] for e in events},
        len(event_errors),
        0,
        event_errors,
    )


def add_pool_history(generator, host, geometry, nodes, segments):
    from plume_advanced.stages.network import _SelectedPath

    if not generator.config.emplacement_history.drained_pool_enabled:
        return segments
    eligible = [
        s
        for s in segments
        if s.metadata.get("topology_role") == "trunk" and len(s.metadata.get("system_ids", [])) >= 2
    ]
    paths = []
    for s in eligible:
        cells: list[tuple[int, int]] = []
        for p in s.points:
            cell = generator._world_to_cell(host, p.x, p.y)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        paths.append(
            _SelectedPath(
                kind="backbone", path=tuple(cells), metadata=dict(s.metadata, coalesced=True)
            )
        )
    annotated = generator._annotate_drained_pool_chambers(
        host_field=host,
        geometry=geometry,
        selected_paths=tuple(paths),
        rng=procedural_rng(generator.config.random_seed, "drained-lava-pools"),
    )
    for s, path in zip(eligible, annotated):
        metadata = path.metadata or {}
        if metadata.get("chamber_type") != "drained_lava_pool":
            continue
        index = min(
            range(1, len(s.points) - 1),
            key=lambda i: math.hypot(
                s.points[i].x - metadata["pool_center_x_m"],
                s.points[i].y - metadata["pool_center_y_m"],
            ),
        )
        if (
            min(s.points[index].arc_length, s.total_length - s.points[index].arc_length)
            < 2 * generator.config.base_passage_radius
        ):
            continue
        point = s.points[index]
        metadata = dict(
            metadata,
            pool_center_x_m=point.x,
            pool_center_y_m=point.y,
            pool_center_along_m=(point.x - geometry.seed_x) * geometry.flow_x
            + (point.y - geometry.seed_y) * geometry.flow_y,
        )
        _split_segment(nodes, segments, s.segment_id, index, node_kind="chamber", metadata=metadata)
    return segments
