"""Bounded graph growth over regional host routing, with local repair and inspection."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from plume_advanced.procedural import procedural_rng
from plume_advanced.progress import report_progress
from plume_advanced.stages.network_metadata import SegmentMetadataValue
from plume_advanced.stages.network_regional_routing import RegionalPlanner


def _edges(paths):
    return sorted({(a, b) for path in paths for a, b in zip(path, path[1:])})


def _assess_feeders(network, host, *, defer_connectivity=False):
    """Inspect feeders before branches and, optionally, before their connections."""
    from plume_advanced.stages.network_quality import assess_network

    config = replace(network.config, systems=replace(network.config.systems, require_split=False))
    report = assess_network(replace(network, config=config), host)
    if defer_connectivity:
        # Only the internal feeder phase can defer this check. The connected
        # phase and final acceptance always inspect the complete graph.
        # Separate feeders may have no confluences yet; creating them is the
        # next phase's job. Geometry, sources and supplied termini still apply.
        deferred = {"regional_connected_components", "system_merge_opportunity", "layer_connections"}
        report["checks"] = [c for c in report["checks"] if c["name"] not in deferred]
        report["accepted"] = all(c["passed"] for c in report["checks"] if c["severity"] == "error")
    return report


def _network_from_paths(generator, host, planner, paths, sources, events, path_types, previous=None):
    from plume_advanced.stages.network import CaveNode, CaveSegment
    from plume_advanced.stages.network_gallery_growth import _points
    from plume_advanced.stages.network_morphology import route_capacity
    from plume_advanced.stages.network_regional_geometry import AcceptedRoutes

    accepted_routes = AcceptedRoutes(previous)

    outgoing, incoming = defaultdict(list), defaultdict(list)
    owners: dict[tuple[int, int], int] = {}
    for index, path in enumerate(paths):
        for edge in zip(path, path[1:]):
            owners.setdefault(edge, index - len(sources))
    for a, b in _edges(paths):
        outgoing[a].append(b)
        incoming[b].append(a)
    cells = sorted(set(outgoing) | set(incoming))
    transitions = {a for a, b in _edges(paths) if planner.layer_ids[a] != planner.layer_ids[b]}
    if generator.config.layers.enabled and generator.config.layers.preserve_layer_trunks:
        transitions.update(
            b for a, b in _edges(paths) if planner.layer_ids[a] != planner.layer_ids[b]
        )
    junctions = sorted(
        c
        for c in cells
        if c in sources
        or c in planner.goals
        or c in transitions
        or len(outgoing[c]) != 1
        or len(incoming[c]) != 1
    )
    node_ids = {cell: i for i, cell in enumerate(junctions)}
    g, cfg = planner.geometry, generator.config
    nodes = []
    for cell in junctions:
        x, y = accepted_routes.position(planner, cell)
        nodes.append(
            CaveNode(
                node_ids[cell],
                float(x),
                float(y),
                float((x - g.seed_x) * g.flow_x + (y - g.seed_y) * g.flow_y),
                float((x - g.seed_x) * g.cross_x + (y - g.seed_y) * g.cross_y),
                "entry"
                if cell in sources
                else "exit"
                if cell in planner.goals
                else "terminal"
                if not outgoing[cell]
                else "junction",
            )
        )
    segments: list[CaveSegment] = []
    preserved_segments = set()
    relaxation = (list(previous.backend_provenance.get("route_relaxation", []))
                  if previous is not None else [])
    for first in junctions:
        for successor in sorted(outgoing[first]):
            route = [first, successor]
            while route[-1] not in node_ids:
                route.append(outgoing[route[-1]][0])
            sid = len(segments)
            meta: dict[str, SegmentMetadataValue] = dict(
                network_process="regional_growth_v1",
                topology_style="interconnected",
                topology_role="trunk",
                regional_start_potential=float(planner.potential[first]),
                regional_end_potential=float(planner.potential[route[-1]]),
            )
            if first in sources:
                meta["source_system_id"] = sources.index(first)
            if cfg.layers.enabled:
                meta.update(
                    regional_start_layer=int(planner.layer_ids[first]),
                    regional_end_layer=int(planner.layer_ids[route[-1]]),
                )
            owner = owners[(route[0], route[1])]
            role = path_types[owner + len(sources)]
            meta.update(
                regional_path_id=owner + len(sources),
                regional_route_type=(
                    "descending_ramp"
                    if planner.layer_ids[first] != planner.layer_ids[route[-1]]
                    else role
                ),
                regional_capacity=route_capacity(
                    cfg, owner, retained=role in {"retained_trunk", "distributary", "connection"}, blind=role == "blind_branch"
                ),
            )
            if role in {"bypass", "blind_branch"}:
                meta["regional_branch_id"] = owner
            if role == "blind_branch":
                meta["regional_blind_terminal"] = not outgoing[route[-1]]
            if cfg.layers.enabled:
                meta["regional_layer_depths_m"] = planner.depths.tolist()
            # Preserve source lineage separately from front identity. After a
            # split both arms may transport material from the same sources.
            segment_width = np.clip(
                0.8 * planner.width,
                2 * cfg.minimum_passage_radius,
                1.9 * cfg.maximum_passage_radius,
            )
            metric, fractions, preserved = accepted_routes.assemble(planner, route, segment_width)
            if (cfg.regional.branch_growth == "front" and not preserved
                    and planner.layer_ids[first] == planner.layer_ids[route[-1]]
                    and not any((a, b) in accepted_routes.edges for a, b in zip(route, route[1:]))):
                from plume_advanced.stages.network_route_relaxation import relax_route

                metric[:, :2], audit = relax_route(planner, metric[:, :2], planner.layer_ids[first])
                relaxation.append(dict(start_cell=first, end_cell=route[-1], **audit))
                # Cell ownership follows the displaced vertices, preserving
                # subsequent cuts at the same metric positions.
                lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(metric[:, :2], axis=0), axis=1))]
                fractions = (lengths / lengths[-1]).tolist()
            meta.update(regional_cell_path=route, regional_cell_fractions=fractions)
            if preserved:
                preserved_segments.add(sid)
            segments.append(
                CaveSegment(
                    sid,
                    node_ids[first],
                    node_ids[route[-1]],
                    "abandoned_lobe" if role == "blind_branch" else "backbone",
                    int(planner.layer_ids[first]),
                    _points(host, metric[:, :2], metric[:, 2]),
                    meta,
                )
            )
    skeleton = np.zeros_like(host.growth_cost, dtype=bool)
    for xy in planner.xy[cells]:
        skeleton[generator._world_to_cell(host, *xy)] = True
    result = generator._finish_network(
        host,
        g,
        nodes,
        segments,
        backend_provenance=dict(
            backend="internal",
            version="regional_growth_v1",
            scope="network_only",
            **(
                dict(
                    layer_count=cfg.layers.count,
                    layer_connections=sum(
                        s.metadata.get("regional_start_layer")
                        != s.metadata.get("regional_end_layer")
                        for s in segments
                    ),
                    layer_model="host-relative levels with descending ramps and reference height/rock separation",
                    layer_depths_m=planner.depths.tolist(),
                    layer_extents_m=planner.extents.tolist(),
                )
                if cfg.layers.enabled
                else {}
            ),
            flow_direction=[g.flow_x, g.flow_y],
            planning_grid_cells=len(planner.xy),
            planning_step_m=float(planner.step),
            growth_events=list(events),
            route_relaxation=relaxation,
            local_repair_history=(list(previous.backend_provenance.get("local_repair_history", []))
                                  if previous is not None else []),
            source_cells=sources,
            **(dict(connectivity_repair=planner.connectivity_audit)
               if hasattr(planner, "connectivity_audit") else {}),
            **(
                dict(outlet_cells=planner.goals)
                if (cfg.layers.enabled and cfg.layers.preserve_layer_trunks) or cfg.regional.outlet_count > 1
                else {}
            ),
            routing_potential=("positive directed host cost to nearest terminus; not hydraulic head"
                               if cfg.regional.outlet_count > 1 else
                               "positive directed host cost to outlet; not hydraulic head"),
        ),
        skeleton_mask=skeleton,
        total_flux=np.zeros_like(host.growth_cost),
        preserved_segments=preserved_segments,
    )
    return result


def generate_regional_network(generator, host):
    """Grow source routes and locally screened branches over the same host."""
    from plume_advanced.stages.network_quality import assess_network
    from plume_advanced.stages.network_systems import GenerationDomainError

    cfg = generator.config
    if (
        cfg.systems.count < 2
        or cfg.emplacement_backend != "internal"
        or cfg.emplacement_history.stacked_lobe_fraction
    ):
        raise GenerationDomainError(
            "Regional growth requires multiple internal systems without stacked_lobe_fraction"
        )
    planner: RegionalPlanner
    if cfg.layers.enabled:
        from plume_advanced.stages.network_layer_routing import LayeredRegionalPlanner

        planner = LayeredRegionalPlanner(generator, host)
    else:
        planner = RegionalPlanner(generator, host)
    g = planner.geometry
    sources = planner.sources
    paths: list[list[int]] = []
    path_types = ["source_route"] * len(sources)
    for source_index, source in enumerate(sources):
        report_progress(
            "Regional source feeders",
            source_index,
            len(sources),
            "routing with inlet separation and downstream capture checks",
        )
        path = planner.primary_path(source)
        destination = path[-1]
        if paths:
            occupied = sorted({p for route in paths for p in route})
            distance = cKDTree(planner.routing_positions[occupied]).query(
                planner.routing_positions
            )[0]
            travel = np.linalg.norm(planner.xy - planner.xy[source], axis=1)
            independent = cfg.systems.minimum_independent_length_widths * planner.width
            penalty = 1 + 12 * np.exp(-distance / (2 * planner.width)) * np.exp(
                -travel / (2 * independent)
            )
            edge_penalty = (penalty[planner.rows] + penalty[planner.cols]) / 2
            if cfg.layers.enabled:
                assert isinstance(planner, LayeredRegionalPlanner)
                edge_penalty = planner.feeder_penalties(paths)
            graph = csr_matrix(
                (
                    planner.weights * edge_penalty,
                    (planner.rows, planner.cols),
                ),
                shape=planner.graph.shape,
            )
            _, previous = dijkstra(graph.T.tocsr(), indices=destination, return_predecessors=True)
            path = [source]
            while path[-1] != destination:
                next_cell = int(previous[path[-1]])
                if next_cell < 0:
                    raise GenerationDomainError(
                        "No independent regional feeder can reach the outlet"
                    )
                path.append(next_cell)
            existing_next = defaultdict(list)
            for a, b in _edges(paths):
                existing_next[a].append(b)
            occupied = sorted({p for route in paths for p in route})
            # Capture can reuse a passage, but must not abandon this feeder's
            # distinct terminus and collapse the result back to one outlet.
            can_reach_destination = {destination}
            if cfg.regional.outlet_count > 1:
                for cell in sorted(occupied, key=lambda q: (planner.potential[q], q)):
                    if any(q in can_reach_destination for q in existing_next[cell]):
                        can_reach_destination.add(cell)
            for index in range(2, len(path)):
                first = path[index - 1]
                distance_from_inlet = (
                    planner.xy[first] - np.array([g.seed_x, g.seed_y])
                ) @ np.array([g.flow_x, g.flow_y])
                if distance_from_inlet < planner.independent_length:
                    continue
                distance = np.linalg.norm(planner.xy[occupied] - planner.xy[first], axis=1)
                choices = sorted(
                    (float(d), q)
                    for d, q in zip(distance, occupied)
                    if d <= 2 * planner.width
                    and q not in sources
                    and planner.layer_ids[q] == planner.layer_ids[first]
                    and planner.potential[q] < planner.potential[first]
                    and (cfg.regional.outlet_count == 1 or q in can_reach_destination)
                )
                if not choices:
                    continue
                if cfg.layers.enabled:
                    aligned = []
                    incoming_direction = planner.xy[first] - planner.xy[path[index - 2]]
                    for distance_to_target, q in choices:
                        connection = planner.xy[q] - planner.xy[first]
                        following_cells = existing_next.get(q, [])
                        if distance_to_target < 1e-9 or not following_cells:
                            continue
                        outgoing_direction = planner.xy[sorted(following_cells)[0]] - planner.xy[q]
                        if all(
                            float(connection @ d) >= 0.75 * distance_to_target * np.linalg.norm(d)
                            for d in (incoming_direction, outgoing_direction)
                        ):
                            aligned.append((distance_to_target, q))
                    choices = aligned
                    if not choices:
                        continue
                _, target = choices[0]
                sample = planner.sample_host(np.linspace(planner.xy[first], planner.xy[target], 32))
                grade = np.diff(sample[:, 0]) / max(
                    np.linalg.norm(planner.xy[target] - planner.xy[first]) / 31, 1e-9
                )
                if not (
                    np.all(sample[:, 1:] > 0) and np.all(grade <= cfg.quality.maximum_uphill_grade)
                ):
                    continue
                path = path[:index] + [target]
                while path[-1] != destination:
                    choices = sorted(existing_next[path[-1]])
                    if cfg.regional.outlet_count > 1:
                        choices = [q for q in choices if q in can_reach_destination]
                    path.append(choices[0])
                break
        paths.append(path)
    report_progress(
        "Regional source feeders", len(sources), len(sources),
        f"feeders routed to {len({path[-1] for path in paths})} downstream terminus/termini"
    )
    feeder_history = []
    if cfg.regional.outlet_count > 1:
        continuations = planner.terminal_continuations(paths)
        paths.extend(continuations)
        path_types.extend(["distributary"] * len(continuations))
    if cfg.layers.enabled and cfg.layers.preserve_layer_trunks:
        assert isinstance(planner, LayeredRegionalPlanner)
        report_progress(
            "Retained layer trunks", detail="continuing upper passages beyond descending junctions"
        )
        retained = planner.retained_trunks(paths)
        paths.extend(retained)
        path_types.extend(["retained_trunk"] * len(retained))
    events: list[dict] = []
    network = _network_from_paths(generator, host, planner, paths, sources, events, path_types)
    controls = cfg.regional
    from plume_advanced.stages.network_morphology import branch_weights, branch_zones

    zones = branch_zones(planner)
    target = min(
        controls.maximum_branches, int(math.ceil(g.along_extent * controls.branches_per_km / 1000))
    )
    front = None
    work_budget = None
    if controls.branch_growth == "front":
        from plume_advanced.stages.network_front_growth import FrontGrower, branch_opportunities
        from plume_advanced.stages.network_local_repair import LocalRepairBudget

        front = FrontGrower(planner)
        target = branch_opportunities(cfg, g.along_extent)
        work_budget = LocalRepairBudget(2400 * cfg.quality.repair_passes)
    trials = min(4, controls.attempts_per_branch) if front else controls.attempts_per_branch
    successful_births = set()
    accepted = 0
    needs_connections = cfg.regional.outlet_count > 1 or cfg.layers.enabled
    if front or needs_connections:
        from plume_advanced.stages.network_acceptance import repair_network

        phases = ("feeders", "connections") if needs_connections else ("feeders",)
        for phase in phases:
            defer_connectivity = phase == "feeders" and needs_connections
            if phase == "connections":
                from plume_advanced.stages.network_connectivity import connect_paths

                connected_paths, connected_types = list(paths), list(path_types)

                def validate_connection(path):
                    nonlocal network
                    # Preserve inspected passages and test finite-width
                    # geometry before committing a graph connection. Failed
                    # proposals leave both the network and routes unchanged.
                    candidate = _network_from_paths(
                        generator, host, planner, connected_paths + [path], sources,
                        events, connected_types + ["connection"], previous=network
                    )
                    if not _assess_feeders(candidate, host, defer_connectivity=True)["accepted"]:
                        return False
                    network = candidate
                    connected_paths.append(path)
                    connected_types.append("connection")
                    return True

                _, planner.connectivity_audit = connect_paths(planner, paths, validate=validate_connection)
                paths, path_types = connected_paths, connected_types
                network = replace(network, backend_provenance=dict(
                    network.backend_provenance, connectivity_repair=planner.connectivity_audit
                ))
            report = _assess_feeders(network, host, defer_connectivity=defer_connectivity)
            # Use the existing feeder + outer-repair allowance here, while growth
            # can still resume. Never retry a completed phase from coarse paths.
            for repair in range(2 * cfg.quality.repair_passes):
                if report["accepted"]:
                    break
                feeder_history.append(dict(
                    repair=repair + 1,
                    **(dict(phase=phase) if needs_connections else {}),
                    failed_checks=[c["name"] for c in report["checks"] if not c["passed"]],
                ))
                report_progress("Regional feeder repair", repair, 2 * cfg.quality.repair_passes,
                                "repairing connections before resuming branch growth")
                network = repair_network(
                    generator,
                    host,
                    network,
                    repair,
                    failed_checks=[c for c in report["checks"] if not c["passed"]],
                    work_budget=work_budget,
                )
                report = _assess_feeders(network, host, defer_connectivity=defer_connectivity)
                feeder_history[-1].update(
                    accepted_after=report["accepted"],
                    remaining_failures=[c["name"] for c in report["checks"] if not c["passed"]],
                    local_repair_history=network.backend_provenance.get("local_repair_history", []),
                )
            report_progress("Regional feeder repair", len(feeder_history), len(feeder_history),
                            "feeders accepted; growing branches" if report["accepted"]
                            else "feeder repair exhausted; trying the next candidate")
            if not report["accepted"]:
                # Branches cannot repair invalid feeder geometry. Return the
                # inspected base to the bounded candidate loop immediately.
                return replace(
                    network,
                    backend_provenance=dict(
                        network.backend_provenance,
                        growth_events=[
                            dict(
                                accepted=False,
                                reason="invalid_feeder_geometry",
                                checks=[c["name"] for c in report["checks"] if not c["passed"]],
                            )
                        ],
                        accepted_branches=0,
                        requested_branches=target,
                        branch_budget_exhausted=True,
                        regional_growth_completed=False,
                        feeder_repair_history=feeder_history,
                        **(work_budget.provenance() if work_budget else {}),
                    ),
                )
    for attempt in range(target * trials):
        if accepted >= target:
            break
        if front and attempt // trials in successful_births:
            continue
        report_progress(
            "Regional branch proposals",
            attempt,
            target * trials,
            f"{accepted}/{target} accepted; host routing, junction and clearance checks",
        )
        rng = procedural_rng(cfg.random_seed, "regional-branch", attempt)
        edges = _edges(paths)
        outgoing: defaultdict[int, list[int]] = defaultdict(list)
        for path, role in zip(paths, path_types):
            if role != "blind_branch" or front:
                for a, b in zip(path, path[1:]):
                    if b not in outgoing[a]:
                        outgoing[a].append(b)
        occupied = sorted({p for path in paths for p in path})
        candidates = sorted(
            p
            for p in outgoing
            if p not in sources
            and np.linalg.norm(planner.xy[p] - planner.xy[planner.goal])
            >= controls.minimum_branch_length_m
        )
        if not candidates:
            break
        if cfg.layers.enabled and cfg.layers.preserve_layer_trunks:
            # Rotate the proposal layer, rather than letting a long deep
            # trunk absorb nearly every bypass in the growth budget.
            selected_layer = attempt % cfg.layers.count
            candidates = [p for p in candidates if planner.layer_ids[p] == selected_layer]
            if not candidates:
                events.append(dict(attempt=attempt, accepted=False, reason="no_layer_branch_start"))
                continue
        probabilities = branch_weights(planner, candidates, zones)
        start = (
            candidates[int(rng.integers(len(candidates)))]
            if probabilities is None
            else int(rng.choice(candidates, p=probabilities))
        )
        growth_detail = {}
        if front:
            grown = front.grow(start, paths, attempt)
            proposal, role = list(grown.path), grown.role
            growth_detail = dict(growth_method="front", birth=attempt // trials,
                                 termination=grown.termination, growth_steps=grown.steps,
                                 capture_started_m=grown.capture_started_m,
                                 capture_expanded_states=grown.capture_expanded_states,
                                 capture_searches=grown.capture_searches,
                                 destination_preselected=False)
            if grown.termination in {"insufficient_independent_growth", "no_parent_direction"}:
                events.append(dict(attempt=attempt, accepted=False, reason=grown.termination,
                                   **growth_detail))
                continue
        else:
            proposal, role, reason = _detour_proposal(planner, start, outgoing, occupied, rng, attempt)
            if proposal is None:
                events.append(dict(attempt=attempt, accepted=False, reason=reason))
                continue
        added = set(zip(proposal, proposal[1:])) - set(edges)
        added_length = float(
            sum(np.linalg.norm(planner.xy[b] - planner.xy[a]) for a, b in sorted(added))
        )
        if added_length < controls.minimum_branch_length_m:
            events.append(dict(attempt=attempt, accepted=False,
                               reason="insufficient_new_passage_length", **growth_detail))
            continue
        if not added or max(cKDTree(planner.routing_positions[occupied]).query(
                planner.routing_positions[proposal])[0]) < planner.width:
            events.append(dict(attempt=attempt, accepted=False, reason="no_separate_passage", **growth_detail))
            continue
        candidate = _network_from_paths(
            generator, host, planner, paths + [proposal], sources, events, path_types + [role],
            previous=network,
        )
        report = assess_network(candidate, host)
        from plume_advanced.stages.network_acceptance import repair_network

        repairs = 0
        while not report["accepted"] and repairs < cfg.quality.repair_passes:
            candidate = repair_network(
                generator, host, candidate, repairs,
                failed_checks=[c for c in report["checks"] if not c["passed"]], localize=True,
                work_budget=work_budget,
            )
            report = assess_network(candidate, host)
            repairs += 1
        failures = [c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "error"]
        if front:
            growth_detail["local_repairs"] = candidate.backend_provenance.get("local_repair_history", [])[len(
                network.backend_provenance.get("local_repair_history", [])):]
            growth_detail["route_relaxation"] = candidate.backend_provenance.get("route_relaxation", [])[len(
                network.backend_provenance.get("route_relaxation", [])):]
        if failures:
            events.append(dict(attempt=attempt, accepted=False, reason="geometry_or_flow",
                               checks=failures, **growth_detail))
            continue
        events.append(dict(
            attempt=attempt, accepted=True, repair_passes=repairs, start_cell=start,
            end_cell=proposal[-1], route_type=role, **growth_detail,
            interpretation="construction order, not eruption chronology",
            branch_id=len(paths) - len(sources),
            added_length_m=sum(s.total_length for s in candidate.segments
                              if s.metadata.get("regional_branch_id") == len(paths) - len(sources)),
            coarse_added_length_m=added_length,
        ))
        paths.append(proposal)
        path_types.append(role)
        network = candidate
        accepted += 1
        successful_births.add(attempt // trials)

    report_progress("Regional branch proposals", len(events), len(events),
                    f"{accepted}/{target} branch opportunities survived inspection")
    if controls.extra_connections:
        from plume_advanced.stages.network_connectivity import connect_paths

        extra_paths, extra_types = list(paths), list(path_types)
        rejected_checks: dict[str, int] = defaultdict(int)
        construction_errors = []

        def validate_extra(path):
            nonlocal network
            # Optional enrichment cannot move existing routes to manufacture
            # room. Retain only a connection that passes complete inspection.
            try:
                candidate = _network_from_paths(
                    generator, host, planner, extra_paths + [path], sources, events,
                    extra_types + ["connection"], previous=network,
                )
            except ValueError as error:
                # A rejected optional fit must not discard an otherwise valid
                # cave or consume a new generation seed.
                rejected_checks["connection_construction"] += 1
                construction_errors.append(str(error))
                return False
            assessment = assess_network(candidate, host)
            if not assessment["accepted"]:
                for check in assessment["checks"]:
                    if not check["passed"] and check["severity"] == "error":
                        rejected_checks[check["name"]] += 1
                return False
            network = candidate
            extra_paths.append(path)
            extra_types.append("connection")
            return True

        _, extra_audit = connect_paths(planner, paths, validate=validate_extra,
                                      extra_connections=controls.extra_connections)
        extra_audit["rejected_checks"] = dict(sorted(rejected_checks.items()))
        extra_audit["construction_errors"] = construction_errors
        network = replace(network, backend_provenance=dict(
            network.backend_provenance, extra_connections=extra_audit,
        ))
    return replace(network, backend_provenance=dict(
        network.backend_provenance, growth_events=events, branch_zones=zones,
        branch_growth=controls.branch_growth, accepted_branches=accepted,
        requested_branches=target, branch_budget_exhausted=accepted < target and front is None,
        branch_opportunities=target if front else None,
        unsuccessful_opportunities=target-accepted if front else None,
        regional_growth_completed=True, feeder_repair_history=feeder_history,
        **(work_budget.provenance() if work_budget else {}),
    ))


def _detour_proposal(planner, start, outgoing, occupied, rng, attempt):
    """Explicit fixed-destination comparison model."""
    cfg, controls = planner.config, planner.config.regional
    length = float(
        rng.uniform(controls.minimum_branch_length_m, controls.maximum_branch_length_m)
    )
    chain, travelled = [start], 0.0
    while chain[-1] in outgoing and travelled < length:
        successors = sorted(outgoing[chain[-1]])
        following = successors[int(rng.integers(len(successors)))]
        travelled += float(np.linalg.norm(planner.xy[following] - planner.xy[chain[-1]]))
        chain.append(following)
    if travelled < controls.minimum_branch_length_m:
        return None, "bypass", "insufficient_downstream_length"
    proposal = planner.detour(start, chain[-1], occupied, rng)
    if proposal is None:
        return None, "bypass", "no_viable_detour"
    role = "bypass"
    blind_rng = procedural_rng(cfg.random_seed, "regional-blind-proposal", attempt)
    if blind_rng.random() < controls.blind_branch_fraction:
        # Keep a separated prefix of a feasible detour. No fictional
        # collapse is assigned: this is an explicit blind-termination proxy.
        lengths = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(planner.xy[proposal], axis=0), axis=1))
        ]
        separation = cKDTree(planner.routing_positions[occupied]).query(
            planner.routing_positions[proposal]
        )[0]
        eligible = np.flatnonzero(
            (lengths >= controls.minimum_branch_length_m)
            & (lengths <= 0.8 * lengths[-1])
            & (planner.layer_ids[proposal] == planner.layer_ids[start])
            & (separation > 2 * planner.width)
        )
        if not len(eligible):
            return None, "blind_branch", "no_blind_termination"
        stop = int(blind_rng.choice(eligible))
        proposal = proposal[: stop + 1]
        role = "blind_branch"
    return proposal, role, "detour"


def assess_regional_systems(network, check):
    """Source lineage is allowed on both split arms; discharge cannot duplicate."""
    from plume_advanced.stages.network import CaveNetworkGenerator
    from plume_advanced.stages.network_systems import annotate_source_lineage

    incoming, outgoing = defaultdict(list), defaultdict(list)
    for segment in network.segments:
        incoming[segment.end_node_id].append(segment)
        outgoing[segment.start_node_id].append(segment)
    sources = [
        s.metadata.get("source_system_id")
        for n in network.nodes
        if n.kind == "entry"
        for s in outgoing[n.node_id]
    ]
    check(
        "system_sources",
        all(type(s) is int for s in sources)
        and sorted(s for s in sources if type(s) is int)
        == list(range(network.config.systems.count)),
        len(sources),
        network.config.systems.count,
    )
    try:
        order = CaveNetworkGenerator._topological_node_ids(
            list(network.nodes), list(network.segments)
        )
        expected = annotate_source_lineage(network.segments, order)
        bad = [
            s.segment_id
            for s, t in zip(network.segments, expected)
            if s.metadata.get("contributing_system_ids") != t.metadata["contributing_system_ids"]
        ]
    except (ValueError, KeyError, TypeError):
        bad = [s.segment_id for s in network.segments]
    check("system_source_lineage", not bad, len(bad), 0, bad)
    bad_sources = [
        n.node_id
        for n in network.nodes
        if n.kind == "entry" and (incoming[n.node_id] or len(outgoing[n.node_id]) != 1)
    ]
    check("regional_source_isolation", not bad_sources, len(bad_sources), 0)
    independent = [
        outgoing[n.node_id][0].total_length
        for n in network.nodes
        if n.kind == "entry" and outgoing[n.node_id]
    ]
    minimum = (
        network.config.systems.minimum_independent_length_widths
        * 2
        * network.config.base_passage_radius
    )
    check(
        "regional_source_persistence",
        bool(independent) and min(independent) >= minimum,
        min(independent, default=0),
        minimum,
    )
    merges = sum(len(v) > 1 for v in incoming.values())
    splits = sum(len(v) > 1 for v in outgoing.values())
    check(
        "system_merge_opportunity",
        not network.config.systems.require_merge or merges > 0,
        merges,
        1,
    )
    check(
        "system_split_opportunity",
        not network.config.systems.require_split or splits > 0,
        splits,
        1,
    )
    expected_flux = network.config.source_flux * network.config.systems.count
    inlet = sum(
        s.mean_flux for n in network.nodes if n.kind == "entry" for s in outgoing[n.node_id]
    )
    outlet = sum(s.mean_flux for s in network.segments if not outgoing[s.end_node_id])
    error = max(abs(inlet - expected_flux), abs(outlet - expected_flux)) / max(expected_flux, 1e-9)
    check("system_total_discharge", error < 1e-8, error, 1e-8)


def assess_regional_geometry(network, host, check):
    """Metric checks independent of a fixed global downstream coordinate."""
    # Outer geometric repair cannot resume skipped growth. Keep an unfinished
    # candidate rejected so bounded search tries the next reproducible seed.
    completed = network.backend_provenance.get("regional_growth_completed", True)
    check("regional_growth_completed", completed is True, completed, True)
    from plume_advanced.stages.network_connectivity import component_labels

    labels = component_labels((n.node_id for n in network.nodes),
                              ((s.start_node_id, s.end_node_id) for s in network.segments))
    components = len(set(labels.values()))
    check("regional_connected_components", components == 1, components, 1)
    if network.config.regional.outlet_count > 1:
        exits = {node.node_id for node in network.nodes if node.kind == "exit"}
        expected = network.config.regional.outlet_count
        check("regional_terminal_count", len(exits) == expected, len(exits), expected)
        outgoing_exits = sorted({s.start_node_id for s in network.segments} & exits)
        check("regional_terminal_sinks", not outgoing_exits, len(outgoing_exits), 0, outgoing_exits)
    roles = {"source_route", "retained_trunk", "distributary", "connection", "bypass", "blind_branch", "descending_ramp"}
    invalid = []
    for s in network.segments:
        capacity = s.metadata.get("regional_capacity")
        role = s.metadata.get("regional_route_type")
        if (
            role not in roles
            or type(capacity) not in (int, float)
            or not math.isfinite(capacity)
            or capacity <= 0
        ):
            invalid.append(s.segment_id)
        if network.config.layers.enabled:
            a, b = (s.metadata.get(f"regional_{k}_layer") for k in ("start", "end"))
            if (a != b) != (role == "descending_ramp"):
                invalid.append(s.segment_id)
    check("regional_route_semantics", not invalid, len(set(invalid)), 0, invalid)
    bad = [
        s.segment_id
        for s in network.segments
        if s.metadata.get("regional_start_potential", 0)
        <= s.metadata.get("regional_end_potential", 0)
    ]
    check("regional_directed_potential", not bad, len(bad), 0, bad)
    branch_lengths: defaultdict[int, float] = defaultdict(float)
    for s in network.segments:
        if "regional_branch_id" in s.metadata:
            branch_id = s.metadata["regional_branch_id"]
            if not isinstance(branch_id, int):
                raise ValueError("regional_branch_id must be an integer")
            branch_lengths[branch_id] += s.total_length
    short = [
        bid
        for bid, length in branch_lengths.items()
        if length < network.config.regional.minimum_branch_length_m
    ]
    check("regional_sustained_branch_length", not short, len(short), 0)
    substrate = []
    arrays = []
    for s in network.segments:
        xy = np.array([(p.x, p.y) for p in s.points])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        step = (
            min(1.0, host.x_coords[1] - host.x_coords[0], host.y_coords[1] - host.y_coords[0]) / 2
            if host is not None else .5
        )
        distances = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1] / step)) + 1))
        dense = np.column_stack([np.interp(distances, arc, xy[:, k]) for k in range(2)])
        arrays.append((s, dense))
        if host is None:
            continue
        coords = [
            (dense[:, 1] - host.y_coords[0]) / (host.y_coords[1] - host.y_coords[0]),
            (dense[:, 0] - host.x_coords[0]) / (host.x_coords[1] - host.x_coords[0]),
        ]
        fields = np.stack(
            [
                map_coordinates(getattr(host, name), coords, order=1, mode="nearest")
                for name in ("cover_thickness", "roof_competence", "growth_cost")
            ]
        )
        if not np.isfinite(fields).all() or np.any(fields[:2] <= 0):
            substrate.append(s.segment_id)
    check("regional_viable_substrate", not substrate, len(substrate), 0, substrate)
    if network.config.layers.enabled:
        from plume_advanced.stages.network_layers import assess_layers

        assess_layers(network, host, check)
        return
    from plume_advanced.stages.network_neighborhoods import passage_conflicts

    proximity = passage_conflicts(arrays)
    check("regional_passage_clearance", not proximity, len(set(proximity)), 0, proximity)
