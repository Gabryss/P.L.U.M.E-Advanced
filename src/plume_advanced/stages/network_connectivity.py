"""Bounded host-routed confluences between otherwise separate regional systems."""

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from plume_advanced.progress import report_progress
from plume_advanced.stages.network_connectivity_space import connection_space
from plume_advanced.stages.network_front_capture import capture_path


def component_labels(nodes, edges):
    """Deterministic undirected connectivity, without imposing reverse flow."""
    adjacency: dict[int, set[int]] = {int(node): set() for node in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    labels = {}
    for root in sorted(adjacency):
        if root in labels:
            continue
        labels[root] = root
        pending = [root]
        while pending:
            for other in sorted(adjacency[pending.pop()]):
                if other not in labels:
                    labels[other] = root
                    pending.append(other)
    return labels


def connect_paths(planner, paths, *, validate=None, extra_connections=0):
    """Join components by real forks/merges while retaining each original path.

    Select bounded, spatially distributed endpoint pairs, then search heading
    states on the existing directed host graph. No straight-line fallback,
    source relocation, outlet deletion or upstream edge is permitted. Metric
    geometry must pass the validation callback before a layered path is kept;
    rejected proposals use the same bounded search budget. Optional extra links
    join different passage chains within the connected graph, creating a small
    number of alternate routes. An unavailable optional link is recorded and
    skipped; it never rejects a valid connected network.
    """
    p = planner
    if type(extra_connections) is not int or not 0 <= extra_connections <= 8:
        raise ValueError("extra_connections must be an integer from 0 to 8")
    layered = bool(np.any(p.layer_ids))
    if (layered or extra_connections) and validate is None:
        raise ValueError("Layered connectivity repair or extra connections requires metric geometry validation")
    additions: list[list[int]] = []
    edges = {(a, b) for path in paths for a, b in zip(path, path[1:])}
    labels = component_labels({cell for path in paths for cell in path}, edges)
    initial = len(set(labels.values()))
    required = max(0, initial - 1)
    total = required + extra_connections
    audit: dict[str, Any] = dict(initial_components=initial, final_components=initial, connections=[],
                 searches=0, expanded_states=0, state_limit=total * 8192,
                 maximum_searches_per_connection=64, maximum_sites=512,
                 geometry_rejections=0)
    if extra_connections:
        audit.update(extra_requested=extra_connections, extra_accepted=0, extra_attempts=[])
    extra_endpoints: list[int] = []
    for iteration in range(total):
        optional = iteration >= required
        # Reserve room for the scarcer descending connection first, then
        # alternate with same-level links. Neither type is forced when the
        # host has no viable route; each slot has a bounded search allowance.
        layer_change_target = 1 - (iteration - required) % 2 if layered else 0
        occupied = set(labels)
        positions, passage_distance, costs = connection_space(p, paths + additions)
        # A long ramp edge can cross occupied space even when both endpoint
        # cells are clear. Reuse the feeder's swept-volume penalties when
        # seeking optional layer links, so safe alternatives rank ahead of
        # cheap routes that full geometry inspection would reject.
        search_weights = (p.weights * p.feeder_penalties(paths + additions)
                          if optional and layered else p.weights)
        outgoing, incoming = defaultdict(set), defaultdict(set)
        for a, b in sorted(edges):
            outgoing[a].add(b)
            incoming[b].add(a)
        junctions = [cell for cell in sorted(occupied)
                     if len(outgoing[cell]) > 1 or len(incoming[cell]) > 1]
        g = p.geometry
        stations = (p.xy - [g.seed_x, g.seed_y]) @ [g.flow_x, g.flow_y]
        earliest = float(stations[p.sources].max()) + p.independent_length
        sites = [cell for cell in sorted(occupied)
                 if cell not in p.sources and cell not in p.goals
                 and len(outgoing[cell]) == len(incoming[cell]) == 1
                 and all(p.layer_ids[other] == p.layer_ids[cell]
                         for other in outgoing[cell] | incoming[cell])
                 and stations[cell] >= earliest]
        if junctions and sites:
            clearance = cKDTree(positions[junctions]).query(positions[sites])[0]
            sites = [cell for cell, distance in zip(sites, clearance) if distance >= 4 * p.width]
        if optional and extra_endpoints and sites:
            spacing = max(6 * p.width, p.config.regional.minimum_branch_length_m)
            clearance = cKDTree(positions[extra_endpoints]).query(positions[sites])[0]
            sites = [cell for cell, distance in zip(sites, clearance) if distance >= spacing]
        if len(sites) > 512:
            sites = [sites[i] for i in np.linspace(0, len(sites) - 1, 512).astype(int)]
        directions = {cell: p.xy[min(outgoing[cell])] - p.xy[cell] for cell in sites}
        directions = {cell: delta / np.linalg.norm(delta) for cell, delta in directions.items()}
        minimum = max(p.config.regional.minimum_branch_length_m, 4 * p.width)
        maximum = p.config.regional.maximum_branch_length_m
        angle_limit = np.cos(np.radians(min(45, p.config.quality.maximum_turn_degrees)))
        # Removing junctions divides existing geometry into unbranched chains.
        # Optional links must connect different chains, not duplicate a reach
        # of the same passage with a nearly parallel shortcut.
        chains = {}
        if optional:
            interior = occupied - set(junctions)
            chains = component_labels(interior, [(a, b) for a, b in edges if a in interior and b in interior])
        pairs = []
        for start in sites:
            for end in sites:
                layer_change = int(p.layer_ids[end] - p.layer_ids[start])
                if ((labels[start] != labels[end] if optional else labels[start] == labels[end])
                        or (optional and (layer_change != layer_change_target or chains[start] == chains[end]))
                        or layer_change not in (0, 1)
                        or p.potential[start] <= p.potential[end]):
                    continue
                delta = p.xy[end] - p.xy[start]
                distance = float(np.linalg.norm(delta))
                if not minimum <= distance <= maximum:
                    continue
                direction = delta / distance
                if min(direction @ directions[start], direction @ directions[end]) < angle_limit - 1e-12:
                    continue
                # A nearly tangential shortcut runs beside its parent for too
                # long and overlaps once given a finite passage width. Require
                # room to separate from both local passage tangents; do not
                # solve connectivity with a second almost coincident tube.
                offsets = [abs(delta[0] * directions[cell][1] - delta[1] * directions[cell][0])
                           for cell in (start, end)]
                if layer_change == 0 and min(offsets) < 2.5 * p.width:
                    continue
                # Cost ranks proposals only. The search evaluates every host
                # edge, including barriers, competence and grade restrictions.
                cost = distance * (costs[start] + costs[end]) / 2
                pairs.append((float(cost), start, end))
        by_start = defaultdict(list)
        start_cost: dict[int, float] = {}
        for cost, start, end in sorted(pairs):
            by_start[start].append(end)
            start_cost.setdefault(start, cost)
        seen: dict[tuple, int] = defaultdict(int)
        proposals = []
        ordered = sorted(by_start, key=lambda cell: (start_cost[cell], cell))
        if optional:
            # A single cheap reach must not consume every optional search.
            # Round-robin across passage chains (and hence across levels),
            # retaining cost order within each chain and stable tie breaking.
            groups = defaultdict(list)
            for cell in ordered:
                groups[int(p.layer_ids[cell]), chains[cell]].append(cell)
            queues = sorted(groups.values(), key=lambda cells: (start_cost[cells[0]], cells[0]))
            ordered = [cells[index] for index in range(max(map(len, queues), default=0))
                       for cells in queues if index < len(cells)]
        for start in ordered:
            key = (int(p.layer_ids[start]), *np.floor(p.xy[start] / (4 * p.width)).astype(int))
            if seen[key] >= (1 if optional else 3):
                continue
            seen[key] += 1
            proposals.append((start, sorted(set(by_start[start]))))
            if len(proposals) == 64:
                break
        spent = 0
        searches_before, rejections_before = audit['searches'], audit['geometry_rejections']
        committed = False
        for start, targets in proposals:
            if spent >= 8192:
                break
            status = (f"{audit['extra_accepted']}/{extra_connections} links accepted; "
                      f"{'ramp' if layer_change_target else 'same-level'} search" if optional
                      else f"{len(set(labels.values()))} components")
            report_progress("Extra network connections" if optional else "Regional connectivity", iteration, total,
                            f"{status}; "
                            f"{audit['searches']} route searches, {audit['expanded_states']} states")
            ds = np.linalg.norm(positions - positions[start], axis=1)
            de = cKDTree(positions[targets]).query(positions)[0]
            departure, receiving = ds <= 4 * p.width, de <= 4 * p.width
            if optional:
                departure &= p.layer_ids == p.layer_ids[start]
                receiving &= np.isin(p.layer_ids, p.layer_ids[targets])
            allowed = (passage_distance >= 1.1 * p.width) | departure | receiving
            allowed &= (p.layer_ids >= p.layer_ids[start]) & (p.layer_ids <= max(p.layer_ids[targets]))
            keep = allowed[p.rows] & allowed[p.cols]
            graph = csr_matrix((search_weights[keep], (p.rows[keep], p.cols[keep])), shape=p.graph.shape)
            # Ramps can be far from the nearest plan-view target. Reverse host
            # costs give an admissible lower bound (turn costs are additional),
            # guiding the same bounded heading search toward a feasible ramp.
            guide = dijkstra(graph.T.tocsr(), indices=targets, min_only=True) if layered else None
            path, expanded = capture_path(
                p, graph, start, directions[start], p.step, outgoing, occupied,
                targets, maximum, maximum_length_m=maximum,
                maximum_expansions=min(2048, 8192 - spent),
                cost_to_targets=guide,
            )
            spent += expanded
            audit['expanded_states'] += expanded
            audit['searches'] += 1
            if path is None:
                continue
            if validate is not None and not validate(path):
                audit['geometry_rejections'] += 1
                continue
            end = path[-1]
            additions.append(path)
            edges.update(zip(path, path[1:]))
            labels = component_labels(occupied | set(path), edges)
            count = len(set(labels.values()))
            if count != (1 if optional else initial - iteration - 1):
                raise ValueError("Regional connectivity route failed to join exactly two components")
            audit['connections'].append(dict(start_cell=start, end_cell=end,
                                              length_m=float(np.linalg.norm(np.diff(p.xy[path], axis=0), axis=1).sum()),
                                              remaining_components=count))
            if layered:
                audit['connections'][-1].update(
                    start_layer=int(p.layer_ids[start]), end_layer=int(p.layer_ids[end]),
                    ramp_count=int(sum(p.layer_ids[a] != p.layer_ids[b] for a, b in zip(path, path[1:]))),
                )
            audit['final_components'] = count
            committed = True
            if optional:
                extra_endpoints.extend((start, end))
                audit['extra_accepted'] += 1
                audit['connections'][-1]['optional'] = True
            break
        if optional:
            audit['extra_attempts'].append(dict(
                kind="descending_ramp" if layer_change_target else "same_layer",
                accepted=committed, searches=audit['searches'] - searches_before,
                expanded_states=spent, geometry_rejections=audit['geometry_rejections'] - rejections_before,
                reason="accepted" if committed else "no_valid_connection_within_budget",
            ))
        elif not committed:
            raise ValueError(
                f"Regional connectivity repair exhausted its bounded search: "
                f"{audit['final_components']} components remain; {audit['searches']} searches, "
                f"{audit['expanded_states']} states"
            )
    report_progress("Extra network connections" if extra_connections else "Regional connectivity", total, total,
                    f"{audit.get('extra_accepted', 0)}/{extra_connections} extra connections accepted"
                    if extra_connections else "one connected network; inspecting connections next")
    return additions, audit
