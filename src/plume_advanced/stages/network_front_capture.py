"""Bounded look-ahead for a front that has encountered a receiving passage."""

import heapq

import numpy as np


def capture_path(planner, graph, start, heading, previous_step, outgoing, occupied,
                 targets, remaining_length, *, maximum_expansions=256, maximum_length_m=None,
                 cost_to_targets=None):
    """Search position/heading states on the same directed host graph.

    Front targets are discovered locally after independent growth. Connectivity
    routing can explicitly supply a longer metric limit. No new graph edges,
    or upstream moves are introduced. Front graphs exclude ramps; connectivity
    graphs may include existing host-screened descending ramps. A valid terminal must
    align with the receiving passage; other occupied cells are barriers.
    """
    p = planner
    if not targets or remaining_length <= 0:
        return None, 0
    limit = min(float(remaining_length), 6 * p.width if maximum_length_m is None else maximum_length_m)
    targets = set(targets)
    # Immutable state ids retain a consistent parent chain when a cheaper
    # arrival supersedes an existing (previous-cell, current-cell) state.
    states = [(start, np.asarray(heading), previous_step, -1, 0.)]
    def lower_bound(cell):
        return float(cost_to_targets[cell]) if cost_to_targets is not None else 0.

    if not np.isfinite(lower_bound(start)):
        return None, 0
    pending = [(lower_bound(start), 0., 0)]
    best = {(-1, start): 0.}
    expanded = 0
    while pending and expanded < maximum_expansions:
        _, cost, index = heapq.heappop(pending)
        cell, direction, step_before, parent, length = states[index]
        key = (states[parent][0] if parent >= 0 else -1, cell)
        if cost > best[key] + 1e-9:
            continue
        expanded += 1
        if cell in targets:
            route = [cell]
            while parent >= 0:
                route.append(states[parent][0])
                parent = states[parent][3]
            return route[::-1], expanded
        first, last = graph.indptr[cell:cell + 2]
        for other, edge_cost in zip(graph.indices[first:last], graph.data[first:last]):
            other = int(other)
            if not np.isfinite(lower_bound(other)):
                continue
            delta = p.xy[other] - p.xy[cell]
            step = float(np.linalg.norm(delta))
            following = delta / step
            turn = float(np.arccos(np.clip(following @ direction, -1, 1)))
            if (length + step > limit
                    or np.degrees(turn) > p.config.quality.maximum_turn_degrees
                    or (step_before + step) / max(2 * turn, 1e-9)
                    < p.width * p.config.quality.minimum_bend_radius_widths):
                continue
            if other in occupied:
                if other not in targets:
                    continue
                receiving = p.xy[min(outgoing[other])] - p.xy[other]
                receiving /= np.linalg.norm(receiving)
                if following @ receiving < np.cos(np.radians(45)) - 1e-12:
                    continue
            next_cost = cost + float(edge_cost) * (1 + turn*turn)
            if next_cost >= best.get((cell, other), float("inf")) - 1e-9:
                continue
            best[cell, other] = next_cost
            states.append((other, following, step, index, length + step))
            heapq.heappush(pending, (next_cost + lower_bound(other), next_cost, len(states)-1))
    return None, expanded
