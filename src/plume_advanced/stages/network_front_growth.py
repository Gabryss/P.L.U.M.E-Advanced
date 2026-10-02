"""Endpoint-free branch proposals on the immutable, directed host graph.

Branches follow local host cost, a persistent heading and passage occupancy.
There is no selected destination or hidden detour to truncate. Formation supply
is a bounded procedural lifetime, not a thermal or hydraulic calculation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from plume_advanced.procedural import procedural_rng
from plume_advanced.stages.network_front_capture import capture_path


@dataclass(frozen=True)
class FrontProposal:
    path: tuple[int, ...]
    role: str
    termination: str
    length_m: float
    steps: int
    capture_started_m: float | None = None
    capture_expanded_states: int = 0
    capture_searches: int = 0


def branch_opportunities(config, extent):
    """Sample a bounded birth count, not a quota of successful connections."""
    mean = extent * config.regional.branches_per_km / 1000
    if mean <= 0:
        return 0
    rng = procedural_rng(config.random_seed, "regional-front-births")
    # Clip before sampling too: huge user density must not overflow Poisson.
    return min(config.regional.maximum_branches, int(rng.poisson(min(mean, 1e6))))


class FrontGrower:
    def __init__(self, planner):
        self.planner = planner
        # Exclude ramps: new branches grow within their current level. Existing
        # retained trunks and screened ramps provide inter-level connectivity.
        same = planner.layer_ids[planner.rows] == planner.layer_ids[planner.cols]
        self.graph = csr_matrix((planner.weights[same],
                                (planner.rows[same], planner.cols[same])),
                               shape=planner.graph.shape)
        self.graph.sort_indices()

    def grow(self, start, paths, attempt):
        p, cfg = self.planner, self.planner.config
        controls = cfg.regional
        rng = procedural_rng(cfg.random_seed, "regional-free-front", attempt)
        occupied = sorted({cell for path in paths for cell in path})
        occupied_set = set(occupied)
        distance = cKDTree(p.routing_positions[occupied]).query(p.routing_positions)[0]
        outgoing: dict[int, list[int]] = {}
        for path in paths:
            for a, b in zip(path, path[1:]):
                outgoing.setdefault(a, []).append(b)
        following = sorted(set(outgoing.get(start, [])))
        if not following:
            return FrontProposal((start,), "blind_branch", "no_parent_direction", 0., 0)
        heading = p.xy[following[0]] - p.xy[start]
        heading /= np.linalg.norm(heading)
        initial = heading.copy()
        side = -1 if rng.random() < .5 else 1
        # Log-distributed travel budget separates short-lived branches from
        # sustained ones without selecting an endpoint or a reconnection.
        lifetime = float(np.exp(rng.uniform(np.log(controls.minimum_branch_length_m),
                                          np.log(controls.maximum_branch_length_m))))
        path, length, previous_step = [start], 0., p.step
        separated = False
        max_steps = int(np.ceil(controls.maximum_branch_length_m / min(
            np.diff(p.x).min(), np.diff(p.y).min()))) + 2
        termination = "step_budget"
        capture_started = None
        capture_expanded = capture_searches = 0
        for _ in range(max_steps):
            cell = path[-1]
            capture = None
            # Encounter a receiving passage only within a local forward cone,
            # after independent growth. This target is discovered during growth,
            # never selected at the birth of the branch.
            if separated and length >= controls.minimum_branch_length_m:
                near = np.flatnonzero(np.linalg.norm(p.xy[occupied] - p.xy[cell], axis=1)
                                      <= 4 * p.width)
                candidates = []
                for index in near:
                    other = occupied[index]
                    delta = p.xy[other] - p.xy[cell]
                    distance_to = float(np.linalg.norm(delta))
                    if (other not in outgoing or other in p.sources or distance_to < 1e-9
                            or p.layer_ids[other] != p.layer_ids[cell]
                            or p.potential[other] >= p.potential[cell]):
                        continue
                    direction = delta / distance_to
                    receiving = p.xy[min(outgoing[other])] - p.xy[other]
                    receiving /= np.linalg.norm(receiving)
                    if direction @ heading > .5 and direction @ receiving > .5:
                        candidates.append((distance_to, other, direction))
                if candidates:
                    _, _, capture = min(candidates, key=lambda item: item[:2])
                    if capture_started is None:
                        capture_started = length
                    if capture_searches < 4:
                        tail, expanded = capture_path(
                            p, self.graph, cell, heading, previous_step, outgoing, occupied_set,
                            [item[1] for item in candidates], controls.maximum_branch_length_m-length)
                        capture_expanded += expanded
                        capture_searches += 1
                        if tail is not None:
                            path.extend(tail[1:])
                            length += float(np.linalg.norm(np.diff(p.xy[tail], axis=0), axis=1).sum())
                            return FrontProposal(tuple(path), "bypass", "local_capture", length,
                                                 len(path)-1, capture_started,
                                                 capture_expanded, capture_searches)
            first, last = self.graph.indptr[cell:cell + 2]
            choices = []
            for other, cost in zip(self.graph.indices[first:last], self.graph.data[first:last]):
                other = int(other)
                delta = p.xy[other] - p.xy[cell]
                step = float(np.linalg.norm(delta))
                direction = delta / step
                turn = float(np.arccos(np.clip(direction @ heading, -1, 1)))
                radius = (previous_step + step) / max(2 * turn, 1e-9)
                if (np.degrees(turn) > cfg.quality.maximum_turn_degrees
                        or radius < p.width * cfg.quality.minimum_bend_radius_widths):
                    continue
                merge = other in occupied_set
                if merge:
                    if not separated or length + step < controls.minimum_branch_length_m:
                        continue
                    if other in p.sources or other not in outgoing:
                        continue
                    receiving = p.xy[min(outgoing[other])] - p.xy[other]
                    receiving /= np.linalg.norm(receiving)
                    if direction @ receiving < np.cos(np.radians(45)):
                        continue
                else:
                    from_birth = np.linalg.norm(p.xy[other] - p.xy[start])
                    # A front can approach a possible future merge only after
                    # sustaining an independent passage. Final metric screening
                    # still checks the complete accepted, smoothed geometry.
                    if (from_birth > 3 * p.width and distance[other] < .8 * p.width
                            and not (separated and length >= controls.minimum_branch_length_m)):
                        continue
                if length + step > controls.maximum_branch_length_m:
                    continue
                lateral = initial[0] * direction[1] - initial[1] * direction[0]
                birth_bias = 2.0 * side * lateral * np.exp(-length / (4 * p.width))
                clearance_penalty = .6 * np.exp(-distance[other] / p.width)
                # Cost/length is local host preference. Heading persistence
                # suppresses cell-to-cell white-noise zigzags.
                # A local-cost-only walker locks onto grid axes. The common
                # outlet potential provides downstream steering without naming
                # a receiving passage. Its influence grows after separation.
                descent = (p.potential[cell] - p.potential[other]) / max(cost, 1e-9)
                steering = (1 - np.exp(-length / (5 * p.width))) * (1 - descent)
                score = float(.2 * cost / step + .2 * turn * turn
                              + 2 * steering + clearance_penalty - birth_bias)
                if capture is not None:
                    score += 3 * (1 - direction @ capture)
                if merge:
                    score -= 1.0
                choices.append((score, other, step, direction, merge))
            if not choices:
                termination = "blocked"
                break
            choices.sort(key=lambda value: (value[0], value[1]))
            # Only near-equivalent local choices are stochastic; named streams
            # make a rejected proposal independent of future random draws.
            shortlist = [item for item in choices if item[0] <= choices[0][0] + .15]
            chosen = shortlist[int(rng.integers(len(shortlist)))]
            _, other, step, heading, merge = chosen
            path.append(other)
            length += step
            previous_step = step
            separated |= distance[other] >= 1.4 * p.width
            if merge:
                return FrontProposal(tuple(path), "bypass", "encountered_passage", length,
                                     len(path)-1, capture_started, capture_expanded, capture_searches)
            if length >= lifetime and separated:
                termination = "supply_budget"
                break
        if length < controls.minimum_branch_length_m or not separated:
            termination = "insufficient_independent_growth"
        return FrontProposal(tuple(path), "blind_branch", termination, length, len(path)-1,
                             capture_started, capture_expanded, capture_searches)
