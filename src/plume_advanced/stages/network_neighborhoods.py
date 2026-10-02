"""Bounded passage neighborhoods, independent of degree-two subdivision.

Only connected arms can share a junction exemption. Both samples must lie
inside its geodesic radius; a nearby unrelated passage remains a conflict.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree


class PassageNeighborhoods:
    def __init__(self, arrays, *, widths=None):
        self.access = []
        self.widths = []
        incident = defaultdict(list)
        node_width: defaultdict[int, float] = defaultdict(float)
        lengths = []
        stations = []
        for i, (segment, xyz) in enumerate(arrays):
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            stations.append(arc)
            lengths.append(float(arc[-1]))
            raw = np.array([p.arc_length for p in segment.points])
            # XYZ length can differ from plan length on a ramp. Sample widths
            # by the same normalized stations used to densify that passage.
            plan = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
            self.widths.append(np.interp(plan, raw, [p.width for p in segment.points])
                               if widths is None else np.asarray(widths[i]))
            for node, width in ((segment.start_node_id, self.widths[-1][0]),
                                (segment.end_node_id, self.widths[-1][-1])):
                incident[node].append(i)
                node_width[node] = max(node_width[node], width)
        self.degrees = {n: len(v) for n, v in incident.items()}
        self.stations = stations
        self.radii = {n: 4 * w for n, w in node_width.items()}
        bound = 4 * max((float(w.max()) for w in self.widths), default=0.)
        for i, (segment, _) in enumerate(arrays):
            access: dict[int, np.ndarray] = {}
            for node, distance in ((segment.start_node_id, stations[i]),
                                   (segment.end_node_id, lengths[i] - stations[i])):
                previous, walked, visited = i, 0., {i}
                while walked <= bound:
                    values = distance + walked
                    access[node] = np.minimum(access.get(node, values), values)
                    if len(incident[node]) != 2:
                        break
                    following = next(j for j in incident[node] if j != previous)
                    if following in visited:
                        break
                    visited.add(following)
                    other = arrays[following][0]
                    walked += lengths[following]
                    node = (other.end_node_id if other.start_node_id == node
                            else other.start_node_id)
                    previous = following
            self.access.append(access)

    def local(self, first, point, second, candidates, *, point_fraction=0., candidate_fraction=0.):
        """Mask a connected local region, optionally at interpolated stations."""
        candidates = np.asarray(candidates, dtype=int)

        def at(values, indices, fraction):
            if fraction == 0.:
                return values[indices]
            return values[indices] * (1 - fraction) + values[indices + 1] * fraction

        radius = 2 * (at(self.widths[first], point, point_fraction)
                      + at(self.widths[second], candidates, candidate_fraction))
        result = np.zeros(len(candidates), dtype=bool)
        if first == second:
            result |= abs(at(self.stations[first], point, point_fraction)
                          - at(self.stations[second], candidates, candidate_fraction)) <= radius
        a, b = self.access[first], self.access[second]
        for node in a.keys() & b.keys():
            da = at(a[node], point, point_fraction)
            db = at(b[node], candidates, candidate_fraction)
            if self.degrees[node] > 2:
                result |= (da <= self.radii[node]) & (db <= self.radii[node])
            else:
                # A continuous passage can be arbitrarily subdivided. Limit
                # the exemption by along-passage distance, not proximity alone.
                result |= da + db <= radius
        return result


def passage_conflicts(arrays, *, vertical_clearance=None):
    """Sample-pair clearance with bounded temporary arrays and local widths."""
    neighborhoods = PassageNeighborhoods(arrays)
    trees = [cKDTree(xyz[:, :2]) for _, xyz in arrays]
    low = [xyz[:, :2].min(axis=0) for _, xyz in arrays]
    high = [xyz[:, :2].max(axis=0) for _, xyz in arrays]
    conflicts: set[int] = set()
    for i, (a, xyz) in enumerate(arrays):
        for j in range(i, len(arrays)):
            b, other = arrays[j]
            limit = .7 * (neighborhoods.widths[i].max() + neighborhoods.widths[j].max())
            if np.any(low[i] > high[j] + limit) or np.any(low[j] > high[i] + limit):
                continue
            for offset in range(0, len(xyz), 512):
                neighbors = trees[j].query_ball_point(xyz[offset:offset+512, :2], limit)
                counts = np.fromiter(map(len, neighbors), dtype=int, count=len(neighbors))
                if not counts.sum():
                    continue
                first = np.repeat(np.arange(offset, offset + len(neighbors)), counts)
                second = np.concatenate(neighbors).astype(int)
                if i == j:
                    # Check distant portions of one passage too. Otherwise a
                    # loopback is rejected only after graph subdivision.
                    unique = first < second
                    first, second = first[unique], second[unique]
                    if not len(first):
                        continue
                widths = neighborhoods.widths[i][first] + neighborhoods.widths[j][second]
                close = np.linalg.norm(xyz[first, :2] - other[second, :2], axis=1) < .7 * widths
                close &= ~neighborhoods.local(i, first, j, second)
                if vertical_clearance is not None:
                    close &= abs(xyz[first, 2] - other[second, 2]) < vertical_clearance
                if close.any():
                    conflicts.update((a.segment_id, b.segment_id))
                    break
    return sorted(conflicts)
