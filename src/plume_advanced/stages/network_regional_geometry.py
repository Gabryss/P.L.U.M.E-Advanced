"""Carry accepted metric routes through subsequent regional graph additions."""

import numpy as np


class AcceptedRoutes:
    def __init__(self, network=None):
        self.edges = {}
        self.positions = {}
        if network is None:
            return
        for segment in network.segments:
            cells = segment.metadata["regional_cell_path"]
            fractions = np.array(segment.metadata["regional_cell_fractions"])
            arc = np.array([p.arc_length for p in segment.points])
            points = np.array([[p.x, p.y, p.width] for p in segment.points])
            stations = fractions * arc[-1]
            for a, b, first, last in zip(cells, cells[1:], stations, stations[1:]):
                sampled = np.r_[first, arc[(arc > first) & (arc < last)], last]
                part = np.column_stack([np.interp(sampled, arc, points[:, k]) for k in range(3)])
                self.edges[a, b] = part
                self.positions[a], self.positions[b] = part[0, :2], part[-1, :2]

    def position(self, planner, cell):
        return self.positions.get(cell, planner.xy[cell])

    def assemble(self, planner, route, width):
        parts: list[np.ndarray] = []
        distances = [0.]
        preserved = True
        for a, b in zip(route, route[1:]):
            part = self.edges.get((a, b))
            if part is None:
                preserved = False
                part = np.column_stack((np.array([self.position(planner, a),
                                                  self.position(planner, b)]), [width, width]))
            distances.append(distances[-1] + np.linalg.norm(np.diff(part[:, :2], axis=0), axis=1).sum())
            parts.append(part if not parts else part[1:])
        points = np.concatenate(parts)
        # A cut may round to an existing sample. Do not feed zero-length
        # intervals into metric interpolation or cubic junction fitting.
        points = points[np.r_[True, np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > 1e-9]]
        return points, (np.array(distances) / distances[-1]).tolist(), preserved
