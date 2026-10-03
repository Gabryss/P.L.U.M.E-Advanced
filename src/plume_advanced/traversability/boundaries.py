"""Exact cell-edge rings for sampled masks, preserving holes and islands."""

from collections import defaultdict

import numpy as np
from scipy import ndimage

from plume_advanced.progress import report_progress


def mask_rings(mask, origin, resolution):
    """Vectorize a four-connected cell mask without smoothing or filling holes.

    Each ring has occupied cells to its left: outer rings are counterclockwise,
    holes clockwise. Coordinates are grid corners, not sample centres. Regions
    meeting at one corner remain separate. Component IDs are local to this mask.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError("Boundary mask must be two-dimensional")
    labels, _ = ndimage.label(mask)
    padded = np.pad(mask, 1)
    # Directions E, N, W, S. The interior always lies on the left of an edge.
    neighbours = (padded[:-2, 1:-1], padded[1:-1, 2:],
                  padded[2:, 1:-1], padded[1:-1, :-2])
    offsets = ((0, 0), (1, 0), (1, 1), (0, 1))
    steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
    edges = {}
    outgoing = defaultdict(list)
    for direction, (neighbour, offset) in enumerate(zip(neighbours, offsets)):
        for row, col in zip(*np.nonzero(mask & ~neighbour)):
            start = (int(col) + offset[0], int(row) + offset[1])
            edge = (*start, direction)
            edges[edge] = int(labels[row, col])
            outgoing[start].append(edge)
    remaining = set(edges)
    rings = []
    # Stable insertion order avoids an O(perimeter^2) repeated min(set) scan.
    for first in edges:
        if first not in remaining:
            continue
        component = edges[first]
        edge = first
        points = []
        while True:
            remaining.remove(edge)
            x, y, direction = edge
            points.append((x, y))
            dx, dy = steps[direction]
            end = (x + dx, y + dy)
            choices = [e for e in outgoing[end] if edges[e] == component
                       and (e in remaining or e == first)]
            if not choices:
                raise ValueError("Open cell boundary encountered")
            # Prefer a left turn at a diagonal contact to keep rings distinct.
            order = {1: 0, 0: 1, 3: 2, 2: 3}
            edge = min(choices, key=lambda e: order[(e[2] - direction) % 4])
            if edge == first:
                break
            if len(points) % 10000 == 0:
                report_progress("Vector cell boundaries", len(edges) - len(remaining), len(edges))
        grid = np.asarray(points, dtype=np.int64)
        before = grid - np.roll(grid, 1, axis=0)
        after = np.roll(grid, -1, axis=0) - grid
        turn = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
        grid = grid[turn != 0]  # Lossless removal of collinear grid vertices.
        area_cells = float(np.sum(grid[:, 0] * np.roll(grid[:, 1], -1)
                                  - grid[:, 1] * np.roll(grid[:, 0], -1)) / 2)
        xy = np.asarray(origin) + np.vstack([grid, grid[0]]) * resolution
        rings.append(dict(component_id=component, role="outer" if area_cells > 0 else "hole",
                          signed_area_m2=area_cells * resolution**2, xy_m=xy.tolist()))
    return rings
