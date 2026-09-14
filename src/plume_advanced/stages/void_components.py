"""Discard detached relief voids that have no support in the sampled network.

Keep *every* component containing a section centre, including disconnected
branches: the final mesh gate must diagnose those instead of hiding them by
keeping only the largest component. Sparse grids are labelled twice, retaining
only boundary planes between passes; no full dense volume is allocated.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from plume_advanced.stages.geometry_types import TiledVoxelGrid, VoxelGrid


def remove_unseeded_voids(grid, route_points, progress=None) -> int:
    points = np.asarray(route_points, dtype=float).reshape(-1, 3)
    if not len(points):
        return 0  # Local studies without a complete network cannot classify voids.
    indices = np.rint((points - grid.origin) / grid.voxel_size).astype(np.int64)
    if isinstance(grid, VoxelGrid):
        tiles = {(0, 0, 0): grid.density}
        step = np.asarray(grid.shape)
    elif isinstance(grid, TiledVoxelGrid):
        grid.synchronize_halos()
        tiles = grid.tiles
        step = np.full(3, grid.tile_size)
    else:
        raise TypeError("Unsupported voxel grid")
    parent = [0]
    records = {}
    boundaries: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
    seeds: set[int] = set()

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for index, (key, tile) in enumerate(sorted(tiles.items()), 1):
        labels, count = ndimage.label(tile >= grid.iso_level)
        offset = len(parent) - 1
        parent.extend(range(len(parent), len(parent) + count))
        records[key] = (offset, count)
        local = indices - np.asarray(key) * step
        inside = np.all((local >= 0) & (local < tile.shape), axis=1)
        seeds.update(offset + int(v) for v in labels[tuple(local[inside].T)] if v)
        for axis in range(3):
            previous = tuple(key[a] - (a == axis) for a in range(3))
            low = np.take(labels, 0, axis=axis)
            other = boundaries.get((previous, axis))
            if other is not None:
                valid = (low > 0) & (other > 0)
                pairs = np.unique(np.column_stack((low[valid] + offset, other[valid])), axis=0)
                for a, b in pairs:
                    a, b = find(int(a)), find(int(b))
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            high = np.take(labels, -1, axis=axis)
            boundaries[key, axis] = np.where(high > 0, high + offset, 0)
        if progress:
            progress("void-connectivity", index, 2 * len(tiles), "joining air regions across tiles")
    if not seeds:
        raise ValueError("No sampled section centre lies in the carved volume")
    roots = {find(v) for v in seeds}
    keep = np.array([False] + [find(v) in roots for v in range(1, len(parent))])
    removed = 0
    for index, (key, tile) in enumerate(sorted(tiles.items()), 1):
        offset, count = records[key]
        if np.all(keep[offset + 1:offset + count + 1]):
            continue
        labels, _ = ndimage.label(tile >= grid.iso_level)
        discard = (labels > 0) & ~keep[labels + offset]
        # Count each shared boundary sample once.
        core = tuple(slice(0, tile.shape[a] if key[a] * step[a] + tile.shape[a] == grid.shape[a]
                           else int(step[a])) for a in range(3))
        removed += int(np.count_nonzero(discard[core]))
        tile[discard] = grid.iso_level - 1.
        if progress:
            progress("void-connectivity", len(tiles) + index, 2 * len(tiles),
                     f"removed {removed:,} unsupported air samples")
    if isinstance(grid, TiledVoxelGrid):
        grid.synchronize_halos()
    if progress:
        progress("void-connectivity", 2 * len(tiles), 2 * len(tiles),
                 f"removed {removed:,} unsupported air samples; sampled routes retained")
    return removed
