"""Classify cavity air and host-connected rock across sparse grid boundaries.

Every sampled air region is retained, including disconnected real routes that
must fail downstream inspection. Solid accretion completely detached from the
host is removable; wall benches and pillars connected to the host are retained
regardless of their size. This is volume repair, never a largest-mesh filter.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from plume_advanced.stages.geometry_types import TiledVoxelGrid, VoxelGrid


def remove_unseeded_voids(grid, route_points, progress=None) -> int:
    points = np.asarray(route_points, dtype=float).reshape(-1, 3)
    if not len(points):
        return 0
    indices = np.rint((points-grid.origin)/grid.voxel_size).astype(np.int64)
    return _remove_components(grid, indices=indices, solid=False, progress=progress)


def remove_floating_solids(grid, progress=None) -> int:
    """Remove only rock disconnected from the surrounding host.

    Full 26-neighbour solid connectivity conservatively protects even diagonal
    contacts. Air still uses face connectivity. A final mesh/route/roof gate is
    required; this cleanup does not establish surface validity by itself.
    """
    return _remove_components(grid, indices=np.empty((0, 3), dtype=np.int64),
                              solid=True, progress=progress)


def _remove_components(grid, *, indices, solid, progress):
    if isinstance(grid, VoxelGrid):
        tiles = {(0, 0, 0): grid.density}
        step = np.asarray(grid.shape)
    elif isinstance(grid, TiledVoxelGrid):
        grid.synchronize_halos()
        tiles = grid.tiles
        step = np.full(3, grid.tile_size)
    else:
        raise TypeError('Unsupported voxel grid')
    structure = ndimage.generate_binary_structure(3, 3 if solid else 1)
    phase = 'solid-connectivity' if solid else 'void-connectivity'
    material = 'floating solid' if solid else 'unsupported air'
    parent = [0]
    records = {}
    boundaries: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
    seeds: set[int] = set()

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def label(tile):
        return ndimage.label(tile < grid.iso_level if solid else tile >= grid.iso_level,
                             structure=structure)

    for index, (key, tile) in enumerate(sorted(tiles.items()), 1):
        labels, count = label(tile)
        offset = len(parent)-1
        parent.extend(range(len(parent), len(parent)+count))
        records[key] = (offset, count)
        if not solid:
            local = indices-np.asarray(key)*step
            inside = np.all((local >= 0) & (local < tile.shape), axis=1)
            seeds.update(offset+int(v) for v in labels[tuple(local[inside].T)] if v)
        for axis in range(3):
            previous = tuple(key[a]-(a == axis) for a in range(3))
            following = tuple(key[a]+(a == axis) for a in range(3))
            low = np.take(labels, 0, axis=axis)
            other = boundaries.get((previous, axis))
            if other is not None:
                valid = (low > 0) & (other > 0)
                pairs = np.unique(np.column_stack((low[valid]+offset, other[valid])), axis=0)
                for a, b in pairs:
                    a, b = find(int(a)), find(int(b))
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            high = np.take(labels, -1, axis=axis)
            boundaries[key, axis] = np.where(high > 0, high+offset, 0)
            if solid:
                # Unallocated sparse tiles and the global exterior are host
                # rock. Any solid label touching that boundary is anchored.
                for neighbor, plane in ((previous, low), (following, high)):
                    if neighbor not in tiles:
                        seeds.update(offset+int(v) for v in np.unique(plane) if v)
        if progress:
            progress(phase, index, 2*len(tiles),
                     'joining host rock across tiles' if solid else 'joining air regions across tiles')
    if not seeds and (not solid or len(parent) > 1):
        raise ValueError('No solid samples connect to the host boundary' if solid
                         else 'No sampled section centre lies in the carved volume')
    roots = {find(v) for v in seeds}
    keep = np.array([False]+[find(v) in roots for v in range(1, len(parent))])
    removed = 0
    for index, (key, tile) in enumerate(sorted(tiles.items()), 1):
        offset, count = records[key]
        if np.all(keep[offset+1:offset+count+1]):
            continue
        labels, _ = label(tile)
        discard = (labels > 0) & ~keep[labels+offset]
        core = tuple(slice(0, tile.shape[a] if key[a]*step[a]+tile.shape[a] == grid.shape[a]
                           else int(step[a])) for a in range(3))
        removed += int(np.count_nonzero(discard[core]))
        tile[discard] = grid.iso_level+(1. if solid else -1.)
        if progress:
            progress(phase, len(tiles)+index, 2*len(tiles),
                     f'removed {removed:,} {material} samples')
    if isinstance(grid, TiledVoxelGrid):
        grid.synchronize_halos()
    if progress:
        retained = 'host-connected rock' if solid else 'sampled routes'
        progress(phase, 2*len(tiles), 2*len(tiles),
                 f'removed {removed:,} {material} samples; {retained} retained')
    return removed
