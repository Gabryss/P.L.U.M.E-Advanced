"""Optional grid-scale closing before roof stability and polygonization.

A radius-one grayscale closing repairs narrow solid fissures in a continuous
cavity. It can connect voids separated by up to two cells, so it is opt-in and
must be followed by stability and graph/mesh topology checks. It does not
remove disconnected mesh components or edit the accepted network.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage

from plume_advanced.stages.geometry_types import TiledVoxelGrid, VoxelGrid


def close_density_fissures(grid: VoxelGrid | TiledVoxelGrid, radius: int) -> None:
    if type(radius) is not int or radius not in (0, 1):
        raise ValueError("density_closing_voxels must be 0 or 1")
    if not radius:
        return

    def close(values):
        # Preserve the original density at global boundaries. The repair may
        # only enlarge the cavity; subsequent roof checks assess that change.
        return np.maximum(
            values, ndimage.grey_closing(values, size=3, mode="constant", cval=grid.iso_level - 8)
        )

    _filter_grid(grid, close)


def open_density_necks(grid: VoxelGrid | TiledVoxelGrid, radius: int, *, regions=()) -> None:
    """Remove unresolved air bridges, without enlarging the void.

    Used only as a bounded surface-acceptance candidate. All sampled
    centres, manifold connectivity, graph genus and roof stability must pass
    afterwards; splitting a real route is a rejection, never hidden cleanup.
    """
    if type(radius) is not int or radius not in (0, 1, 2):
        raise ValueError("density opening radius must be 0, 1 or 2")
    if radius:
        _filter_grid(grid, lambda values: np.minimum(values, ndimage.grey_opening(
            values, size=2*radius+1, mode="constant", cval=grid.iso_level - 8)),
            regions=regions, support_radius=2*radius)


def _blend_regions(original, filtered, origin, size, regions):
    if not regions:
        return filtered
    weight = np.zeros(original.shape, dtype=np.float32)
    for region in regions:
        lower, upper = np.asarray(region["lower_m"]), np.asarray(region["upper_m"])
        if np.any(upper+2*size < origin) or np.any(lower-2*size > origin+(np.asarray(original.shape)-1)*size):
            continue
        local = np.ones(original.shape, dtype=np.float32)
        for axis in range(3):
            coords = origin[axis]+np.arange(original.shape[axis])*size
            distance = np.maximum(np.maximum(lower[axis]-coords, coords-upper[axis]), 0)
            t = np.clip(1-distance/(2*size), 0, 1)
            shape = [1, 1, 1]
            shape[axis] = len(t)
            local *= (t*t*(3-2*t)).reshape(shape)
        np.maximum(weight, local, out=weight)
    return original+(filtered-original)*weight


def _filter_grid(
    grid: VoxelGrid | TiledVoxelGrid, operation: Callable[[np.ndarray], np.ndarray], *, regions=(),
    support_radius: int = 2,
) -> None:
    """Retain the full erosion/dilation support across tile boundaries."""
    if isinstance(grid, VoxelGrid):
        grid.density[...] = _blend_regions(grid.density, operation(grid.density),
                                           np.asarray(grid.origin), grid.voxel_size, regions)
        return
    grid.synchronize_halos()
    result = {}
    halo = support_radius
    for key, tile in sorted(grid.tiles.items()):
        start = np.asarray(key) * grid.tile_size - halo
        shape = np.asarray(tile.shape) + 2 * halo
        padded = np.full(tuple(shape), grid.iso_level - 8, dtype=tile.dtype)
        # Include a primary tile owning only the last support sample when a
        # whole intervening tile is absent and halo is a multiple of tile size.
        reach = 1 + halo // grid.tile_size
        for delta in np.ndindex(*((2 * reach + 1,) * 3)):
            neighbor_key = tuple(key[i] + delta[i] - reach for i in range(3))
            neighbor = grid.tiles.get(neighbor_key)
            if neighbor is None:
                continue
            neighbor_start = np.asarray(neighbor_key) * grid.tile_size
            lower = np.maximum(start, neighbor_start)
            upper = np.minimum(start + shape, neighbor_start + neighbor.shape)
            if np.any(upper <= lower):
                continue
            target = tuple(slice(int(a), int(b)) for a, b in zip(lower - start, upper - start))
            source = tuple(
                slice(int(a), int(b))
                for a, b in zip(lower - neighbor_start, upper - neighbor_start)
            )
            padded[target] = neighbor[source]
        filtered = operation(padded)[tuple(slice(halo, halo + n) for n in tile.shape)]
        result[key] = _blend_regions(tile, filtered,
            np.asarray(grid.origin)+np.asarray(key)*grid.tile_size*grid.voxel_size,
            grid.voxel_size, regions).copy()
    grid.tiles = result
    grid.synchronize_halos()
