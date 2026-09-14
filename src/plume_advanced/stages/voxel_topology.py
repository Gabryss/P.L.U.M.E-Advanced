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


def open_density_necks(grid: VoxelGrid | TiledVoxelGrid, radius: int) -> None:
    """Remove unresolved air bridges, without enlarging the void.

    Used only as a final bounded surface-acceptance candidate. All sampled
    centres, manifold connectivity, graph genus and roof stability must pass
    afterwards; splitting a real route is a rejection, never hidden cleanup.
    """
    if type(radius) is not int or radius not in (0, 1):
        raise ValueError("density opening radius must be 0 or 1")
    if radius:
        _filter_grid(grid, lambda values: np.minimum(values, ndimage.grey_opening(
            values, size=3, mode="constant", cval=grid.iso_level - 8)))


def _filter_grid(
    grid: VoxelGrid | TiledVoxelGrid, operation: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Apply a two-sample-support filter with identical dense/tiled interiors."""
    if isinstance(grid, VoxelGrid):
        grid.density[...] = operation(grid.density)
        return
    grid.synchronize_halos()
    result = {}
    halo = 2  # Dilation followed by erosion has a two-cell dependency radius.
    for key, tile in sorted(grid.tiles.items()):
        start = np.asarray(key) * grid.tile_size - halo
        shape = np.asarray(tile.shape) + 2 * halo
        padded = np.full(tuple(shape), grid.iso_level - 8, dtype=tile.dtype)
        reach = int(np.ceil(halo / grid.tile_size))
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
        result[key] = operation(padded)[tuple(slice(halo, halo + n) for n in tile.shape)].copy()
    grid.tiles = result
    grid.synchronize_halos()
