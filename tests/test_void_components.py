"""A relief remnant must not be mistaken for a sampled passage."""

import numpy as np
import pytest

from plume_advanced.stages.geometry_types import TiledVoxelGrid, VoxelGrid
from plume_advanced.stages.void_components import remove_unseeded_voids


@pytest.mark.parametrize("iso", [0., 1.25])
def test_sparse_cleanup_preserves_all_seeded_components_and_cross_tile_connections(iso):
    source = np.full((49, 49, 49), iso - 4, np.float32)
    source[4:44, 4:11, 4:11] = iso + 2  # a route crossing three tiles
    source[8:14, 25:44, 6:14] = iso + 2  # a disconnected but sampled branch
    source[19:43, 19:43, 29:32] = iso + 1  # large unsupported air sheet
    source[38:42, 8:12, 37:42] = iso + 1  # small unsupported pocket
    points = ((6., 6., 6.), (10., 30., 8.))
    dense = VoxelGrid((0., 0., 0.), 1., source.copy(), iso)
    tiles = {key: source[tuple(slice(k * 16, k * 16 + 17) for k in key)].copy()
             for key in np.ndindex(3, 3, 3)}
    # Deliberately reverse insertion order; results may not depend on it.
    tiled = TiledVoxelGrid(dense.origin, 1., dense.shape, iso, 16,
                           dict(reversed(list(tiles.items()))))
    removed = remove_unseeded_voids(dense, points)
    assert removed == 24 * 24 * 3 + 4 * 4 * 5
    assert remove_unseeded_voids(tiled, points) == removed
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(tile, dense.density[tuple(slice(k * 16, k * 16 + 17)
                                                               for k in key)])
    assert dense.component_count == 2  # retain the evidence of a disconnected route!
    assert np.all(dense.density[4:44, 4:11, 4:11] == iso + 2)
    assert remove_unseeded_voids(dense, points) == 0


def test_no_network_context_does_not_discard_real_cavities():
    grid = VoxelGrid((0., 0., 0.), 1., np.ones((3, 3, 3), np.float32), 0.)
    original = grid.density.copy()
    assert remove_unseeded_voids(grid, ()) == 0
    np.testing.assert_array_equal(grid.density, original)


def test_absent_route_does_not_erase_the_entire_cave():
    grid = VoxelGrid((0., 0., 0.), 1., np.ones((3, 3, 3), np.float32), 0.)
    original = grid.density.copy()
    with pytest.raises(ValueError, match="No sampled section"):
        remove_unseeded_voids(grid, [(100., 100., 100.)])
    np.testing.assert_array_equal(grid.density, original)
