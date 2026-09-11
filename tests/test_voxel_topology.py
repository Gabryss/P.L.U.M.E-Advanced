"""Grid-scale repairs connect thin fissures without filling resolved islands."""

import numpy as np
import pytest
from scipy import ndimage

from plume_advanced.stages.geometry_types import TiledVoxelGrid, VoxelGrid
from plume_advanced.stages.voxel_topology import close_density_fissures


@pytest.mark.parametrize("iso", [0.0, 1.25])
def test_closing_preserves_large_islands_and_agrees_across_tile_edges(iso):
    original = np.full((49, 49, 49), iso - 4, dtype=np.float32)
    original[5:44, 5:44, 15:35] = iso + 2
    original[16:18, 5:44, 15:35] = iso - 2  # two-cell fissure, straddling the tile edge
    original[28:37, 28:37, :] = iso - 4  # resolved rock island attached to roof/floor
    original[21, 21, :] = iso - 1  # isolated one-cell pillar
    dense = VoxelGrid((0, 0, 0), 0.2, original.copy(), iso)
    tiles = {
        k: original[tuple(slice(i * 16, i * 16 + 17) for i in k)].copy()
        for k in np.ndindex(3, 3, 3)
    }
    tiled = TiledVoxelGrid(dense.origin, 0.2, dense.shape, iso, 16, tiles)
    close_density_fissures(dense, 1)
    close_density_fissures(tiled, 1)
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(
            tile, dense.density[tuple(slice(k * 16, k * 16 + 17) for k in key)]
        )
    assert np.all(dense.density >= original)
    assert ndimage.label(dense.density >= iso)[1] == 1
    plan = np.any(dense.density >= iso, axis=2)
    holes = ndimage.binary_fill_holes(plan) & ~plan
    assert ndimage.label(holes)[1] == 1
    assert np.all(holes[28:37, 28:37])


def test_disabled_closing_is_exact_noop_and_does_not_remove_distant_cavities():
    source = np.full((20, 20, 20), -2, dtype=np.float32)
    source[3:7, 3:7, 3:7] = 2
    source[12:17, 12:17, 12:17] = 2
    grid = VoxelGrid((0, 0, 0), 0.2, source.copy(), 0)
    close_density_fissures(grid, 0)
    np.testing.assert_array_equal(source, grid.density)
    close_density_fissures(grid, 1)
    assert ndimage.label(grid.density >= 0)[1] == 2


@pytest.mark.parametrize("radius", [-1, 2, True, 0.5])
def test_invalid_radius_is_rejected(radius):
    grid = VoxelGrid((0, 0, 0), 1, np.ones((3, 3, 3)), 0)
    with pytest.raises(ValueError, match="0 or 1"):
        close_density_fissures(grid, radius)
