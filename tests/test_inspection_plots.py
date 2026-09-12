"""Sparse diagnostics preserve dense projections without creating a 3-D array."""

from types import SimpleNamespace

import numpy as np
import pytest

from plume_advanced.visualization.inspection import SavedGeometryPlotter


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_sparse_and_dense_plot_envelopes_match(seed):
    rng = np.random.default_rng(seed)
    density = rng.uniform(-1, 1, (7, 9, 5))
    size = 4
    shared = dict(shape=density.shape, voxel_size=.2, origin=(-3, 8, -12), iso_level=.1)
    dense = SimpleNamespace(voxel_grid=SimpleNamespace(density=density, **shared))
    tiles = {}
    for x in range(0, 7, size):
        for y in range(0, 9, size):
            for z in range(0, 5, size):
                tiles[x // size, y // size, z // size] = density[x:x+size, y:y+size, z:z+size]
    sparse = SimpleNamespace(voxel_grid=SimpleNamespace(tiles=tiles, tile_size=size, **shared))
    a, b = SavedGeometryPlotter(dense), SavedGeometryPlotter(sparse)
    np.testing.assert_array_equal(a.plan, b.plan)
    np.testing.assert_array_equal(a.profile, b.profile)
    np.testing.assert_array_equal(a._carved_profile(dense), b._carved_profile(sparse))
    assert a._carved_footprint(dense)[1] == b._carved_footprint(sparse)[1]


def test_empty_sparse_plot_has_empty_profile():
    geometry = SimpleNamespace(voxel_grid=SimpleNamespace(
        tiles={}, tile_size=4, shape=(7, 9, 5), origin=(0, 0, 0), voxel_size=.2, iso_level=0,
    ))
    plotter = SavedGeometryPlotter(geometry)
    assert not plotter.plan.any() and not plotter.profile.any()
    assert all(len(values) == 0 for values in plotter._carved_profile(geometry))
