"""A relief remnant must not be mistaken for a sampled passage."""

from pathlib import Path

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


@pytest.mark.parametrize('tiled', [False, True])
def test_opening_detached_air_is_reclassified_without_hiding_a_broken_route(tiled):
    from plume_advanced.stages.voxel_topology import open_density_necks

    source = np.full((49, 33, 33), -4., np.float32)
    source[4:19, 5:18, 5:18] = 2.  # sampled main cavity
    source[19:30, 11:12, 11:12] = 2.  # unresolved bridge
    source[30:41, 6:17, 6:17] = 2.  # remnant, previously connected to the main cavity
    source[5:16, 23:30, 5:16] = 2.  # disconnected but sampled real passage
    points = ((10., 10., 10.), (10., 26., 10.))
    dense = VoxelGrid((0., 0., 0.), 1., source.copy(), 0.)
    grid = dense
    if tiled:
        tiles = {key: source[tuple(slice(k*16, k*16+17) for k in key)].copy()
                 for key in np.ndindex(3, 2, 2)}
        grid = TiledVoxelGrid(dense.origin, 1., dense.shape, 0., 16, tiles)
    assert remove_unseeded_voids(grid, points) == 0
    open_density_necks(grid, 1)
    assert grid.component_count == 3
    assert remove_unseeded_voids(grid, points) > 0
    assert grid.component_count == 2
    assert grid.sample_density((35., 10., 10.)) < 0
    assert all(grid.sample_density(point) > 0 for point in points)


def _tiled(dense, size=8):
    counts = np.ceil((np.asarray(dense.shape)-1)/size).astype(int)
    tiles = {key: dense.density[tuple(slice(k*size, k*size+size+1) for k in key)].copy()
             for key in np.ndindex(*counts)}
    return TiledVoxelGrid(dense.origin, dense.voxel_size, dense.shape, dense.iso_level,
                         size, dict(reversed(list(tiles.items()))))


@pytest.mark.parametrize('iso', [0., 1.25])
def test_large_floating_solid_removed_without_erasing_host_attached_pillar(iso):
    from plume_advanced.stages.void_components import remove_floating_solids

    source = np.full((41, 41, 41), iso-4., np.float32)
    source[4:37, 4:37, 4:37] = iso+2.
    source[16:24, 16:24, 16:24] = iso-2.  # 512 samples, spanning tile seams
    source[10:14, 10:14, :] = iso-3.  # a true floor-to-roof pillar
    dense = VoxelGrid((2., -3., 10.), .04, source.copy(), iso)
    tiled = _tiled(dense)
    assert remove_floating_solids(dense) == 512
    assert remove_floating_solids(tiled) == 512
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(tile, dense.density[tuple(slice(k*8, k*8+9) for k in key)])
    assert np.all(dense.density[10:14, 10:14, :] == iso-3.)
    assert np.count_nonzero(source != dense.density) == 512
    assert remove_floating_solids(dense) == remove_floating_solids(tiled) == 0


@pytest.mark.parametrize('tiled', [False, True])
def test_diagonal_host_contacts_are_conservatively_retained(tiled):
    from plume_advanced.stages.void_components import remove_floating_solids

    source = np.full((41, 41, 41), -4., np.float32)
    source[4:37, 4:37, 4:37] = 2.
    source[16:24, 16:24, 16:24] = -2.
    for i in range(4, 17):
        source[i, i, i] = -2.  # diagonal contacts continue across tile boundaries
    dense = VoxelGrid((0., 0., 0.), .04, source.copy(), 0.)
    grid = _tiled(dense) if tiled else dense
    assert remove_floating_solids(grid) == 0
    if tiled:
        for key, tile in grid.tiles.items():
            np.testing.assert_array_equal(tile, source[tuple(slice(k*8, k*8+9) for k in key)])
    else:
        np.testing.assert_array_equal(grid.density, source)


def test_missing_host_anchor_does_not_erase_all_rock():
    from plume_advanced.stages.void_components import remove_floating_solids

    source = np.ones((12, 12, 12), np.float32)
    source[4:8, 4:8, 4:8] = -1.
    grid = VoxelGrid((0., 0., 0.), .04, source.copy(), 0.)
    with pytest.raises(ValueError, match='host boundary'):
        remove_floating_solids(grid)
    np.testing.assert_array_equal(grid.density, source)


@pytest.mark.parametrize('tiled', [False, True])
def test_measured_seed_zero_relief_fragment_is_removed(tiled):
    from skimage.measure import marching_cubes

    from plume_advanced.stages.surface_topology import component_count
    from plume_advanced.stages.void_components import remove_floating_solids

    path = Path(__file__).parent/'fixtures/volume_connectivity/floating_relief_seed0.npz'
    with np.load(path) as data:
        density = data['density'].copy()
        dense = VoxelGrid(tuple(data['origin']), float(data['voxel_size']),
                          density.copy(), float(data['iso_level']))
    before = component_count(marching_cubes(density, level=dense.iso_level)[1])
    grid = _tiled(dense, 16) if tiled else dense
    assert remove_floating_solids(grid) == 286
    if tiled:
        for key, tile in sorted(grid.tiles.items()):
            dense.density[tuple(slice(k*16, k*16+17) for k in key)] = tile
    after = component_count(marching_cubes(dense.density, level=dense.iso_level)[1])
    assert before == after+1
    assert np.count_nonzero(density != dense.density) == 286
