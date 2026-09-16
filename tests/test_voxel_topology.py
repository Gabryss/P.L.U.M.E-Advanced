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


def test_neck_opening_is_bounded_and_does_not_join_or_enlarge_air():
    from plume_advanced.stages.voxel_topology import open_density_necks
    original = np.full((49, 49, 49), -4, dtype=np.float32)
    original[4:22, 5:42, 5:42] = 3
    original[26:44, 5:42, 5:42] = 3
    original[21:27, 24:25, 24:25] = 1  # unresolved single-cell shortcut
    original[10:16, 15:25, :] = -4  # resolved rock island through first gallery
    dense = VoxelGrid((0, 0, 0), .1, original.copy(), 0)
    tiles = {k: original[tuple(slice(i*16, i*16+17) for i in k)].copy()
             for k in np.ndindex(3, 3, 3)}
    tiled = TiledVoxelGrid(dense.origin, .1, dense.shape, 0, 16, tiles)
    open_density_necks(dense, 1)
    open_density_necks(tiled, 1)
    assert np.all(dense.density <= original)
    assert ndimage.label(original >= 0)[1] == 1
    assert ndimage.label(dense.density >= 0)[1] == 2
    assert np.all(dense.density[10:16, 15:25, :] < 0)
    assert dense.sample_density((.8, 1., 2.)) > 0
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(tile, dense.density[
            tuple(slice(k*16, k*16+17) for k in key)])
    once = dense.density.copy()
    open_density_necks(dense, 1)
    np.testing.assert_array_equal(once, dense.density)


def test_local_opening_preserves_distant_thin_passage_and_matches_tiles():
    from plume_advanced.stages.voxel_topology import open_density_necks
    source = np.full((49, 49, 49), -4, dtype=np.float32)
    source[4:44, 4:44, 10:35] = 2
    source[4:44, 20:23, 10:35] = -4
    source[16:17, 20:23, 22:23] = 2  # unwanted bridge at tile boundary
    source[38:44, 38:39, 35:41] = 2  # distant narrow feature must survive
    dense = VoxelGrid((-2, -2, -2), .1, source.copy(), 0)
    tiles = {k: source[tuple(slice(i*16, i*16+17) for i in k)].copy()
             for k in np.ndindex(3, 3, 3)}
    tiled = TiledVoxelGrid(dense.origin, .1, dense.shape, 0, 16, tiles)
    region = dict(lower_m=[-.6, -.1, .1], upper_m=[-.2, .4, .5])
    open_density_necks(dense, 1, regions=[region])
    open_density_necks(tiled, 1, regions=[region])
    assert dense.density[16, 21, 22] < 0
    np.testing.assert_array_equal(dense.density[30:], source[30:])
    assert np.all(dense.density <= source)
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(tile, dense.density[
            tuple(slice(k*16, k*16+17) for k in key)])


@pytest.mark.parametrize("tile_size", [4, 16])
@pytest.mark.parametrize("iso", [0., 1.25])
def test_radius_two_repairs_wider_neck_with_full_tile_support(tile_size, iso):
    from plume_advanced.stages.voxel_topology import open_density_necks

    source = np.full((49, 49, 49), iso-4, np.float32)
    source[4:22, 5:42, 5:42] = iso+3
    source[26:44, 5:42, 5:42] = iso+3
    source[21:27, 23:26, 23:26] = iso+1
    source[40:44, 42:43, 36:40] = iso+2  # distant thin feature
    origin = np.array([-2.03, 1.07, -3.11])
    size = .04
    region = dict(lower_m=(origin+np.array([19, 19, 19])*size).tolist(),
                  upper_m=(origin+np.array([29, 30, 30])*size).tolist())
    narrow = VoxelGrid(tuple(origin), size, source.copy(), iso)
    open_density_necks(narrow, 1, regions=[region])
    assert narrow.density[24, 24, 24] > iso
    dense = VoxelGrid(tuple(origin), size, source.copy(), iso)
    n = 48//tile_size
    tiles = {k: source[tuple(slice(i*tile_size, i*tile_size+tile_size+1) for i in k)].copy()
             for k in np.ndindex(n, n, n)}
    tiled = TiledVoxelGrid(dense.origin, size, dense.shape, iso, tile_size, tiles)
    open_density_necks(dense, 2, regions=[region])
    open_density_necks(tiled, 2, regions=[region])
    assert dense.density[24, 24, 24] < iso
    assert np.all(dense.density <= source)
    np.testing.assert_array_equal(dense.density[36:], source[36:])
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(tile, dense.density[
            tuple(slice(k*tile_size, k*tile_size+tile_size+1) for k in key)])


@pytest.mark.parametrize("radius", [-1, 3, True, 1.5])
def test_opening_cannot_exceed_its_two_voxel_budget(radius):
    from plume_advanced.stages.voxel_topology import open_density_necks
    grid = VoxelGrid((0., 0., 0.), .04, np.ones((9, 9, 9)), 0.)
    with pytest.raises(ValueError, match="0, 1 or 2"):
        open_density_necks(grid, radius)


def test_measured_multi_source_relief_neck_needs_radius_two():
    from pathlib import Path

    from skimage import measure

    from plume_advanced.stages.surface_topology import (
        check_closed_surface_topology,
        component_count,
    )
    from plume_advanced.stages.voxel_topology import open_density_necks

    path = Path(__file__).parent/'fixtures/geometry/relief_neck_seed4294967295.npz'
    with np.load(path) as saved:
        source = saved['density'].copy()
        origin, size = tuple(saved['origin']), float(saved['voxel_size'])
    for radius, expected in ((1, (2, 1.)), (2, (1, 0.))):
        grid = VoxelGrid(origin, size, source.copy(), 0.)
        open_density_necks(grid, radius)
        assert np.all(grid.density <= source)
        # Artificial diagnostic crop boundaries have no host-cover meaning.
        for axis in range(3):
            for end in (0, -1):
                cap = [slice(None)]*3
                cap[axis] = end
                grid.density[tuple(cap)] = -.08
        vertices, faces, _, _ = measure.marching_cubes(grid.density, level=0.,
            spacing=(size,)*3, allow_degenerate=False)
        edges = np.unique(np.sort(np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
        components = component_count(faces)
        genus = (2*components-len(vertices)+len(edges)-len(faces))/2
        assert (components, genus) == expected
        if radius == 2:
            check_closed_surface_topology(vertices, faces, components, expected_genus=0)
