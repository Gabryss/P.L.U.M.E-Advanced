"""Accretion preserves the envelope, physical scale and tile continuity."""

from dataclasses import replace

import numpy as np
import pytest

from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryConfig, TiledVoxelGrid, VoxelGrid
from plume_advanced.stages.surface_relief import apply_surface_relief, relief_depth, value_noise


def config(**kw):
    return GeometryConfig(random_seed=2, voxel_size=.2, surface_wall_relief_m=.55,
                          surface_roof_relief_m=.4, surface_floor_relief_m=.22, **kw)


def test_field_has_no_lattice_jumps_and_is_seeded():
    points = np.array([[a, -.37, .81] for a in [-1.-1e-7, -1.+1e-7, 1.-1e-7, 1.+1e-7]])
    values = value_noise(points, 2)
    assert abs(values[1]-values[0]) < 1e-10
    assert abs(values[3]-values[2]) < 1e-10
    assert not np.allclose(values, value_noise(points, 3))


def test_depth_is_in_metres_bounded_and_preserves_different_surface_zones():
    points = np.random.default_rng(5).uniform(-20., 20., (1000, 3))
    c = config()
    for upward, bound in [(0., .55), (1., .4), (-1., .22)]:
        depth = relief_depth(points, np.full(1000, upward), c)
        assert np.all((depth >= 0.) & (depth <= bound))
        assert np.std(depth) > .012
        # Refining the grid does not halve the relief amplitude.
        np.testing.assert_array_equal(depth, relief_depth(points, np.full(1000, upward), replace(c, voxel_size=.1)))


def test_unresolved_crust_is_attenuated_and_cannot_exceed_its_bound():
    points = np.random.default_rng(7).uniform(-20., 20., (1000, 3))
    c = GeometryConfig(random_seed=2, voxel_size=.05, surface_crust_relief_m=.1)
    fine = relief_depth(points, np.zeros(1000), c)
    coarse = relief_depth(points, np.zeros(1000), replace(c, voxel_size=.4))
    assert 0. < fine.max() <= .1
    assert 0. < coarse.max() <= .025


@pytest.mark.parametrize("iso", [0., 1.25])
def test_dense_and_tiled_accretion_agree_and_only_shrink_the_void(iso):
    axis = np.arange(49)*.2-4.8
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    original = (iso+(2.-np.sqrt(x*x+y*y+z*z))/.2).astype(np.float32)
    grid = VoxelGrid(origin=(-4.8,)*3, voxel_size=.2, density=original.copy(), iso_level=iso)
    tiles = {key: original[tuple(slice(k*16, k*16+17) for k in key)].copy()
             for key in np.ndindex(3, 3, 3)}
    tiled = TiledVoxelGrid(origin=grid.origin, voxel_size=.2, global_shape=grid.shape,
                           iso_level=iso, tile_size=16, tiles=tiles)
    apply_surface_relief(grid, config())
    apply_surface_relief(tiled, config())
    for key, tile in tiled.tiles.items():
        expected = grid.density[tuple(slice(k*16, k*16+17) for k in key)]
        np.testing.assert_allclose(tile, expected, atol=2e-6)
    assert np.all(grid.density <= original)
    assert np.count_nonzero(grid.density >= iso) < np.count_nonzero(original >= iso)
    assert grid.component_count == 1


def test_shallow_passage_keeps_clearance_and_disabled_mode_is_exact_noop():
    x, y, z = np.meshgrid(np.arange(61)*.1, np.arange(41)*.1-2., np.arange(21)*.1-1., indexing="ij")
    density = (np.minimum(.25-np.abs(z), 1.5-np.abs(y))/.1).astype(np.float32)
    grid = VoxelGrid(origin=(0., -2., -1.), voxel_size=.1, density=density.copy(), iso_level=0.)
    apply_surface_relief(grid, GeometryConfig())
    np.testing.assert_array_equal(grid.density, density)
    apply_surface_relief(grid, replace(config(), voxel_size=.1))
    assert np.all(grid.density[:, 20, 10] > 0.)
    assert grid.component_count == 1


def test_post_relief_cleanup_removes_both_speck_phases_across_tiles_but_keeps_real_cavities():
    density = np.full((49, 49, 49), -4., dtype=np.float32)
    density[5:25, 5:25, 5:25] = 4.
    density[16, 12, 12] = -.02  # detached flake on a tile boundary
    density[31, 30, 20] = .02  # isolated air left by an almost-closed neck
    density[30:33, 35:38, 25:28] = 4.  # resolved, separate 27-voxel cavity
    density[12, :20, 12] = -4.  # connected divider must survive
    grid = VoxelGrid(origin=(0.,)*3, voxel_size=.2, density=density.copy(), iso_level=0.)
    tiles = {key: density[tuple(slice(k*16, k*16+17) for k in key)].copy()
             for key in np.ndindex(3, 3, 3)}
    tiled = TiledVoxelGrid(origin=grid.origin, voxel_size=.2, global_shape=grid.shape,
                           iso_level=0., tile_size=16, tiles=tiles)
    GeometryGenerator._remove_small_solid_pockets(grid, include_void=True)
    GeometryGenerator._remove_small_solid_pockets(tiled, include_void=True)
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(tile, grid.density[tuple(slice(k*16, k*16+17) for k in key)])
    assert grid.density[16, 12, 12] > 0.
    assert grid.density[31, 30, 20] < 0.
    assert np.all(grid.density[30:33, 35:38, 25:28] > 0.)
    assert np.all(grid.density[12, :20, 12] < 0.)


def test_pocket_cleanup_does_not_invert_a_tiny_air_shell_into_an_air_speck():
    density = np.full((17, 17, 17), -1., dtype=np.float32)
    center = np.array([8, 8, 8])
    for axis in range(3):
        for sign in (-1, 1):
            point = center.copy()
            point[axis] += sign
            density[tuple(point)] = 1.
    grid = VoxelGrid(origin=(0.,)*3, voxel_size=.2, density=density, iso_level=0.)
    GeometryGenerator._remove_small_solid_pockets(grid, include_void=True)
    assert np.all(grid.density < 0.)
