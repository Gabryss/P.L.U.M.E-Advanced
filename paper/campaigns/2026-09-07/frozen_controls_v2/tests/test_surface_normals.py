"""Distance-field shading must be continuous across processing tiles."""

import numpy as np
import pytest

from plume_advanced.stages.geometry_export import density_surface_normals
from plume_advanced.stages.geometry_types import TiledVoxelGrid, VoxelGrid


@pytest.mark.parametrize("filter_voxels", [1.2, .75])
def test_density_normals_are_unit_length_and_match_across_tile_seams(filter_voxels):
    axis = np.linspace(-8., 8., 65)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    density = (4.-np.sqrt(x*x+y*y+z*z)).astype(np.float32)
    dense = VoxelGrid(origin=(-8., -8., -8.), voxel_size=.25, density=density, iso_level=0.)
    tiles = {key: density[tuple(slice(k*16, k*16+17) for k in key)].copy()
             for key in np.ndindex(4, 4, 4)}
    tiled = TiledVoxelGrid(origin=dense.origin, voxel_size=.25, global_shape=density.shape,
                           iso_level=0., tile_size=16, tiles=tiles)
    points = np.array([[4., 0., 0.], [0., -4., 0.], [0., 0., 4.],
                       [2., 2., np.sqrt(8.)], [-2., -2., -np.sqrt(8.)]])
    reference = points/4.
    actual = density_surface_normals(dense, points, reference, filter_voxels=filter_voxels)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=1e-7)
    np.testing.assert_allclose(actual, reference, atol=.002)
    np.testing.assert_allclose(density_surface_normals(tiled, points, reference, filter_voxels=filter_voxels), actual, atol=1e-6)
    np.testing.assert_allclose(density_surface_normals(tiled, points, -reference, filter_voxels=filter_voxels), -actual, atol=1e-6)
    # Changing the local triangle-normal bias must not imprint that bias on
    # the reconstructed smooth sphere.
    biased = reference + np.array([.2, .1, -.1])
    biased /= np.linalg.norm(biased, axis=1, keepdims=True)
    np.testing.assert_allclose(density_surface_normals(dense, points, biased, filter_voxels=filter_voxels), actual, atol=1e-6)


def test_flat_or_missing_density_keeps_valid_mesh_normals():
    grid = VoxelGrid(origin=(0., 0., 0.), voxel_size=1., density=np.zeros((3, 3, 3), dtype=np.float32), iso_level=0.)
    points = np.array([[1., 1., 1.], [9., 9., 9.]])
    normals = np.array([[0., 0., 1.], [1., 0., 0.]])
    np.testing.assert_array_equal(density_surface_normals(grid, points, normals), normals)
