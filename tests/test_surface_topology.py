"""Closed surfaces can still realize the wrong cave topology."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.stages.surface_topology import (
    SurfaceTopologyError,
    check_closed_surface_topology,
    component_count,
)


@pytest.mark.parametrize("genus", [0, 1])
def test_expected_sphere_or_loop_passes_but_wrong_loop_count_fails(genus):
    mesh = trimesh.creation.icosphere(subdivisions=1) if genus == 0 else trimesh.creation.torus(
        major_radius=3., minor_radius=1.)
    vertices = np.vstack([mesh.vertices, [123, 456, 789]])  # unused coordinates do not count
    assert mesh.is_watertight
    count = component_count(mesh.faces)
    assert count == 1
    assert check_closed_surface_topology(vertices, mesh.faces, count, genus)["genus"] == genus
    with pytest.raises(SurfaceTopologyError, match="accepted network"):
        check_closed_surface_topology(vertices, mesh.faces, count, 1 - genus)


def test_detached_shell_fails_even_though_both_parts_are_closed():
    first = trimesh.creation.icosphere(subdivisions=1)
    second = first.copy()
    second.apply_translation([10, 0, 0])
    mesh = first + second
    assert mesh.is_watertight and component_count(mesh.faces) == 2
    with pytest.raises(SurfaceTopologyError, match="accepted network"):
        check_closed_surface_topology(mesh.vertices, mesh.faces, 2, 0)


def test_components_handle_empty_and_sparse_vertex_identifiers():
    assert component_count(()) == 0
    assert component_count(((100, 200, 300), (300, 200, 400), (900, 901, 902))) == 2


def test_finalization_rejects_a_closed_extra_handle():
    x, y, z = np.mgrid[-5:5:33j, -5:5:33j, -2:2:17j]
    density = (0.8 - np.sqrt((np.hypot(x, y) - 3.)**2 + z**2)).astype(np.float32)
    grid = VoxelGrid((-5, -5, -2), .3, density, 0.)
    base = CaveGeometry(GeometryConfig(voxel_size=.3), grid, (), (), (), 0, 0, (),
                        expected_surface_genus=0)
    with pytest.raises(SurfaceTopologyError, match="accepted network"):
        GeometryGenerator(base.config).finalize(base)
    actual = GeometryGenerator(base.config).finalize(replace(base, expected_surface_genus=1))
    assert actual.component_count == 1 and actual.expected_surface_genus == 1
