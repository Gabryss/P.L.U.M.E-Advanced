"""Actual thin-branch regression plus bounded deterministic surface acceptance."""

import json
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.stages.surface_topology import SurfaceTopologyError


def fixture():
    path = Path(__file__).parent / "fixtures/geometry/relief_branch_seed20260912.npz"
    with np.load(path) as saved:
        config = GeometryConfig(**json.loads(str(saved["config"])))
        grid = VoxelGrid(tuple(saved["origin"]), config.voxel_size,
                         saved["density"].copy(), config.iso_level)
        points = tuple(map(tuple, saved["route_points"]))
    base = CaveGeometry(config, grid, (), (), (), 0, 0, (),
                        route_centers=points, expected_surface_genus=0)
    return base


def test_real_branch_relaxes_detail_without_changing_seed_or_input(monkeypatch):
    base = fixture()
    original = base.voxel_grid.density.copy()
    generator = GeometryGenerator(base.config)
    # Artificial crop caps have no host-cover interpretation. Roof stability
    # is exercised by full generation tests, not by this local density fixture.
    monkeypatch.setattr(generator, "_enforce_roof_stability", lambda *args: ())
    first = generator._accept_base_surface(base, None, [], None)
    second = generator._accept_base_surface(base, None, [], None)
    records = [dict(r) for r in first.surface_quality_records]
    assert records[0]["accepted"] is False
    assert records[-1]["accepted"] is True
    assert first.effective_surface_relief_scale == 1.
    assert first.effective_local_relief_regions
    assert records[-1]["local_relief_regions"]
    assert first.config == base.config
    assert first.surface_quality_records == second.surface_quality_records
    np.testing.assert_array_equal(first.assembled_vertices, second.assembled_vertices)
    np.testing.assert_array_equal(first.assembled_faces, second.assembled_faces)
    np.testing.assert_array_equal(base.voxel_grid.density, original)
    assert all(first.voxel_grid.sample_density(p) > base.config.iso_level for p in base.route_centers)


def test_unrecoverable_topology_exhausts_finite_candidates_and_does_not_mutate_input(monkeypatch):
    base = fixture()
    generator = GeometryGenerator(base.config)
    original = base.voxel_grid.density.copy()
    calls = []
    monkeypatch.setattr(generator, "_enforce_roof_stability", lambda *args: ())

    def reject(candidate, **kwargs):
        calls.append(candidate.effective_surface_relief_scale)
        raise SurfaceTopologyError("unrecoverable graph/volume mismatch")

    monkeypatch.setattr(generator, "finalize", reject)
    with pytest.raises(SurfaceTopologyError, match="No surface candidate"):
        generator._accept_base_surface(base, None, [], None)
    assert len(calls) <= 6
    assert calls[-1] == 0.
    np.testing.assert_array_equal(base.voxel_grid.density, original)


def test_programming_errors_are_not_treated_as_bad_detail(monkeypatch):
    base = fixture()
    generator = GeometryGenerator(base.config)
    monkeypatch.setattr(generator, "_enforce_roof_stability", lambda *args: ())
    monkeypatch.setattr(generator, "finalize", lambda *args, **kwargs: (_ for _ in ()).throw(
        TypeError("programming error")))
    with pytest.raises(TypeError, match="programming error"):
        generator._accept_base_surface(base, None, [], None)


def test_real_merge_neck_needs_final_opening_and_preserves_all_centres(monkeypatch):
    with np.load(Path(__file__).parent/'fixtures/geometry/merge_neck_seed17.npz') as saved:
        config = GeometryConfig(**json.loads(str(saved['config'])))
        grid = VoxelGrid(tuple(saved['origin']), config.voxel_size,
                         saved['density'].copy(), config.iso_level)
        points = tuple(map(tuple, saved['route_points']))
    base = CaveGeometry(config, grid, (), (), (), 0, 0, (),
                        route_centers=points, expected_surface_genus=0)
    original = grid.density.copy()
    generator = GeometryGenerator(config)
    monkeypatch.setattr(generator, '_enforce_roof_stability', lambda *args: ())
    result = generator._accept_base_surface(base, None, [], None)
    records = [dict(r) for r in result.surface_quality_records]
    assert not records[0]['accepted']
    assert records[-1]['accepted'] and records[-1]['opening_voxels'] == 1
    assert result.effective_density_opening_voxels == 1
    assert all(result.voxel_grid.sample_density(p) > grid.iso_level for p in points)
    np.testing.assert_array_equal(original, grid.density)
