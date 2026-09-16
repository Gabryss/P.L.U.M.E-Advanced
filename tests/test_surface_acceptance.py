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


def test_ground_rejection_does_not_erase_relief_or_try_unrelated_filter_repairs(monkeypatch):
    base = fixture()
    records, calls = [], []
    generator = GeometryGenerator(base.config, surface_diagnostic=records.append)
    monkeypatch.setattr(generator, '_enforce_roof_stability', lambda *args: ())
    def reject(candidate, **kwargs):
        calls.append(candidate.effective_surface_relief_scale)
        raise SurfaceTopologyError('ground contract failed', report={'ground_traversal': {'passed': False}})
    monkeypatch.setattr(generator, 'finalize', reject)
    with pytest.raises(SurfaceTopologyError, match='ground contract'):
        generator._accept_base_surface(base, None, [], None)
    assert calls == [1.]
    assert len(records) == 1 and records[0]['inspection']['ground_traversal']['passed'] is False


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


def test_simulation_tries_local_topology_repair_without_attenuating_relief(monkeypatch):
    from plume_advanced.acceptance import build_acceptance_policy
    base = fixture()
    diagnostics = []
    generator = GeometryGenerator(base.config, acceptance=build_acceptance_policy({'profile': 'simulation'}),
                                  surface_diagnostic=diagnostics.append)
    monkeypatch.setattr(generator, "_enforce_roof_stability", lambda *a: ())
    seen = []
    def accept_local(candidate, **kwargs):
        seen.append(candidate)
        assert candidate.effective_surface_relief_scale == 1
        assert not candidate.effective_local_relief_regions
        if not candidate.effective_local_opening_regions:
            raise SurfaceTopologyError('handle', report={'defect_regions': [dict(
                kind='handle_patch', lower_m=[0, 0, 0], upper_m=[1, 1, 1])]})
        assert diagnostics and diagnostics[0]['accepted'] is False
        return candidate
    monkeypatch.setattr(generator, 'finalize', accept_local)
    result = generator._accept_base_surface(base, None, [], None)
    assert len(seen) == 2
    assert dict(result.surface_quality_records[-1])['local_opening_regions']
    assert [r['candidate_index'] for r in diagnostics] == [1, 2]
    assert diagnostics[-1]['accepted'] is True
    assert all(r['voxel_size_m'] == base.config.voxel_size for r in diagnostics)


@pytest.mark.parametrize("blocked", [False, True])
def test_second_local_opening_keeps_all_acceptance_gates(monkeypatch, blocked):
    from plume_advanced.acceptance import build_acceptance_policy

    base = fixture()
    source = base.voxel_grid.density.copy()
    generator = GeometryGenerator(base.config,
        acceptance=build_acceptance_policy({'profile': 'simulation'}))
    monkeypatch.setattr(generator, '_enforce_roof_stability', lambda *args: ())
    seen = []
    region = dict(kind='handle_patch', lower_m=[0, 0, 0], upper_m=[1, 1, 1])
    def inspect(candidate, **kwargs):
        radius = candidate.effective_density_opening_voxels
        seen.append(radius)
        assert candidate.effective_surface_relief_scale == 1.
        if not candidate.effective_local_opening_regions or radius < 2:
            raise SurfaceTopologyError('residual handle', report={'defect_regions': [region]})
        if blocked:
            raise SurfaceTopologyError('required body route remains blocked')
        return candidate
    monkeypatch.setattr(generator, 'finalize', inspect)
    if blocked:
        with pytest.raises(SurfaceTopologyError, match='No surface candidate'):
            generator._accept_base_surface(base, None, [], None)
    else:
        result = generator._accept_base_surface(base, None, [], None)
        assert result.effective_density_opening_voxels == 2
        assert dict(result.surface_quality_records[-1])['local_opening_regions'] == [region]
    assert seen[:3] == [0, 1, 2]
    np.testing.assert_array_equal(base.voxel_grid.density, source)


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


def test_surface_acceptance_reclassifies_remnants_after_roof_clipping(monkeypatch):
    from dataclasses import replace

    density = np.full((49, 33, 33), -4., np.float32)
    density[4:41, 5:18, 5:18] = 2.
    config = replace(fixture().config, voxel_size=1., surface_wall_relief_m=0.,
                     surface_roof_relief_m=0., surface_floor_relief_m=0.,
                     surface_crust_relief_m=0., density_closing_voxels=0,
                     required_route_height_m=0.)
    base = CaveGeometry(config, VoxelGrid((0., 0., 0.), 1., density, 0.), (), (), (),
                        0, 0, (), route_centers=((10., 10., 10.),), expected_surface_genus=0)
    generator = GeometryGenerator(config)
    def clip(grid, *args):
        grid.density[19:30] = -4.
        return ()
    monkeypatch.setattr(generator, '_enforce_roof_stability', clip)
    result = generator._accept_base_surface(base, None, [], None)
    assert result.component_count == 1
    record = dict(result.surface_quality_records[-1])
    assert record['accepted'] and record['unsupported_air_samples_removed_after_filters'] == 11*13*13
    assert base.voxel_grid.sample_density((35., 10., 10.)) > 0  # source remains unchanged


def test_complete_required_route_collapse_has_a_structured_rejection(monkeypatch):
    base = fixture()
    original = base.voxel_grid.density.copy()
    generator = GeometryGenerator(base.config)
    def collapse(grid, *args):
        grid.density.fill(-8.)
        return ()
    monkeypatch.setattr(generator, '_enforce_roof_stability', collapse)
    with pytest.raises(SurfaceTopologyError, match='No surface candidate') as caught:
        generator._accept_base_surface(base, None, [], None)
    assert caught.value.report['attempts']
    assert all(row['inspection']['blocked_points_m'] for row in caught.value.report['attempts'])
    np.testing.assert_array_equal(base.voxel_grid.density, original)
