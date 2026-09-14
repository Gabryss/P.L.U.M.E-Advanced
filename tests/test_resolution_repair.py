from dataclasses import replace

import numpy as np
import pytest
import trimesh
from test_embedded_inspection import geometry
from test_network_quality import network_fixture

from plume_advanced.pipeline.resolution import (
    ResolutionBudgetError,
    build_with_resolution_checks,
    compare_mesh_clearance,
)
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.section_field import SectionFieldGenerator


def inputs():
    network = network_fixture()
    sections = SectionFieldGenerator().generate(network)
    config = GeometryConfig(voxel_size=1, resolution_refinement_attempts=2,
                            resolution_min_voxel_size_m=.1)
    return network, sections, config


def test_clearance_convergence_detects_translation_despite_equal_total_height():
    first = geometry(trimesh.creation.box(extents=(4, 4, 2)))
    mesh = trimesh.creation.box(extents=(4, 4, 2))
    mesh.apply_translation((0, 0, .2))
    result = compare_mesh_clearance(first, geometry(mesh), ((0, 0, 0),), .03)
    assert not result["passed"]
    assert result["max_floor_roof_change_m"] == pytest.approx(.2)


def test_refinement_preserves_inputs_and_records_effective_resolution(monkeypatch):
    network, sections, config = inputs()
    mesh = trimesh.creation.box(extents=(300, 100, 300))
    mesh.apply_translation((50, 0, 80))
    calls = []
    def build(self, n, s, **kwargs):
        assert n is network and s is sections
        calls.append(self.config.voxel_size)
        base = geometry(mesh)
        return replace(base, config=self.config,
                       voxel_grid=replace(base.voxel_grid, voxel_size=self.config.voxel_size))
    monkeypatch.setattr(GeometryGenerator, "build_base_volume", build)
    result = build_with_resolution_checks(network, sections, config)
    assert calls == [1, .5]
    assert result.config.voxel_size == .5 and config.voxel_size == 1
    record = dict(result.resolution_repair)
    assert record["passed"] and record["outcome"] == "converged"
    assert record["attempts"][-1]["comparison"]["compared_samples"] > 0


def test_unconverged_refinement_exhausts_without_silent_acceptance(monkeypatch):
    network, sections, config = inputs()
    def build(self, *args, **kwargs):
        mesh = trimesh.creation.box(extents=(300, 100, 300))
        mesh.apply_translation((50, 0, 80+self.config.voxel_size))
        return replace(geometry(mesh), config=self.config)
    monkeypatch.setattr(GeometryGenerator, "build_base_volume", build)
    with pytest.raises(ResolutionBudgetError) as caught:
        build_with_resolution_checks(network, sections, config)
    assert len(caught.value.report["attempts"]) == 3
    assert not caught.value.report["passed"]


def test_refinement_budget_stops_before_allocating_dense_volume():
    network, sections, config = inputs()
    config = replace(config, resolution_max_allocated_voxels=8, storage_mode="dense")
    with pytest.raises(ResolutionBudgetError, match="budget") as caught:
        build_with_resolution_checks(network, sections, config)
    assert len(caught.value.report["attempts"]) == 1
    assert caught.value.report["attempts"][0]["allocation"]["allocated_samples"] > 8


def test_zero_refinement_budget_does_not_change_requested_grid(monkeypatch):
    network, sections, config = inputs()
    mesh = geometry()
    monkeypatch.setattr(GeometryGenerator, "build_base_volume", lambda *a, **k: mesh)
    assert build_with_resolution_checks(network, sections, replace(config, resolution_refinement_attempts=0)) is mesh


@pytest.mark.parametrize("name", ["route_repair_attempts", "resolution_refinement_attempts",
                                  "collision_repair_attempts", "surface_local_repair_attempts"])
@pytest.mark.parametrize("value", [-1, True, 1.2, "1", 100])
def test_attempt_budgets_are_strictly_bounded(name, value):
    with pytest.raises(ValueError, match=name):
        GeometryConfig(**{name: value})


def test_local_relief_mask_preserves_remote_detail_and_is_continuous():
    from plume_advanced.stages.surface_relief import local_relief_weights
    region = dict(lower_m=[-1, -1, -1], upper_m=[1, 1, 1], blend_m=2., scale=.25)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [100, 0, 0]])
    weights = local_relief_weights(points, [region])
    np.testing.assert_allclose(weights, [.25, .25, .625, 1., 1.])
    np.testing.assert_array_equal(weights, local_relief_weights(points, [region]))
    assert abs(local_relief_weights(np.array([[1+1e-6, 0, 0]]), [region])[0]-.25) < 1e-12


def real_short_inputs():
    network = network_fixture([(0, 0), (5, 0), (10, 0), (15, 0), (20, 0)], widths=[2]*5)
    network = replace(network, config=replace(network.config, target_route_length_m=20))
    sections = SectionFieldGenerator().generate(network)
    config = GeometryConfig(voxel_size=.2, density_margin=1, minimum_radius=.2,
        wall_roughness_amplitude=0, cave_diffuse_texture='', cave_normal_texture='',
        cave_roughness_texture='', cave_displacement_texture='', required_route_height_m=.5,
        required_route_width_m=.5, resolution_refinement_attempts=2,
        resolution_min_voxel_size_m=.05, resolution_max_allocated_voxels=8000000)
    from plume_advanced.stages.route_clearance import fit_required_sections
    sections, _ = fit_required_sections(network, sections, config)
    return network, sections, config


def test_real_volume_refines_and_checks_intermediate_route_probes():
    network, sections, config = real_short_inputs()
    result = build_with_resolution_checks(network, sections, config)
    journal = dict(result.resolution_repair)
    assert journal['passed'] and result.config.voxel_size < config.voxel_size
    comparison = journal['attempts'][-1]['comparison']
    assert comparison['compared_samples'] > 80
    assert comparison['max_floor_roof_change_m'] <= .03
    assert dict(result.mesh_inspection)['traversal']['passed']


def test_real_unconverged_volume_is_not_published_when_budget_exhausts():
    network, sections, config = real_short_inputs()
    config = replace(config, resolution_refinement_attempts=1, resolution_convergence_m=1e-12)
    with pytest.raises(ResolutionBudgetError) as caught:
        build_with_resolution_checks(network, sections, config)
    assert len(caught.value.report['attempts']) == 2
    assert not caught.value.report['passed']
