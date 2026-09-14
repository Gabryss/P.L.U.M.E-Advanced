from dataclasses import replace

import numpy as np
import pytest
import trimesh
from test_network_quality import network_fixture

from plume_advanced.evaluation.artifacts import section_semantic_hash
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.mesh_inspection import MeshInspectionError, inspect_surface
from plume_advanced.stages.route_clearance import fit_required_sections, required_paths
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.stages.surface_topology import SurfaceTopologyError


def inspect(mesh, paths=(((0, -1, 0), (0, 1, 0)),), height=.5, width=.5):
    return inspect_surface(mesh.vertices, mesh.faces, points=((0, 0, 0),),
                           required_paths=paths, route_segment_ids=tuple(range(len(paths))),
                           route_height_m=height, route_width_m=width, route_margin_m=.02)


@pytest.mark.parametrize("extents", [(2, 4, .4), (.4, 4, 2)])
def test_point_inside_does_not_imply_capsule_fits(extents):
    mesh = trimesh.creation.box(extents=extents)
    assert inspect_surface(mesh.vertices, mesh.faces, points=((0, 0, 0),))["passed"]
    with pytest.raises(MeshInspectionError, match="capsule") as error:
        inspect(mesh)
    assert error.value.report["traversal"]["failures"]
    assert error.value.report["defect_regions"][0]["kind"] == "route_clearance"


def test_finite_capsule_passes_and_replays_exactly():
    mesh = trimesh.creation.box(extents=(2, 4, 2))
    first = inspect(mesh, height=1.5)
    assert first["traversal"]["passed"]
    assert first == inspect(mesh, height=1.5)


def test_required_routes_cannot_be_omitted_to_pass():
    with pytest.raises(MeshInspectionError, match="missing"):
        inspect(trimesh.creation.box(), paths=())


def test_unrelated_paths_are_not_joined_through_rock():
    first = trimesh.creation.box(extents=(2, 4, 2))
    second = first.copy()
    second.apply_translation((5, 0, 0))
    paths = (((0, -1, 0), (0, 1, 0)), ((5, -1, 0), (5, 1, 0)))
    assert inspect(first+second, paths)["traversal"]["passed"]
    with pytest.raises(MeshInspectionError):
        inspect(first+second, (((0, 0, 0), (5, 0, 0)),))


def test_thin_obstruction_between_stations_is_rejected():
    cave = trimesh.creation.box(extents=(2, 6, 2))
    obstacle = trimesh.creation.box(extents=(1, .01, 1))
    obstacle.apply_translation((0, .5, 0))
    with pytest.raises(MeshInspectionError, match="capsule"):
        inspect(cave+obstacle)


def test_envelope_repair_preserves_seed_centres_and_optional_fields():
    network = network_fixture()
    fields = SectionFieldGenerator().generate(network)
    source = fields.segment_fields[0]
    narrow = tuple(SectionFieldGenerator(fields.config)._assess_roof(replace(
        s, tube_height=s.tube_height*.1,
        profile_points=tuple((u, v*.1) for u, v in s.profile_points))) for s in source.samples)
    fields = replace(fields, segment_fields=(replace(source, samples=narrow), replace(source, segment_id=99)))
    config = GeometryConfig(voxel_size=.12, required_route_height_m=.5, required_route_width_m=.5)
    before = section_semantic_hash(fields)
    result, report = fit_required_sections(network, fields, config)
    assert report["changed_samples"] > 0
    assert result.segment_fields[1] is fields.segment_fields[1] or result.segment_fields[1] == fields.segment_fields[1]
    assert result.config == fields.config
    for a, b in zip(fields.segment_fields[0].samples, result.segment_fields[0].samples, strict=True):
        assert (a.x, a.y, a.z) == (b.x, b.y, b.z)
        assert b.roof_demand_ratio <= 1 and not b.collapse_required
    assert section_semantic_hash(fields) == before
    again, journal = fit_required_sections(network, fields, config)
    assert journal == report and section_semantic_hash(again) == section_semantic_hash(result)


def test_impossible_clearance_is_not_repaired_by_weakening_roof_limits():
    network = network_fixture()
    sections = SectionFieldGenerator().generate(network)
    config = GeometryConfig(required_route_height_m=100, required_route_width_m=100)
    with pytest.raises(SurfaceTopologyError, match="physical constraints"):
        fit_required_sections(network, sections, config)


def test_required_polyline_retains_corners_and_trims_only_terminal_caps():
    network = network_fixture()
    sections = SectionFieldGenerator().generate(network)
    config = GeometryConfig(required_route_height_m=.5, required_route_width_m=.5)
    paths, ids = required_paths(network, sections, config)
    assert ids == (0,)
    assert max(np.linalg.norm(np.diff(paths[0], axis=0), axis=1)) <= .25+1e-10
    assert np.linalg.norm(np.array(paths[0][0])-np.array([
        sections.segment_fields[0].samples[0].x, sections.segment_fields[0].samples[0].y,
        sections.segment_fields[0].samples[0].z])) > .5


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "0.5"])
@pytest.mark.parametrize("name", ["required_route_height_m", "required_route_width_m",
                                  "collision_max_error_m", "resolution_convergence_m"])
def test_new_physical_controls_reject_invalid_values(name, value):
    with pytest.raises(ValueError):
        GeometryConfig(**{name: value})


def test_separated_clearance_failures_do_not_mask_healthy_passage_between_them():
    mesh = trimesh.creation.box(extents=(2, 20, 2))
    for y in (-5., 5.):
        obstacle = trimesh.creation.box(extents=(1, .01, 1))
        obstacle.apply_translation((0, y, 0))
        mesh += obstacle
    path = tuple((0, float(y), 0) for y in np.arange(-8, 8.01, .25))
    with pytest.raises(MeshInspectionError) as error:
        inspect(mesh, paths=(path,))
    regions = error.value.report['defect_regions']
    assert len(regions) == 2
    assert regions[0]['upper_m'][1] < -4
    assert regions[1]['lower_m'][1] > 4
