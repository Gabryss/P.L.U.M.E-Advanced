from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from test_network_quality import network_fixture

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.mesh_inspection import MeshInspectionError, inspect_surface
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import assess_network
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
        assert (a.tangent, a.normal, a.binormal) == (b.tangent, b.normal, b.binormal)
        assert b.floor_world_z == pytest.approx(a.floor_world_z, abs=1e-10)
        assert b.roof_demand_ratio <= 1 and not b.collapse_required
    assert section_semantic_hash(fields) == before
    again, journal = fit_required_sections(network, fields, config)
    assert journal == report and section_semantic_hash(again) == section_semantic_hash(result)


def rectangular_section(*, pitch=0., roll=0., width=3., height=.2, cover=20.):
    network = network_fixture()
    sections = SectionFieldGenerator().generate(network)
    generator = SectionFieldGenerator(sections.config)
    tangent = np.array([np.cos(pitch), 0., np.sin(pitch)])
    normal = np.array([0., 1., 0.])
    binormal = np.cross(tangent, normal)
    normal, binormal = (np.cos(roll)*normal+np.sin(roll)*binormal,
                        -np.sin(roll)*normal+np.cos(roll)*binormal)
    sample = generator._assess_roof(replace(
        sections.segment_fields[0].samples[0], x=0., y=0., z=0., surface_z=cover,
        tangent=tuple(tangent), normal=tuple(normal), binormal=tuple(binormal),
        tube_width=width, tube_height=height,
        profile_points=((-width/2,-height/2),(width/2,-height/2),
                        (width/2,height/2),(-width/2,height/2)),
    ))
    field = replace(sections.segment_fields[0], samples=(sample,))
    return network, replace(sections, segment_fields=(field,)), sample


def test_required_free_air_reserves_retained_relief_without_lowering_floor():
    network, sections, original = rectangular_section()
    config = GeometryConfig(voxel_size=.12, required_route_height_m=.5, required_route_width_m=.5,
        surface_roof_relief_m=.65, surface_floor_relief_m=.22,
        surface_wall_relief_m=.55, surface_crust_relief_m=.1)
    enlarged, journal = fit_required_sections(network, sections, config, minimum_relief_scale=1.)
    sample = enlarged.segment_fields[0].samples[0]
    assert journal["changed_samples"] == 1
    assert journal["relief_reserve_height_m"] == pytest.approx(1.07)
    assert sample.roof_world_z-sample.floor_world_z >= .5+.24+.04+1.07-1e-9
    assert sample.floor_world_z == pytest.approx(original.floor_world_z, abs=1e-12)
    assert not sample.collapse_required


def test_detail_reservation_still_rejects_insufficient_roof_cover():
    network, sections, original = rectangular_section(cover=.3)
    config = GeometryConfig(voxel_size=.12, required_route_height_m=.5, required_route_width_m=.5,
        surface_roof_relief_m=.65, surface_floor_relief_m=.22)
    with pytest.raises(SurfaceTopologyError) as caught:
        fit_required_sections(network, sections, config, minimum_relief_scale=1.)
    assert sections.segment_fields[0].samples[0] == original
    assert caught.value.report["route_section_repair"]["rejected_enlargements"]


@pytest.mark.parametrize("pitch,roll", [(0.,0.),(.2,0.),(-.2,.4),(.2,-.4),(0.,np.pi)])
@pytest.mark.parametrize("width_only", [False, True])
def test_enlargement_preserves_actual_world_floor_in_tilted_frames(pitch, roll, width_only):
    network, sections, sample = rectangular_section(
        pitch=pitch, roll=roll, width=.6 if width_only else 3., height=1.2 if width_only else .2)
    config = GeometryConfig(voxel_size=.12, required_route_height_m=.5,
                            required_route_width_m=.5)
    repaired, report = fit_required_sections(network, sections, config)
    changed = repaired.segment_fields[0].samples[0]
    assert report['changed_samples'] == 1
    before = np.asarray(sample.profile_points) @ np.array([sample.normal[2],sample.binormal[2]])
    after = np.asarray(changed.profile_points) @ np.array([changed.normal[2],changed.binormal[2]])
    assert min(after) == pytest.approx(min(before),abs=1e-10)
    assert changed.floor_world_z == pytest.approx(sample.floor_world_z,abs=1e-10)
    assert np.ptp(after) >= config.required_route_height_m+2*config.voxel_size+2*config.route_clearance_margin_m
    assert (changed.tangent,changed.normal,changed.binormal) == (sample.tangent,sample.normal,sample.binormal)
    assert changed.roof_world_z > sample.roof_world_z or width_only
    assert report['changes'][0]['after_floor_world_z'] == changed.floor_world_z
    again, repeated = fit_required_sections(network,repaired,config)
    assert repeated['changed_samples'] == 0
    assert section_semantic_hash(again) == section_semantic_hash(repaired)


def test_floor_anchoring_rechecks_cover_instead_of_lowering_the_floor():
    network, sections, sample = rectangular_section(cover=6.6)
    assert sample.roof_thickness > sections.config.minimum_roof_thickness
    config = GeometryConfig(voxel_size=.12,required_route_height_m=.5,required_route_width_m=.5)
    with pytest.raises(SurfaceTopologyError,match='physical constraints') as caught:
        fit_required_sections(network,sections,config)
    assert caught.value.report['route_section_repair']['rejected_enlargements'][0]['reason'].startswith('Enlargement violates')
    assert sections.segment_fields[0].samples[0] is sample


def test_floor_anchored_tilt_cannot_move_an_enlarged_contour_outside_host():
    network, sections, sample = rectangular_section(pitch=.2)
    # Centre-based enlargement fits this footprint; raising the floor-anchored
    # tilted contour moves its upper edge outside. Check the final world points.
    class Host:
        def contains(self,x,y):
            return -.1 <= x <= .1 and -2 <= y <= 2
    config = GeometryConfig(voxel_size=.12,required_route_height_m=.5,required_route_width_m=.5)
    with pytest.raises(SurfaceTopologyError,match='physical constraints') as caught:
        fit_required_sections(network,sections,config,Host())
    assert caught.value.report['route_section_repair']['rejected_enlargements'][0]['segment_id'] == 0
    assert sections.segment_fields[0].samples[0] is sample


@pytest.mark.integration
@pytest.mark.parametrize("retention", [0., 1.])
def test_seed0_clearance_enlargement_preserves_all_junction_floors(retention):
    project = load_project_config(Path(__file__).parent/'fixtures/route_clearance/seed0.toml',seed_override=0)
    host = HostFieldGenerator(project.host_field).generate()
    network = CaveNetworkGenerator(project.network).generate(host,section_config=project.section_field)
    sections = SectionFieldGenerator(project.section_field).generate(network)
    before = host_semantic_hash(host),network_semantic_hash(network),section_semantic_hash(sections)
    assert assess_network(network,host,sections)['accepted']
    config = GeometryConfig(voxel_size=.12,required_route_height_m=.5,required_route_width_m=.5,
        surface_roof_relief_m=.65,surface_floor_relief_m=.22,
        surface_wall_relief_m=.55,surface_crust_relief_m=.1)
    repaired, report = fit_required_sections(network,sections,config,host,minimum_relief_scale=retention)
    assert report['changed_samples'] > 0
    assessment = assess_network(network,host,repaired)
    assert assessment['accepted'], [r for r in assessment['checks'] if not r['passed']]
    for a,b in zip(sections.segment_fields,repaired.segment_fields,strict=True):
        np.testing.assert_allclose([s.floor_world_z for s in a.samples],
                                   [s.floor_world_z for s in b.samples],atol=1e-10,rtol=0)
        if a.segment_id not in sections.dominant_route_segment_ids:
            assert a == b
    assert before == (host_semantic_hash(host),network_semantic_hash(network),section_semantic_hash(sections))


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
