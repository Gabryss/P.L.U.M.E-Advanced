"""Measured ramp/obstruction repair, immutable inputs and constrained replay."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh
from test_network_quality import network_fixture

from plume_advanced.acceptance import build_acceptance_policy
from plume_advanced.evaluation.artifacts import section_semantic_hash
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.ground_routes import inspect_ground_routes
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.stages.section_repairs import (
    grade_floor,
    repair_ground_sections,
    repair_obstructed_sections,
)
from plume_advanced.stages.surface_topology import SurfaceTopologyError


def ramp_sections(height=.6):
    network = network_fixture([(-8, 0), (0, 0), (8, 0)], widths=[6]*3)
    original = SectionFieldGenerator().generate(network)
    config = replace(original.config, maximum_height_ratio=.8, minimum_tube_height=.3)
    generator = SectionFieldGenerator(config)
    template = original.segment_fields[0].samples[0]
    samples = []
    for i, x in enumerate(np.linspace(-8, 8, 81)):
        floor = height if x > 0 else 0.
        samples.append(generator._assess_roof(replace(template,
            index=i, segment_arc_length=x+8, x=x, y=0., z=1.5,
            surface_z=25., cover_thickness=30., centerline_depth=23.5,
            tangent=(1., 0., 0.), normal=(0., 1., 0.), binormal=(0., 0., 1.),
            tube_width=6., tube_height=3.-floor,
            profile_points=((-3., floor-1.5), (3., floor-1.5), (3., 1.5), (-3., 1.5)))))
    field = replace(original.segment_fields[0], samples=tuple(samples))
    sections = replace(original, config=config, segment_fields=(field,), dominant_route_segment_ids=(field.segment_id,))
    controls = GeometryConfig(voxel_size=.1, required_route_height_m=.5,
        required_route_width_m=.5, ground_robot_length_m=.7)
    return network, sections, controls


def section_mesh(sections):
    vertices, faces = [], []
    for s in sections.segment_fields[0].samples:
        profile = np.array(s.profile_points)
        vertices.extend(np.array([s.x, s.y, s.z]) + profile[:, 0, None]*s.normal + profile[:, 1, None]*s.binormal)
    for i in range(len(vertices)//4-1):
        for j in range(4):
            a, b = 4*i+j, 4*i+(j+1)%4
            faces.extend([(a, b, a+4), (b, b+4, a+4)])
    last = len(vertices)-4
    faces.extend([(0, 1, 2), (0, 2, 3), (last, last+2, last+1), (last, last+3, last+2)])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.fix_normals()
    assert mesh.is_watertight
    return mesh


def ground(sections):
    mesh = section_mesh(sections)
    sid = sections.segment_fields[0].segment_id
    return inspect_ground_routes(mesh.vertices, mesh.faces,
        (((-6., 0., 1.5), (6., 0., 1.5)),), (sid,), repair_attempts=0)


def test_real_floor_step_becomes_a_passing_ramp_without_relaxing_robot():
    _, sections, controls = ramp_sections()
    before_hash = section_semantic_hash(sections)
    failure = ground(sections)
    assert not failure["passed"]
    assert any(p.get("floor_points_m") for p in failure["paths"][0]["poses"] if not p["passed"])
    repaired, report = repair_ground_sections(sections, controls, None, failure)
    assert report["changed_samples"] > 0
    checked = ground(repaired)
    assert checked["passed"]
    assert checked["robot"] == failure["robot"]
    again, replay = repair_ground_sections(sections, controls, None, failure)
    assert report == replay and section_semantic_hash(again) == section_semantic_hash(repaired)
    assert section_semantic_hash(sections) == before_hash
    assert max(abs(c["before_floor_m"]-c["after_floor_m"]) for c in report["changes"]) <= .5+1e-8
    for a, b in zip(sections.segment_fields[0].samples, repaired.segment_fields[0].samples, strict=True):
        assert (a.x, a.y, a.z, a.tangent, a.normal, a.binormal) == (b.x, b.y, b.z, b.tangent, b.normal, b.binormal)
        assert a.roof_world_z == pytest.approx(b.roof_world_z)
        if abs(a.x) >= report["reach_m"]+.2:
            assert a == b
    assert sections.segment_fields[0].samples[0] == repaired.segment_fields[0].samples[0]
    assert sections.segment_fields[0].samples[-1] == repaired.segment_fields[0].samples[-1]


def test_unrepairable_step_is_retained_and_not_declared_qualified():
    _, sections, controls = ramp_sections(height=1.5)
    failure = ground(sections)
    repaired, report = repair_ground_sections(sections, replace(controls, ground_ramp_max_change_m=.05), None, failure)
    assert report["changed_samples"] == 0 and report["rejected_segments"]
    assert not ground(repaired)["passed"]


def test_ground_repair_never_edits_optional_route_or_unknown_failure():
    _, sections, controls = ramp_sections()
    failure = ground(sections)
    optional = replace(sections, dominant_route_segment_ids=())
    result, report = repair_ground_sections(optional, controls, None, failure)
    assert result == optional and report["changed_samples"] == 0
    for pose in failure["paths"][0]["poses"]:
        pose["failure"] = "missing_floor_or_wrong_cavity"
    result, report = repair_ground_sections(sections, controls, None, failure)
    assert result == sections and report["changed_samples"] == 0


def test_measured_cross_slope_is_repaired_without_changing_the_roof():
    network, sections, controls = ramp_sections(height=0)
    generator = SectionFieldGenerator(sections.config)
    slope = np.tan(np.radians(24.))
    # A narrow lower contour resolves the robot's support patch, with tall walls.
    profile = ((-3., -.9), (-.7, -.9-.7*slope), (.7, -.9+.7*slope),
               (3., -.9), (3., 1.5), (-3., 1.5))
    samples = tuple(generator._assess_roof(replace(s, profile_points=profile))
                    for s in sections.segment_fields[0].samples)
    sections = replace(sections, segment_fields=(replace(sections.segment_fields[0], samples=samples),))
    # Use a generic loft for this six-vertex contour.
    def measure(field):
        points = np.array([np.array([s.x, s.y, s.z])+np.array(s.profile_points)[:, 0, None]*s.normal
                           +np.array(s.profile_points)[:, 1, None]*s.binormal for s in field.segment_fields[0].samples])
        count, width, _ = points.shape
        faces = []
        for i in range(count-1):
            for j in range(width):
                a, b = i*width+j, i*width+(j+1)%width
                faces.extend([(a, b, a+width), (b, b+width, a+width)])
        for j in range(1, width-1):
            faces.extend([(0,j+1,j), ((count-1)*width,(count-1)*width+j,(count-1)*width+j+1)])
        mesh = trimesh.Trimesh(points.reshape(-1,3), faces, process=False)
        mesh.fix_normals()
        return inspect_ground_routes(mesh.vertices, mesh.faces,
            (((-6., 0., 1.5), (6., 0., 1.5)),), (sections.segment_fields[0].segment_id,), repair_attempts=0)
    failed = measure(sections)
    assert not failed['passed']
    repaired, report = repair_ground_sections(sections, controls, None, failed, network=network)
    assert report['changed_samples'] > 0
    assert any(abs(c['cross_grade_correction']) > 0 for c in report['changes'])
    assert measure(repaired)['passed']
    assert all(a.roof_world_z == pytest.approx(b.roof_world_z) for a,b in zip(samples,repaired.segment_fields[0].samples,strict=True))


def test_ramp_cannot_violate_host_bounds():
    _, sections, controls = ramp_sections()
    class RejectHost:
        def contains(self, x, y):
            return False
    with pytest.raises(SurfaceTopologyError, match="host"):
        repair_ground_sections(sections, controls, RejectHost(), ground(sections))


def test_obstruction_expansion_is_local_bounded_and_reproducible():
    _, sections, controls = ramp_sections(height=0)
    result, report = repair_obstructed_sections(sections, controls, None, [[0., 0., 1.5]], attempt=1)
    assert report["changed_samples"] > 0
    assert all(c["maximum_profile_change_m"] <= report["maximum_change_m"]+1e-9 for c in report["changes"])
    assert result.segment_fields[0].samples[0] == sections.segment_fields[0].samples[0]
    assert result.segment_fields[0].samples[-1] == sections.segment_fields[0].samples[-1]
    assert repair_obstructed_sections(sections, controls, None, [[0., 0., 1.5]], attempt=1) == (result, report)
    # It may propose an edit, but cannot report inspection/qualification success.
    assert "passed" not in report


def test_off_centre_relief_obstruction_is_repaired_on_the_actual_volume():
    from plume_advanced.acceptance import AcceptancePolicy
    from plume_advanced.stages.geometry import GeometryGenerator
    from plume_advanced.stages.surface_topology import PassageObstructionError

    network, sections, controls = ramp_sections(height=0)
    network = replace(network, config=replace(network.config, target_route_length_m=16.))
    generator = SectionFieldGenerator(sections.config)
    samples = []
    for sample in sections.segment_fields[0].samples:
        floor = 1.48*np.exp(-(sample.x/1.5)**2)
        samples.append(generator._assess_roof(replace(sample, tube_height=3-floor,
            profile_points=((-3., floor-1.5), (3., floor-1.5), (3., 1.5), (-3., 1.5)))))
    sections = replace(sections, segment_fields=(replace(sections.segment_fields[0], samples=tuple(samples)),))
    controls = replace(controls, wall_roughness_amplitude=0, density_margin=1,
        cave_smoothing_iterations=0, cave_displacement_scale_m=0, storage_mode='dense',
        ground_robot_length_m=0, surface_floor_relief_m=.22)
    policy = AcceptancePolicy(minimum_relief_scale=1.)
    with pytest.raises(PassageObstructionError) as caught:
        GeometryGenerator(controls, acceptance=policy).build_base_volume(network, sections)
    repaired, report = repair_obstructed_sections(sections, controls, None,
        caught.value.report['blocked_points_m'], attempt=2)
    assert report['maximum_change_m'] == .5
    result = GeometryGenerator(controls, acceptance=policy).build_base_volume(network, repaired)
    assert dict(result.mesh_inspection)['passed']
    assert result.config.voxel_size == controls.voxel_size
    assert result.effective_surface_relief_scale == 1.


def test_grade_solver_rejects_impossible_ramp_and_preserves_fixed_elevations():
    x = np.arange(5.)
    z = x.copy()
    assert grade_floor(x, z, z, z, max_grade=.2, max_curvature=.1) is None
    flat = np.full(5, 1_000_000.)
    np.testing.assert_array_equal(grade_floor(x, flat, flat, flat, max_grade=.2, max_curvature=.1), flat)


@pytest.mark.parametrize("value", [True, 0., -1., 1.01, float("nan"), float("inf"), "0.5"])
def test_ramp_edit_budget_validated(value):
    with pytest.raises(ValueError, match="ground_ramp_max_change_m"):
        GeometryConfig(ground_ramp_max_change_m=value)


def test_floor_edit_requires_explicit_robot_qualification_opt_in():
    assert not build_acceptance_policy().repair_ground_routes
    with pytest.raises(ValueError, match="require_ground_routes"):
        build_acceptance_policy(dict(repair_ground_routes=True))
    assert build_acceptance_policy(dict(require_ground_routes=True, repair_ground_routes=True)).repair_ground_routes
