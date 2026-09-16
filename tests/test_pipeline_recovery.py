"""Recovery orchestration, real diagnostics, failure containment and replay contracts."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from test_network_quality import network_fixture

from plume_advanced.acceptance import AcceptancePolicy, build_acceptance_policy
from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.pipeline import StageCheckpointStore
from plume_advanced.pipeline.recovery import (
    SEED_DOMAIN,
    PipelineRecoveryError,
    _local_candidate,
    _resample_sections,
    build_accepted_base,
    locate_affected_sections,
    write_recovery_report,
)
from plume_advanced.procedural import derive_subseed
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import NetworkQualityError, assess_network
from plume_advanced.stages.network_systems import GenerationDomainError
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.stages.surface_defects import localize_surface_defects
from plume_advanced.stages.surface_topology import SurfaceTopologyError


@pytest.fixture
def inputs():
    network = network_fixture()
    host = HostFieldGenerator(
        HostFieldConfig(grid=GridConfig(width=400, height=400, nx=24, ny=24))
    ).generate()
    sections = SectionFieldGenerator().generate(network)
    config = GeometryConfig(
        voxel_size=0.3,
        density_margin=2,
        minimum_radius=0.3,
        tunnel_radius_scale=1,
        chamber_radius_scale=1,
        junction_radius_scale=1,
        wall_roughness_amplitude=0,
        surface_wall_relief_m=0,
    )
    project = SimpleNamespace(
        procedural_seed=17, network=network.config, section_field=sections.config, geometry=config,
        acceptance=AcceptancePolicy(),
    )
    return project, host, network, sections


def rejected_surface(sections):
    point = sections.segment_fields[0].samples[1]
    return SurfaceTopologyError(
        "extra handle",
        report={
            "defect_regions": [
                dict(
                    kind="handle_patch",
                    lower_m=[point.x - 2, point.y - 2, point.z - 2],
                    upper_m=[point.x + 2, point.y + 2, point.z + 2],
                    center_m=[point.x, point.y, point.z],
                )
            ],
            "attempts": [],
        },
    )


def accepted_surface(config, network, sections, **kwargs):
    # A simple mesh is sufficient for orchestration tests. Separate integration
    # tests exercise stamping, real defect inspection and the full gates.
    mesh = trimesh.creation.box(extents=(220, 50, 100))
    mesh.apply_translation([50, 0, 80])
    points = tuple((s.x, s.y, s.z) for f in sections.segment_fields for s in f.samples)
    return CaveGeometry(
        config.config,
        VoxelGrid((0, 0, 0), 1, np.ones((2, 2, 2)), 0),
        (),
        tuple(map(tuple, mesh.vertices)),
        tuple(map(tuple, mesh.faces)),
        1,
        len(points),
        (0,),
        expected_surface_genus=0,
        route_centers=points,
    )


@pytest.mark.parametrize(
    "name,maximum", [("recovery_local_attempts", 2), ("recovery_network_attempts", 8)]
)
@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None, 100])
def test_recovery_budget_rejects_invalid_type_or_range(name, maximum, value):
    with pytest.raises(ValueError, match=name):
        GeometryConfig(**{name: value})


def test_config_accepts_explicit_zero_and_caps():
    assert GeometryConfig(recovery_local_attempts=0, recovery_network_attempts=0)
    assert GeometryConfig(recovery_local_attempts=2, recovery_network_attempts=8)


def test_measured_handle_localization_is_deterministic_and_does_not_modify_mesh():
    mesh = trimesh.creation.torus(major_radius=3, minor_radius=1)
    original = mesh.vertices.copy()
    regions = localize_surface_defects(mesh.vertices, mesh.faces, window_m=12)
    assert any(r["kind"] == "handle_patch" and r["genus"] == 1 for r in regions)
    assert regions == localize_surface_defects(mesh.vertices, mesh.faces, window_m=12)
    np.testing.assert_array_equal(mesh.vertices, original)
    assert len(localize_surface_defects(mesh.vertices, mesh.faces, window_m=12, max_regions=1)) == 1


def test_localizer_distinguishes_sphere_and_detached_shell():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    assert localize_surface_defects(mesh.vertices, mesh.faces, window_m=12) == []
    small = trimesh.creation.icosphere(radius=0.2, subdivisions=1)
    small.apply_translation([10, 0, 0])
    union = mesh + small
    regions = localize_surface_defects(union.vertices, union.faces, window_m=12)
    detached = [r for r in regions if r["kind"] == "detached_surface"]
    assert detached and detached[0]["center_m"] == [10, 0, 0]


@pytest.mark.parametrize('translation', [0., 1_000_000.])
def test_detached_fragment_diagnostic_retains_oriented_volume(translation):
    main = trimesh.creation.box(extents=(10, 10, 10))
    main.invert()  # cavity boundary points inward
    small = trimesh.creation.box(extents=(.4, .5, .2))
    # These signed measurements are diagnostic; host anchoring decides removal.
    union = main+small
    union.apply_translation([translation]*3)
    regions = localize_surface_defects(union.vertices, union.faces, window_m=12)
    detached = next(r for r in regions if r['kind'] == 'detached_surface')
    assert detached['signed_volume_m3'] == pytest.approx(.04, abs=1e-8)
    assert detached['main_signed_volume_m3'] == pytest.approx(-1000.)
    assert detached['triangles'] == 12


def test_handle_crossing_first_partition_is_found_by_overlapping_slabs():
    mesh = trimesh.creation.torus(major_radius=2, minor_radius=0.5)
    mesh.apply_translation([6, 0, 0])
    # Used triangles on a remote sphere extend the partition domain.
    extra = trimesh.creation.icosphere(radius=1, subdivisions=1)
    extra.apply_translation([-3, 0, 0])
    mesh += extra
    assert any(
        r["kind"] == "handle_patch"
        for r in localize_surface_defects(mesh.vertices, mesh.faces, window_m=12)
    )


def test_localization_maps_to_segments_and_keeps_uncertainty_explicit(inputs):
    project, host, network, sections = inputs
    result = locate_affected_sections(network, sections, rejected_surface(sections).report, 0.3)
    assert result["segment_ids"] == [0]
    assert result["regions"][0]["segment_ids"] == [0]
    assert not result["under_resolved_junction_hypotheses"]
    assert locate_affected_sections(network, sections, {}, 0.01)["segment_ids"] == []


def test_mesh_invariant_mapping_keeps_spatial_localization(inputs):
    _, _, network, sections = inputs
    failure = rejected_surface(sections).report
    failure['checks'] = dict(finite=True, positive_triangle_area=True,
                             closed=True, winding_consistent=True)
    before = deepcopy(failure)
    assert locate_affected_sections(network, sections, failure, .3)['segment_ids'] == [0]
    assert failure == before


def test_section_diagnostics_localize_without_mesh_regions_or_mutating_evidence(inputs):
    _, _, network, sections = inputs
    failure = {'checks': [
        dict(name='junction_floor_continuity',passed=False,segment_ids=[0,999]),
        dict(name='section_grade',passed=True,segment_ids=[0]),
        dict(name='unknown_segment',passed=False,segment_ids=[999]),
    ]}
    before = deepcopy(failure)
    report = locate_affected_sections(network,sections,failure,.01)
    assert report['segment_ids'] == [0]
    assert report['reported_failures'] == [dict(name='junction_floor_continuity',segment_ids=[0])]
    assert report['regions'] == [] and report['under_resolved_junction_hypotheses'] == []
    assert report == locate_affected_sections(network,sections,failure,.01)
    assert failure == before


def test_enlargement_rejection_identifies_its_section_and_ignores_unknown_ids(inputs):
    _, _, network, sections = inputs
    failure = {'route_section_repair': {'rejected_enlargements': [
        dict(segment_id=0,sample_index=2,reason='Insufficient cover'),
        dict(segment_id=999,sample_index=3,reason='Unknown segment'),
    ]}}
    result = locate_affected_sections(network,sections,failure,.01)
    assert result['segment_ids'] == [0]
    assert result['reported_failures'] == [dict(name='route_section_enlargement',segment_ids=[0],
                                               sample_index=2,reason='Insufficient cover')]


def test_spatial_localization_does_not_overwrite_original_region_ids(inputs):
    _, _, network, sections = inputs
    failure = rejected_surface(sections).report
    failure['defect_regions'][0]['segment_ids'] = [-1]
    before = deepcopy(failure)
    assert locate_affected_sections(network,sections,failure,.3)['segment_ids'] == [0]
    assert failure == before


def test_reported_junction_failure_reaches_local_repair_before_regeneration(inputs,monkeypatch):
    import plume_advanced.pipeline.recovery as module

    project,host,network,sections = inputs
    project.geometry = replace(project.geometry,recovery_network_attempts=0)
    def assess(candidate,field,profiles):
        if profiles is sections:
            return dict(accepted=False,checks=[dict(name='junction_floor_continuity',passed=False,
                        severity='error',value=.08,limit=.01,segment_ids=[0])])
        return assess_network(candidate,field,profiles)
    monkeypatch.setattr(module,'assess_network',assess)
    monkeypatch.setattr(GeometryGenerator,'build_base_volume',accepted_surface)
    accepted = build_accepted_base(project,host,network,sections)
    assert accepted.report['outcome'] == 'locally_repaired'
    initial,repaired = accepted.report['attempts']
    assert initial['error_type'] == 'NetworkQualityError'
    assert repaired['kind'] == 'local_section_resampling' and repaired['status'] == 'accepted'
    assert repaired['localization']['reported_failures'][0]['segment_ids'] == [0]
    assert repaired['localization']['regions'] == []
    assert accepted.network.config.random_seed == network.config.random_seed


def test_resampling_preserves_unaffected_fields_and_minimum_envelopes(inputs):
    project, host, network, sections = inputs
    assert _resample_sections(network, sections, set()).segment_fields == sections.segment_fields
    first, second = [_resample_sections(network, sections, {0}) for _ in range(2)]
    assert len(first.segment_fields[0].samples) > len(sections.segment_fields[0].samples)
    assert section_semantic_hash(first) == section_semantic_hash(second)
    assert assess_network(network, host, first)["accepted"]
    assert first.config == sections.config
    assert all(
        s.tube_width >= sections.config.minimum_tube_width
        and s.tube_height >= sections.config.minimum_tube_height
        for f in first.segment_fields
        for s in f.samples
    )


def test_width_repair_retains_graph_nodes_host_and_route_endpoints(inputs):
    project, host, network, sections = inputs
    before = host_semantic_hash(host)
    repaired, profiles = _local_candidate(
        "local_width_clearance", project, host, network, sections, [0]
    )
    assert repaired.nodes == network.nodes
    assert [(s.start_node_id, s.end_node_id) for s in repaired.segments] == [(0, 1)]
    for original, changed in zip(network.segments, repaired.segments, strict=True):
        assert changed.points[0].x == original.points[0].x
        assert changed.points[-1].x == original.points[-1].x
        assert max(p.width for p in changed.points) <= max(p.width for p in original.points)
    assert host_semantic_hash(host) == before
    assert profiles.config == sections.config


def test_first_success_has_no_repair_and_round_trips_atomic_checkpoint(
    inputs, monkeypatch, tmp_path
):
    monkeypatch.setattr(GeometryGenerator, "build_base_volume", accepted_surface)
    result = build_accepted_base(*inputs, report_path=tmp_path / "report.json")
    assert result.report["outcome"] == "unchanged"
    assert len(result.report["attempts"]) == 1
    assert result.network is inputs[2] and result.sections is inputs[3]
    store = StageCheckpointStore(tmp_path / "cache", "frozen")
    store.save("accepted_base", result)
    restored = store.load("accepted_base")
    assert restored.context_sha256 == result.context_sha256
    assert restored.report == result.report
    write_recovery_report(restored, tmp_path / "reused.json")
    assert (tmp_path / "report.json").read_bytes() == (tmp_path / "reused.json").read_bytes()


def test_local_success_preserves_original_centers_and_replays(inputs, monkeypatch):
    project, host, network, sections = inputs
    original = (
        host_semantic_hash(host),
        network_semantic_hash(network),
        section_semantic_hash(sections),
    )

    def build(generator, candidate, profiles, **kwargs):
        if profiles is sections:
            raise rejected_surface(profiles)
        return accepted_surface(generator, candidate, profiles)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", build)
    first, second = [build_accepted_base(*inputs) for _ in range(2)]
    assert first.report == second.report
    assert first.report["outcome"] == "locally_repaired"
    assert first.report["attempts"][-1]["preserved_original_centers"] == 4
    assert first.report["attempts"][0]["status"] == "rejected"
    assert set((s.x, s.y, s.z) for f in sections.segment_fields for s in f.samples) <= set(
        first.geometry.route_centers
    )
    assert original == (
        host_semantic_hash(host),
        network_semantic_hash(network),
        section_semantic_hash(sections),
    )
    assert first.report["accepted_identity"]["network"] == network_semantic_hash(first.network)
    assert first.report["accepted_identity"]["sections"] == section_semantic_hash(first.sections)


def test_local_repair_that_loses_old_route_is_rejected(inputs, monkeypatch):
    import plume_advanced.pipeline.recovery as module

    project, host, network, sections = inputs
    project.geometry = replace(project.geometry, recovery_network_attempts=0)

    def build(generator, candidate, profiles, **kwargs):
        if profiles is sections:
            raise rejected_surface(profiles)
        return accepted_surface(generator, candidate, profiles)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", build)
    monkeypatch.setattr(
        module,
        "inspect_surface",
        lambda *a, **kw: (_ for _ in ()).throw(SurfaceTopologyError("old route lost")),
    )
    with pytest.raises(PipelineRecoveryError) as caught:
        build_accepted_base(*inputs)
    assert any(r.get("reason") == "old route lost" for r in caught.value.report["attempts"])


def test_regeneration_uses_original_stage_seed_and_same_host(inputs, monkeypatch):
    project, host, network, sections = inputs
    project.geometry = replace(project.geometry, recovery_local_attempts=0)
    seen = []

    def candidate(generator, same_host):
        assert same_host is host
        seen.append(generator.config.random_seed)
        return network_fixture(config=generator.config, widths=[5.2] * 5)

    monkeypatch.setattr(CaveNetworkGenerator, "_generate_candidate", candidate)

    def build(generator, candidate, profiles, **kwargs):
        if candidate is network:
            raise rejected_surface(profiles)
        return accepted_surface(generator, candidate, profiles)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", build)
    first, second = [build_accepted_base(*inputs) for _ in range(2)]
    seed = derive_subseed(project.network.random_seed, SEED_DOMAIN, 1)
    assert seen == [seed, seed]
    assert first.report == second.report and first.report["outcome"] == "regenerated"
    assert first.network.config.random_seed == seed
    assert first.network.config.quality == network.config.quality
    assert first.sections.config == sections.config
    assert first.report["host_unchanged"]


def test_exhaustion_is_finite_and_retains_diagnostics(inputs, monkeypatch, tmp_path):
    project, host, network, sections = inputs
    calls = []

    def reject(generator, candidate, profiles, **kwargs):
        calls.append(network_semantic_hash(candidate))
        raise rejected_surface(profiles)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", reject)
    monkeypatch.setattr(
        CaveNetworkGenerator,
        "_generate_candidate",
        lambda g, h: network_fixture(
            config=g.config, widths=[5.1 + (g.config.random_seed % 3) * 0.1] * 5
        ),
    )
    with pytest.raises(PipelineRecoveryError) as caught:
        build_accepted_base(*inputs, report_path=tmp_path / "report.json")
    assert caught.value.report["status"] == "exhausted"
    assert len(caught.value.report["attempts"]) == 5
    assert 1 <= len(calls) <= 5
    assert all(r["status"] in ("rejected", "skipped") for r in caught.value.report["attempts"])
    assert not caught.value.report["accepted"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_ground_placement_exhaustion_retains_failure_without_geological_edits(inputs, monkeypatch, tmp_path):
    calls = []
    def reject(self, *args, **kwargs):
        calls.append(True)
        raise SurfaceTopologyError('ground contract failed', report={'ground_traversal': {'passed': False}})
    monkeypatch.setattr(GeometryGenerator, 'build_base_volume', reject)
    with pytest.raises(PipelineRecoveryError) as error:
        build_accepted_base(*inputs, report_path=tmp_path/'report.json')
    assert len(calls) == len(error.value.report['attempts']) == 1
    assert 'ground-route' in error.value.report['stop_reason']
    assert error.value.report['host_unchanged']


def test_measured_seed0_ground_report_rejects_without_localization_crash(inputs, monkeypatch, tmp_path):
    import json
    from pathlib import Path

    import plume_advanced.pipeline.recovery as module
    from plume_advanced.stages.mesh_inspection import MeshInspectionError

    failure = json.loads((Path(__file__).parent / 'fixtures/recovery/ground_failure_seed0.json').read_text())
    calls = []
    def reject(*args, **kwargs):
        calls.append(True)
        raise MeshInspectionError(failure)
    def must_not_localize(*args, **kwargs):
        raise AssertionError('Ground exhaustion must not enter geological localization')
    monkeypatch.setattr(GeometryGenerator, 'build_base_volume', reject)
    monkeypatch.setattr(module, 'locate_affected_sections', must_not_localize)
    with pytest.raises(PipelineRecoveryError) as error:
        build_accepted_base(*inputs, report_path=tmp_path/'report.json')
    report = error.value.report
    assert report['status'] == 'exhausted' and report['host_unchanged']
    assert len(calls) == len(report['attempts']) == 1
    assert report['attempts'][0]['inspection'] == failure
    assert json.loads((tmp_path/'report.json').read_text()) == report


@pytest.mark.parametrize(
    "error",
    [
        TypeError("bug"),
        MemoryError("budget"),
        GenerationDomainError("domain"),
        FileNotFoundError("input"),
    ],
)
def test_nonrepairable_errors_escape_without_seed_retries(inputs, monkeypatch, tmp_path, error):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", fail)
    with pytest.raises(type(error), match=str(error)):
        build_accepted_base(*inputs, report_path=tmp_path / "report.json")
    assert len(calls) == 1
    import json

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "interrupted" and not report["accepted"]


def test_surface_diagnostics_survive_interruption_before_next_grid(inputs, monkeypatch, tmp_path):
    import json

    measurement = dict(candidate_index=1, voxel_size_m=.04, accepted=False,
                       floating_solid_samples_removed=286,
                       inspection=dict(components=2, expected_genus=2))
    def interrupted(self, *args, **kwargs):
        self.surface_diagnostic(measurement)
        # The measurement must already be on disk before the next expensive step.
        saved = json.loads((tmp_path/'report.json').read_text())
        assert saved['attempts'][0]['surface_candidates'] == [measurement]
        raise KeyboardInterrupt
    monkeypatch.setattr(GeometryGenerator, 'build_base_volume', interrupted)
    with pytest.raises(KeyboardInterrupt):
        build_accepted_base(*inputs, report_path=tmp_path/'report.json')
    saved = json.loads((tmp_path/'report.json').read_text())
    assert not saved['accepted']
    assert saved['attempts'][0]['surface_candidates'] == [measurement]


def test_identical_replacement_skips_expensive_geometry(inputs, monkeypatch):
    inputs[0].geometry = replace(inputs[0].geometry, recovery_local_attempts=0)
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        raise SurfaceTopologyError("same mesh")

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", reject)
    monkeypatch.setattr(CaveNetworkGenerator, "generate", lambda *a, **kw: inputs[2])
    with pytest.raises(PipelineRecoveryError) as caught:
        build_accepted_base(*inputs)
    assert len(calls) == 1
    assert [r["status"] for r in caught.value.report["attempts"]] == [
        "rejected",
        "skipped",
        "skipped",
    ]


def test_rejected_network_replacements_do_not_reach_meshing(inputs, monkeypatch):
    inputs[0].geometry = replace(inputs[0].geometry, recovery_local_attempts=0)
    monkeypatch.setattr(
        GeometryGenerator,
        "build_base_volume",
        lambda *a, **kw: (_ for _ in ()).throw(SurfaceTopologyError("original failed")),
    )
    monkeypatch.setattr(
        CaveNetworkGenerator,
        "generate",
        lambda *a, **kw: (_ for _ in ()).throw(
            NetworkQualityError({"accepted": False, "checks": []})
        ),
    )
    with pytest.raises(PipelineRecoveryError) as caught:
        build_accepted_base(*inputs)
    assert [r["error_type"] for r in caught.value.report["attempts"]] == [
        "SurfaceTopologyError",
        "NetworkQualityError",
        "NetworkQualityError",
    ]


@pytest.mark.integration
def test_actual_stamping_after_local_resampling_with_all_original_route_gates(inputs, monkeypatch):
    original_build = GeometryGenerator.build_base_volume
    original_sections = inputs[3]

    def fail_original_once(generator, network, sections, **kwargs):
        if sections is original_sections:
            raise rejected_surface(sections)
        return original_build(generator, network, sections, **kwargs)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", fail_original_once)
    result = build_accepted_base(*inputs)
    assert result.report["outcome"] == "locally_repaired"
    assert dict(result.geometry.mesh_inspection)["passed"]
    assert result.geometry.component_count == 1


@pytest.mark.integration
@pytest.mark.parametrize("host_consistent", [True, False])
def test_actual_stamping_after_local_width_repair_preserves_original_routes(
    inputs, monkeypatch, host_consistent
):
    """Force preceding failures, then stamp and inspect the actual narrowed network."""
    from plume_advanced.stages.network_acceptance import repair_network

    project, host, network, sections = inputs
    # The generic synthetic fixture deliberately has elevations unrelated to
    # this host. Test both that unsafe case and a host-sampled original route.
    if host_consistent:
        network = repair_network(CaveNetworkGenerator(network.config), host, network, 0)
        sections = SectionFieldGenerator(project.section_field).generate(network)
    project.geometry = replace(project.geometry, recovery_network_attempts=0)
    inputs = project, host, network, sections
    original_build = GeometryGenerator.build_base_volume
    original_network = inputs[2]
    before = (host_semantic_hash(inputs[1]), network_semantic_hash(original_network),
              section_semantic_hash(inputs[3]))

    def fail_before_width_repair(generator, network, sections, **kwargs):
        if network is original_network:
            raise rejected_surface(sections)
        return original_build(generator, network, sections, **kwargs)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", fail_before_width_repair)
    if not host_consistent:
        with pytest.raises(PipelineRecoveryError) as caught:
            build_accepted_base(*inputs)
        last = caught.value.report["attempts"][-1]
        assert last["kind"] == "local_width_clearance" and last["status"] == "rejected"
        assert last["inspection"]["outside_centers"]
        assert not caught.value.report["accepted"]
        assert before == (host_semantic_hash(host), network_semantic_hash(network),
                          section_semantic_hash(sections))
        return
    first, second = [build_accepted_base(*inputs) for _ in range(2)]
    assert first.report["outcome"] == "locally_repaired"
    assert [(r["kind"], r["status"]) for r in first.report["attempts"]] == [
        ("original", "rejected"), ("local_section_resampling", "rejected"),
        ("local_width_clearance", "accepted"),
    ]
    assert dict(first.geometry.mesh_inspection)["passed"]
    assert first.geometry.component_count == 1
    assert first.report == second.report
    np.testing.assert_array_equal(first.geometry.assembled_vertices, second.geometry.assembled_vertices)
    np.testing.assert_array_equal(first.geometry.assembled_faces, second.geometry.assembled_faces)
    assert before == (host_semantic_hash(inputs[1]), network_semantic_hash(original_network),
                      section_semantic_hash(inputs[3]))
    original_centers = {(s.x, s.y, s.z) for f in inputs[3].segment_fields for s in f.samples}
    assert original_centers <= set(first.geometry.route_centers)


@pytest.mark.integration
@pytest.mark.parametrize("profile", ["research", "inspection"])
def test_cli_regeneration_publishes_matching_artifacts_and_resume_reuses_triple(
    inputs, monkeypatch, tmp_path, profile
):
    """Real exports and checkpoints must use the replacement, including after resume."""
    import json

    from plume_advanced import cli
    from plume_advanced.config import load_project_config
    from plume_advanced.identity import sha256_file

    fixture_project, host, network, sections = inputs
    project = load_project_config(cli.PACKAGED_CONFIG)
    project = replace(
        project,
        acceptance=build_acceptance_policy({"profile": profile}),
        export=replace(project.export, target="blender"),
        host_field=host.config,
        network=network.config,
        section_field=sections.config,
        geometry=replace(
            fixture_project.geometry,
            recovery_local_attempts=0,
            cave_diffuse_texture="",
            cave_normal_texture="",
            cave_roughness_texture="",
            cave_displacement_texture="",
            cave_smoothing_iterations=0,
            cave_displacement_scale_m=0,
            required_route_height_m=.5 if profile == "inspection" else 0.,
            required_route_width_m=.5 if profile == "inspection" else 0.,
        ),
    )
    monkeypatch.setattr(cli, "load_project_config", lambda *a, **kw: project)
    monkeypatch.setattr(
        CaveNetworkGenerator,
        "_generate_candidate",
        lambda g, h: network_fixture(
            config=g.config, widths=[5 if g.config.random_seed is None else 5.2] * 5
        ),
    )
    build = GeometryGenerator.build_base_volume

    def reject_initial(generator, candidate, profiles, **kwargs):
        if candidate.config.random_seed is None:
            raise rejected_surface(profiles)
        return build(generator, candidate, profiles, **kwargs)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", reject_initial)
    output = tmp_path / "run" / "network.png"
    assert cli.main(["--output", str(output)]) == 0
    directory = output.parent
    recovery = json.loads((directory / "pipeline_recovery.json").read_text())
    assert recovery["outcome"] == "regenerated"
    assert (
        json.loads((directory / "stage_b_network.json").read_text())["semantic_sha256"]
        == recovery["accepted_identity"]["network"]
    )
    assert (
        json.loads((directory / "stage_c_sections.json").read_text())["semantic_sha256"]
        == recovery["accepted_identity"]["sections"]
    )
    assert (
        json.loads((directory / "pipeline_quality_report.json").read_text())["upstream_recovery"]
        == recovery
    )
    asset = next(directory.rglob("plume_cave_scene.glb"))
    original_hash = sha256_file(asset)

    def must_reuse(*args, **kwargs):
        raise AssertionError("Accepted triple should have been restored")

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", must_reuse)
    monkeypatch.setattr(CaveNetworkGenerator, "generate", must_reuse)
    assert cli.main(["--output", str(output), "--resume"]) == 0
    assert json.loads((directory / "pipeline_recovery.json").read_text()) == recovery
    assert sha256_file(asset) == original_hash
    assert any(
        p.name.startswith("base_floor_atlas_")
        for p in (directory / ".plume-checkpoints").glob("*.pickle")
    )
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    quality = json.loads((directory / "pipeline_quality_report.json").read_text())
    assert quality["acceptance"]["policy"]["profile"] == profile
    assert quality["acceptance"]["checks"]["clearance"]["status"] == (
        "passed" if profile == "inspection" else "not_requested")
    assert "pipeline_recovery.json" in {row["path"] for row in manifest["outputs"]}
    for row in manifest["outputs"]:
        assert sha256_file(directory / row["path"]) == row["sha256"]

    import runpy
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    loader = runpy.run_path(str(scripts / "render_run_diagnostics.py"))["load_saved_stages"]
    stages, evidence, _ = loader(directory)
    assert network_semantic_hash(stages[1]) == recovery["accepted_identity"]["network"]
    assert section_semantic_hash(stages[2]) == recovery["accepted_identity"]["sections"]
    assert evidence["final_geometry"]["stage"].startswith("final_geometry_")
    # A renderer must not fall back to a damaged provisional network.
    (directory / ".plume-checkpoints" / "network.pickle").write_bytes(b"obsolete initial candidate")
    assert (
        network_semantic_hash(loader(directory)[0][1]) == recovery["accepted_identity"]["network"]
    )
    monkeypatch.setattr("sys.argv", ["check_mesh_topology.py", str(directory)])
    checker = runpy.run_path(str(scripts / "check_mesh_topology.py"))
    assert checker["main"]() == 0


def test_host_mutation_is_fatal_and_not_retried(inputs, monkeypatch):
    def mutate(generator, network, sections, **kwargs):
        inputs[1].elevation.flat[0] += 1
        return accepted_surface(generator, network, sections)

    monkeypatch.setattr(GeometryGenerator, "build_base_volume", mutate)
    with pytest.raises(AssertionError, match="mutated"):
        build_accepted_base(*inputs)


def test_failure_progress_finishes_and_disabled_recovery_keeps_original(inputs, monkeypatch):
    inputs[0].geometry = replace(
        inputs[0].geometry, recovery_local_attempts=0, recovery_network_attempts=0
    )
    monkeypatch.setattr(
        GeometryGenerator,
        "build_base_volume",
        lambda *a, **kw: (_ for _ in ()).throw(SurfaceTopologyError("unresolved")),
    )
    events = []
    with pytest.raises(PipelineRecoveryError) as caught:
        build_accepted_base(*inputs, progress=lambda *args: events.append(args))
    assert len(caught.value.report["attempts"]) == 1
    assert events[-1][1:3] == (1, 1)


def test_retained_seed1_recipe_loads_without_deleted_outputs_or_external_maps():
    from pathlib import Path

    from plume_advanced.config import load_project_config

    recipe = Path(__file__).parent / "fixtures/recovery/seed1_multi_250m.toml"
    project = load_project_config(recipe)
    assert project.procedural_seed == 1
    assert project.network.systems.count == 3
    assert project.network.target_route_length_m == 250
    assert project.geometry.voxel_size == 0.12
    assert project.network.quality.enabled
    assert not project.events.enabled
    assert not project.geometry.cave_diffuse_texture
    assert (
        project.geometry.recovery_local_attempts == project.geometry.recovery_network_attempts == 2
    )
