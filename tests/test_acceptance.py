"""Acceptance is an executable publication contract, including cold replay/resume."""

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from test_embedded_inspection import geometry
from test_resolution_repair import real_short_inputs
from test_surface_acceptance import fixture as branch_fixture

from plume_advanced import cli
from plume_advanced.acceptance import (
    AcceptanceError,
    AcceptancePolicy,
    apply_acceptance_defaults,
    build_acceptance_policy,
    evaluate_acceptance,
    validate_acceptance_configuration,
)
from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.evaluation.reliability import ReliabilityCase
from plume_advanced.evaluation.reliability_state import preflight
from plume_advanced.exporters import export_target_asset
from plume_advanced.exporters.targets import ExportBudgetError
from plume_advanced.identity import sha256_file
from plume_advanced.pipeline.checkpoints import StageCheckpointStore, pipeline_fingerprint
from plume_advanced.pipeline.inspection import complete_inspection
from plume_advanced.pipeline.resolution import build_with_resolution_checks
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.surface_topology import SurfaceTopologyError
from plume_advanced.world import ExportConfig


def test_unspecified_policy_preserves_explicit_research_behavior(tmp_path):
    path = tmp_path / "research.toml"
    path.write_text('schema_version = 4\n[export]\ngenerate_collision = false\n')
    assert load_project_config(path).acceptance == AcceptancePolicy()


@pytest.fixture
def passage():
    cave = geometry(trimesh.creation.box(extents=(4, 4, 2)), smoothing=0)
    cave = replace(cave, config=replace(cave.config, required_route_height_m=.5,
        required_route_width_m=.5, route_clearance_margin_m=.02),
        required_route_paths=(((-1., 0., 0.), (1., 0., 0.)),
                              ((-1., .5, 0.), (1., .5, 0.))), route_path_segment_ids=(0, 1))
    export = ExportConfig(target="blender", generate_collision=True,
                          max_visual_triangles=10000, max_asset_bytes=1024 * 1024)
    return cave, export


@pytest.mark.parametrize('part', ['raw', 'visual', 'collision'])
def test_ground_contract_propagates_to_actual_export_and_rejects_missing_evidence(passage, tmp_path, part):
    cave, export = passage
    export = replace(export, max_asset_bytes=4*1024*1024)
    cave = replace(cave, config=replace(cave.config, ground_robot_length_m=.7))
    policy = build_acceptance_policy({'profile': 'inspection', 'require_ground_routes': True})
    result = export_target_asset(cave, export, tmp_path/'out', acceptance=policy)
    path = result.primary_asset.parent/'pipeline_inspection.json'
    report = json.loads(path.read_text())
    assert report['acceptance']['checks']['ground_routes']['status'] == 'passed'
    for name in ('raw', 'visual', 'collision'):
        target = report[name] if name != 'collision' else report[name]['inspection']
        assert target['ground_traversal']['passed']
    target = report[part] if part != 'collision' else report[part]['inspection']
    del target['ground_traversal']
    path.write_text(json.dumps(report))
    with pytest.raises(AcceptanceError, match='ground_routes: failed'):
        complete_inspection(cave, result, {}, tmp_path, acceptance=policy, export_config=export)
    assert not (tmp_path/'pipeline_quality_report.json').exists()


@pytest.mark.parametrize("profile,required", [
    ("research", set()),
    ("inspection", {"clearance", "collision", "export_budgets"}),
    ("simulation", {"clearance", "collision", "export_budgets", "resolution"}),
])
def test_profile_resolves_and_fills_only_absent_controls(profile, required):
    policy = build_acceptance_policy({"profile": profile})
    assert {key.removeprefix("require_") for key, value in asdict(policy).items()
            if key.startswith("require_") and value} == required
    source_geometry, source_export = {"voxel_size": .2}, {"target": "blender"}
    geo, export = apply_acceptance_defaults(policy, source_geometry, source_export)
    assert source_geometry == {"voxel_size": .2} and source_export == {"target": "blender"}
    if required:
        assert geo["required_route_height_m"] == geo["required_route_width_m"] == .5
        assert export["generate_collision"] and export["max_visual_triangles"] == 2_000_000
        assert export["max_asset_bytes"] == 256 * 1024 * 1024
    if profile == "simulation":
        assert policy.minimum_relief_scale == 1 and geo["resolution_refinement_attempts"] == 2
        assert export["visual_max_error_m"] == .01
    explicit, _ = apply_acceptance_defaults(policy, {"required_route_height_m": 0}, {})
    assert explicit["required_route_height_m"] == 0


@pytest.mark.parametrize("raw", [
    {"profile": "unknown"}, {"profile": []}, {"requirements": True}, [],
    {"require_native": "true"}, {"require_collision": 1},
    {"profile": "inspection", "require_clearance": False},
    {"profile": "inspection", "require_collision": False},
    {"profile": "inspection", "require_export_budgets": False},
    {"profile": "simulation", "require_resolution": False},
    {"profile": "simulation", "minimum_relief_scale": 0},
    {"route_height_m": float("nan")}, {"route_margin_m": float("inf")},
    {"route_margin_m": -1}, {"route_height_m": True}, {"route_width_m": 0},
    {"route_width_m": 2}, {"minimum_relief_scale": 1.01},
])
def test_invalid_or_downgraded_policy_is_rejected(raw):
    with pytest.raises(ValueError):
        build_acceptance_policy(raw)


@pytest.mark.parametrize("text,match", [
    ('[geometry]\nrequired_route_height_m = 0.0', 'required route height'),
    ('[geometry]\nrequired_route_width_m = 0.4', 'required_route_width_m'),
    ('[geometry]\nroute_clearance_margin_m = 0.0', 'route_clearance_margin_m'),
    ('[export]\ngenerate_collision = false', 'generate_collision'),
    ('[export]\nmax_visual_triangles = 0', 'max_visual_triangles'),
    ('[export]\nmax_asset_bytes = 0', 'max_asset_bytes'),
])
def test_explicit_contradiction_is_not_silently_overridden(tmp_path, text, match):
    path = tmp_path / "case.toml"
    path.write_text('schema_version = 4\n[acceptance]\nprofile = "inspection"\n' + text + '\n')
    with pytest.raises(ValueError, match=match):
        load_project_config(path)


def test_config_manifest_and_checkpoints_include_policy_even_without_geometry_changes(tmp_path):
    path = tmp_path / "case.toml"
    path.write_text('schema_version = 4\n[acceptance]\nprofile = "inspection"\n')
    project = load_project_config(path)
    assert project_config_manifest(project)["acceptance"] == asdict(project.acceptance)
    modified = replace(project, acceptance=replace(project.acceptance, minimum_relief_scale=.5))
    first, second = [pipeline_fingerprint(p, inputs=(), source_root=tmp_path)
                     for p in (project, modified)]
    assert first != second
    root = tmp_path / "checkpoints"
    StageCheckpointStore(root, first).save("geometry", "old evidence")
    assert StageCheckpointStore(root, second).load("geometry") is None


def test_inspection_exports_actual_passage_and_collider_identically_on_cold_replay(passage, tmp_path):
    cave, export = passage
    policy = build_acceptance_policy({"profile": "inspection"})
    results = [export_target_asset(cave, export, tmp_path / name, acceptance=policy)
               for name in ("first", "second")]
    assert sha256_file(results[0].primary_asset) == sha256_file(results[1].primary_asset)
    reports = [json.loads((r.primary_asset.parent / "pipeline_inspection.json").read_text())
               for r in results]
    assert reports[0] == reports[1]
    checks = reports[0]["acceptance"]["checks"]
    assert all(checks[key]["status"] == "passed" for key in
               ("mesh", "clearance", "collision", "texture_integrity", "export_budgets"))
    assert checks["native"]["status"] == checks["resolution"]["status"] == "not_requested"
    assert reports[0]["collision"]["inspection"]["traversal"]["paths"][0]["samples"] == 2
    quality, figure = complete_inspection(cave, results[0],
        {"section_count": 1, "under_resolved_count": 0, "minimum_samples": 8}, tmp_path / "completed",
        acceptance=policy, export_config=export)
    assert figure.is_file() and json.loads(quality.read_text())["acceptance"]["passed"]


def test_research_export_does_not_claim_omitted_requirements(passage, tmp_path):
    cave, export = passage
    result = export_target_asset(cave, replace(export, generate_collision=False), tmp_path / "out")
    report = json.loads((result.primary_asset.parent / "pipeline_inspection.json").read_text())
    assert report["acceptance"]["passed"]
    assert all(row["status"] == "not_requested" for key, row in
               report["acceptance"]["checks"].items() if key not in ("mesh", "texture_integrity"))


@pytest.mark.parametrize("resolution,status", [
    (None, "unavailable"), ({}, "unavailable"),
    ({"section_count": 0, "under_resolved_count": 0}, "failed"),
    ({"section_count": 1, "under_resolved_count": 2}, "failed"),
    ({"section_count": 1, "under_resolved_count": 1, "minimum_samples": 8}, "failed"),
    ({"section_count": 1, "under_resolved_count": 0}, "unavailable"),
    ({"section_count": 1, "under_resolved_count": 0, "minimum_samples": 2}, "failed"),
])
def test_resolution_gate_preserves_previous_export(passage, tmp_path, resolution, status):
    cave, export = passage
    output = tmp_path / "export"
    output.mkdir()
    (output / "previous.txt").write_text("keep")
    policy = build_acceptance_policy({"profile": "simulation"})
    with pytest.raises(AcceptanceError) as caught:
        export_target_asset(cave, export, output, acceptance=policy, resolution=resolution)
    assert caught.value.report["checks"]["resolution"]["status"] == status
    assert [p.name for p in output.iterdir()] == ["previous.txt"]
    assert not list(tmp_path.glob(".*staging*"))


@pytest.mark.parametrize("part", ["raw", "visual", "collision"])
@pytest.mark.parametrize("damage", ["missing", "undersized", "empty", "failed", "dropped_path", "wrong_segment"])
def test_passing_summary_does_not_hide_invalid_route_evidence(passage, tmp_path, part, damage):
    cave, export = passage
    policy = build_acceptance_policy({"profile": "inspection"})
    exported = export_target_asset(cave, export, tmp_path / "out", acceptance=policy)
    path = exported.primary_asset.parent / "pipeline_inspection.json"
    report = json.loads(path.read_text())
    target = report[part] if part != "collision" else report[part]["inspection"]
    route = target["traversal"]
    if damage == "missing":
        del target["traversal"]
    elif damage == "undersized":
        route["width_m"] = .4
    elif damage == "empty":
        route["paths"] = []
    elif damage == "dropped_path":
        route["paths"].pop()
    elif damage == "wrong_segment":
        route["paths"][0]["segment_id"] = 99
    else:
        route["paths"][0]["passed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(AcceptanceError, match="clearance: failed"):
        complete_inspection(cave, exported, {}, tmp_path, acceptance=policy, export_config=export)
    assert not (tmp_path / "pipeline_quality_report.json").exists()


@pytest.mark.parametrize("damage", ["missing_policy", "changed_policy", "changed_resolution"])
def test_completion_rejects_stale_acceptance_evidence(passage, tmp_path, damage):
    cave, export = passage
    policy = build_acceptance_policy({"profile": "simulation"})
    resolution = {"section_count": 1, "under_resolved_count": 0, "minimum_samples": 8}
    exported = export_target_asset(cave, export, tmp_path / "out",
                                   acceptance=policy, resolution=resolution)
    path = exported.primary_asset.parent / "pipeline_inspection.json"
    if damage == "missing_policy":
        report = json.loads(path.read_text())
        del report["acceptance"]
        path.write_text(json.dumps(report))
    elif damage == "changed_policy":
        policy = replace(policy, minimum_relief_scale=.5)
    else:
        resolution["section_count"] = 2
    with pytest.raises(ValueError, match="Acceptance policy/evidence changed"):
        complete_inspection(cave, exported, resolution, tmp_path,
                            acceptance=policy, export_config=export)


@pytest.mark.parametrize("global_scale,local_scale", [(0., 1.), (.5, 1.), (1., .25), (.75, .75)])
def test_relief_budget_checks_global_and_local_factors(passage, global_scale, local_scale):
    cave, export = passage
    cave = replace(cave, config=replace(cave.config, surface_wall_relief_m=.1),
        effective_surface_relief_scale=global_scale,
        effective_local_relief_regions=(tuple({"scale": local_scale}.items()),))
    policy = build_acceptance_policy({"minimum_relief_scale": .6})
    report = evaluate_acceptance(policy, cave, {}, None, export)
    assert report["checks"]["relief"]["status"] == "failed"
    assert report["checks"]["relief"]["effective_minimum_scale"] == global_scale * local_scale
    with pytest.raises(FrozenInstanceError):
        policy.minimum_relief_scale = 0


def test_real_branch_cannot_erase_relief_to_satisfy_strict_profile(monkeypatch):
    base = branch_fixture()
    original = base.voxel_grid.density.copy()
    generator = GeometryGenerator(base.config,
        acceptance=build_acceptance_policy({"profile": "simulation"}))
    monkeypatch.setattr(generator, "_enforce_roof_stability", lambda *args: ())
    with pytest.raises(SurfaceTopologyError, match="No surface candidate") as caught:
        generator._accept_base_surface(base, None, [], None)
    assert any("reduction budget" in r.get("reason", "") for r in caught.value.report["attempts"])
    np.testing.assert_array_equal(base.voxel_grid.density, original)


@pytest.mark.integration
def test_real_refinement_satisfies_simulation_profile_and_completion(tmp_path):
    network, sections, config = real_short_inputs()
    policy = build_acceptance_policy({"profile": "simulation"})
    config = replace(config, cave_smoothing_iterations=0, cave_displacement_scale_m=0)
    cave = build_with_resolution_checks(network, sections, config, acceptance=policy)
    resolution = section_resolution_report(sections, config.voxel_size)
    assert resolution["under_resolved_count"] > 0 and cave.config.voxel_size < config.voxel_size
    export = ExportConfig(target="blender", generate_collision=True,
                          max_visual_triangles=2_000_000, max_asset_bytes=256 * 1024 * 1024)
    result = export_target_asset(cave, export, tmp_path / "out",
                                 acceptance=policy, resolution=resolution)
    quality, _ = complete_inspection(cave, result, resolution, tmp_path,
                                     acceptance=policy, export_config=export)
    report = json.loads(quality.read_text())
    assert report["acceptance"]["passed"]
    assert report["acceptance"]["checks"]["resolution"]["status"] == "passed"
    assert report["resolution_repair"]["outcome"] == "input_sampling_sufficient"
    assert report["resolution_repair"]["attempts"][-1]["under_resolved"] == 0


@pytest.mark.parametrize("change", [dict(under_resolved=1), dict(resolution_section_count=2),
                                  dict(minimum_samples=4), dict(voxel_size_m=.5)])
def test_refinement_screen_must_match_current_geometry_and_all_profiles(passage, change):
    from plume_advanced.acceptance import _resolution_check
    cave, _ = passage
    row = dict(accepted=True, under_resolved=0, resolution_section_count=1,
               minimum_samples=8, voxel_size_m=cave.config.voxel_size)
    journal = dict(passed=True, outcome="input_sampling_sufficient",
                   effective_voxel_size_m=cave.config.voxel_size, attempts=[row | change])
    candidate = replace(cave, resolution_repair=tuple(journal.items()))
    passed, _ = _resolution_check(dict(section_count=1, under_resolved_count=1,
                                       minimum_samples=8), candidate)
    assert not passed


def test_missing_required_textures_fail_before_export(passage, tmp_path):
    cave, export = passage
    with pytest.raises(ValueError, match="diffuse, normal and roughness"):
        export_target_asset(cave, export, tmp_path / "out",
                             acceptance=build_acceptance_policy({"require_textures": True}))
    assert not (tmp_path / "out").exists()


def test_required_pbr_maps_are_inspected_in_real_package(passage, tmp_path):
    from PIL import Image

    cave, export = passage
    maps = {}
    for role, color in (("diffuse", (100, 90, 80)), ("normal", (128, 128, 255)),
                        ("roughness", (200, 200, 200))):
        path = tmp_path / f"{role}.png"
        Image.new("RGB", (16, 16), color).save(path)
        maps[f"cave_{role}_texture"] = str(path)
    cave = replace(cave, config=replace(cave.config, **maps))
    policy = build_acceptance_policy({"profile": "inspection", "require_textures": True})
    result = export_target_asset(cave, export, tmp_path / "out", acceptance=policy)
    report = json.loads((result.primary_asset.parent / "pipeline_inspection.json").read_text())
    assert report["acceptance"]["checks"]["pbr_textures"]["status"] == "passed"
    assert report["textures"]["package_attempts"][-1]["checked_files"]


def test_tube_only_helper_rejects_full_policy_before_creating_outputs(tmp_path, monkeypatch):
    import runpy
    import sys

    helper = Path(__file__).resolve().parents[1] / "scripts/generate_tube_only.py"
    monkeypatch.setattr(sys, "argv", [str(helper), "--config", str(cli.PACKAGED_CONFIG),
                                    "--output-directory", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(helper), run_name="__main__")
    assert caught.value.code == 2 and not (tmp_path / "out").exists()


@pytest.mark.integration
def test_full_campaign_worker_uses_same_simulation_contract(tmp_path, monkeypatch):
    from plume_advanced import config as config_module
    from plume_advanced.evaluation.reliability import execute_case
    from plume_advanced.stages.host_field import GridConfig
    from plume_advanced.stages.network import CaveNetworkGenerator

    network, sections, geometry_config = real_short_inputs()
    project = load_project_config(cli.PACKAGED_CONFIG)
    project = replace(project, network=network.config, section_field=sections.config,
        host_field=replace(project.host_field, grid=GridConfig(width=200, height=200, nx=24, ny=24)),
        geometry=replace(geometry_config, cave_smoothing_iterations=0, cave_displacement_scale_m=0),
        acceptance=build_acceptance_policy({"profile": "simulation"}))
    # Fixed proposal; production network screening, meshing, recovery, export and
    # completion still execute. This tests wiring, not stochastic coverage.
    monkeypatch.setattr(config_module, "load_project_config", lambda *a, **kw: project)
    monkeypatch.setattr(CaveNetworkGenerator, "_generate_candidate", lambda *a, **kw: network)
    result = execute_case(ReliabilityCase(str(cli.PACKAGED_CONFIG), 3, scope="full"), tmp_path)
    assert result["status"] == "passed" and result["acceptance"]["passed"]
    assert result["acceptance"]["policy"]["profile"] == "simulation"
    assert result["acceptance"]["checks"]["resolution"]["status"] == "passed"
    assert json.loads((tmp_path / "pipeline_quality_report.json").read_text())["acceptance"] == result["acceptance"]


@pytest.mark.parametrize("limit", ["max_visual_triangles", "max_asset_bytes"])
def test_actual_budget_overrun_never_publishes(passage, tmp_path, limit):
    cave, export = passage
    export = replace(export, **{limit: 1})
    with pytest.raises(ExportBudgetError):
        export_target_asset(cave, export, tmp_path / "out",
                             acceptance=build_acceptance_policy({"profile": "inspection"}))
    assert not (tmp_path / "out").exists()


def test_native_requirement_fails_preflight_and_cli_without_starting_generation(tmp_path, monkeypatch):
    config = tmp_path / "case.toml"
    config.write_text(cli.PACKAGED_CONFIG.read_text()
                      + '\n[acceptance]\nprofile = "simulation"\nrequire_native = true\n')
    report = preflight([ReliabilityCase(str(config), 42, scope="full")])
    assert not report["passed"]
    assert report["cases"][0]["diagnostic"]["category"] == "acceptance_requirements"
    assert report["cases"][0]["inspection"]["checks"]["native"]["status"] == "unavailable"
    monkeypatch.setattr(cli.HostFieldGenerator, "generate",
        lambda *a, **kw: pytest.fail("Native requirement must fail before allocating a host"))
    output = tmp_path / "out" / "network.png"
    with pytest.raises(AcceptanceError, match="native: unavailable"):
        cli.main(["--config", str(config), "--output", str(output)])
    failure = json.loads((output.parent / "pipeline_quality_report.json").read_text())
    assert not failure["passed"] and failure["inspection"]["phase"] == "preflight"
    assert not list(output.parent.rglob("*.glb"))
    stage_only = preflight([ReliabilityCase(str(config), 42, scope="network")])
    assert stage_only["passed"] and "not evaluated" in stage_only["cases"][0]["warnings"][0]


def test_native_preflight_failure_respects_output_overwrite_refusal(tmp_path, monkeypatch):
    project = load_project_config(cli.PACKAGED_CONFIG)
    project = replace(project, acceptance=replace(project.acceptance, require_native=True))
    monkeypatch.setattr(cli, "load_project_config", lambda *a, **kw: project)
    output = tmp_path / "network.png"
    previous = tmp_path / "run_manifest.json"
    previous.write_text('{"status": "complete"}\n')

    def refuse(*a, **kw):
        raise cli.OutputOverwriteRefused("Keep existing results")

    monkeypatch.setattr(cli, "require_output_overwrite_confirmation", refuse)
    assert cli.main(["--output", str(output)]) == 2
    assert previous.read_text() == '{"status": "complete"}\n'
    assert list(tmp_path.iterdir()) == [previous]


def test_packaged_and_comparison_presets_declare_intended_profile():
    inspection = {"project", "short-single", "long-single", "short-multi", "long-multi",
                  "gallery-long", "simulator-check"}
    simulation = {"simulation-single", "simulation-multi"}
    for path in sorted((Path(__file__).resolve().parents[1] / "config").glob("*.toml")):
        project = load_project_config(path)
        expected = ("simulation" if path.stem in simulation
                    else "inspection" if path.stem in inspection else "research")
        assert project.acceptance.profile == expected, path
        validate_acceptance_configuration(project.acceptance, project.geometry, project.export)
    default = load_project_config(cli.PACKAGED_CONFIG)
    assert default.acceptance.profile == "inspection" and default.export.generate_collision


@pytest.mark.parametrize('failure', [KeyboardInterrupt, EOFError, OSError])
def test_cancel_or_prompt_failure_preserves_previous_run(tmp_path, monkeypatch, failure):
    previous = tmp_path / 'run_manifest.json'
    previous.write_text('{"status":"complete","current_stage":"finished","outputs":[]}\n')
    quality = tmp_path / 'pipeline_quality_report.json'
    quality.write_text('{"passed":true}\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def cancel(*args, **kwargs):
        raise failure('prompt interrupted')

    monkeypatch.setattr(cli, 'require_output_overwrite_confirmation', cancel)
    with pytest.raises(failure):
        cli.main(['--config', str(cli.PACKAGED_CONFIG), '--output', str(tmp_path / 'network.png')])
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_failure_after_generation_starts_records_only_current_run(tmp_path, monkeypatch):
    previous = tmp_path / 'run_manifest.json'
    previous.write_text('{"status":"complete","current_stage":"old-stage","outputs":[]}\n')

    def fail(*args, **kwargs):
        raise RuntimeError('new generation failed')

    monkeypatch.setattr(cli, 'write_project_config_manifest', fail)
    with pytest.raises(RuntimeError, match='new generation failed'):
        cli.main(['--config', str(cli.PACKAGED_CONFIG), '--force-overwrite',
                  '--output', str(tmp_path / 'network.png')])
    manifest = json.loads(previous.read_text())
    assert manifest['status'] == 'failed'
    assert manifest['failure']['stage'] == 'configuration'
    assert manifest['outputs'] == []
    assert (tmp_path / 'pipeline_quality_report.json').is_file()
