"""Full-seed retries must be replayable and cannot convert failures into acceptance."""

import json
from dataclasses import asdict

import pytest

from plume_advanced import cli
from plume_advanced.config import load_project_config
from plume_advanced.pipeline.recovery import PipelineRecoveryError
from plume_advanced.pipeline.seed_search import SeedSearch, next_seed, retryable_rejection


def search(path, **overrides):
    return SeedSearch(path, **(dict(root_seed=3, fingerprint="source-and-inputs", required=True,
                                    max_attempts=3) | overrides))


def test_journal_records_before_work_and_resumes_without_skipping(tmp_path):
    path = tmp_path / "seeds.json"
    first = search(path)
    assert not path.exists()
    assert first.seed == 3 and first.attempt == 1
    first.begin({"host": 12})
    assert json.loads(path.read_text())["attempts"][0]["status"] == "running"
    resumed = search(path, resume=True)
    assert resumed.seed == 3
    resumed.begin({"host": 12})
    resumed.finish("rejected", stage="network", diagnosis=["bad branch"])
    second = search(path, resume=True)
    assert second.seed == next_seed(3, [3]) != 3
    seed = second.seed
    second.begin({"host": 13})
    second.finish("interrupted")
    resumed = search(path, resume=True)
    assert resumed.seed == seed and resumed.attempt == 2
    resumed.begin({"host": 13})
    resumed.finish("accepted", robot_qualification={"qualified": True})
    saved = json.loads(path.read_text())
    assert [r["status"] for r in saved["attempts"]] == ["rejected", "accepted"]
    assert saved["accepted_seed"] == seed
    # Revalidation of a completed run starts with its winner, not another seed.
    assert search(path, resume=True).seed == seed


@pytest.mark.parametrize("changed", [dict(root_seed=4), dict(fingerprint="changed"), dict(required=False),
                                      dict(max_attempts=4)])
def test_resume_rejects_identity_mismatch(tmp_path, changed):
    path = tmp_path / "seeds.json"
    original = search(path)
    original.begin({})
    with pytest.raises(ValueError, match="does not match"):
        search(path, resume=True, **changed)


def test_tampered_seed_order_rejected(tmp_path):
    path = tmp_path / "seeds.json"
    original = search(path)
    original.begin({})
    data = json.loads(path.read_text())
    data["attempts"][0]["seed"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid deterministic"):
        search(path, resume=True)


def test_seed_budget_and_unlimited_search(tmp_path):
    limited = search(tmp_path / "limited.json", max_attempts=1)
    limited.begin({})
    limited.finish("rejected")
    assert limited.seed is None
    unlimited = search(tmp_path / "unlimited.json", max_attempts=0)
    seeds = []
    for _ in range(20):
        seeds.append(unlimited.seed)
        unlimited.begin({})
        unlimited.finish("rejected")
    replay = []
    for _ in range(20):
        replay.append(next_seed(3, replay))
    assert seeds == replay and len(set(seeds)) == 20
    assert unlimited.seed is not None


@pytest.mark.parametrize("error", [RuntimeError("bug"), OSError("disk full"), MemoryError(),
                                   ValueError("missing texture"), KeyboardInterrupt()])
def test_operational_errors_never_trigger_seed_retry(error):
    assert not retryable_rejection(error)


def test_nested_resource_exhaustion_stops_search():
    error = PipelineRecoveryError("exhausted", report={"attempts": [{"error_type": "ResolutionBudgetError"}]})
    assert not retryable_rejection(error)


def test_export_budget_and_time_budget_never_change_seed():
    from plume_advanced.exporters.errors import ExportBudgetError
    from plume_advanced.progress import GenerationTimeBudgetError
    assert not retryable_rejection(ExportBudgetError('too many triangles'))
    assert not retryable_rejection(GenerationTimeBudgetError('deadline'))


@pytest.mark.parametrize('value', [-1, True, float('inf'), float('nan'), 'slow'])
def test_invalid_attempt_time_budget(value):
    from plume_advanced.world import build_run_config
    with pytest.raises(ValueError, match='max_attempt_seconds'):
        build_run_config(dict(max_attempt_seconds=value))


def test_seed_search_is_bounded_but_time_limit_is_opt_in():
    from plume_advanced.world import build_run_config
    config = build_run_config({})
    assert config.max_seed_attempts == 8 and config.max_attempt_seconds == 0
    assert build_run_config(dict(max_attempt_seconds=1800)).max_attempt_seconds == 1800


@pytest.mark.parametrize("qualified", [False, True])
def test_cli_search_reloads_all_seeded_inputs_after_rejection(tmp_path, monkeypatch, qualified):
    cfg = tmp_path / "recipe.toml"
    cfg.write_text('recipe_version = 1\npreset = "preview"\nprocedural_seed = 13\n'
                   '[run]\nmax_seed_attempts = 3\n[acceptance]\nrequire_ground_routes = '
                   + str(qualified).lower() + '\n')
    output = tmp_path / "out/network.png"
    seen = []

    def candidate(argv, *, state):
        project = state.project
        seen.append(project)
        state.started, state.stage = True, "base_geometry"
        state.search.begin(asdict(project.stage_seeds))
        if len(seen) == 1:
            assert not state.overwrite_authorized
            raise PipelineRecoveryError("candidate defect", report={"attempts": []})
        assert state.overwrite_authorized
        state.search.finish("accepted")
        return 0

    monkeypatch.setattr(cli, "_run_pipeline", candidate)
    assert cli.main(["--config", str(cfg), "--output", str(output)]) == 0
    assert len(seen) == 2 and seen[0].procedural_seed == 13
    assert seen[0].stage_seeds.host != seen[1].stage_seeds.host
    assert seen[0].stage_seeds.network != seen[1].stage_seeds.network
    assert seen[0].acceptance == seen[1].acceptance
    assert seen[1] == load_project_config(cfg, seed_override=seen[1].procedural_seed)
    journal = json.loads(output.with_name("seed_attempts.json").read_text())
    assert journal["accepted_seed"] == seen[1].procedural_seed
    assert journal["robot_qualification_required"] is qualified
    assert [r["status"] for r in journal["attempts"]] == ["rejected", "accepted"]


@pytest.mark.parametrize("preset", ["preview", "simulation-single", "simulation-multi"])
@pytest.mark.parametrize("flag", [None, False, True])
def test_robot_qualification_only_enabled_explicitly(tmp_path, preset, flag):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'recipe_version = 1\npreset = "{preset}"\n'
                   + (f'[acceptance]\nrequire_ground_routes = {str(flag).lower()}\n' if flag is not None else ""))
    config = load_project_config(cfg)
    assert config.acceptance.require_ground_routes is (flag is True)
    assert bool(config.geometry.ground_robot_length_m) is (flag is True)


def test_old_geometry_robot_length_cannot_enable_qualification(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('recipe_version = 1\npreset = "preview"\n[geometry]\nground_robot_length_m = 0.7\n')
    assert load_project_config(cfg).geometry.ground_robot_length_m == 0


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_bad_seed_budget_is_configuration_error(tmp_path, limit):
    cfg = tmp_path / "config.toml"
    cfg.write_text('recipe_version = 1\npreset = "preview"\n[run]\nmax_seed_attempts = ' + str(limit).lower())
    with pytest.raises(ValueError, match="max_seed_attempts"):
        load_project_config(cfg)


def test_cli_seed_override_can_replay_winner(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('recipe_version = 1\npreset = "preview"\nprocedural_seed = 2\n')
    assert cli.main(["--config", str(cfg), "--seed", "999", "--show-config"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["procedural_seed"] == 999
    assert manifest["stage_seeds"] == asdict(load_project_config(cfg, seed_override=999).stage_seeds)


@pytest.mark.parametrize("error_type,status", [(KeyboardInterrupt, "interrupted"), (OSError, "failed"),
                                              (MemoryError, "failed"), (RuntimeError, "failed")])
def test_cli_does_not_retry_operational_failure_and_preserves_resume_seed(tmp_path, monkeypatch, error_type, status):
    cfg = tmp_path / "config.toml"
    cfg.write_text('recipe_version = 1\npreset = "preview"\nprocedural_seed = 29\n')
    output = tmp_path / "out/network.png"
    calls = []

    def fail(generator):
        calls.append(generator)
        raise error_type("stop here")

    monkeypatch.setattr(cli.HostFieldGenerator, "generate", fail)
    with pytest.raises(error_type):
        cli.main(["--config", str(cfg), "--output", str(output)])
    saved = json.loads(output.with_name("seed_attempts.json").read_text())
    assert len(calls) == len(saved["attempts"]) == 1
    assert saved["attempts"][0]["seed"] == 29
    assert saved["status"] == status and saved["accepted_seed"] is None
    monkeypatch.setattr(cli.HostFieldGenerator, "generate", fail)
    with pytest.raises(error_type):
        cli.main(["--config", str(cfg), "--output", str(output), "--resume"])
    resumed = json.loads(output.with_name("seed_attempts.json").read_text())
    assert len(resumed["attempts"]) == 1 and resumed["attempts"][0]["seed"] == 29


@pytest.mark.parametrize('required,expected_attempts', [(False, 1), (True, 2)])
def test_seed_search_with_real_ground_checks_and_atomic_export(tmp_path, monkeypatch, required, expected_attempts):
    """Controlled steep/flat meshes exercise the real qualification gate, not a passing mock."""
    from dataclasses import replace

    import numpy as np
    import trimesh
    from test_embedded_inspection import geometry

    from plume_advanced.exporters import export_target_asset
    from plume_advanced.world import ExportConfig

    cfg = tmp_path / 'recipe.toml'
    cfg.write_text('recipe_version = 1\npreset = "preview"\nprocedural_seed = 13\n'
                   '[run]\nmax_seed_attempts = 2\n[acceptance]\nrequire_ground_routes = '
                   + str(required).lower() + '\n')
    output = tmp_path / 'out/network.png'
    attempts = []

    def candidate(argv, *, state):
        project = state.project
        attempts.append(project.procedural_seed)
        state.started, state.stage = True, 'export'
        state.search.begin(asdict(project.stage_seeds))
        mesh = trimesh.creation.box(extents=(4, 4, 2))
        if len(attempts) == 1:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [0, 1, 0]))
        cave = geometry(mesh, smoothing=0)
        cave = replace(cave, config=replace(cave.config, required_route_height_m=.5,
            required_route_width_m=.5, route_clearance_margin_m=.02,
            ground_robot_length_m=project.geometry.ground_robot_length_m),
            required_route_paths=(((-1., 0., 0.), (1., 0., 0.)),), route_path_segment_ids=(0,))
        export = ExportConfig(target='blender', generate_collision=True,
                              max_visual_triangles=10000, max_asset_bytes=4*1024*1024)
        result = export_target_asset(cave, export, output.parent / 'export_blender', acceptance=project.acceptance)
        label = json.loads((result.primary_asset.parent / 'robot_qualification.json').read_text())
        assert label['qualified'] is required
        state.search.finish('accepted', robot_qualification=label)
        return 0

    monkeypatch.setattr(cli, '_run_pipeline', candidate)
    assert cli.main(['--config', str(cfg), '--output', str(output)]) == 0
    assert len(attempts) == expected_attempts
    history = json.loads(output.with_name('seed_attempts.json').read_text())
    assert history['accepted_seed'] == attempts[-1]
    assert history['attempts'][-1]['robot_qualification']['qualified'] is required
    if required:
        assert history['attempts'][0]['status'] == 'rejected'
        assert attempts[0] != attempts[1]


def test_changed_recipe_between_retries_is_not_silently_accepted(tmp_path, monkeypatch):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('recipe_version = 1\npreset = "preview"\nprocedural_seed = 3\n')
    output = tmp_path / 'out/network.png'
    seen = []

    def reject(argv, *, state):
        seen.append(state.project)
        state.started = True
        state.search.begin(asdict(state.project.stage_seeds))
        cfg.write_text(cfg.read_text().replace('procedural_seed = 3', 'procedural_seed = 4'))
        raise PipelineRecoveryError('candidate defect', report={'attempts': []})

    monkeypatch.setattr(cli, '_run_pipeline', reject)
    with pytest.raises(ValueError, match='changed during seed search'):
        cli.main(['--config', str(cfg), '--output', str(output)])
    assert len(seen) == 1
    assert json.loads(output.with_name('seed_attempts.json').read_text())['accepted_seed'] is None
