"""Command-line packaging and failure-record regressions."""

import json
from pathlib import Path

import pytest

from plume_advanced import cli
from plume_advanced.config import load_project_config
from plume_advanced.pipeline import StageCheckpointStore, pipeline_fingerprint
from plume_advanced.stages.host_field import HostFieldGenerator


def test_packaged_default_config_is_available() -> None:
    assert cli.PACKAGED_CONFIG.is_file()
    config = cli.load_project_config(cli.PACKAGED_CONFIG)
    assert config.run.dev_mode
    assert not config.geometry.cave_diffuse_texture
    assert not config.events.include_rock_props
    assert not config.events.use_rocky_meshes


@pytest.mark.parametrize("debug", [False, True])
def test_deadline_has_actionable_message_and_no_seed_retry(tmp_path, monkeypatch, capsys, debug):
    from plume_advanced.progress import GenerationTimeBudgetError

    config = tmp_path / 'project.toml'
    config.write_text('recipe_version = 1\npreset = "preview"\n[acceptance]\nrequire_ground_routes = true\n')
    output = tmp_path / 'out/network.png'

    def expire(_generator):
        raise GenerationTimeBudgetError('attempt expired')

    monkeypatch.setattr(cli.HostFieldGenerator, 'generate', expire)
    arguments = ['--config', str(config), '--output', str(output)]
    if debug:
        with pytest.raises(GenerationTimeBudgetError, match='attempt expired'):
            cli.main(arguments + ['--debug'])
    else:
        assert cli.main(arguments) == 2
        error = capsys.readouterr().err
        assert 'stopped at host_field' in error
        assert 'existing files are not a validated delivery' in error
        assert 'run.max_attempt_seconds' in error
        assert 'Traceback' not in error
    report = json.loads(output.with_name('seed_attempts.json').read_text())
    assert report['status'] == 'failed' and report['accepted_seed'] is None
    assert len(report['attempts']) == 1
    assert report['attempts'][0]['error_type'] == 'GenerationTimeBudgetError'


def test_default_complete_scene_uses_portable_asset_name() -> None:
    args = cli.parse_args([])
    assert args.geometry_glb_output is None
    assert args.geometry_mesh_output is None
    expected_glb = args.output.with_name("plume_cave_scene.glb")
    expected_obj = args.output.with_name("plume_cave_scene.obj")
    assert expected_glb.name == "plume_cave_scene.glb"
    assert expected_obj.name == "plume_cave_scene.obj"


def test_resume_cli_uses_default_or_explicit_checkpoint_directory(tmp_path: Path) -> None:
    default_args = cli.parse_args(["--output", str(tmp_path / "outputs" / "network.png")])
    assert not default_args.resume
    assert default_args.checkpoint_directory is None

    checkpoint_directory = tmp_path / "cache"
    resumed = cli.parse_args(
        [
            "--resume",
            "--checkpoint-directory",
            str(checkpoint_directory),
        ]
    )
    assert resumed.resume
    assert resumed.checkpoint_directory == checkpoint_directory


def test_pipeline_failure_writes_failed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        """
schema_version = 4
procedural_seed = 7

[run]
dev_mode = true
render_diagnostics = false

[geometry]
cave_diffuse_texture = ""
cave_normal_texture = ""
cave_roughness_texture = ""
cave_displacement_texture = ""
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "network.png"

    def fail_host_generation(_generator):
        raise RuntimeError("synthetic host failure")

    monkeypatch.setattr(
        cli.HostFieldGenerator,
        "generate",
        fail_host_generation,
    )

    with pytest.raises(RuntimeError, match="synthetic host failure"):
        cli.main(
            [
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--force-overwrite",
            ]
        )

    payload = json.loads(output.with_name("run_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure"]["stage"] == "host_field"
    assert "synthetic host failure" in payload["failure"]["error"]
    quality = json.loads(output.with_name("pipeline_quality_report.json").read_text())
    assert not quality["passed"]
    assert quality["stage"] == "host_field"
    assert quality["error_type"] == "RuntimeError"
    assert "synthetic host failure" in quality["error"]


@pytest.mark.parametrize('debug', [False, True])
def test_expected_recovery_rejection_is_readable_and_still_fails(tmp_path, monkeypatch, capsys, debug):
    from plume_advanced.pipeline.recovery import PipelineRecoveryError

    output = tmp_path / 'network.png'
    config_path = tmp_path / 'bounded.toml'
    config_path.write_text('recipe_version = 1\npreset = "preview"\n[run]\nmax_seed_attempts = 1\n')
    project = load_project_config(config_path)
    report = dict(root_seed=1, stop_reason='Reference ground-route placement exhausted', attempts=[
        dict(status='rejected', reason='Reference robot limits failed', inspection={'ground_traversal': {
            'passed': False, 'robot': {'max_slope_deg': 20., 'max_step_m': .1},
            'paths': [{'passed': False, 'poses': [{'slope_deg': 40.569, 'step_m': .254598}]}],
        }}),
    ])

    def reject(argv, *, state):
        from dataclasses import asdict
        state.search.begin(asdict(project.stage_seeds))
        state.started, state.project, state.stage = True, project, 'base_geometry'
        raise PipelineRecoveryError('Reference ground-route placement exhausted', report=report)

    monkeypatch.setattr(cli, '_run_pipeline', reject)
    args = ['--config', str(config_path), '--output', str(output), *(['--debug'] if debug else [])]
    if debug:
        with pytest.raises(PipelineRecoveryError):
            cli.main(args)
    else:
        assert cli.main(args) == 2
        message = capsys.readouterr().err
        assert 'Generation rejected at base_geometry' in message
        assert '1/1 inspected routes rejected' in message
        assert '40.6 deg; limit 20 deg' in message
        assert str(output.with_name('pipeline_quality_report.json')) in message
        assert 'Traceback' not in message
    assert json.loads(output.with_name('run_manifest.json').read_text())['status'] == 'failed'
    assert json.loads(output.with_name('pipeline_quality_report.json').read_text())['inspection'] == report
    assert not list(tmp_path.rglob('*.glb'))


def test_pipeline_resume_reuses_stage_before_continuing_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        """
schema_version = 4
procedural_seed = 11

[run]
dev_mode = true
render_diagnostics = false

[events]
enabled = false
include_rock_props = false
use_rocky_meshes = false

[geometry]
cave_diffuse_texture = ""
cave_normal_texture = ""
cave_roughness_texture = ""
cave_displacement_texture = ""
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "network.png"
    project_config = load_project_config(config_path)
    store = StageCheckpointStore(
        output.parent / ".plume-checkpoints",
        pipeline_fingerprint(
            project_config,
            inputs=cli._run_inputs(config_path, project_config),
            source_root=cli.SOURCE_ROOT,
        ),
    )
    store.save("host_field", HostFieldGenerator(project_config.host_field).generate())

    def fail_if_host_rebuilt(_generator):
        raise AssertionError("validated host checkpoint was not reused")

    def stop_after_host_resume(_generator, _host_field, **_quality_options):
        raise RuntimeError("stop after resumed host")

    monkeypatch.setattr(cli.HostFieldGenerator, "generate", fail_if_host_rebuilt)
    monkeypatch.setattr(cli.CaveNetworkGenerator, "generate", stop_after_host_resume)

    with pytest.raises(RuntimeError, match="stop after resumed host"):
        cli.main(
            [
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--resume",
            ]
        )

    payload = json.loads(output.with_name("run_manifest.json").read_text(encoding="utf-8"))
    assert payload["failure"]["stage"] == "network"


@pytest.mark.integration
def test_complete_cli_finishes_progress_and_records_export_provenance(tmp_path: Path) -> None:
    """Exercise orchestration through finalization, including the real exporters."""
    from plume_advanced.identity import sha256_file
    from plume_advanced.progress import TerminalProgress

    output = tmp_path / "run" / "network.png"
    assert cli.main(["--config", str(cli.PACKAGED_CONFIG), "--output", str(output)]) == 0
    assert TerminalProgress._current is None
    rows = [
        json.loads(line) for line in output.with_name("progress.jsonl").read_text().splitlines()
    ]
    started = [row["stage"] for row in rows if row["event"] == "stage_start"]
    finished = [row["stage"] for row in rows if row["event"] == "stage_finish"]
    assert len(started) == 12 and started == finished
    assert finished[-1] == "Finalize run"
    assert "Pipeline acceptance" in finished
    quality = json.loads(output.with_name("pipeline_quality_report.json").read_text())
    assert quality["passed"] and quality["export_inspection"]["serialized"]["passed"]
    assert output.with_name("pipeline_inspection.png").is_file()
    work = {row["step"] for row in rows if row["event"] == "work"}
    assert {"Host fields", "Cross sections", "UV charts", "Tangent triangles"} <= work
    payload = json.loads(output.with_name("run_manifest.json").read_text())
    assert payload["status"] == payload["current_stage"] == "complete"
    assert payload["robot_qualification"] == quality["robot_qualification"]
    assert not payload["robot_qualification"]["qualified"]
    history = json.loads(output.with_name("seed_attempts.json").read_text())
    assert history["accepted_seed"] == payload["resolved_config"]["procedural_seed"]
    assert history["attempts"][-1]["status"] == "accepted"
    records = payload["outputs"]
    names = {Path(record["path"]).name for record in records}
    assert {
        "network_quality_report.json",
        "export_size_report.json",
        "plume_cave_scene.glb",
        "pipeline_quality_report.json",
        "pipeline_inspection.png",
        "section_resolution_report.json",
    } <= names
    for record in records:
        assert sha256_file(output.parent / record["path"]) == record["sha256"]
    # Maps are part of the delivered package and its provenance, not loose diagnostics.
    mapped = output.parent / "export_neutral/traversability"
    map_manifest = json.loads((mapped / "manifest.json").read_text())
    assert len(map_manifest["charts"]) == 1
    assert map_manifest["surface"]["kind"] == "collision"
    assert {"layer_0.npz", "layer_0.png", "layer_0_occupancy.png"} <= names
    import numpy as np

    from plume_advanced.traversability.cli import main as map_saved
    assert map_saved(["--source", str(output.parent)]) == 0
    with np.load(mapped / "layer_0.npz") as inline, np.load(output.parent / "traversability/layer_0.npz") as saved:
        np.testing.assert_array_equal(inline["status"], saved["status"])
        np.testing.assert_allclose(inline["floor_z_m"], saved["floor_z_m"], atol=1e-5)
    # Backfilling never rewrites source data. A modified input cannot claim the run's identity.
    assert all(sha256_file(output.parent / row["path"]) == row["sha256"] for row in records)
    sections = output.with_name("stage_c_sections.npz")
    sections.write_bytes(sections.read_bytes() + b"modified")
    with pytest.raises(SystemExit) as caught:
        map_saved(["--source", str(output.parent)])
    assert caught.value.code == 2
    assert (output.parent / "traversability/manifest.json").exists()
