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
schema_version = 2
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

    payload = json.loads(
        output.with_name("run_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["stage"] == "host_field"
    assert "synthetic host failure" in payload["failure"]["error"]


def test_pipeline_resume_reuses_stage_before_continuing_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        """
schema_version = 3
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

    def stop_after_host_resume(_generator, _host_field):
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

    payload = json.loads(
        output.with_name("run_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["failure"]["stage"] == "network"
