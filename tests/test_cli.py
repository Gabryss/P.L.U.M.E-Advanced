"""Command-line packaging and failure-record regressions."""

import json
from pathlib import Path

import pytest

from plume_advanced import cli


def test_packaged_default_config_is_available() -> None:
    assert cli.PACKAGED_CONFIG.is_file()
    config = cli.load_project_config(cli.PACKAGED_CONFIG)
    assert config.run.dev_mode
    assert not config.geometry.cave_diffuse_texture


def test_default_complete_scene_uses_portable_asset_name() -> None:
    args = cli.parse_args([])
    assert args.geometry_glb_output is None
    assert args.geometry_mesh_output is None
    expected_glb = args.output.with_name("plume_cave_scene.glb")
    expected_obj = args.output.with_name("plume_cave_scene.obj")
    assert expected_glb.name == "plume_cave_scene.glb"
    assert expected_obj.name == "plume_cave_scene.obj"


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
