"""Exercise the real evaluation command through meshing and package checks."""

import json
from pathlib import Path

import pytest
import trimesh

from plume_advanced.evaluation import cli
from plume_advanced.evaluation.experiments import determinism, export_consistency

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment_config(tmp_path: Path) -> Path:
    project = ROOT / "src" / "plume_advanced" / "default_project.toml"
    (tmp_path / "seeds.txt").write_text("1\n")
    path = tmp_path / "experiments.toml"
    path.write_text(
        f'''schema_version = 1
[general]
project_config = "{project}"
output_root = "results"
[export_consistency]
seed_file = "seeds.txt"
targets = ["blender", "ue5", "unity", "gazebo", "omniverse"]
'''
    )
    return path


@pytest.mark.integration
def test_export_evaluation_meshes_packages_and_reuses_matching_case(
    experiment_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = ["--config", str(experiment_config), "export-consistency"]
    assert cli.main(arguments) == 0
    output = experiment_config.parent / "results" / "export_consistency"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete_n"] == summary["planned_n"] == 1
    assert summary["failed_n"] == 0
    assert summary["passed"]
    case = output / "cases" / "seed-000001-all-targets.json"
    original = case.read_bytes()
    record = json.loads(original)
    assert len(record["target_checks"]) == 5
    assert all(check["visual_asset_present"] for check in record["target_checks"])
    asset = output / "packages" / "seed-000001" / "blender" / "plume_seed_000001.glb"
    scene = trimesh.load(asset, force="scene", process=False)
    assert sum(len(mesh.faces) for mesh in scene.geometry.values()) > 0

    def fail_if_rebuilt(_project):
        raise AssertionError("matching case should have been reused")

    monkeypatch.setattr(export_consistency, "generate_sections", fail_if_rebuilt)
    assert cli.main(arguments) == 0
    assert case.read_bytes() == original


def test_failed_export_evaluation_returns_failure_without_success_claims(
    experiment_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_project):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(export_consistency, "generate_sections", fail)
    assert cli.main(["--config", str(experiment_config), "export-consistency"]) == 1
    output = experiment_config.parent / "results" / "export_consistency"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete_n"] == 0
    assert summary["failed_n"] == 1
    assert not summary["all_packages_present"]
    assert not summary["all_collisions_present"]
    assert not summary["passed"]
    record = json.loads((output / "cases" / "seed-000001-all-targets.json").read_text())
    assert "synthetic generation failure" in record["failure_reason"]


def test_all_command_propagates_failed_experiment_summary(
    experiment_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "audit_pdc", lambda *_args: {})
    for name in (
        "run_morphometry", "run_controllability", "run_host_ablation", "run_sampling_ablation",
        "run_scalability", "run_export_consistency", "run_determinism",
    ):
        monkeypatch.setattr(cli, name, lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        cli, "aggregate_results",
        lambda *_args: {"experiments": {"export_consistency": {"complete_n": 0, "failed_n": 1}}},
    )
    monkeypatch.setattr("plume_advanced.evaluation.plotting.generate_figures", lambda *_args: [])
    monkeypatch.setattr(cli, "generate_latex", lambda *_args: [])
    assert cli.main([
        "--config", str(experiment_config), "all", "--data-root", str(experiment_config.parent),
    ]) == 1


def test_determinism_reports_all_failed_cases_without_indexing_empty_results(
    experiment_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_config.write_text(
        experiment_config.read_text() + '\n[determinism]\nseed_file = "seeds.txt"\n'
    )

    def fail(_project):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(determinism, "generate_sections", fail)
    assert cli.main(["--config", str(experiment_config), "determinism"]) == 1
    summary = json.loads(
        (experiment_config.parent / "results" / "determinism" / "summary.json").read_text()
    )
    assert summary["complete_n"] == 0
    assert summary["failed_n"] == 1
    assert summary["checks"] == {}
