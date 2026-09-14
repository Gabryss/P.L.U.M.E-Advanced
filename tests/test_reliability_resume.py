"""Fault injection for the supported unattended workflow, without expensive meshing."""

import errno
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from plume_advanced.evaluation import reliability
from plume_advanced.evaluation.reliability import ReliabilityCase, run_campaign
from plume_advanced.evaluation.reliability_reports import diagnose, seal_result, verify_result
from plume_advanced.evaluation.reliability_state import campaign_lock, plan_identity, preflight


@pytest.fixture
def worker(monkeypatch):
    calls = []

    def run(case, output, **kwargs):
        calls.append((case, output, kwargs))
        output.mkdir(parents=True)
        (output / "evidence.json").write_text('{"measured":true}')
        result = dict(status="passed", case=asdict(case), identity={"network": str(case.seed)})
        return seal_result(output, result)

    monkeypatch.setattr(reliability, "run_isolated", run)
    return calls


def test_resume_retains_verified_original_and_cold_replay(tmp_path, worker):
    cases = [ReliabilityCase("unused.toml", seed) for seed in (0, 17)]
    output = tmp_path / "campaign"
    first = run_campaign(cases, output)
    assert first["passed"] and len(worker) == 4
    original = (output / "case_0000/attempt_0000/result.json").read_bytes()
    second = run_campaign(cases, output, resume=True, timeout_s=900, memory_limit_mib=0)
    assert second["passed"] and len(worker) == 4
    assert all(r["reused"] for r in second["cases"])
    assert (output / "case_0000/attempt_0000/result.json").read_bytes() == original
    assert worker[0][2]["checkpoint_directory"].name == "checkpoints"
    assert worker[1][2]["checkpoint_directory"] is None
    assert worker[1][2]["hash_seed"] == 37


def test_resume_after_interruption_only_runs_unfinished_work(tmp_path, worker, monkeypatch):
    cases = [ReliabilityCase("unused.toml", seed) for seed in (0, 1)]
    output = tmp_path / "campaign"
    original = reliability.run_isolated

    def interrupt(case, directory, **kwargs):
        if case.seed == 1:
            raise KeyboardInterrupt
        return original(case, directory, **kwargs)

    monkeypatch.setattr(reliability, "run_isolated", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_campaign(cases, output)
    saved = json.loads((output / "summary.json").read_text())
    assert not saved["complete"] and not saved["passed"]
    assert len(saved["cases"]) == 1 and "Interrupted" in saved["campaign_error"]
    monkeypatch.setattr(reliability, "run_isolated", original)
    assert run_campaign(cases, output, resume=True)["passed"]
    assert [c.seed for c, _, _ in worker] == [0, 0, 1, 1]


@pytest.mark.parametrize("damage", ["missing", "modified", "receipt", "identity", "traversal"])
def test_damaged_evidence_never_passes_and_is_rebuilt(tmp_path, worker, damage):
    cases = [ReliabilityCase("unused.toml", 42)]
    output = tmp_path / "campaign"
    run_campaign(cases, output)
    original = output / "case_0000/attempt_0000"
    if damage == "missing":
        (original / "evidence.json").unlink()
    elif damage == "modified":
        (original / "evidence.json").write_text("wrong mesh")
    elif damage == "receipt":
        (original / "result.sha256").unlink()
    elif damage == "identity":
        result = json.loads((original / "result.json").read_text())
        result["identity"] = {"network": "changed"}
        (original / "result.json").write_text(json.dumps(result))
    else:
        from plume_advanced.identity import sha256_file

        result = json.loads((original / "result.json").read_text())
        result["artifacts"] = {"../../plan.json": sha256_file(output / "plan.json")}
        (original / "result.json").write_text(json.dumps(result))
        (original / "result.sha256").write_text(sha256_file(original / "result.json"))
    report = run_campaign(cases, output, resume=True, report_only=True)
    assert not report["passed"] and len(worker) == 2
    assert report["cases"][0]["diagnostic"]["category"] == "artifact_integrity"
    repaired = run_campaign(cases, output, resume=True)
    assert repaired["passed"] and len(worker) == 3
    assert original.is_dir() and (original.parent / "attempt_0001/result.json").is_file()


def test_replay_damage_replays_only_and_does_not_reuse_checkpoints(tmp_path, worker):
    output = tmp_path / "campaign"
    cases = [ReliabilityCase("unused.toml", 42)]
    run_campaign(cases, output)
    (output / "replay_0000/attempt_0000/evidence.json").unlink()
    assert run_campaign(cases, output, resume=True)["passed"]
    assert len(worker) == 3
    assert worker[-1][2]["hash_seed"] == 37
    assert worker[-1][2]["checkpoint_directory"] is None


def test_persistent_failure_has_one_attempt_per_explicit_resume(tmp_path, monkeypatch):
    calls = []

    def fail(case, output, **kwargs):
        calls.append(case.seed)
        output.mkdir(parents=True)
        return seal_result(
            output,
            dict(
                status="failed",
                case=asdict(case),
                diagnostic=diagnose(TypeError("new defect"), "network"),
            ),
        )

    monkeypatch.setattr(reliability, "run_isolated", fail)
    output = tmp_path / "campaign"
    cases = [ReliabilityCase("unused.toml", 42)]
    assert not run_campaign(cases, output)["passed"]
    assert not run_campaign(cases, output, resume=True)["passed"]
    assert calls == [42, 42]
    assert len(list((output / "case_0000").glob("attempt_*"))) == 2


@pytest.mark.parametrize("change", ["seed", "replay", "config", "source", "runtime"])
def test_changed_plan_is_refused_without_mutating_evidence(tmp_path, worker, monkeypatch, change):
    config = tmp_path / "invalid.toml"
    config.write_text("invalid toml")
    output = tmp_path / "campaign"
    cases = [ReliabilityCase(str(config), 42)]
    run_campaign(cases, output)
    saved = (output / "summary.json").read_bytes()
    replay = True
    if change == "seed":
        cases = [ReliabilityCase(str(config), 43)]
    elif change == "replay":
        replay = False
    elif change == "config":
        config.write_text("different toml")
    elif change == "source":
        monkeypatch.setattr(reliability, "package_source_hash", lambda: "new source")
    else:
        from plume_advanced.evaluation import reliability_state

        monkeypatch.setattr(reliability_state, "runtime_identity", lambda: {"new": "runtime"})
    with pytest.raises(ValueError, match="changed"):
        run_campaign(cases, output, resume=True, replay=replay)
    assert len(worker) == 2
    assert (output / "summary.json").read_bytes() == saved


def test_missing_plan_refused(tmp_path):
    with pytest.raises(ValueError, match="No valid resumable"):
        run_campaign([ReliabilityCase("config", 0)], tmp_path, resume=True)


def test_report_only_does_not_create_workers(tmp_path, worker):
    output = tmp_path / "campaign"
    cases = [ReliabilityCase("unused.toml", 42)]
    output.mkdir()
    from plume_advanced.evaluation.reliability_reports import write_json
    from plume_advanced.identity import package_source_hash

    write_json(
        output / "plan.json", plan_identity(cases, source=package_source_hash(), replay=True)
    )
    report = run_campaign(cases, output, resume=True, report_only=True)
    assert not report["complete"] and not report["passed"] and not worker


def test_report_escapes_config_and_error_text(tmp_path, worker, monkeypatch):
    def fail(case, output, **kwargs):
        return dict(
            status="failed", case=asdict(case), diagnostic={"message": "<script>bad</script>"}
        )

    monkeypatch.setattr(reliability, "run_isolated", fail)
    output = tmp_path / "campaign"
    run_campaign([ReliabilityCase("<svg onload=x>.toml", 42)], output)
    text = (output / "report.html").read_text()
    assert "<script>" not in text and "<svg" not in text
    assert "&lt;script&gt;" in text and "NOT CHECKED" not in text


def test_real_process_lock_rejects_concurrent_campaign_then_releases(tmp_path):
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; from plume_advanced.evaluation.reliability_state import campaign_lock; "
        f"\nwith campaign_lock(Path({str(tmp_path)!r})): pass",
    ]
    with campaign_lock(tmp_path):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and "Another process" in result.stderr
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0


@pytest.mark.parametrize(
    "error,category",
    [
        (MemoryError(), "memory_budget"),
        (OSError(errno.ENOSPC, "full"), "disk_space"),
        (FileNotFoundError("map.png"), "input_or_environment"),
        (PermissionError("denied"), "input_or_environment"),
        (ImportError("library"), "input_or_environment"),
        (TypeError("bug"), "unexpected_error"),
        (ValueError("not a known geometry error"), "unexpected_error"),
    ],
)
def test_diagnostics_never_guess_unknown_errors_are_repairable(error, category):
    report = diagnose(error, "geometry")
    assert report["category"] == category and report["action"]


def test_known_quality_and_budget_diagnostics():
    from plume_advanced.exporters.targets import ExportBudgetError
    from plume_advanced.stages.network_quality import NetworkQualityError
    from plume_advanced.stages.network_systems import GenerationDomainError
    from plume_advanced.stages.surface_topology import SurfaceTopologyError

    for error, category in [
        (ExportBudgetError("too large"), "export_budget"),
        (GenerationDomainError("too narrow"), "host_domain"),
        (NetworkQualityError({}), "network_rejected"),
        (SurfaceTopologyError("bad genus"), "surface_rejected"),
    ]:
        assert diagnose(error, "geometry")["category"] == category


@pytest.mark.parametrize("payload", ["garbage", "[]", '{"status":"unknown"}'])
def test_invalid_worker_result_is_retained_as_failure(tmp_path, monkeypatch, payload):
    def run(command, **kwargs):
        output = Path(command[command.index("--worker") + 1])
        (output / "result.json").write_text(payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reliability.subprocess, "run", run)
    result = reliability.run_isolated(ReliabilityCase("config", 0), tmp_path / "case", timeout_s=1)
    assert result["status"] == "failed" and result["diagnostic"]["category"] == "worker_crash"


def test_cli_resume_loads_saved_recipe(tmp_path, worker):
    output = tmp_path / "campaign"
    run_campaign([ReliabilityCase("unused.toml", 0)], output, replay=False)
    assert reliability.main(["--output", str(output), "--resume"]) == 0
    assert len(worker) == 1


def test_preflight_reports_all_invalid_inputs_without_creating_output(tmp_path):
    output = tmp_path / "no_write"
    report = preflight([ReliabilityCase(str(tmp_path / "missing.toml"), s) for s in (0, 1)], output)
    assert not report["passed"] and len(report["cases"]) == 2 and not output.exists()


def test_preflight_packaged_neutral_config():
    from plume_advanced.cli import PACKAGED_CONFIG

    report = preflight([ReliabilityCase(str(PACKAGED_CONFIG), 0, scope="full")])
    assert report["passed"] and report["cases"][0]["warnings"]


def test_configured_texture_content_is_part_of_resume_identity(tmp_path):
    from plume_advanced.cli import PACKAGED_CONFIG

    config = tmp_path / "case.toml"
    config.write_text(
        PACKAGED_CONFIG.read_text().replace(
            'cave_diffuse_texture = ""', 'cave_diffuse_texture = "tile.png"'
        )
    )
    tile = tmp_path / "tile.png"
    tile.write_bytes(b"first")
    cases = [ReliabilityCase(str(config), 0)]
    first = plan_identity(cases, source="same", replay=True)
    tile.write_bytes(b"second")
    assert plan_identity(cases, source="same", replay=True) != first


def test_hash_receipt_requires_identity_and_evidence(tmp_path):
    with pytest.raises(ValueError, match="retain evidence"):
        seal_result(tmp_path, dict(status="passed", identity={"mesh": "abc"}))
    (tmp_path / "mesh.bin").write_bytes(b"mesh")
    seal_result(tmp_path, dict(status="passed"))
    assert verify_result(tmp_path)[0] is None


@pytest.mark.integration
def test_real_stage_checkpoint_resume_after_artifact_write_failure(tmp_path, monkeypatch):
    from plume_advanced.cli import PACKAGED_CONFIG
    from plume_advanced.evaluation import artifacts
    from plume_advanced.stages.host_field import HostFieldGenerator
    from plume_advanced.stages.network import CaveNetworkGenerator
    from plume_advanced.stages.section_field import SectionFieldGenerator

    case = ReliabilityCase(str(PACKAGED_CONFIG), 3)
    checkpoints = tmp_path / "checkpoints"
    export = artifacts.export_section_artifact

    def interrupted_write(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected disk-full after sections were checkpointed")

    monkeypatch.setattr(artifacts, "export_section_artifact", interrupted_write)
    with pytest.raises(OSError, match="injected"):
        reliability.execute_case(case, tmp_path / "failed", checkpoint_directory=checkpoints)
    assert (checkpoints / "sections.pickle").is_file()
    monkeypatch.setattr(artifacts, "export_section_artifact", export)

    def reject_recomputation(*args, **kwargs):
        raise AssertionError("Compatible completed stage was unexpectedly recomputed")

    monkeypatch.setattr(HostFieldGenerator, "generate", reject_recomputation)
    monkeypatch.setattr(CaveNetworkGenerator, "generate", reject_recomputation)
    monkeypatch.setattr(SectionFieldGenerator, "generate", reject_recomputation)
    result = reliability.execute_case(case, tmp_path / "resumed", checkpoint_directory=checkpoints)
    assert result["status"] == "passed"
    assert (tmp_path / "resumed/stage_c_sections.npz").is_file()
    assert (tmp_path / "resumed/network_quality.json").is_file()
    events = [
        json.loads(line) for line in (tmp_path / "resumed/progress.jsonl").read_text().splitlines()
    ]
    assert {e["detail"] for e in events if e["step"] == "Checkpoint"} == {
        "host: reused",
        "network: reused",
        "sections: reused",
    }
    # A replay must not share those checkpoints, even if the original exists.
    with pytest.raises(AssertionError, match="recomputed"):
        reliability.execute_case(case, tmp_path / "cold")


def test_report_only_preserves_failed_attempt_diagnosis(tmp_path, monkeypatch):
    def fail(case, output, **kwargs):
        output.mkdir(parents=True)
        return seal_result(
            output,
            dict(
                status="failed",
                case=asdict(case),
                diagnostic=diagnose(MemoryError("budget"), "geometry"),
            ),
        )

    monkeypatch.setattr(reliability, "run_isolated", fail)
    cases = [ReliabilityCase("unused.toml", 0)]
    output = tmp_path / "campaign"
    run_campaign(cases, output)
    result = run_campaign(cases, output, resume=True, report_only=True)
    assert result["cases"][0]["diagnostic"]["category"] == "memory_budget"


@pytest.mark.parametrize("voxel", [0, -1, float("inf"), float("nan")])
def test_preflight_rejects_invalid_resolution_override(voxel):
    from plume_advanced.cli import PACKAGED_CONFIG

    report = preflight([ReliabilityCase(str(PACKAGED_CONFIG), 0, voxel_size=voxel)])
    assert not report["passed"]
    assert report["cases"][0]["diagnostic"]["category"] == "configuration"


def test_input_change_during_campaign_invalidates_pass(tmp_path, worker, monkeypatch):
    config = tmp_path / "bad.toml"
    config.write_text("invalid")
    original = reliability.run_isolated

    def mutate(case, output, **kwargs):
        result = original(case, output, **kwargs)
        config.write_text("changed while running")
        return result

    monkeypatch.setattr(reliability, "run_isolated", mutate)
    result = run_campaign([ReliabilityCase(str(config), 0)], tmp_path / "campaign")
    assert result["complete"] and not result["passed"] and not result["inputs_unchanged"]
