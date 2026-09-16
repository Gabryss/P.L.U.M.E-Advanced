"""Promotion checks with injected editor outcomes; no native editor is launched."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from plume_advanced.evaluation.reliability_reports import seal_result
from plume_advanced.identity import package_source_hash, sha256_file

spec = importlib.util.spec_from_file_location(
    "simulation_delivery", Path(__file__).resolve().parents[1] / "scripts/qualify_simulation.py")
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


def seal_pair(root, **overrides):
    result = dict(status="passed", identity={"fixture": "promotion-test"},
                  source_sha256=package_source_hash(), runtime={"python": "fixture"},
                  case={"seed": 0, "scope": "full"}, python_hash_seed=11)
    result.update(overrides)
    seal_result(root, result)
    replay = root.parent.parent / "replay_0000/attempt_0000"
    shutil.copytree(root, replay, dirs_exist_ok=True)
    seal_result(replay, dict(result, python_hash_seed=37))
    delivery.write(root.parent.parent / "summary.json", dict(cases=[dict(
        directory="case_0000/attempt_0000", replay_directory="replay_0000/attempt_0000",
        status="passed", replay_passed=True)]))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "campaign/case_0000/attempt_0000"
    package = root / "export"
    package.mkdir(parents=True)
    for name in ("plume_cave.glb", "plume_cave_collision.obj", "material.py"):
        (package / name).write_bytes(b"Provider fixture: promotion only, not a geometry test")
    checks = {key: {"status": "passed"} for key in
              ("mesh", "clearance", "collision", "resolution", "relief",
               "texture_integrity", "pbr_textures", "export_budgets")}
    files = [dict(path=p.name, passed=True, sha256=sha256_file(p)) for p in package.iterdir()]
    delivery.write(root / "pipeline_quality_report.json", dict(passed=True,
        acceptance=dict(passed=True, policy=dict(profile="simulation"), checks=checks),
        export_inspection=dict(serialized=dict(files=files[:2]), textures=dict(
            passed=True, package_attempts=[dict(passed=True, checked_files=files)]))))
    for name in ("stage_c_sections.json", "stage_c_sections.npz", "resolved_config.json", "stage_b_network.json"):
        (root / name).write_bytes(b"fixture")
    seal_pair(root)
    return root


def fake_editor(monkeypatch, run, output, *, missing=None, mutate=None, stale=False):
    def execute(*args):
        native = output / "native"
        native.mkdir()
        delivery.write(native / "native_summary.json", dict(passed=True, checks={
            key: dict(native=dict(passed=True), body_validation=dict(passed=True)) for key in ("unity", "unreal") if key != missing}))
        delivery.write(native / "native_input_receipt.json", dict(
            glb_sha256="stale" if stale else sha256_file(run / "export/plume_cave.glb"),
            quality_sha256=sha256_file(run / "pipeline_quality_report.json"),
            collision_obj_sha256=sha256_file(run / "export/plume_cave_collision.obj"),
            adapter_source=package_source_hash(), fixture_sha256={
                name: sha256_file(delivery.REPO / "tests/fixtures" / name)
                for name in ("unity/PlumeNativeCheck.cs", "unreal/native_check.py")}))
        if mutate:
            (run / mutate).write_bytes(b"changed during editor check")
    monkeypatch.setattr(delivery, "run_command", execute)


def test_only_complete_native_results_promote_exact_input_bytes(run, tmp_path, monkeypatch):
    output = tmp_path / "check"
    output.mkdir()
    expected = delivery.checked_inputs(run)
    fake_editor(monkeypatch, run, output)
    result = delivery.qualify(run, output, {"unity": Path("unity"), "unreal": Path("ue")}, 1)
    assert result["passed"]
    for name, digest in expected.items():
        assert sha256_file(output / "ready" / name) == digest
    assert (output / "ready/simulation_ready.json").is_file()
    assert result["cold_replay"]["passed"]
    assert sha256_file(output / "ready/replay_result.json") == result["cold_replay"]["receipt_sha256"]


@pytest.mark.parametrize("failure", ["missing", "stale", "asset_mutation", "material_mutation"])
def test_incomplete_or_mismatched_engine_evidence_never_publishes(run, tmp_path, monkeypatch, failure):
    output = tmp_path / "check"
    output.mkdir()
    mutation = {"asset_mutation": "export/plume_cave.glb",
                "material_mutation": "export/material.py"}.get(failure)
    fake_editor(monkeypatch, run, output, missing="unreal" if failure == "missing" else None,
                stale=failure == "stale", mutate=mutation)
    with pytest.raises(ValueError):
        delivery.qualify(run, output, {"unity": Path("unity"), "unreal": Path("ue")}, 1)
    assert not (output / "ready").exists()
    assert (output / "native/native_summary.json").is_file()


def test_inspection_profile_cannot_be_promoted_as_simulation(run):
    path = run / "pipeline_quality_report.json"
    quality = json.loads(path.read_text())
    quality["acceptance"]["policy"]["profile"] = "inspection"
    delivery.write(path, quality)
    seal_result(run, dict(status="passed", identity={"fixture": "inspection-provider"}))
    with pytest.raises(ValueError, match="simulation acceptance"):
        delivery.checked_inputs(run)


def test_changed_material_is_rejected_before_launching_an_editor(run):
    (run / "export/material.py").write_bytes(b"uninspected material")
    with pytest.raises(ValueError, match="changed"):
        delivery.checked_inputs(run)


def test_uninspected_extra_files_are_not_bundled_in_a_delivery(run):
    (run / "export/unchecked.py").write_bytes(b"unexpected helper")
    with pytest.raises(ValueError, match="not covered"):
        delivery.checked_inputs(run)


def test_changed_world_configuration_is_not_promoted_with_old_assets(run):
    (run / "resolved_config.json").write_text('{"world": "changed"}')
    with pytest.raises(ValueError, match="resolved_config"):
        delivery.checked_inputs(run)


def test_native_failure_records_failure_without_ready_delivery(run, tmp_path, monkeypatch):
    binary = tmp_path / "editor"
    binary.touch()
    output = tmp_path / "qualification"
    def failed(*args):
        raise RuntimeError("editor crashed")
    monkeypatch.setattr(delivery, "run_command", failed)
    assert delivery.main(["--run", str(run), "--output", str(output), "--unity", str(binary)]) == 1
    assert not (output / "ready").exists()
    result = json.loads((output / "qualification.json").read_text())
    assert result["passed"] is False and "editor crashed" in result["error"]


def test_timeout_gives_adapter_a_chance_to_clean_up_its_editors(tmp_path):
    marker = tmp_path / "cleanup.txt"
    program = ("import time,pathlib\ntry:\n time.sleep(60)\n"
               f"except KeyboardInterrupt:\n pathlib.Path({str(marker)!r}).write_text('cleaned')\n")
    with pytest.raises(subprocess.TimeoutExpired):
        delivery.run_command([sys.executable, "-c", program], tmp_path / "timeout.log", .5)
    assert marker.read_text() == "cleaned"


def test_failed_worker_with_still_passing_export_cannot_be_promoted(run):
    seal_result(run, dict(status='failed', reason='source changed after export'))
    with pytest.raises(ValueError, match='did not pass'):
        delivery.checked_inputs(run)


def test_ray_only_native_success_cannot_promote_simulation(run, tmp_path, monkeypatch):
    output = tmp_path / 'check'
    output.mkdir()
    fake_editor(monkeypatch, run, output)
    execute = delivery.run_command
    def ray_only(*args):
        execute(*args)
        path = output / 'native/native_summary.json'
        data = json.loads(path.read_text())
        data['checks']['unity'].pop('body_validation')
        delivery.write(path, data)
    monkeypatch.setattr(delivery, 'run_command', ray_only)
    with pytest.raises(ValueError, match='Every requested'):
        delivery.qualify(run, output, {'unity': Path('unity')}, 1)
    assert not (output / 'ready').exists()


def test_delivery_distinguishes_generation_source_from_native_checker(run, tmp_path, monkeypatch):
    output = tmp_path / 'check'
    output.mkdir()
    seal_pair(run, source_sha256='a'*64, identity={'network': 'prior generation'})
    fake_editor(monkeypatch, run, output)
    result = delivery.qualify(run, output, {'unity': Path('unity')}, 1)
    assert result['generation']['source_sha256'] == 'a'*64
    assert result['generation']['identity']['network'] == 'prior generation'
    assert result['source_sha256'] == package_source_hash()
    assert result['generation']['receipt_sha256'] == sha256_file(run / 'result.json')


@pytest.mark.parametrize('fault', ['missing', 'failed', 'identity', 'case', 'source_sha256',
                                 'runtime', 'python_hash_seed', 'mutated_asset', 'self_reference'])
def test_cold_replay_must_be_intact_independent_and_equivalent(run, tmp_path, monkeypatch, fault):
    replay = run.parent.parent / 'replay_0000/attempt_0000'
    result = json.loads((replay / 'result.json').read_text())
    if fault == 'missing':
        (run.parent.parent / 'summary.json').unlink()
    elif fault == 'failed':
        seal_result(replay, dict(result, status='failed'))
    elif fault == 'mutated_asset':
        (replay / 'export/plume_cave.glb').write_bytes(b'changed replay')
    elif fault == 'self_reference':
        delivery.write(run.parent.parent / 'summary.json', dict(cases=[dict(
            directory='case_0000/attempt_0000', replay_directory='case_0000/attempt_0000',
            status='passed', replay_passed=True)]))
    else:
        result[fault] = 11 if fault == 'python_hash_seed' else 'mismatched'
        seal_result(replay, result)
    monkeypatch.setattr(delivery, 'run_command', lambda *args: pytest.fail('Editor must not start'))
    output = tmp_path / 'blocked'
    output.mkdir()
    with pytest.raises(ValueError, match='[Cc]old replay'):
        delivery.qualify(run, output, {'unity': Path('unity')}, 1)
    assert not (output / 'ready').exists()


def test_replay_changed_during_native_checks_cannot_promote(run, tmp_path, monkeypatch):
    output = tmp_path / 'check'
    output.mkdir()
    fake_editor(monkeypatch, run, output)
    execute = delivery.run_command
    def mutate(*args):
        execute(*args)
        replay = run.parent.parent / 'replay_0000/attempt_0000/export/plume_cave.glb'
        replay.write_bytes(b'changed during native checks')
    monkeypatch.setattr(delivery, 'run_command', mutate)
    with pytest.raises(ValueError, match='Cold replay integrity'):
        delivery.qualify(run, output, {'unity': Path('unity')}, 1)
    assert not (output / 'ready').exists()
