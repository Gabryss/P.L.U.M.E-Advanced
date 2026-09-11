import json
from dataclasses import replace
from pathlib import Path

import pytest

from plume_advanced.evaluation import provenance
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult


def _template() -> ExperimentResult:
    return ExperimentResult(
        experiment_name="smoke",
        run_id="seed-000001-full",
        condition_id="full",
        seed=1,
        status="complete",
    )


def test_complete_case_resumes_without_rerun_and_force_archives(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "smoke")
    calls = []
    assert run_case(store, _template(), lambda: calls.append(1) or {"answer": 42})
    assert run_case(store, _template(), lambda: calls.append(2) or {"answer": 0}) is None
    assert calls == [1]

    run_case(store, _template(), lambda: {"answer": 43}, force=True)
    assert store.load(_template().run_id)["answer"] == 43
    assert len(list((store.case_root / "attempts").glob("*.json"))) == 1


def test_failed_case_is_preserved_and_can_resume(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "smoke")

    def fail():
        raise RuntimeError("synthetic failure")

    result = run_case(store, _template(), fail)
    assert result is not None and result.status == "failed"
    assert "synthetic failure" in store.load(_template().run_id)["failure_reason"]

    rerun = run_case(store, _template(), lambda: {"recovered": True})
    assert rerun is not None and rerun.status == "complete"
    assert list((store.case_root / "attempts").glob("*.json"))


@pytest.mark.parametrize(
    "change",
    [
        {"resolved_config_sha256": "new-config"},
        {"git_commit": "new-revision"},
        {"git_dirty": True},
        {"provenance_sha256": "new-source-or-inputs"},
        {"input_dataset_sha256_or_version": "new-dataset"},
        {"python_version": "changed-python"},
        {"model_schema_version": 999},
    ],
)
def test_changed_case_identity_recomputes_and_archives(tmp_path: Path, change: dict) -> None:
    store = ResultStore(tmp_path, "smoke")
    first = _template()
    run_case(store, first, lambda: {"answer": 42})
    changed = replace(first, **change)

    result = run_case(store, changed, lambda: {"answer": 99})

    assert result is not None
    assert store.load(first.run_id)["answer"] == 99
    archive = list((store.case_root / "attempts").glob("*.json"))
    assert len(archive) == 1
    assert json.loads(archive[0].read_text())["answer"] == 42


def test_legacy_results_without_fingerprint_are_not_reused(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "smoke")
    legacy = _template().to_dict()
    legacy.pop("provenance_sha256")
    store.case_path(_template().run_id).write_text(json.dumps(legacy))

    assert run_case(store, _template(), lambda: {"recomputed": True}) is not None
    assert store.load(_template().run_id)["recomputed"]


def test_campaign_indexes_exclude_old_cases_but_preserve_them(tmp_path: Path) -> None:
    old = ResultStore(tmp_path, "smoke", provenance_sha256="old")
    first = replace(_template(), provenance_sha256="old")
    run_case(old, first, lambda: {"answer": 1})
    current = ResultStore(tmp_path, "smoke", provenance_sha256="current")
    second = replace(first, run_id="seed-000002-full", seed=2, provenance_sha256="current")
    run_case(current, second, lambda: {"answer": 2})

    assert [row["seed"] for row in current.rows()] == [2]
    assert [row["seed"] for row in json.loads((current.root / "raw_results.json").read_text())] == [2]
    assert old.case_path(first.run_id).is_file()


def test_provenance_tracks_dirty_sources_inputs_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "src" / "plume_advanced"
    module = package / "evaluation" / "provenance.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n")
    input_file = tmp_path / "project.toml"
    input_file.write_text("procedural_seed = 1\n")
    monkeypatch.setattr(provenance, "__file__", str(module))
    monkeypatch.setattr(provenance, "_git", lambda *_args: "unchanged-dirty-revision")

    def capture():
        return provenance.capture_provenance(tmp_path, inputs=(input_file,))

    first = capture()
    assert first["identity_sha256"] == capture()["identity_sha256"]
    module.write_text("VALUE = 2\n")
    changed_source = capture()
    assert first["git_commit"] == changed_source["git_commit"]
    assert first["git_dirty"] == changed_source["git_dirty"]
    assert first["identity_sha256"] != changed_source["identity_sha256"]
    input_file.write_text("procedural_seed = 2\n")
    changed_input = capture()
    assert changed_source["identity_sha256"] != changed_input["identity_sha256"]
    monkeypatch.setattr(provenance, "_version", lambda _name: "changed-version")
    assert changed_input["identity_sha256"] != capture()["identity_sha256"]
