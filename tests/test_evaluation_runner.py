from pathlib import Path

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
