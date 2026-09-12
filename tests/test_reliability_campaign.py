"""Campaigns retain failures, respect timeouts, and verify fresh-process identity."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from plume_advanced.evaluation import reliability
from plume_advanced.evaluation.reliability import ReliabilityCase, run_campaign, run_isolated


def test_campaign_continues_after_failure_and_keeps_every_case(tmp_path, monkeypatch):
    calls = []

    def run(case, output, **kwargs):
        calls.append(case.seed)
        return {
            "status": "failed" if case.seed == 1 else "passed",
            "identity": {"network": str(case.seed)},
        }

    monkeypatch.setattr(reliability, "run_isolated", run)
    cases = [ReliabilityCase("config", s) for s in (0, 1, 2)]
    result = run_campaign(cases, tmp_path / "campaign")
    assert result["complete"] and not result["passed"]
    assert calls == [0, 0, 1, 2, 2]
    assert [row["status"] for row in result["cases"]] == ["passed", "failed", "passed"]
    assert json.loads((tmp_path / "campaign/summary.json").read_text()) == result


def test_replay_mismatch_is_a_failure(tmp_path, monkeypatch):
    def run(case, output, hash_seed=11, **kwargs):
        return {"status": "passed", "identity": {"network": str(hash_seed)}}

    monkeypatch.setattr(reliability, "run_isolated", run)
    result = run_campaign([ReliabilityCase("config", 42)], tmp_path / "campaign")
    assert not result["passed"]
    assert not result["cases"][0]["replay_passed"]


@pytest.mark.parametrize("failure", ["timeout", "crash", "missing_result"])
def test_worker_failure_always_has_a_result(tmp_path, monkeypatch, failure):
    def run(command, **kwargs):
        assert kwargs["timeout"] == 0.5
        assert kwargs["env"]["PYTHONHASHSEED"] == "37"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 0.5)
        return SimpleNamespace(returncode=-11 if failure == "crash" else 0)

    monkeypatch.setattr(reliability.subprocess, "run", run)
    result = run_isolated(
        ReliabilityCase("config", 0), tmp_path / "case", timeout_s=0.5, hash_seed=37
    )
    assert result["status"] == ("timeout" if failure == "timeout" else "failed")
    assert (tmp_path / "case/result.json").is_file()


def test_campaign_does_not_overwrite_previous_results(tmp_path):
    path = tmp_path / "campaign"
    path.mkdir()
    (path / "summary.json").write_text("existing")
    with pytest.raises(FileExistsError):
        run_campaign([ReliabilityCase("config", 0)], path)
    assert (path / "summary.json").read_text() == "existing"


def test_real_worker_records_invalid_input(tmp_path):
    result = run_isolated(
        ReliabilityCase(str(tmp_path / "missing.toml"), 0), tmp_path / "case", timeout_s=30
    )
    assert result["status"] == "failed" and "FileNotFoundError" in result["reason"]


@pytest.mark.parametrize(
    "error_type", [ValueError, FloatingPointError, OverflowError, ZeroDivisionError]
)
def test_recoverable_seed_failure_uses_reproducible_next_seed(error_type):
    from dataclasses import replace
    from unittest.mock import patch

    from test_network_quality import network_fixture

    from plume_advanced.procedural import derive_subseed
    from plume_advanced.stages.network import CaveNetworkGenerator
    from plume_advanced.stages.network_quality import NetworkQualityConfig

    network = network_fixture()
    config = replace(
        network.config,
        random_seed=42,
        quality=NetworkQualityConfig(max_attempts=2, repair_passes=0),
    )
    visited = []

    def candidate(worker, host):
        visited.append(worker.config.random_seed)
        if worker.config.random_seed == 42:
            raise error_type("unusable candidate")
        return network_fixture(config=worker.config)

    with patch.object(CaveNetworkGenerator, "_generate_candidate", candidate):
        results = [CaveNetworkGenerator(config).generate(None) for _ in range(2)]
    second = derive_subseed(42, "network-quality-v1", 1)
    assert visited == [42, second, 42, second]
    assert results[0].quality_report == results[1].quality_report
    assert results[0].quality_report["selected_seed"] == second


def test_domain_failure_is_not_retried():
    from unittest.mock import patch

    from plume_advanced.stages.network import CaveNetworkGenerator
    from plume_advanced.stages.network_systems import GenerationDomainError

    with patch.object(
        CaveNetworkGenerator,
        "_generate_candidate",
        side_effect=GenerationDomainError("host too narrow"),
    ) as build:
        with pytest.raises(GenerationDomainError, match="host too narrow"):
            CaveNetworkGenerator().generate(None)
    assert build.call_count == 1


def test_programming_error_is_not_hidden_as_a_bad_seed():
    from unittest.mock import patch

    from plume_advanced.stages.network import CaveNetworkGenerator

    with patch.object(
        CaveNetworkGenerator, "_generate_candidate", side_effect=TypeError("programming error")
    ) as build:
        with pytest.raises(TypeError, match="programming error"):
            CaveNetworkGenerator().generate(None)
    assert build.call_count == 1


def test_section_evaluation_uses_production_acceptance():
    from unittest.mock import patch

    from plume_advanced.config import load_project_config
    from plume_advanced.evaluation.experiments.common import generate_sections
    from plume_advanced.stages.network import CaveNetworkGenerator

    project = load_project_config(Path(__file__).parents[1] / "config/earth_short_single.toml")
    with patch.object(
        CaveNetworkGenerator, "generate", side_effect=RuntimeError("boundary")
    ) as generate:
        with pytest.raises(RuntimeError, match="boundary"):
            generate_sections(project)
    assert generate.call_args.kwargs["section_config"] is project.section_field


def test_source_edit_during_campaign_invalidates_overall_pass(tmp_path, monkeypatch):
    identities = iter(["before", "after"])
    monkeypatch.setattr(reliability, "package_source_hash", lambda: next(identities))
    monkeypatch.setattr(reliability, "run_isolated", lambda *args, **kwargs: {
        "status": "passed", "identity": {"network": "same"},
    })
    result = run_campaign([ReliabilityCase("config", 42)], tmp_path / "campaign")
    assert result["cases"][0]["replay_passed"]
    assert result["complete"] and not result["source_unchanged"] and not result["passed"]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_worker_timeout_rejected_before_creating_output(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        run_isolated(ReliabilityCase("config", 0), tmp_path / "case", timeout_s=timeout)
    assert not (tmp_path / "case").exists()


@pytest.mark.parametrize("limit", [-1, True, 0.5, "8192"])
def test_invalid_worker_memory_budget_rejected_before_launch(tmp_path, limit):
    with pytest.raises(ValueError, match="memory limit"):
        run_isolated(ReliabilityCase("config", 0), tmp_path / "case", timeout_s=1,
                     memory_limit_mib=limit)
    assert not (tmp_path / "case").exists()
