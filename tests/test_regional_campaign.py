"""Rejected regional cases must participate in the replay audit."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plume_advanced.stages.network_quality import NetworkQualityError, write_quality_report
from plume_advanced.stages.network_systems import GenerationDomainError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(monkeypatch):
    spec = importlib.util.spec_from_file_location("regional_campaign", ROOT / "scripts/evaluate_regional_networks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.HostFieldGenerator, "generate", lambda self: object())
    monkeypatch.setattr(module, "host_semantic_hash", lambda host: "unchanged")
    monkeypatch.setattr(module, "render_campaign_gallery", lambda root: None)
    return module


@pytest.mark.parametrize("mismatch", [False, True])
def test_campaign_replays_failures_without_promoting_them(tmp_path, monkeypatch, campaign, mismatch):
    module = campaign
    calls = []

    def generate(self, host, *, quality_report_path):
        calls.append(quality_report_path)
        report = dict(status="rejected", accepted=False, checks=[dict(
            name="fixture_clearance", passed=False, severity="error",
            value=int(mismatch and len(calls) > 1), limit=0, segment_ids=[1, 2])])
        write_quality_report(report, quality_report_path)
        raise NetworkQualityError(report)

    monkeypatch.setattr(module.CaveNetworkGenerator, "generate", generate)
    result = module.main(["--config", str(ROOT / "config/regional-network.toml"),
                          "--seeds", "2", "--layers", "1", "--replay", "--output", str(tmp_path)])
    receipt = json.loads((tmp_path / "campaign.json").read_text())
    assert len(calls) == 2
    assert result == 1 and receipt["failed"] == 1 and receipt["accepted"] == 0
    assert receipt["cases"][0]["replay_matches"] is (not mismatch)


@pytest.mark.parametrize("outcomes,matches,accepted", [
    (("accepted", "accepted"), True, True),
    (("accepted", "rejected"), False, False),
    (("rejected", "accepted"), False, False),
    (("domain", "domain"), True, False),
    (("rejected", "domain"), False, False),
    (("accepted", "different_report"), False, False),
])
def test_replay_audits_exactly_two_independent_outcomes(
    tmp_path, monkeypatch, campaign, outcomes, matches, accepted
):
    calls = []
    for name in ("export_network_artifact", "render_network_comparison", "render_network_morphology"):
        monkeypatch.setattr(campaign, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign, "assess_network", lambda n, host: dict(accepted=True))
    monkeypatch.setattr(campaign, "network_semantic_hash", lambda n: "same-network")

    def generate(self, host, *, quality_report_path):
        outcome = outcomes[len(calls)]
        calls.append(quality_report_path)
        if outcome == "domain":
            # An infeasible host can fail before any quality report exists.
            raise GenerationDomainError("no feasible exit")
        report = dict(accepted=outcome != "rejected", checks=[dict(
            name="fixture_clearance", passed=outcome != "rejected", severity="error",
            value=int(outcome == "different_report"))])
        write_quality_report(report, quality_report_path)
        if outcome == "rejected":
            raise NetworkQualityError(report)
        return SimpleNamespace(config=self.config, summary=lambda: {},
                               backend_provenance=dict(accepted_branches=1, requested_branches=1))

    monkeypatch.setattr(campaign.CaveNetworkGenerator, "generate", generate)
    result = campaign.main(["--config", str(ROOT / "config/regional-network.toml"),
                            "--seeds", "2", "--layers", "1", "--replay", "--output", str(tmp_path)])
    receipt = json.loads((tmp_path / "campaign.json").read_text())
    assert len(calls) == 2
    assert result == int(not accepted)
    assert receipt["cases"][0]["replay_matches"] is matches
    assert receipt["cases"][0]["accepted"] is accepted


def test_campaign_refuses_to_mix_old_evidence(tmp_path, campaign):
    old = tmp_path / "network.png"
    old.write_bytes(b"previous successful case")
    with pytest.raises(SystemExit, match="2"):
        campaign.main(["--output", str(tmp_path)])
    assert old.read_bytes() == b"previous successful case"
