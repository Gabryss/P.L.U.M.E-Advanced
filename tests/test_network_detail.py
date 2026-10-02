"""Contracts for refinement: bounded changes, replay, rollback and actual XYZ."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_payload,
    network_semantic_hash,
)
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_detail import (
    NetworkDetailConfig,
    adaptive_indices,
    refine_network,
)
from plume_advanced.stages.network_layers import segment_xyz
from plume_advanced.stages.network_quality import assess_network, shape_hash

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["regional-network", "regional-multilayer"])
def example(request):
    cfg = load_project_config(ROOT / f"config/{request.param}.toml")
    host = HostFieldGenerator(cfg.host_field).generate()
    coarse = CaveNetworkGenerator(cfg.network).generate(host)
    enabled = replace(coarse, config=replace(coarse.config, detail=NetworkDetailConfig(enabled=True)))
    refined = refine_network(CaveNetworkGenerator(enabled.config), host, enabled)
    return host, coarse, enabled, refined


@pytest.mark.parametrize("field,value", [
    ("enabled", 1), ("strength", True), ("strength", -1), ("strength", 1.1),
    ("strength", float("nan")), ("feature_scale_m", 0), ("feature_scale_m", 201),
    ("sampling_error_m", 0), ("sampling_error_m", .2), ("maximum_samples", True),
    ("maximum_samples", 99), ("maximum_samples", 1e6),
])
def test_invalid_controls(field, value):
    with pytest.raises(ValueError, match="network.detail"):
        NetworkDetailConfig(**{field: value})


def test_adaptive_error_and_spacing_bound_with_protected_points():
    s = np.linspace(0, 100, 2001)
    values = np.column_stack((s, np.sin(s/4), .02*s, 4+.3*np.cos(s/7)))
    keep = adaptive_indices(s, values, .01, 2., mandatory=[87, 101, 923])
    rebuilt = np.column_stack([np.interp(s, s[keep], c[keep]) for c in values.T])
    assert np.max(abs(rebuilt-values)) <= .01 + 1e-12
    assert max(np.diff(s[keep])) <= 2.
    assert {87, 101, 923} <= set(keep)
    assert len(keep) < len(s)/4


def test_optional_and_zero_strength_preserve_geometry(example):
    host, coarse, enabled, _ = example
    assert refine_network(CaveNetworkGenerator(coarse.config), host, coarse) is coarse
    zero = replace(enabled, config=replace(enabled.config, detail=replace(enabled.config.detail, strength=0)))
    output = refine_network(CaveNetworkGenerator(zero.config), host, zero)
    assert shape_hash(output) == shape_hash(coarse)
    assert output.backend_provenance["detail"]["skip_reason"] == "zero_strength"


def test_refinement_preserves_graph_anchors_flux_and_host(example):
    host, coarse, enabled, refined = example
    assert assess_network(refined, host)["accepted"]
    assert coarse.nodes == refined.nodes
    assert shape_hash(refined) != shape_hash(coarse)
    assert refined.max_flow_conservation_error() < 1e-8
    assert refined.backend_provenance["detail"]["graph_preserved"]
    assert refined.backend_provenance["detail"]["final_samples"] <= enabled.config.detail.maximum_samples
    assert refined.quality_report["selected_shape_sha256"] == shape_hash(refined)
    for a, b in zip(coarse.segments, refined.segments):
        assert (a.segment_id, a.start_node_id, a.end_node_id, a.kind, a.z_level) == (
            b.segment_id, b.start_node_id, b.end_node_id, b.kind, b.z_level)
        np.testing.assert_allclose(segment_xyz(a, coarse.config.layers)[[0, -1]],
                                   segment_xyz(b, refined.config.layers)[[0, -1]], atol=1e-10)
        assert a.points[0].width == b.points[0].width
        assert a.points[-1].width == b.points[-1].width
        assert a.points[0].flux == b.points[0].flux
        assert a.metadata["regional_capacity"] == b.metadata["regional_capacity"]
        for source, target in ((a.points[:2], b.points[:2]), (a.points[-2:], b.points[-2:])):
            da = np.array([source[1].x-source[0].x, source[1].y-source[0].y])
            db = np.array([target[1].x-target[0].x, target[1].y-target[0].y])
            np.testing.assert_allclose(da/np.linalg.norm(da), db/np.linalg.norm(db), atol=1e-9)
        assert all(abs(p.elevation-host.sample(p.x, p.y).elevation) < 1e-9 for p in b.points)
    original_host = host_semantic_hash(host)
    original_network = network_semantic_hash(enabled)
    replay = refine_network(CaveNetworkGenerator(enabled.config), host, enabled)
    assert host_semantic_hash(host) == original_host
    assert network_semantic_hash(enabled) == original_network
    assert network_semantic_hash(replay) == network_semantic_hash(refined)
    assert refine_network(CaveNetworkGenerator(enabled.config), host, replay) is replay


def test_rejected_edits_keep_valid_baseline_and_are_bounded(example, monkeypatch):
    import plume_advanced.stages.network_detail as detail

    host, coarse, enabled, _ = example
    baseline = shape_hash(coarse)
    original = detail.assess_network

    def reject_changes(network, host):
        report = original(network, host)
        if shape_hash(network) != baseline:
            report.update(accepted=False, checks=[dict(name="injected_collision", passed=False)])
        return report

    monkeypatch.setattr(detail, "assess_network", reject_changes)
    output = refine_network(CaveNetworkGenerator(enabled.config), host, enabled)
    assert shape_hash(output) == baseline
    report = output.backend_provenance["detail"]
    assert report["accepted_edits"] == 0 and report["status"] == "unchanged"
    assert all(len(edit["trials"]) <= 3 for edit in report["edits"])
    assert any(edit["trials"] for edit in report["edits"])


def test_budget_exhaustion_retains_output(example):
    host, coarse, enabled, _ = example
    limited = replace(enabled, config=replace(enabled.config, detail=replace(enabled.config.detail, maximum_samples=100)))
    output = refine_network(CaveNetworkGenerator(limited.config), host, limited)
    assert shape_hash(output) == shape_hash(coarse)
    assert output.backend_provenance["detail"]["skip_reason"] == "baseline_exceeds_sample_budget"


def test_refinement_cannot_silently_ignore_changed_settings(example):
    host, _, _, refined = example
    changed = replace(refined, config=replace(refined.config, detail=replace(refined.config.detail, strength=.7)))
    with pytest.raises(ValueError, match="original coarse"):
        refine_network(CaveNetworkGenerator(changed.config), host, changed)


def test_final_rebuild_failure_rolls_back_the_entire_stage(example, monkeypatch):
    import plume_advanced.stages.network_acceptance as acceptance

    host, coarse, enabled, _ = example

    def corrupt(generator, host, network, segments):
        broken = replace(segments[0], points=tuple(replace(p, width=1000.) for p in segments[0].points))
        return replace(network, segments=(broken, *segments[1:]))

    monkeypatch.setattr(acceptance, "rebuild_network_geometry", corrupt)
    output = refine_network(CaveNetworkGenerator(enabled.config), host, enabled)
    assert shape_hash(output) == shape_hash(coarse)
    assert output.backend_provenance["detail"]["status"] == "rolled_back"
    assert output.quality_report["final_assessment"]["accepted"]


def test_uniform_host_gains_bounded_seeded_plan_variety(example):
    from plume_advanced.stages.network_detail import _proposal

    host, _, enabled, _ = example
    segment = max(enabled.segments, key=lambda s: s.total_length)
    a, b = segment.points[0], segment.points[-1]
    distance = np.hypot(b.x-a.x, b.y-a.y)
    points = tuple(replace(a, index=i, x=a.x+t*(b.x-a.x), y=a.y+t*(b.y-a.y), arc_length=t*distance)
                   for i, t in enumerate(np.linspace(0, 1, 81)))
    straight = replace(segment, points=points)
    flat = replace(host, growth_cost=np.ones_like(host.growth_cost))
    proposal, _ = _proposal(straight, enabled.config, flat)
    assert proposal is not None
    displacement = np.linalg.norm(proposal[4][:, :2], axis=1)
    assert displacement.max() > .1
    assert displacement.max() <= 1.25*straight.mean_width*enabled.config.detail.strength
    assert np.mean(displacement == 0) > .25
    assert displacement[0] == displacement[-1] == 0
    replay, _ = _proposal(straight, enabled.config, flat)
    np.testing.assert_array_equal(proposal[4], replay[4])


def test_actual_layer_offsets_are_exported_hashed_and_inspected(example):
    host, _, _, refined = example
    if not refined.config.layers.enabled:
        return
    segment = refined.segments[0]
    invalid = replace(segment, metadata=dict(segment.metadata,
                      network_detail_vertical_offsets_m=[50.]*len(segment.points)))
    bad = replace(refined, segments=(invalid, *refined.segments[1:]))
    assert shape_hash(bad) != shape_hash(refined)
    assert not assess_network(bad, host)["accepted"]
    payload = network_payload(refined)
    for s, p in zip(sorted(refined.segments, key=lambda x: x.segment_id), payload["segments"]):
        np.testing.assert_array_equal(p["centerline_xyz_m"], segment_xyz(s, refined.config.layers))
    malformed = replace(segment, metadata=dict(segment.metadata, network_detail_vertical_offsets_m=[0.]))
    assert not np.isfinite(segment_xyz(malformed, refined.config.layers)).all()


def test_single_layer_repair_receives_finite_elevations(example):
    from plume_advanced.stages.network_local_repair import _Corridor

    host, coarse, _, _ = example
    if coarse.config.layers.enabled:
        return
    segment = coarse.segments[0]
    xy = np.array([(p.x, p.y) for p in segment.points])
    corridor = _Corridor(coarse, host, segment, xy, 0, len(xy)-1, segment.mean_width)
    np.testing.assert_array_equal(corridor.xyz[:, 2], [p.elevation for p in segment.points])
    np.testing.assert_array_equal(corridor.depths, np.zeros(len(xy)))


def test_generation_integrates_refinement_and_final_quality_receipt(example, tmp_path):
    host, _, enabled, refined = example
    generated = CaveNetworkGenerator(enabled.config).generate(host, quality_report_path=tmp_path / "quality.json")
    assert network_semantic_hash(generated) == network_semantic_hash(refined)
    assert generated.quality_report["selected_shape_sha256"] == shape_hash(generated)
    assert generated.quality_report["final_assessment"]["accepted"]
    assert generated.quality_report["detail"]["baseline_shape_sha256"] != shape_hash(generated)


def test_unknown_or_unsupported_detail_settings_fail_early(tmp_path):
    for line in ('enabled = true\nmisspelled = 1', 'enabled = true'):
        path = tmp_path / "config.toml"
        path.write_text('recipe_version = 1\npreset = "short-multi"\n[network.detail]\n'+line)
        with pytest.raises(ValueError, match="detail"):
            load_project_config(path)


def test_accepted_features_leave_original_quiet_vertices_exact(example):
    _, coarse, _, refined = example
    report = refined.backend_provenance['detail']
    for edit in report['edits']:
        if not edit['accepted']:
            continue
        original = next(s for s in coarse.segments if s.segment_id == edit['segment_id'])
        target = next(s for s in refined.segments if s.segment_id == original.segment_id)
        features = edit['proposal']['features']
        for p in original.points:
            if any(f['start_m'] < p.arc_length < f['end_m'] for f in features):
                continue
            nearest = min(target.points, key=lambda q: (q.x-p.x)**2+(q.y-p.y)**2)
            np.testing.assert_allclose([nearest.x, nearest.y, nearest.width], [p.x, p.y, p.width], atol=1e-10)


def test_plan_width_and_burial_have_one_feature_catalogue(example):
    from plume_advanced.stages.network_detail import _proposal

    host, _, enabled, _ = example
    segment = max(enabled.segments, key=lambda s: s.total_length)
    proposal, audit = _proposal(segment, enabled.config, host)
    assert proposal is not None
    station, _, _, _, delta, _ = proposal
    active = np.zeros(len(station), dtype=bool)
    for feature in audit['features']:
        active |= (station > feature['start_m']) & (station < feature['end_m'])
    assert np.all(delta[~active] == 0)
    assert np.all(delta[:, 2] <= 0)
    assert not np.any(delta[:, 2]) if not enabled.config.layers.enabled else True
