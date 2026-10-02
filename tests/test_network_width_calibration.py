"""Statistical calibration invariants, independent of external survey files."""

from dataclasses import replace

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.metrics.network_calibration import (
    common_footprint_support,
    fit_width_parameters,
    projected_width_series,
    spatial_scale,
    width_targets,
)
from plume_advanced.stages.network import CavePoint, CaveSegment
from plume_advanced.stages.network_morphology import width_profile
from plume_advanced.stages.network_regional_routing import RegionalGrowthConfig
from plume_advanced.stages.network_width_field import log_width_field


def test_width_field_queries_do_not_change_with_sampling_order_or_batch_size():
    xy = np.random.default_rng(91).normal(size=(2400, 2))*100
    original = log_width_field(xy, 117, 10.)
    assert np.array_equal(original, log_width_field(xy, 117, 10.))
    assert np.array_equal(original[::11], log_width_field(xy[::11], 117, 10.))
    assert np.array_equal(original[::-1], log_width_field(xy[::-1], 117, 10.))
    assert not np.array_equal(original, log_width_field(xy, 118, 10.))
    assert not np.array_equal(original, log_width_field(xy, 117, 10., layer=1))


def test_field_ensemble_has_requested_covariance_and_no_axis_preference():
    xy = np.array([[0., 0.], [8., 0.], [0., 8.], [80., 0.]])
    samples = np.stack([log_width_field(xy, seed, 8.) for seed in range(512)])
    assert np.max(abs(samples.mean(axis=0))) < .12
    assert np.allclose(samples.std(axis=0), 1., atol=.12)
    correlation = np.corrcoef(samples.T)[0]
    assert correlation[1:3] == pytest.approx(np.exp(-.5), abs=.1)
    assert abs(correlation[3]) < .13


@pytest.mark.parametrize("name,value", [("width_log_sigma", -.1), ("width_log_sigma", 1.1),
    ("width_log_sigma", float("nan")), ("width_log_sigma", True), ("width_correlation_m", 0),
    ("width_correlation_m", float("inf")), ("width_correlation_m", True)])
def test_invalid_width_controls_rejected(name, value):
    with pytest.raises(ValueError, match="network.regional"):
        RegionalGrowthConfig(**{name: value})


def test_disabling_width_field_preserves_existing_profile_exactly():
    cfg = load_project_config("config/detailed-network.toml").network
    cfg = replace(cfg, regional=replace(cfg.regional, hierarchy_strength=.8))
    p = CavePoint(0, 0., 0., 0., 0., 10., .7, .5, 0., 5., 1., 1200., 0.)
    segment = CaveSegment(12, 0, 1, "trunk", 0,
                          tuple(replace(p, x=float(x), arc_length=float(x)) for x in range(401)), {})
    expected = width_profile(cfg, segment, 1.)
    changed = replace(cfg, regional=replace(cfg.regional, width_log_sigma=0, width_correlation_m=.01))
    assert np.array_equal(expected, width_profile(changed, segment, 1.))
    active = replace(cfg, regional=replace(cfg.regional, width_log_sigma=.7, width_correlation_m=6.))
    widths = width_profile(active, segment, 1.)
    assert widths.min() >= 2*cfg.minimum_passage_radius - 1e-12
    assert widths.max() <= 1.9*cfg.maximum_passage_radius + 1e-12
    assert np.max(abs(np.diff(widths))) <= .9*cfg.quality.maximum_width_gradient + 1e-12
    assert np.array_equal(widths, width_profile(active, replace(segment, segment_id=999), 1.))


def test_caves_receive_equal_weight_regardless_of_section_counts():
    rows = [dict(cave_id="a", width_m=w) for w in (2, 4, 8)]
    rows += [dict(cave_id="b", width_m=w) for w in (4, 8, 16)]
    first = width_targets(rows)
    repeated = width_targets(rows+rows[:3]*20)
    assert first["representative_width_quantiles_m"] == repeated["representative_width_quantiles_m"]
    assert first["median_cave_width_m"] == 6
    assert first["relative_width_quantiles"] == repeated["relative_width_quantiles"]


def test_parallel_passages_are_not_measured_as_a_single_wide_chord():
    x, y = np.meshgrid(np.arange(0, 100, .2), np.arange(-2., 2., .2))
    xy = np.vstack((np.c_[x.ravel(), y.ravel()-6], np.c_[x.ravel(), y.ravel()+6]))
    series = projected_width_series(xy, resolution_m=.25)
    assert np.sum(series["intervals"] == 2) > 350
    assert not np.isfinite(series["widths"]).any()


def test_single_passage_chords_match_a_known_rectangle_and_ignore_rotation():
    x, y = np.meshgrid(np.arange(0, 100, .1), np.arange(-2., 2.01, .1))
    xy = np.c_[x.ravel(), y.ravel()]
    theta = .75
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    original = projected_width_series(xy)
    rotated = projected_width_series(xy@rotation)
    assert np.nanmedian(original["widths"]) == pytest.approx(4., abs=.26)
    assert np.nanmedian(rotated["widths"]) == pytest.approx(4., abs=.26)


def test_constant_footprint_does_not_invent_a_correlation_length():
    with pytest.raises(ValueError, match="no identifiable"):
        spatial_scale(dict(widths=np.full(200, 5.), resolution_m=.25))


def test_resolution_comparison_uses_common_reaches_and_never_fills_missing_data():
    stations = np.arange(100, dtype=float)
    first = 5.+np.sin(stations/10)
    second = first.copy()
    first[20:24] = np.nan
    second[50:55] = np.nan
    paired = common_footprint_support([
        dict(stations=stations, widths=first), dict(stations=stations, widths=second)])
    assert np.array_equal(np.isnan(paired[0]["widths"]), np.isnan(paired[1]["widths"]))
    assert np.isnan(paired[0]["widths"]).sum() == 9
    assert paired[0]["widths"][10] == first[10]


def test_spatial_pairs_never_cross_missing_or_multiple_passage_stations():
    widths = np.exp(.3*np.sin(np.arange(1000)/30))
    widths[::20] = np.nan
    report = spatial_scale(dict(widths=widths, resolution_m=1.))
    # Every block has 19 samples. Long lag pairs spanning a gap are excluded.
    twelve = next(row for row in report["variogram"] if row["lag_m"] == 12)
    assert twelve["pairs"] == 50*7


def test_bounded_fit_improves_central_quantiles_without_hiding_unreachable_tail():
    arc = np.arange(0., 400., 2.)
    fields = np.stack([log_width_field(np.c_[arc, np.zeros(len(arc))], s, 10.) for s in range(4)])
    targets = dict(median_cave_width_m=5., representative_width_quantiles_m=[1., 3.8, 5., 6.8, 15.])
    fitted = fit_width_parameters(targets, fields, arc, minimum_m=1., maximum_m=9.5, gradient=.54)
    best = fitted["best"]
    assert best["loss"] < .005
    assert best["predicted_quantiles_m"][-1] <= 9.5
    assert targets["representative_width_quantiles_m"][-1] == 15.
