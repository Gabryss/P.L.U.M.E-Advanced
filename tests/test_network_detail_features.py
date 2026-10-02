"""Sparse feature placement, host response and absence of per-vertex noise."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.stages.network import CavePoint, CaveSegment
from plume_advanced.stages.network_detail_features import (
    PassageFeature,
    passage_scale,
    plan_features,
)


class UniformHost:
    def sample(self, x, y):
        return SimpleNamespace(growth_cost=.5, roof_competence=.7)


class SlopingCostHost:
    def __init__(self, sign):
        self.sign = sign

    def sample(self, x, y):
        return SimpleNamespace(growth_cost=.5+self.sign*.015*y, roof_competence=.7)


@pytest.fixture
def route():
    config = load_project_config('config/detailed-network.toml').network
    # The feature planner only consumes route identity, role and conserved flux.
    point = CavePoint(0, 0., 0., 0., 0., 10., .7, .5, 0., 5., 1., 1200., 0.)
    segment = CaveSegment(12, 0, 1, 'trunk', 0, (point, replace(point, x=1000., arc_length=1000.)),
                          {'regional_route_type': 'retained_trunk'})
    arc = np.array([0., 1000.])
    xy = np.column_stack((arc, np.zeros_like(arc)))
    return segment, config, arc, xy


def test_profile_is_compact_asymmetric_and_query_independent():
    feature = PassageFeature(10., 23., 60., 1., .1, 0., 0., 0., 'pocket')
    stations = np.linspace(-20, 100, 1201)
    values = feature.profile(stations)
    assert np.all(values[(stations <= 10) | (stations >= 60)] == 0)
    assert values.max() == 1
    assert np.array_equal(values[::7], feature.profile(stations[::7]))
    assert np.array_equal(values[::-1], feature.profile(stations[::-1]))
    assert feature.profile([20])[0] != feature.profile([26])[0]
    # First derivative approaches zero at both support boundaries and the peak.
    for anchor in (10., 23., 60.):
        h = 1e-4
        assert abs((feature.profile([anchor+h])[0]-feature.profile([anchor-h])[0])/(2*h)) < 1e-7


def test_sparse_support_quiet_gaps_replay_and_varied_spacing(route):
    segment, config, arc, xy = route
    features, scale = plan_features(segment, config, UniformHost(), arc, xy, 5., 10.)
    assert len(features) >= 3
    assert features == plan_features(segment, config, UniformHost(), arc, xy, 5., 10.)[0]
    assert features != plan_features(segment, replace(config, random_seed=99), UniformHost(), arc, xy, 5., 10.)[0]
    assert sum(f.end_m-f.start_m for f in features) < .75*arc[-1]
    assert all(b.start_m-a.end_m >= .45*scale for a, b in zip(features, features[1:]))
    assert np.std([f.end_m-f.start_m for f in features]) > 1
    assert np.std(np.diff([f.peak_m for f in features])) > 1
    assert all(f.burial_m == 0 for f in features)
    assert all(abs(f.lateral_m) <= .25*5*config.detail.strength for f in features)


def test_feature_catalogue_ignores_collinear_resampling(route):
    segment, config, arc, xy = route
    coarse = plan_features(segment, config, UniformHost(), arc, xy, 5., 10.)[0]
    fine_arc = np.sort(np.r_[arc, np.linspace(.1, 999, 230)])
    fine = plan_features(segment, config, UniformHost(), fine_arc, np.c_[fine_arc, np.zeros_like(fine_arc)], 5., 10.)[0]
    assert coarse == fine


def test_host_cost_reversal_reverses_bends_not_the_random_catalogue(route):
    segment, config, arc, xy = route
    positive = plan_features(segment, config, SlopingCostHost(1), arc, xy, 5., 10.)[0]
    negative = plan_features(segment, config, SlopingCostHost(-1), arc, xy, 5., 10.)[0]
    np.testing.assert_allclose([f.peak_m for f in positive], [f.peak_m for f in negative])
    assert all(f.lateral_m < 0 for f in positive)
    assert all(f.lateral_m > 0 for f in negative)


def test_passage_hierarchy_changes_feature_extent(route):
    segment, config, *_ = route
    branch = replace(segment, points=tuple(replace(p, flux=.1) for p in segment.points),
                     metadata={'regional_route_type': 'blind_branch'})
    assert passage_scale(segment, config, 2.) > passage_scale(branch, config, 2.)
    assert passage_scale(branch, config, 6.) >= 36.


def test_roof_weakening_couples_constriction_and_burial(route):
    segment, config, arc, xy = route
    class WeakRoof:
        def sample(self, x, y):
            # A convex competence field is weaker than both flanks at any peak.
            return SimpleNamespace(growth_cost=.5, roof_competence=.2+5e-7*(x-500)**2)
    features = plan_features(segment, config, WeakRoof(), arc, xy, 5., 10.)[0]
    assert all(f.burial_m > 0 and f.roof_anomaly < 0 for f in features)
    # A bounded synthetic sequence of weak patches tests the strong response.
    class PatchedRoof:
        def sample(self, x, y):
            return SimpleNamespace(growth_cost=.5, roof_competence=.8-.65*np.cos(2*np.pi*x/100)**2)
    features = plan_features(segment, config, PatchedRoof(), arc, xy, 5., 10.)[0]
    weak = [f for f in features if f.roof_anomaly < -.018]
    assert weak and all(f.width_fraction < 0 and f.burial_m > 0 for f in weak)


def test_short_reaches_do_not_force_a_feature(route):
    segment, config, *_ = route
    features, _ = plan_features(segment, config, UniformHost(), np.array([0., 30.]), np.array([[0.,0.],[30.,0.]]), 5., 10.)
    assert features == []


def test_strength_scales_amplitude_without_moving_feature_locations(route):
    segment, config, arc, xy = route
    weak = replace(config, detail=replace(config.detail, strength=.2))
    strong = replace(config, detail=replace(config.detail, strength=.8))
    a = plan_features(segment, weak, SlopingCostHost(1), arc, xy, 5., 10.)[0]
    b = plan_features(segment, strong, SlopingCostHost(1), arc, xy, 5., 10.)[0]
    assert len(a) == len(b) > 0
    np.testing.assert_array_equal([[f.start_m, f.peak_m, f.end_m] for f in a],
                                  [[f.start_m, f.peak_m, f.end_m] for f in b])
    np.testing.assert_allclose([[f.lateral_m, f.width_fraction, f.burial_m] for f in b],
                               np.array([[f.lateral_m, f.width_fraction, f.burial_m] for f in a])*4)
