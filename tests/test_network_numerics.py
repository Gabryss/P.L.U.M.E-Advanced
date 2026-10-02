"""Branch decisions must not amplify CPU rounding into different topology."""

import math
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkConfig, CaveNetworkGenerator
from plume_advanced.stages.network_quality import assess_network

ROOT = Path(__file__).resolve().parents[1]


def _fma(a, b, c):
    # Python 3.12 has no math.fma. Exact rationals emulate the same single
    # rounding without requiring an AVX-512 runner or a platform C library.
    if hasattr(math, "fma"):
        return math.fma(a, b, c)
    return float(Fraction(a) * Fraction(b) + Fraction(c))


@contextmanager
def _vector_arithmetic(fused):
    original_dot, original_norm = np.dot, np.linalg.norm

    def dot(a, b, *args, **kwargs):
        if np.shape(a) == (2,) and np.shape(b) == (2,) and not args and not kwargs:
            x, y, z = float(a[0]), float(b[0]), float(a[1]) * float(b[1])
            return _fma(x, y, z) if fused else x * y + z
        return original_dot(a, b, *args, **kwargs)

    def norm(a, *args, **kwargs):
        if np.shape(a) == (2,) and not args and not kwargs:
            return math.sqrt(dot(a, a))
        return original_norm(a, *args, **kwargs)

    with patch.object(np, "dot", dot), patch.object(np.linalg, "norm", norm):
        yield


def test_arithmetic_fixture_exercises_distinct_rounding():
    a, b = np.array([1 + 2**-27, -1.]), np.array([1 - 2**-27, 1.])
    with _vector_arithmetic(False):
        assert np.dot(a, b) == 0.
    with _vector_arithmetic(True):
        assert np.dot(a, b) == -(2**-54)


@pytest.mark.parametrize("values", [[], [0.2], [0.2, 0.2, 0.2], [0., 0., 0.5, 1.]])
def test_driver_ranks_share_ties_and_ignore_candidate_order(values):
    values = np.array(values)
    rank = CaveNetworkGenerator._rank_breakout_driver
    result = rank(values)
    np.testing.assert_array_equal(rank(values[::-1]), result[::-1])
    for value in values:
        assert np.ptp(result[values == value]) == 0
    if len(values) > 1:
        assert np.all((result >= 0) & (result <= 1))


def test_driver_ranks_ignore_roundoff_but_keep_resolved_variation():
    # These repeated grid-bend curvatures differed only in the last few bits
    # in the CI seed-2 case. Ordinal ranks turned those bits into branch bias.
    values = np.array([0., 0., 0.025270787198668376, 0.025270787198668654,
                       0.1422430953789685, 0.14224309537896884, 1.])
    rank = CaveNetworkGenerator._rank_breakout_driver
    expected = np.array([.5, .5, 2.5, 2.5, 4.5, 4.5, 6.]) / 6
    np.testing.assert_array_equal(rank(values), expected)
    np.testing.assert_array_equal(rank(values + np.array([1, 0, -1, 1, -1, 1, 0]) * 2e-16), expected)
    changed = values.copy()
    changed[3] += 1e-8
    assert rank(changed)[3] > rank(changed)[2]


def _case(name):
    if name == "natural_seed_2":
        host = replace(HostFieldConfig(), random_seed=2,
                       grid=GridConfig(width=1800., height=1400., nx=60, ny=48),
                       target_route_length_m=1000., seed_point=(0., 0.), flow_angle_degrees=0.)
        network = CaveNetworkConfig(random_seed=2, target_route_length_m=1000.,
                                    source_count=2, trace_max_steps=140, network_density=1.,
                                    downflow_ensemble_size=4, flowy_timeout_s=30.)
        return host, network
    project = load_project_config(ROOT / "config/research.toml")
    return project.host_field, replace(project.network, random_seed=5)


def test_insufficient_branch_supply_retires_lobes():
    # Exercise starvation deliberately instead of requiring this optional
    # event to appear by chance in the default research seed.
    project = load_project_config(ROOT / "config/research.toml")
    config = replace(project.network, emplacement_history=replace(
        project.network.emplacement_history, retirement_flux_threshold=.9))
    host = HostFieldGenerator(project.host_field).generate()
    network = CaveNetworkGenerator(config).generate(host)
    branches = [s for s in network.segments if s.metadata.get("lobe_path_id")]
    assert branches
    for segment in branches:
        assert segment.metadata["initial_flux"] < .9 * config.source_flux
        assert segment.kind == "abandoned_lobe"
        assert segment.metadata["formation_state"] == "flux_starved_retired"
        assert segment.metadata["event_type"] == "retired"
        assert not segment.metadata["coalesced"]
    assert network.summary()["flux_starved_retired_count"] > 0
    assert network.summary()["max_phase_budget_utilization"] <= 1. + 1e-9


@pytest.mark.parametrize("case", ["natural_seed_2", "research_seed_5"])
def test_ci_failure_seeds_accept_same_geometry_with_fused_arithmetic(case):
    host_config, config = _case(case)
    host = HostFieldGenerator(host_config).generate()
    networks = []
    for fused in (False, True):
        with _vector_arithmetic(fused):
            network = CaveNetworkGenerator(config).generate(host)
            assert assess_network(network, host)["accepted"]
            networks.append(network)
    first, second = networks
    for key in ("selected_attempt", "selected_seed", "selected_repair_pass"):
        assert first.quality_report[key] == second.quality_report[key]
    assert first.dominant_route_node_ids == second.dominant_route_node_ids
    assert len(first.segments) == len(second.segments)
    for a, b in zip(first.segments, second.segments):
        assert (a.segment_id, a.start_node_id, a.end_node_id, a.kind, a.z_level) == (
            b.segment_id, b.start_node_id, b.end_node_id, b.kind, b.z_level)
        np.testing.assert_allclose(
            [[p.x, p.y, p.elevation, p.width, p.flux] for p in a.points],
            [[p.x, p.y, p.elevation, p.width, p.flux] for p in b.points], rtol=0., atol=1e-9)
