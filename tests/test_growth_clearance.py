"""Branch construction must respect passages before final network screening."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_gallery_growth import breakout_has_clearance
from plume_advanced.stages.network_systems import (
    NetworkSystemsConfig,
    plan_interactions,
    preferred_tracks,
)


def points(xy):
    return tuple(SimpleNamespace(x=x, y=y, width=1.0) for x, y in xy)


@pytest.mark.parametrize("other", [
    [(5, -100), (5, 100)],  # Mid-chord crossing, remote endpoints.
    [(0, 1), (10, 1)],     # Parallel tube walls overlap.
    [(5, 0), (5, 0)],      # Degenerate existing chord is still an obstacle.
])
def test_branch_crossing_or_wall_overlap_is_rejected(other):
    branch = points([(x, 0) for x in range(11)])
    segment = SimpleNamespace(segment_id=2, points=points(other))
    assert not breakout_has_clearance(branch, [segment], parent_id=1)


def test_interval_margin_detects_contact_between_branch_samples():
    branch = points([(0, 0), (10, 0)])
    segment = SimpleNamespace(segment_id=2, points=points([(5, -10), (5, 10)]))
    assert not breakout_has_clearance(branch, [segment], parent_id=1)


def test_safe_branch_and_connected_parent_are_allowed():
    branch = points([(x, 0) for x in range(11)])
    parent = SimpleNamespace(segment_id=1, points=branch)
    other = SimpleNamespace(segment_id=2, points=points([(0, 4), (10, 4)]))
    assert breakout_has_clearance(branch, [parent, other], parent_id=1)
    assert not breakout_has_clearance((), [parent], parent_id=1)


@pytest.mark.parametrize("seed", [0, 2, 17, 4294967295])
@pytest.mark.parametrize("length", [400, 3000, 10000])
def test_trunk_preference_amplitude_stays_bounded_as_extent_grows(seed, length):
    config = CaveNetworkGenerator().config
    config = replace(config, random_seed=seed,
                     topology=replace(config.topology, style="trunk_dominated"),
                     systems=replace(config.systems, count=1, lateral_variation_widths=.75))
    gen = CaveNetworkGenerator(config)
    width = 2 * config.base_passage_radius
    host = SimpleNamespace(x_coords=np.linspace(-1000, 1000, 20),
                           y_coords=np.linspace(-1000, length+1000, 100),
                           growth_cost=np.zeros((100, 20)))
    geometry = SimpleNamespace(seed_x=0, seed_y=0, flow_x=0, flow_y=1,
                               cross_x=1, cross_y=0, cell_scale=3, along_extent=length)
    a, tracks, _ = preferred_tracks(gen, host, geometry)
    _, repeat, _ = preferred_tracks(gen, host, geometry)
    np.testing.assert_array_equal(tracks, repeat)
    assert tracks.shape == (1, len(a))
    assert np.max(np.abs(tracks)) <= .75 * width + 1e-10


def test_gallery_splits_form_local_islands_after_all_sources_merge():
    along = np.arange(101, dtype=float)
    stations = [0, 8, 20, 40, 60, 85, 100]
    tracks = np.array([
        np.interp(along, stations, row) for row in
        [[-2, -.1, -3, 0, -3, 0, 0], [0, 0, 3, 0, 0, 0, 0], [10, 10, 10, 0, 3, 0, 0]]
    ])
    config = NetworkSystemsConfig(
        count=3, source_spacing_widths=2, merge_distance_widths=.65,
        split_distance_widths=1.65, minimum_shared_length_widths=3,
        minimum_independent_length_widths=3, split_confirmation_widths=1,
        interaction_spacing_widths=1,
    )
    general_events, gallery_events = [], []
    plan_interactions(along, tracks, 1, config, events=general_events)
    plan_interactions(along, tracks, 1, config, events=gallery_events, simple_splits=True)
    assert any(e['kind'] == 'split' and e['before'] != [[0, 1, 2]] for e in general_events)
    splits = [e for e in gallery_events if e['kind'] == 'split']
    assert splits and all(e['before'] == [[0, 1, 2]] for e in splits)
    assert all(e['station_m'] > 30 for e in splits)


@pytest.mark.integration
@pytest.mark.parametrize("seed", [0, 2])
def test_previously_exhausted_long_multi_seeds_pass_production_screening(seed):
    from plume_advanced.config import load_project_config
    from plume_advanced.evaluation.artifacts import host_semantic_hash
    from plume_advanced.evaluation.experiments.common import for_seed
    from plume_advanced.stages.host_field import HostFieldGenerator

    config = for_seed(load_project_config(
        Path(__file__).parents[1] / "config/earth_long_multi.toml"), seed)
    host = HostFieldGenerator(config.host_field).generate()
    identity = host_semantic_hash(host)
    network = CaveNetworkGenerator(config.network).generate(host, section_config=config.section_field)
    assert network.quality_report["accepted"]
    assert network.quality_report["scope"] == "network_and_sections"
    assert host_semantic_hash(host) == identity
    assert all(check["passed"] for check in network.quality_report["attempts"][-1]["checks"])
