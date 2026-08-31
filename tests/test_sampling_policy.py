from dataclasses import replace

import numpy as np

from plume_advanced.evaluation.metrics.contours import self_intersection_count
from plume_advanced.stages.host_field import RoutingWeights
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNode,
    CavePoint,
    CaveSegment,
)
from plume_advanced.stages.section_field import SectionFieldConfig, SectionFieldGenerator


def _straight_network() -> CaveNetwork:
    points = tuple(
        CavePoint(index, float(index * 10), 0.0, 20.0, 0.0, 20.0, 0.8, 0.2, float(index * 10), 4.0)
        for index in range(11)
    )
    segment = CaveSegment(0, 0, 1, "backbone", 0, points, {})
    return CaveNetwork(
        CaveNetworkConfig(),
        (CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"), CaveNode(1, 100.0, 0.0, 100.0, 0.0, "exit")),
        (segment,),
        (),
        np.zeros((2, 2), dtype=bool),
        np.zeros((2, 2)),
        (0, 1),
        (),
        (),
        (),
    )


def test_routing_ablation_renormalizes_without_changing_legacy_defaults() -> None:
    weights = RoutingWeights()
    assert weights.resolved() == {
        "slope": 0.12,
        "cover": 0.10,
        "fracture": 0.22,
        "capacity": 0.28,
        "stability": 0.28,
    }
    ablated = weights.without("fracture").resolved()
    assert ablated["fracture"] == 0.0
    assert sum(ablated.values()) == 1.0


def test_sampling_policies_share_profile_state_at_common_positions() -> None:
    network = _straight_network()
    base = SectionFieldConfig(
        random_seed=9,
        minimum_sample_spacing=10.0,
        maximum_sample_spacing=10.0,
        uniform_sample_spacing=10.0,
        reference_sample_spacing=5.0,
    )
    adaptive = SectionFieldGenerator(base).generate(network)
    uniform = SectionFieldGenerator(replace(base, sampling_policy="uniform")).generate(network)
    reference = SectionFieldGenerator(replace(base, sampling_policy="reference")).generate(network)

    assert adaptive.segment_fields == uniform.segment_fields
    assert adaptive.dominant_route_segment_ids == uniform.dominant_route_segment_ids
    assert reference.summary()["sample_count"] > adaptive.summary()["sample_count"]
    reference_by_s = {
        sample.segment_arc_length: sample for sample in reference.segment_fields[0].samples
    }
    for sample in adaptive.segment_fields[0].samples:
        dense = reference_by_s[sample.segment_arc_length]
        assert sample.tube_width == dense.tube_width
        assert sample.tube_height == dense.tube_height
        assert sample.floor_flatness == dense.floor_flatness
        assert sample.roof_arch == dense.roof_arch
        assert sample.lateral_skew == dense.lateral_skew


def test_realism_profiles_are_varied_finite_and_simple() -> None:
    sections = SectionFieldGenerator(
        SectionFieldConfig(
            random_seed=17,
            minimum_sample_spacing=5.0,
            maximum_sample_spacing=5.0,
        )
    ).generate(_straight_network())
    samples = sections.segment_fields[0].samples
    widths = np.asarray([sample.tube_width for sample in samples])
    heights = np.asarray([sample.tube_height for sample in samples])

    assert np.ptp(widths) > 0.25
    assert np.ptp(heights) > 0.25
    for sample in samples:
        profile = np.asarray(sample.profile_points)
        assert np.isfinite(profile).all()
        assert np.allclose(profile[0], profile[-1])
        assert self_intersection_count(profile) == 0
