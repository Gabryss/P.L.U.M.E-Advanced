"""Full generation must retain accepted layer depths and actual ramp profiles."""
from dataclasses import replace

import numpy as np
import pytest

from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNode,
    CavePoint,
    CaveSegment,
)
from plume_advanced.stages.network_layers import NetworkLayersConfig, segment_xyz
from plume_advanced.stages.network_quality import NetworkQualityConfig, assess_sections
from plume_advanced.stages.section_field import SectionFieldConfig, SectionFieldGenerator


def stacked_network(ramp=False):
    controls = NetworkLayersConfig(enabled=True, spacing_m=8.)
    configs = CaveNetworkConfig(random_seed=71, layers=controls, quality=NetworkQualityConfig(enabled=False))
    # Identical plan projections deliberately test separation by true elevation.
    segments, nodes = [], []
    for sid in range(2):
        points = tuple(CavePoint(i, x, 0., 100., 0., 60., .8, .2, x, 6., 1., 1400., x)
                       for i, x in enumerate(np.linspace(0, 80., 41)))
        meta = dict(regional_start_layer=sid, regional_end_layer=sid, regional_layer_depths_m=[4.5, 12.5])
        segments.append(CaveSegment(sid, sid*2, sid*2+1, 'trunk', sid, points, meta))
        nodes += [CaveNode(sid*2, 0., 0., 0., 0., 'entry'), CaveNode(sid*2+1, 80., 0., 80., 0., 'exit')]
    if ramp:
        segments[0] = replace(segments[0], metadata=dict(segments[0].metadata, regional_end_layer=1))
    return CaveNetwork(configs, tuple(nodes), tuple(segments), (), np.zeros((2, 2)), np.zeros((2, 2)),
                       (0, 1), (), (), ())


def sections(network):
    return SectionFieldGenerator(SectionFieldConfig(random_seed=99, minimum_roof_thickness=3.,
        minimum_sample_spacing=3., maximum_sample_spacing=8., centerline_wobble_amplitude=5.)).generate(network)


def test_section_centres_follow_layers_even_with_legacy_wobble_enabled():
    network = stacked_network()
    result = sections(network)
    assert result == sections(network)
    for segment, field in zip(network.segments, result.segment_fields):
        np.testing.assert_allclose([s.z for s in field.samples], 100.-network.config.layers.depth(segment.z_level))
        assert all(s.y == 0 for s in field.samples)
        assert {p.arc_length for p in segment.points} <= {s.segment_arc_length for s in field.samples}
    checks = assess_sections(network, result)
    assert checks['accepted'], [c for c in checks['checks'] if not c['passed']]


def test_sections_preserve_the_entire_descending_ramp_and_frames():
    network = stacked_network(ramp=True)
    result = sections(network)
    segment, field = network.segments[0], result.segment_fields[0]
    reference = segment_xyz(segment, network.config.layers)
    for sample in field.samples:
        z = np.interp(sample.segment_arc_length, [p.arc_length for p in segment.points], reference[:, 2])
        assert sample.z == pytest.approx(z, abs=1e-10)
        frame = np.array([sample.tangent, sample.normal, sample.binormal])
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-10)
        assert sample.binormal[2] > 0
    assert field.samples[-1].z == pytest.approx(87.5)
    assert min(s.tangent[2] for s in field.samples) < -.1


@pytest.mark.parametrize('corruption,failed', [
    ('flatten', 'layer_section_centerline'), ('enlarge', 'layer_section_envelope'),
    ('roof', 'layer_section_roof')])
def test_inspection_rejects_flattening_oversize_or_insufficient_roof(corruption, failed):
    network = stacked_network()
    result = sections(network)
    field = result.segment_fields[1]
    bad = field.samples[1]
    if corruption == 'flatten':
        bad = replace(bad, z=result.segment_fields[0].samples[1].z)
    elif corruption == 'enlarge':
        bad = replace(bad, profile_points=tuple((x*10, y*10) for x, y in bad.profile_points))
    else:
        bad = replace(bad, roof_thickness=0.)
    field = replace(field, samples=(field.samples[0], bad, *field.samples[2:]))
    result = replace(result, segment_fields=(result.segment_fields[0], field))
    checks = {c['name']: c['passed'] for c in assess_sections(network, result)['checks']}
    assert not checks[failed]


def test_voxel_stamping_keeps_rock_between_overlapping_layers():
    from plume_advanced.stages.geometry import GeometryGenerator
    from plume_advanced.stages.geometry_types import GeometryConfig
    network = stacked_network()
    result = sections(network)
    g = GeometryGenerator(GeometryConfig(voxel_size=.3, storage_mode='dense', density_margin=1.,
        wall_roughness_amplitude=0., density_closing_voxels=0, cave_displacement_scale_m=0.))
    grid = g._build_voxel_grid({f.segment_id:f.samples for f in result.segment_fields},network,None)
    assert grid.sample_density((40.,0.,95.5)) > grid.iso_level
    assert grid.sample_density((40.,0.,87.5)) > grid.iso_level
    assert grid.sample_density((40.,0.,91.5)) < grid.iso_level
    assert grid.component_count == 2


def test_ramp_meets_lower_passage_at_the_same_floor_without_moving_centres():
    network = stacked_network(ramp=True)
    first, second = network.segments
    second = replace(second, start_node_id=first.end_node_id,
                     points=tuple(replace(p,x=p.x+80) for p in second.points))
    network = replace(network,segments=(first,second))
    result = sections(network)
    arrival, departure = result.segment_fields[0].samples[-1],result.segment_fields[1].samples[0]
    assert (arrival.x,arrival.y,arrival.z) == (departure.x,departure.y,departure.z) == (80.,0.,87.5)
    assert arrival.floor_world_z == pytest.approx(departure.floor_world_z, abs=1e-10)
    assert arrival.roof_world_z == pytest.approx(departure.roof_world_z, abs=1e-10)


def test_voxel_ramp_is_one_connected_cavity_into_the_lower_layer():
    from plume_advanced.stages.geometry import GeometryGenerator
    from plume_advanced.stages.geometry_types import GeometryConfig
    network = stacked_network(ramp=True)
    first, second = network.segments
    second = replace(second, start_node_id=first.end_node_id,
                     points=tuple(replace(p,x=p.x+80) for p in second.points))
    network = replace(network,segments=(first,second))
    result = sections(network)
    g = GeometryGenerator(GeometryConfig(voxel_size=.3, storage_mode='dense', density_margin=1.,
        wall_roughness_amplitude=0., density_closing_voxels=0, cave_displacement_scale_m=0.))
    grid = g._build_voxel_grid({f.segment_id:f.samples for f in result.segment_fields},network,None)
    assert grid.component_count == 1
    for field in result.segment_fields:
        for sample in field.samples[1:-1]:
            assert grid.sample_density((sample.x,sample.y,sample.z)) > grid.iso_level


def test_missing_layer_layout_cannot_fall_back_to_legacy_depths():
    network = stacked_network()
    first = replace(network.segments[0],metadata={})
    with pytest.raises(ValueError, match='finite accepted XYZ'):
        sections(replace(network,segments=(first,network.segments[1])))
