"""Physical envelope, compulsory collapse, and preserved morphology contracts."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import export_geometry_report
from plume_advanced.stability import RoofStabilityModel
from plume_advanced.stages.events import GeologicalEventConfig, GeologicalEventField
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.network import CaveNetwork, CaveNetworkConfig
from plume_advanced.stages.network_quality import NetworkQualityConfig
from plume_advanced.stages.section_field import (
    SectionField,
    SectionFieldConfig,
    SectionFieldGenerator,
    SectionSample,
    SegmentSectionField,
)

ROOT = Path(__file__).resolve().parents[1]


def test_width_and_height_are_one_coupled_failure_envelope() -> None:
    model = RoofStabilityModel()
    baseline = model.assess(width_m=20, height_m=6, floor_depth_m=15)
    wider = model.assess(width_m=40, height_m=6, floor_depth_m=15)
    taller = model.assess(width_m=20, height_m=10, floor_depth_m=15)
    assert wider.demand_ratio == pytest.approx(4 * baseline.demand_ratio)
    assert taller.maximum_width_m < baseline.maximum_width_m
    assert wider.maximum_height_m < baseline.maximum_height_m
    at_width = model.assess(width_m=baseline.maximum_width_m, height_m=6, floor_depth_m=15)
    assert not at_width.failed
    assert model.assess(width_m=baseline.maximum_width_m * 1.001, height_m=6, floor_depth_m=15).failed
    assert not model.assess(width_m=20, height_m=baseline.maximum_height_m, floor_depth_m=15).failed
    assert model.assess(width_m=20, height_m=baseline.maximum_height_m + .001, floor_depth_m=15).failed


def test_lower_gravity_and_stronger_rock_raise_the_limits() -> None:
    earth = RoofStabilityModel()
    lunar = replace(earth, gravity_m_s2=1.62)
    strong = replace(earth, effective_tensile_strength_pa=6_000_000)
    args = dict(width_m=20., height_m=6., floor_depth_m=10.)
    e, m, s = [model.assess(**args) for model in (earth, lunar, strong)]
    assert e.failed and not m.failed and not s.failed
    assert m.maximum_width_m / e.maximum_width_m == pytest.approx(np.sqrt(9.80665 / 1.62))
    assert m.maximum_height_m > e.maximum_height_m
    assert s.maximum_width_m / e.maximum_width_m == pytest.approx(np.sqrt(2))
    assert earth.assess(width_m=2, height_m=8, floor_depth_m=7).failed


@pytest.mark.parametrize('field', ['gravity_m_s2', 'rock_density_kg_m3', 'effective_tensile_strength_pa', 'safety_factor'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_physics_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        RoofStabilityModel(**{field: value})


def _fixture(gravity: float = 9.80665) -> tuple[CaveNetwork, SectionField]:
    angle = np.linspace(0, 2 * np.pi, 41)
    profile = tuple(zip(10 * np.cos(angle), 3 * np.sin(angle)))
    samples = tuple(SectionSample(
        index=i, segment_id=0, segment_arc_length=i * 10., x=i * 10., y=0., z=0.,
        surface_z=4., cover_thickness=10., roof_thickness=999., centerline_depth=4.,
        tangent=(1., 0., 0.), normal=(0., 1., 0.), binormal=(0., 0., 1.),
        tube_width=20., tube_height=6., floor_flatness=.6, roof_arch=1., lateral_skew=0.,
        junction_blend_weight=0., junction_influences=(), profile_points=profile,
    ) for i in range(3))
    network = CaveNetwork(
        # This isolates roof mechanics using synthetic profiles without a graph.
        config=CaveNetworkConfig(quality=NetworkQualityConfig(enabled=False)),
        nodes=(), segments=(), junctions=(),
        occupancy=np.zeros((2, 2)), width_field=np.zeros((2, 2)),
        dominant_route_node_ids=(), slice_along_positions=(),
        slice_channel_counts=(), slice_visible_channel_counts=(),
    )
    field = SectionField(
        config=SectionFieldConfig(gravity_m_s2=gravity),
        segment_fields=(SegmentSectionField(0, (), samples),), dominant_route_segment_ids=(0,),
    )
    return network, field


@pytest.mark.parametrize('storage', ['dense', 'tiled'])
def test_failed_roof_collapses_even_without_optional_events(storage: str, tmp_path: Path) -> None:
    network, sections = _fixture()
    config = GeometryConfig(
        voxel_size=.5, storage_mode=storage, chunk_size=16, minimum_radius=.5,
        tunnel_radius_scale=1., wall_roughness_amplitude=0., density_margin=2.,
    )
    optional_events = GeologicalEventField(config=GeologicalEventConfig(enabled=False), events=())
    failed = GeometryGenerator(config).generate(network, sections, optional_events)
    lunar_sections = replace(sections, config=replace(sections.config, gravity_m_s2=1.62))
    intact = GeometryGenerator(config).generate(network, lunar_sections, optional_events)
    assert failed.summary()['stability_collapse_count'] == failed.stamped_sample_count
    assert intact.summary()['stability_collapse_count'] == 0
    assert failed.voxel_grid.sample_density((10., 0., 0.)) < 0
    assert intact.voxel_grid.sample_density((10., 0., 0.)) > 0
    assert not failed.assembled_faces  # complete failure may close the entire route
    assert intact.assembled_faces
    import json
    report = json.loads(export_geometry_report(failed, tmp_path / 'report.json').read_text())
    assert all(r['outcome'] == 'blocked_by_breakdown' for r in report['stability_records'])


def test_assessment_uses_actual_transformed_roof_not_stale_metadata() -> None:
    _, sections = _fixture()
    sample = sections.segment_fields[0].samples[0]
    updated = SectionFieldGenerator(sections.config)._assess_roof(sample)
    assert updated.roof_thickness == pytest.approx(1.)
    assert updated.floor_world_z == pytest.approx(-3.)
    assert updated.roof_world_z == pytest.approx(3.)
    assert updated.collapse_required


def test_geometry_scale_cannot_bypass_roof_stability() -> None:
    network, sections = _fixture(gravity=1.62)
    generator = GeometryGenerator(GeometryConfig(
        voxel_size=.5, minimum_radius=.5, wall_roughness_amplitude=0.,
        tunnel_radius_scale=2.,
    ))
    geometry = generator.build_base_volume(network, sections)
    records = [dict(r) for r in geometry.stability_records]
    assert all(r['failed'] and r['width_m'] == pytest.approx(40.) for r in records)
    assert geometry.voxel_grid.sample_density((10., 0., 0.)) < 0


def test_body_maximum_no_longer_sets_a_minimum_passage_size() -> None:
    configs = [load_project_config(ROOT / 'config/project.toml', world_body=b) for b in ('earth', 'mars', 'moon')]
    assert [c.section_field.minimum_tube_width for c in configs] == [.5, .5, .5]
    assert [c.section_field.minimum_tube_height for c in configs] == [.35, .35, .35]
    assert configs[-1].section_field.roof_stability_model.gravity_m_s2 == 1.62


def test_benches_and_incision_change_floor_without_moving_roof() -> None:
    generator = SectionFieldGenerator(SectionFieldConfig(profile_resolution=80))
    args = dict(tube_width=10., tube_height=8., floor_flatness=.7, roof_arch=1., lateral_skew=0.,
                wall_roughness=0., floor_relief=0., shape_bias=0., roof_bias=0., floor_bias=0.,
                asymmetry_bias=0., roughness_phase=0., floor_phase=0.)
    base = np.asarray(generator._build_profile_points(**args))
    shaped = np.asarray(generator._build_profile_points(**args, bench_strength=.8, floor_incision_ratio=.05))
    half = (len(base) - 1) // 2
    np.testing.assert_allclose(base[half:-1], shaped[half:-1])
    assert shaped[:half, 1].min() < base[:half, 1].min()
    side = np.abs(base[:half, 0]) > 3.5
    assert np.any(shaped[:half, 1][side] > base[:half, 1][side])
    from plume_advanced.evaluation.metrics.contours import self_intersection_count
    assert self_intersection_count(shaped) == 0


def test_existing_pillars_survive_but_underpasses_are_never_filled() -> None:
    footprint = np.zeros((11, 11), dtype=bool)
    footprint[2:9, 2:9] = True
    footprint[4:7, 4:7] = False
    pillars = GeometryGenerator._solid_pillar_columns(footprint)
    assert np.count_nonzero(pillars) == 9
    footprint[5, 4:7] = True  # projected passage on another level
    pillars = GeometryGenerator._solid_pillar_columns(footprint)
    assert not pillars[5].any()
    assert np.count_nonzero(pillars) == 6


def test_floor_noise_is_quieter_than_roof_at_equal_distance() -> None:
    generator = GeometryGenerator(GeometryConfig(random_seed=8))
    xyz = np.linspace(0., 30., 100)
    zero = np.zeros_like(xyz)
    floor = generator._wall_roughness(xyz, zero, zero, zero, local_vertical=np.full_like(xyz, -100.))
    roof = generator._wall_roughness(xyz, zero, zero, zero, local_vertical=np.full_like(xyz, 100.))
    assert np.std(floor) < np.std(roof)
    np.testing.assert_allclose(floor, .25 * roof)


@pytest.mark.parametrize('storage', ['dense', 'tiled'])
def test_room_union_keeps_an_existing_loop_pillar(storage: str) -> None:
    from plume_advanced.stages.network import CaveJunction
    from plume_advanced.stages.section_field import SectionJunctionInfluence
    network, field = _fixture()
    template = field.segment_fields[0].samples[0]
    phi = np.linspace(0., 2 * np.pi, 33)
    profile = tuple(zip(np.cos(phi), np.sin(phi)))
    influence = SectionJunctionInfluence(7, 'chamber', 1., '', '', 1., chamber_type='drained_lava_pool')
    samples = tuple(replace(
        template, index=i, segment_arc_length=6. * a, x=6. * np.cos(a), y=6. * np.sin(a),
        surface_z=20., tube_width=2., tube_height=2., profile_points=profile,
        tangent=(-np.sin(a), np.cos(a), 0.), normal=(-np.cos(a), -np.sin(a), 0.),
        junction_influences=(influence,),
    ) for i, a in enumerate(np.linspace(0., 2 * np.pi, 33)))
    junction = CaveJunction(
        junction_id=7, kind='chamber', node_ids=(), segment_ids=(0,), center_x=0., center_y=0.,
        along_position=0., blend_length=8., split_style='', merge_style='', capacity_bias=1.,
        metadata={'chamber_type': 'drained_lava_pool', 'pool_width_m': 14., 'pool_length_m': 18., 'pool_depth_m': 2.},
    )
    network = replace(network, junctions=(junction,))
    field = replace(field, segment_fields=(SegmentSectionField(0, (7,), samples),))
    geometry = GeometryGenerator(GeometryConfig(
        voxel_size=.5, storage_mode=storage, chunk_size=12, minimum_radius=.5,
        wall_roughness_amplitude=0., density_margin=2.,
    )).generate(network, field)
    assert geometry.preserved_pillar_columns > 0
    assert geometry.voxel_grid.sample_density((0., 0., 0.)) < 0
    assert geometry.voxel_grid.sample_density((6., 0., 0.)) > 0
    assert geometry.summary()['stability_collapse_count'] == 0


def test_enlarged_junction_cannot_bypass_roof_stability() -> None:
    from plume_advanced.stages.geometry import _JunctionEnvelope
    from plume_advanced.stages.section_field import SectionJunctionInfluence
    network, field = _fixture()
    samples = tuple(replace(s, profile_points=tuple((x * .1, y * .25) for x, y in s.profile_points),
                            tube_width=2., tube_height=1.5,
                            junction_influences=(SectionJunctionInfluence(7, 'chamber', 1., '', '', 1.),))
                    for s in field.segment_fields[0].samples)
    field = replace(field, segment_fields=(SegmentSectionField(0, (7,), samples),))
    generator = GeometryGenerator(GeometryConfig(voxel_size=.5, minimum_radius=.5, wall_roughness_amplitude=0.))
    stamp = _JunctionEnvelope(center=np.array((10., 0., 0.)), radius_long=20., radius_short=15.,
                           radius_z=3., angle=0., kind='chamber', junction_id=7)
    grid = generator._build_voxel_grid({0: samples}, network, None, junction_stamps=[stamp])
    records = [dict(r) for r in generator._enforce_roof_stability(grid, field, [stamp])]
    assert all(not r['failed'] for r in records if str(r['source']).startswith('section:'))
    assert next(r for r in records if r['source'] == 'junction:7')['failed']
    assert grid.sample_density((10., 0., 0.)) < 0
