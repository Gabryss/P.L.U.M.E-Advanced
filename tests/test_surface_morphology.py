"""Geometry regressions for the chamber, shallow-tube and corrugated-wall defects."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_mesh_continuity import sample

from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.section_field import SectionFieldConfig, SectionFieldGenerator


def test_profile_interpolation_resolves_subpercent_changes():
    gen = GeometryGenerator()
    first = np.array([[-2., -1.], [2., -1.], [2., 1.], [-2., 1.], [-2., -1.]])
    last = first * (3., 1.)
    t = np.linspace(.101, .109, 19)
    distance = gen._interpolated_profile_signed_distance(
        np.full_like(t, 5.), np.zeros_like(t), t, first, last,
    )
    np.testing.assert_allclose(distance, 5. - (2. + 4. * t), atol=1e-12)
    assert np.all(np.diff(distance) < 0.)


def test_shallow_contour_is_not_enlarged_by_a_minimum_radius_capsule():
    config = GeometryConfig(voxel_size=.1, minimum_radius=1.5, tunnel_radius_scale=1., wall_roughness_amplitude=0.)
    gen = GeometryGenerator(config)
    original = sample()
    contour = tuple((x, z * .2) for x, z in original.profile_points)
    shallow = replace(original, tube_height=.6, profile_points=contour)
    density = np.full((141, 61, 31), -8., dtype=np.float32)
    gen._stamp_sample_chain(density=density, origin=np.array([-2., -3., -1.5]), samples=(shallow, replace(shallow, x=10.)))
    assert density[70, 30, 15] > 0
    assert density[70, 30, 20] < 0  # 0.5 m above axis is outside the 0.6 m passage


def test_bent_sweeps_agree_on_their_shared_section_plane():
    middle = replace(sample(10., 1.), tangent=(np.cos(.3), np.sin(.3), 0.),
                     normal=(-np.sin(.3), np.cos(.3), 0.))
    end = replace(sample(20., 5.), tangent=(np.cos(.5), np.sin(.5), 0.),
                  normal=(-np.sin(.5), np.cos(.5), 0.))
    gen = GeometryGenerator(GeometryConfig(voxel_size=.25, tunnel_radius_scale=1., wall_roughness_amplitude=0.))
    for side in (-1., 1.):
        point = np.array((middle.x, middle.y, middle.z)) + side*1.9*np.array(middle.normal)
        distances = []
        for first, last in ((sample(), middle), (middle, end)):
            density = np.full((1, 1, 1), -8., dtype=np.float32)
            gen._stamp_profile_segment(density=density, origin=point, start=first, end=last,
                                       cap_start=False, cap_end=False)
            distances.append(float(density[0, 0, 0]))
        assert distances[0] > 0.
        assert distances[0] == pytest.approx(distances[1], abs=1e-6)


@pytest.mark.parametrize("storage", ["dense", "tiled"])
def test_inner_loop_wall_retains_its_subvoxel_distance(storage):
    config = GeometryConfig(voxel_size=.3, storage_mode=storage, minimum_radius=.5,
                            tunnel_radius_scale=1., wall_roughness_amplitude=0., density_margin=2., chunk_size=24)
    gen = GeometryGenerator(config)
    angles = np.linspace(0., 2 * np.pi, 129)
    chain = tuple(replace(sample(6 * np.cos(a), 6 * np.sin(a)), segment_arc_length=6*a,
                          tangent=(-np.sin(a), np.cos(a), 0.), normal=(-np.cos(a), -np.sin(a), 0.))
                  for a in angles)
    grid = gen._build_voxel_grid({0: chain}, SimpleNamespace(junctions=()), None)
    # Analytical torus inner radius is 4 m. Its surrounding negative band must
    # remain a distance field, not a -8 wall made of vertical voxel columns.
    assert -1.2 < grid.sample_density((3.85, 0., 0.)) < -.05
    assert grid.sample_density((4.3, 0., 0.)) > 0.


def test_level_offset_has_no_flat_deck_or_pointwise_clipped_ramp():
    gen = SectionFieldGenerator(SectionFieldConfig(minimum_roof_thickness=6., maximum_uphill_grade=.02))
    arc = np.linspace(0., 200., 101)
    samples = [replace(sample(float(x), z=-.05*x), surface_z=10.-.05*x,
                       cover_thickness=30., tube_height=2., segment_arc_length=float(x)) for x in arc]
    segment = SimpleNamespace(z_level=1, total_length=200.)
    resolved = gen._apply_vertical_level_profile(segment, samples)
    offset = np.array([s.z - original.z for s, original in zip(resolved, samples, strict=True)])
    assert offset[0] == offset[-1] == 0.
    assert 2. < offset.max() <= 4.
    assert np.count_nonzero(np.isclose(offset, offset.max(), atol=1e-5)) == 1
    assert np.max(np.abs(np.diff(offset, n=2))) < .015


def test_profile_chamber_keeps_passage_roof_and_floor():
    from plume_advanced.stages.section_field import SectionJunctionInfluence

    influence = SectionJunctionInfluence(7, "chamber", 1., "gradual", "gradual", 1., 20., "drained_lava_pool")
    chain = tuple(replace(sample(float(x)), profile_points=tuple((u * scale, v) for u, v in sample().profile_points),
                          tube_width=4.*scale, junction_influences=(influence,))
                  for x, scale in [(-12., 1.), (0., 3.), (12., 1.)])
    network = SimpleNamespace(junctions=(SimpleNamespace(junction_id=7, kind="chamber", center_x=0., center_y=0.,
                              blend_length=20., metadata={"chamber_type":"drained_lava_pool", "pool_width_m":24., "pool_depth_m":8.}),))
    gen = GeometryGenerator(GeometryConfig(voxel_size=.25, tunnel_radius_scale=1., wall_roughness_amplitude=0.))
    grid = gen._build_voxel_grid({0:chain}, network, None)
    assert grid.sample_density((0., 4., 0.)) > 0.  # widened chamber still present
    assert grid.sample_density((0., 9., 0.)) < 0.  # no separately added disk
    assert grid.sample_density((0., 0., 2.)) < 0.  # shaped passage roof is authoritative
