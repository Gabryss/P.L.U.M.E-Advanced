"""Local refinement must preserve neighboring sweeps and detect shallow grids."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from plume_advanced.evaluation.local_geometry import (
    contour_distance,
    local_geometry,
    section_contour,
    section_resolution_report,
)
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.network import CaveNetwork, CaveNetworkConfig
from plume_advanced.stages.network_quality import NetworkQualityConfig
from plume_advanced.stages.section_field import (
    SectionField,
    SectionFieldConfig,
    SectionSample,
    SegmentSectionField,
)


def fixture():
    angle = np.linspace(0, 2 * np.pi, 41)
    profile = tuple(zip(2 * np.cos(angle), np.sin(angle)))
    base = SectionSample(
        index=0,
        segment_id=0,
        segment_arc_length=0.0,
        x=0.0,
        y=0.0,
        z=0.0,
        surface_z=20.0,
        cover_thickness=20.0,
        roof_thickness=19.0,
        centerline_depth=20.0,
        tangent=(1.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        binormal=(0.0, 0.0, 1.0),
        tube_width=4.0,
        tube_height=2.0,
        floor_flatness=0.5,
        roof_arch=1.0,
        lateral_skew=0.0,
        junction_blend_weight=0.0,
        junction_influences=(),
        profile_points=profile,
    )
    main = tuple(
        replace(base, index=i, x=x, segment_arc_length=x + 10.0)
        for i, x in enumerate((-10.0, 0.0, 10.0))
    )
    # A neighboring branch widens the measured section on just one side.
    branch = tuple(replace(s, segment_id=1, y=2.0) for s in main)
    network = CaveNetwork(
        # Local sweep equivalence is tested independently of network generation.
        config=CaveNetworkConfig(quality=NetworkQualityConfig(enabled=False)),
        nodes=(),
        segments=(),
        junctions=(),
        occupancy=np.zeros((2, 2)),
        width_field=np.zeros((2, 2)),
        dominant_route_node_ids=(),
        slice_along_positions=(),
        slice_channel_counts=(),
        slice_visible_channel_counts=(),
    )
    sections = SectionField(
        config=SectionFieldConfig(),
        segment_fields=(SegmentSectionField(0, (), main), SegmentSectionField(1, (), branch)),
        dominant_route_segment_ids=(0,),
    )
    return network, sections, base


def test_local_cut_matches_full_mesh_and_retains_adjacent_branch():
    network, sections, sample = fixture()
    config = GeometryConfig(
        voxel_size=0.2,
        storage_mode="dense",
        wall_roughness_amplitude=0.1,
        surface_wall_relief_m=0.15,
        surface_roof_relief_m=0.2,
        surface_floor_relief_m=0.05,
    )
    full = GeometryGenerator(config).generate(network, sections)
    local = local_geometry(
        network,
        sections,
        config,
        center=np.array([0.0, 1.0, 0.0]),
        half_extent=np.array([3.0, 5.0, 3.0]),
        lattice_origin=np.asarray(full.voxel_grid.origin),
    )
    cuts = [
        section_contour(
            trimesh.Trimesh(vertices=g.assembled_vertices, faces=g.assembled_faces, process=False),
            sample,
        )
        for g in (full, local)
    ]
    assert contour_distance(*cuts) < 2e-5
    assert np.ptp(cuts[1][:, 0]) > 5.5  # isolated original tube is only 4 m wide
    assert local.component_count == 1


def test_resolution_screen_uses_actual_height_instead_of_only_width():
    _, sections, sample = fixture()
    shallow = replace(
        sample,
        tube_height=999.0,
        profile_points=tuple((x * 5, y * 0.2) for x, y in sample.profile_points),
    )
    sections = replace(sections, segment_fields=(SegmentSectionField(0, (), (shallow,)),))
    result = section_resolution_report(sections, 0.2)
    assert result["under_resolved_count"] == 1
    row = result["under_resolved_sections"][0]
    assert row["width_m"] == pytest.approx(20.0)
    assert row["height_m"] == pytest.approx(0.4)
    assert row["samples_across_smallest_dimension"] == pytest.approx(2.0)
    assert row["initial_local_voxel_size_m"] == pytest.approx(0.05)
    assert section_resolution_report(sections, 0.025)["under_resolved_count"] == 0


def test_artificial_cap_cannot_be_reported_as_a_cave_wall():
    _, _, sample = fixture()
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="artificial inspection boundary"):
        section_contour(
            mesh,
            sample,
            region_bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            clip_margin_m=0.1,
        )


def test_region_memory_guard_precedes_allocation():
    network, sections, _ = fixture()
    with pytest.raises(ValueError, match="voxels"):
        local_geometry(
            network,
            sections,
            GeometryConfig(voxel_size=0.001),
            center=np.zeros(3),
            half_extent=np.ones(3),
            max_voxels=1000,
        )


def test_contour_distance_uses_segments_not_vertex_spacing():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    extra_vertices = np.insert(square, 1, [0.3, 0.0], axis=0)
    assert contour_distance(square, extra_vertices) == pytest.approx(0.0)
    assert contour_distance(square, square + [0.1, 0.0]) == pytest.approx(0.1)
