"""End-to-end regression coverage for sparse tiled geometry generation."""

import math

import numpy as np
import pytest

from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryConfig, TiledVoxelGrid
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNode,
    CavePoint,
    CaveSegment,
)
from plume_advanced.stages.section_field import (
    SectionField,
    SectionFieldConfig,
    SectionSample,
    SegmentSectionField,
)


def _profile(width: float, height: float) -> tuple[tuple[float, float], ...]:
    points = tuple(
        (
            0.5 * width * math.cos(angle),
            0.5 * height * math.sin(angle),
        )
        for angle in np.linspace(0.0, 2.0 * math.pi, 17)
    )
    return points


@pytest.mark.parametrize('missing_primary', [False, True])
def test_shared_tile_samples_match_density_queries(missing_primary: bool) -> None:
    tiles = {key: np.full((5, 5, 5), i, dtype=np.float32)
             for i, key in enumerate(np.ndindex(2, 2, 2))}
    if missing_primary:
        del tiles[(1, 1, 1)]
    grid = TiledVoxelGrid((0., 0., 0.), 1., (9, 9, 9), 0., 4, tiles)
    expected = {index: grid._density_at(np.asarray(index)) for index in np.ndindex(grid.shape)}
    grid.synchronize_halos()
    for key, tile in tiles.items():
        for local in np.ndindex(tile.shape):
            index = tuple(k * 4 + i for k, i in zip(key, local, strict=True))
            assert tile[local] == expected[index]


def test_inconsistent_tile_halos_still_produce_a_closed_surface() -> None:
    xyz = np.indices((17, 17, 17), dtype=float)
    density = (5.3 - np.linalg.norm(xyz - 8., axis=0)).astype(np.float32)
    tiles = {key: density[tuple(slice(k * 8, k * 8 + 9) for k in key)].copy()
             for key in np.ndindex(2, 2, 2)}
    # The earlier tile has a different interpolation value at its shared face.
    tiles[(0, 0, 0)][-1] += .15
    grid = TiledVoxelGrid((0., 0., 0.), 1., density.shape, 0., 8, tiles)
    generator = GeometryGenerator(GeometryConfig(voxel_size=1., chunk_size=8))
    chunks = generator._march_chunks(grid, None)
    _, faces = generator._assemble_chunks(chunks)
    assert faces and generator._mesh_is_closed_manifold(faces)


def test_forced_tiled_geometry_builds_connected_watertight_mesh() -> None:
    points = tuple(
        CavePoint(
            index=index,
            x=float(index * 10),
            y=0.0,
            elevation=20.0,
            slope_degrees=0.0,
            cover_thickness=20.0,
            roof_competence=1.0,
            growth_cost=0.0,
            arc_length=float(index * 10),
            width=4.0,
            flux=1.0,
            temperature_k=1400.0,
            age_s=float(index),
        )
        for index in range(2)
    )
    segment = CaveSegment(
        segment_id=0,
        start_node_id=0,
        end_node_id=1,
        kind="backbone",
        z_level=0,
        points=points,
        metadata={},
    )
    network = CaveNetwork(
        config=CaveNetworkConfig(),
        nodes=(
            CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"),
            CaveNode(1, 10.0, 0.0, 10.0, 0.0, "exit"),
        ),
        segments=(segment,),
        junctions=(),
        occupancy=np.ones((2, 2), dtype=bool),
        width_field=np.full((2, 2), 4.0),
        dominant_route_node_ids=(0, 1),
        slice_along_positions=(),
        slice_channel_counts=(),
        slice_visible_channel_counts=(),
    )
    profile = _profile(4.0, 3.0)
    samples = tuple(
        SectionSample(
            index=index,
            segment_id=0,
            segment_arc_length=float(index * 10),
            x=float(index * 10),
            y=0.0,
            z=0.0,
            surface_z=20.0,
            cover_thickness=20.0,
            roof_thickness=10.0,
            centerline_depth=20.0,
            tangent=(1.0, 0.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            binormal=(0.0, 0.0, 1.0),
            tube_width=4.0,
            tube_height=3.0,
            floor_flatness=0.4,
            roof_arch=1.0,
            lateral_skew=0.0,
            junction_blend_weight=0.0,
            junction_influences=(),
            profile_points=profile,
        )
        for index in range(2)
    )
    sections = SectionField(
        config=SectionFieldConfig(),
        segment_fields=(
            SegmentSectionField(
                segment_id=0,
                connected_junction_ids=(),
                samples=samples,
            ),
        ),
        dominant_route_segment_ids=(0,),
    )
    config = GeometryConfig(
        voxel_size=0.5,
        storage_mode="tiled",
        density_margin=2.0,
        chunk_size=8,
        minimum_radius=1.0,
        tunnel_radius_scale=1.0,
        chamber_radius_scale=1.0,
        junction_radius_scale=1.0,
        wall_roughness_amplitude=0.0,
        cave_diffuse_texture="",
        cave_normal_texture="",
        cave_roughness_texture="",
        cave_displacement_texture="",
    )

    geometry = GeometryGenerator(config).generate(network, sections)

    assert isinstance(geometry.voxel_grid, TiledVoxelGrid)
    assert geometry.voxel_grid.active_tile_count > 1
    assert geometry.voxel_grid.component_count == 1
    assert geometry.component_count == 1
    assert geometry.assembled_faces
