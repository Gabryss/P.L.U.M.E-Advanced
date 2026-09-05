"""Regressions for finite sweeps and geometric (not just voxel) continuity."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryChunkMesh, GeometryConfig, VoxelGrid
from plume_advanced.stages.section_field import SectionSample


def sample(x=0.0, y=0.0, z=0.0):
    angles = np.linspace(0, 2 * np.pi, 41)
    return SectionSample(
        index=0,
        segment_id=0,
        segment_arc_length=x,
        x=x,
        y=y,
        z=z,
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
        profile_points=tuple(zip(2 * np.cos(angles), 1.5 * np.sin(angles))),
    )


def test_profile_end_narrows_before_closing_without_box_cutoff():
    gen = GeometryGenerator(
        GeometryConfig(
            voxel_size=0.25,
            minimum_radius=1.0,
            tunnel_radius_scale=1.0,
            wall_roughness_amplitude=0.0,
        )
    )
    density = np.full((81, 41, 41), -1.0, dtype=np.float32)
    gen._stamp_sample_chain(
        density=density, origin=np.array([-5.0, -5.0, -5.0]), samples=(sample(), sample(10.0))
    )
    areas = (density >= 0).sum(axis=(1, 2))
    # At 1 m past the endpoint the section must already be shrinking.
    assert 0 < areas[64] < areas[60]
    assert areas[69] == 0
    # The exact old failure: full-density extrusion to the bounding box.
    assert density[68, 20, 20] < density[60, 20, 20]


def test_welding_joins_near_vertices_across_rounding_buckets():
    chunks = [
        GeometryChunkMesh(
            0,
            (0, 1, 0, 1, 0, 1),
            ((0.0, 0.0, 0.0), (1.0000049, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        ),
        GeometryChunkMesh(
            1,
            (1, 2, 0, 1, 0, 1),
            ((1.0000051, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        ),
    ]
    vertices, faces = GeometryGenerator(GeometryConfig(weld_tolerance=1e-5))._assemble_chunks(
        chunks
    )
    assert len(vertices) == 4
    assert len(set(faces[0]) & set(faces[1])) == 2


@pytest.mark.parametrize("voxel", [0.5, 0.6])
@pytest.mark.parametrize("chunk_size", [8, 13])
def test_translated_nonbinary_chunks_are_closed_after_smoothing(voxel, chunk_size):
    from plume_advanced.stages.geometry_export import _smooth_visual_surface

    origin = np.array([901.12345, -663.43215, 88.3333])
    xyz = np.indices((50, 25, 25)).transpose(1, 2, 3, 0) * voxel
    closest_x = np.clip(xyz[..., 0], 6.0, 16.0)
    distance = np.sqrt(
        (xyz[..., 0] - closest_x) ** 2 + (xyz[..., 1] - 6.0) ** 2 + (xyz[..., 2] - 6.0) ** 2
    )
    grid = VoxelGrid(tuple(origin), voxel, (3.0 - distance).astype(np.float32), 0.0)
    gen = GeometryGenerator(GeometryConfig(voxel_size=voxel, chunk_size=chunk_size))
    vertices, faces = gen._assemble_chunks(gen._march_chunks(grid, None))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight
    assert len(mesh.split(only_watertight=False)) == 1
    moved = _smooth_visual_surface(np.asarray(vertices), np.asarray(faces), iterations=4)
    assert trimesh.Trimesh(vertices=moved, faces=faces, process=False).is_watertight


def test_connected_branches_share_a_floor_and_blend_gradually():
    from types import SimpleNamespace

    from plume_advanced.stages.section_field import SectionFieldGenerator, SegmentSectionField

    first = tuple(sample(float(x)) for x in np.linspace(-30, 0, 16))
    first = tuple(replace(s, segment_arc_length=s.x + 30) for s in first)
    second = tuple(replace(sample(float(x), z=1.8), segment_id=1) for x in np.linspace(0, 30, 16))
    # A nearby grade-separated passage has different node IDs.
    underpass = tuple(replace(s, segment_id=2, z=-8.0) for s in second)
    network = SimpleNamespace(
        segments=[
            SimpleNamespace(segment_id=0, start_node_id=0, end_node_id=1),
            SimpleNamespace(segment_id=1, start_node_id=1, end_node_id=2),
            SimpleNamespace(segment_id=2, start_node_id=3, end_node_id=4),
        ]
    )
    fields = [
        SegmentSectionField(i, (), values) for i, values in enumerate((first, second, underpass))
    ]
    gen = SectionFieldGenerator()
    result = gen._harmonize_connections(network, fields)
    assert gen._sample_floor(result[0].samples[-1]) == pytest.approx(
        gen._sample_floor(result[1].samples[0])
    )
    assert result[1].samples[-1].z == pytest.approx(second[-1].z)
    assert first[-1].z < result[1].samples[1].z < second[1].z
    assert np.allclose([s.z for s in result[2].samples], [s.z for s in underpass])


@pytest.mark.parametrize("tile_size", [4, 8])
def test_small_solid_cleanup_matches_dense_across_tile_boundaries(tile_size):
    from plume_advanced.stages.geometry_types import TiledVoxelGrid

    density = np.ones((25, 25, 25), dtype=np.float32)
    density[:2] = density[-2:] = -1
    density[:, :2] = density[:, -2:] = -1
    density[:, :, :2] = density[:, :, -2:] = -1
    density[7:9, 7:9, 7:9] = -1  # eight-cell pocket straddles three seams
    density[15:18, 15:18, 15:18] = -1  # resolved rock remains
    density[1:12, 4, 4] = -1  # thin connected divider remains
    dense = VoxelGrid((0., 0., 0.), .6, density.copy(), 0.)
    tiles = {key: density[tuple(slice(k * tile_size, k * tile_size + tile_size + 1) for k in key)].copy()
             for key in np.ndindex(*((24 // tile_size,) * 3))}
    tiled = TiledVoxelGrid((0., 0., 0.), .6, density.shape, 0., tile_size, tiles)
    GeometryGenerator._remove_small_solid_pockets(dense)
    GeometryGenerator._remove_small_solid_pockets(tiled)
    assert np.all(dense.density[7:9, 7:9, 7:9] > 0)
    assert np.all(dense.density[15:18, 15:18, 15:18] < 0)
    assert np.all(dense.density[1:12, 4, 4] < 0)
    for key, tile in tiled.tiles.items():
        np.testing.assert_array_equal(tile, dense.density[tuple(slice(k * tile_size, k * tile_size + tile_size + 1) for k in key)])


@pytest.mark.parametrize('open_wall', [False, True])
def test_validator_distinguishes_uv_seams_from_physical_holes(tmp_path, monkeypatch, open_wall):
    from plume_advanced.validation import PortableAssetValidator

    tetra = trimesh.creation.icosphere(subdivisions=0)
    faces = tetra.faces[:-1] if open_wall else tetra.faces
    # Every face has its own UV-chart copies, but their positions still match.
    positions = tetra.vertices[faces].reshape((-1, 3))
    indices = np.arange(len(positions)).reshape((-1, 3))
    validator = object.__new__(PortableAssetValidator)
    validator.asset_path = tmp_path / 'test.glb'
    validator.manifest = {'summary': {'voxel_component_count': 1}}
    monkeypatch.setattr(validator, '_geometry_arrays', lambda: (positions, indices, None, None, None))
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.Trimesh(vertices=positions, faces=indices, process=False), geom_name='cave_wall')
    monkeypatch.setattr(trimesh, 'load', lambda *args, **kwargs: scene)
    checks = validator._geometry_checks()
    topology = next(check for check in checks if check.name == 'Closed manifold cave wall')
    assert topology.passed is not open_wall


def test_collision_simplification_keeps_narrow_tube_closed():
    from plume_advanced.exporters.scene import simplified_collision_arrays
    from plume_advanced.stages.geometry_types import CaveGeometry

    mesh = trimesh.creation.torus(major_radius=6., minor_radius=.6,
                                 major_sections=48, minor_sections=12)
    geometry = CaveGeometry(
        GeometryConfig(), VoxelGrid((0., 0., 0.), .6, np.ones((2, 2, 2)), 0.), (),
        tuple(map(tuple, mesh.vertices)), tuple(map(tuple, mesh.faces)), 1, 0, (),
    )
    vertices, faces = simplified_collision_arrays(geometry)
    collision = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert collision.is_watertight
    assert collision.is_winding_consistent
    assert collision.euler_number == mesh.euler_number


@pytest.mark.parametrize("chunk_size", [8, 13])
def test_junction_blending_has_identical_surface_on_dense_and_tiled_grids(chunk_size):
    from types import SimpleNamespace

    from plume_advanced.stages.section_field import SectionJunctionInfluence

    influence = SectionJunctionInfluence(0, "split", 1.0, "gradual", "gradual", 1.0)
    chains = {
        0: tuple(replace(sample(x), junction_influences=(influence,))
                 for x in np.linspace(-12, 12, 9)),
        1: tuple(replace(sample(x, 0.03 * x * x), segment_id=1,
                         junction_influences=(influence,))
                 for x in np.linspace(0, 12, 5)),
    }
    config = GeometryConfig(voxel_size=.6, chunk_size=chunk_size, storage_mode="dense",
                            wall_roughness_amplitude=0.)
    network = SimpleNamespace(junctions=())
    dense = GeometryGenerator(config)._build_voxel_grid(chains, network, None)
    generator = GeometryGenerator(replace(config, storage_mode="tiled"))
    tiled = generator._build_voxel_grid(chains, network, None)
    for key, tile in tiled.tiles.items():
        start = np.asarray(key) * chunk_size
        region = dense.density[tuple(slice(int(a), int(a + size))
                                     for a, size in zip(start, tile.shape))]
        np.testing.assert_array_equal(tile >= 0, region >= 0)
        # Marching cubes interpolates both sides of the zero crossing.
        near_surface = (np.abs(tile) < 1.5) | (np.abs(region) < 1.5)
        np.testing.assert_allclose(tile[near_surface], region[near_surface], atol=2e-5)
    vertices, faces = generator._assemble_chunks(generator._march_chunks(tiled, None))
    assert trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight


def test_connected_split_merge_get_finite_transition_stamp_and_report():
    from types import SimpleNamespace

    from plume_advanced.stages.section_field import SectionJunctionInfluence

    influence = SectionJunctionInfluence(7, "split_merge", 1.0, "gradual", "gradual", 1.0, 18.0)
    parent = tuple(replace(sample(float(x)), junction_influences=(influence,)) for x in np.linspace(-18, 0, 7))
    daughter = tuple(
        replace(sample(float(x), y=0.8 * x), segment_id=1, junction_influences=(influence,))
        for x in np.linspace(0, 18, 7)
    )
    network = SimpleNamespace(
        junctions=(SimpleNamespace(
            junction_id=7,
            kind="split_merge",
            node_ids=(1, 2),
            center_x=0.0,
            center_y=0.0,
            blend_length=18.0,
        ),)
    )
    generator = GeometryGenerator(GeometryConfig(voxel_size=0.6, wall_roughness_amplitude=0.0))
    stamps = generator._junction_stamp_points({0: parent, 1: daughter}, network)
    assert len(stamps) == 1
    stamp = stamps[0]
    assert stamp.kind == "split_merge"
    assert stamp.blend_length_m >= 2.0 * np.median(stamp.incident_widths)
    assert stamp.radius_short * 2.0 / np.median(stamp.incident_widths) <= 2.5
    report = dict(generator._junction_report(network, {0: parent, 1: daughter}, junction_stamp_points=stamps))
    assert report["junction_max_width_m"] >= report["junction_median_incident_width_m"]
    assert report["junction_refinement_sample_count"] == 9.0


def test_stage_c_blend_metadata_controls_transition_length():
    from types import SimpleNamespace

    from plume_advanced.stages.section_field import SectionJunctionInfluence

    def stamps(length):
        influence = SectionJunctionInfluence(8, "junction", 1.0, "gradual", "gradual", 1.0, 4.0)
        samples = tuple(
            replace(sample(float(x)), junction_blend_length_m=length, junction_influences=(influence,))
            for x in np.linspace(-8, 8, 9)
        )
        network = SimpleNamespace(junctions=(SimpleNamespace(junction_id=8, kind="junction", node_ids=(1,), center_x=0.0, center_y=0.0, blend_length=4.0),))
        return GeometryGenerator(GeometryConfig(voxel_size=0.6))._junction_stamp_points({0: samples}, network)[0]

    assert stamps(12.0).blend_length_m > stamps(4.0).blend_length_m


def test_grade_separated_crossing_does_not_receive_union_stamp_or_fuse():
    from types import SimpleNamespace

    from plume_advanced.stages.section_field import SectionJunctionInfluence

    influence = SectionJunctionInfluence(3, "crossing", 1.0, "constant_envelope_then_divide", "constant_envelope_then_divide", 0.92, 18.0)
    upper = tuple(replace(sample(float(x)), junction_influences=(influence,)) for x in np.linspace(-18, 18, 15))
    lower = tuple(replace(sample(float(x), z=10.0), segment_id=1, junction_influences=(influence,)) for x in np.linspace(-18, 18, 15))
    network = SimpleNamespace(junctions=(SimpleNamespace(junction_id=3, kind="crossing", node_ids=(1,), center_x=0.0, center_y=0.0, blend_length=18.0),))
    generator = GeometryGenerator(GeometryConfig(voxel_size=0.6, wall_roughness_amplitude=0.0))
    stamps = generator._junction_stamp_points({0: upper, 1: lower}, network)
    assert stamps == []
    grid = generator._build_voxel_grid({0: upper, 1: lower}, network, None)
    assert grid.component_count == 2


def test_junction_refinement_is_seed_deterministic():
    from types import SimpleNamespace

    from plume_advanced.stages.section_field import SectionJunctionInfluence

    influence = SectionJunctionInfluence(4, "junction", 1.0, "gradual", "gradual", 1.0, 18.0)
    chains = {0: tuple(replace(sample(float(x)), junction_influences=(influence,)) for x in np.linspace(-18, 18, 15))}
    network = SimpleNamespace(junctions=(SimpleNamespace(junction_id=4, kind="junction", node_ids=(1,), center_x=0.0, center_y=0.0, blend_length=18.0),))
    config = GeometryConfig(voxel_size=0.6, wall_roughness_amplitude=0.0, random_seed=19)
    first = GeometryGenerator(config)._build_voxel_grid(chains, network, None)
    second = GeometryGenerator(config)._build_voxel_grid(chains, network, None)
    np.testing.assert_array_equal(first.density, second.density)
