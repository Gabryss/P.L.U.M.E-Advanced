"""Physical maps retain geometry/topology independently of robot rules."""

import json
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from matplotlib.path import Path as PlotPath
from PIL import Image
from scipy import ndimage

from plume_advanced.traversability.boundaries import mask_rings
from plume_advanced.traversability.config import TraversabilityConfig
from plume_advanced.traversability.export import export_traversability
from plume_advanced.traversability.request import from_paths
from plume_advanced.traversability.vector import network_vectors


def test_mask_rings_preserve_holes_islands_and_world_registration():
    mask = np.ones((9, 12), bool)
    mask[2:7, 2:10] = False
    mask[4, 5] = True
    rings = mask_rings(mask, np.array([100.25, -31.5]), .2)
    assert [r["role"] for r in rings].count("outer") == 2
    assert [r["role"] for r in rings].count("hole") == 1
    assert sum(r["signed_area_m2"] for r in rings) == pytest.approx(mask.sum() * .2**2)
    assert sorted({r["component_id"] for r in rings}) == [1, 2]
    for ring in rings:
        np.testing.assert_array_equal(ring["xy_m"][0], ring["xy_m"][-1])
        corners = (np.array(ring["xy_m"]) - [100.25, -31.5])/.2
        np.testing.assert_allclose(corners, np.rint(corners), atol=1e-10)


@pytest.mark.parametrize("seed", range(12))
def test_vector_boundaries_roundtrip_random_cell_masks(seed):
    mask = np.random.default_rng(seed).random((17, 23)) > .45
    rings = mask_rings(mask, [0., 0.], 1.)
    row, col = np.indices(mask.shape)
    points = np.column_stack([col.ravel()+.5, row.ravel()+.5])
    reconstructed = np.zeros(mask.size, int)
    for ring in rings:
        reconstructed += PlotPath(ring["xy_m"]).contains_points(points) * (
            1 if ring["role"] == "outer" else -1)
    np.testing.assert_array_equal(reconstructed.reshape(mask.shape), mask)
    assert sum(r["signed_area_m2"] for r in rings) == mask.sum()
    assert len({r["component_id"] for r in rings}) == ndimage.label(mask)[1]


def test_diagonal_contacts_are_separate_and_empty_mask_is_empty():
    assert mask_rings(np.zeros((3, 3), bool), [0, 0], .5) == []
    rings = mask_rings(np.eye(3, dtype=bool), [0, 0], .5)
    assert len(rings) == 3
    assert all(r["role"] == "outer" and r["signed_area_m2"] == .25 for r in rings)
    assert all(len(r["xy_m"]) == 5 for r in rings)


def request(config=None):
    paths = {0: np.array([[-3, 0, 0, 2, 2], [3, 0, 0, 2, 2]], float)}
    return from_paths(config or TraversabilityConfig(), [(0, 0, 1, {})], paths)


def test_robot_changes_leave_physical_arrays_outlines_and_graph_unchanged(tmp_path):
    mesh = trimesh.creation.box(extents=[8, 3, 2])
    for name, config in (("small", TraversabilityConfig()),
                         ("tall", replace(TraversabilityConfig(), robot_height_m=3, robot_length_m=1.5))):
        export_traversability(request(config), mesh.vertices, mesh.faces, tmp_path/name)
    with np.load(tmp_path/"small/layer_0.npz") as a, np.load(tmp_path/"tall/layer_0.npz") as b:
        for key in ("floor_z_m", "ceiling_z_m", "vertical_clearance_m", "origin_xy_m",
                    "resolution_m", "physical_state", "sampled_cavity", "chart_domain", "cavity_component_id"):
            np.testing.assert_array_equal(a[key], b[key])
        assert np.any(a["status"] == 1) and not np.any(b["status"] == 1)
        png = np.array(Image.open(tmp_path/"small/layer_0_physical.png"))
        np.testing.assert_array_equal(png, np.flipud(a["physical_state"]))
        assert set(np.unique(png)) == {0, 1}
    for file in ("layer_0_vectors.json", "network_vectors.json"):
        assert (tmp_path/"small"/file).read_bytes() == (tmp_path/"tall"/file).read_bytes()
    graph = json.loads((tmp_path/"small/network_vectors.json").read_text())
    assert graph["edges"][0]["cavity_witness"]["connected"] is True
    assert [n["role"] for n in graph["nodes"]] == ["source_terminal", "sink_terminal"]


def test_overlapping_layers_do_not_become_a_verified_ramp():
    low = trimesh.creation.box(extents=[8, 3, 2])
    high = low.copy()
    high.apply_translation([0, 0, 5])
    mesh = trimesh.util.concatenate([low, high])
    paths = {0: np.array([[-3, 0, 0, 3, 2], [0, 0, 0, 3, 2]]),
             1: np.array([[0, 0, 5, 3, 2], [3, 0, 5, 3, 2]]),
             2: np.array([[0, 0, 0, 3, 2], [0, 0, 5, 3, 2]])}
    segments = [(0, 0, 1, dict(regional_start_layer=0, regional_end_layer=0)),
                (1, 2, 3, dict(regional_start_layer=1, regional_end_layer=1)),
                (2, 1, 2, dict(regional_start_layer=0, regional_end_layer=1))]
    req = from_paths(TraversabilityConfig(), segments, paths, layered=True, layer_count=2)
    graph = network_vectors(req, mesh.vertices, mesh.faces)
    assert [e["cavity_witness"]["status"] for e in graph["edges"]] == ["verified", "verified", "unresolved"]
    assert graph["edges"][2]["kind"] == "ramp"
    assert graph["edges"][2]["cavity_witness"]["surface_contact_intervals"] == [0]
    assert len(graph["nodes"]) == 4


def test_shared_node_id_does_not_snap_incident_paths_through_a_partition():
    left = trimesh.creation.box(extents=[3.9, 3, 2])
    right = left.copy()
    left.apply_translation([-2, 0, 0])
    right.apply_translation([2, 0, 0])
    mesh = trimesh.util.concatenate([left, right])
    paths = {0: np.array([[-3, 0, 0, 3, 2], [-.1, 0, 0, 3, 2]]),
             1: np.array([[.1, 0, 0, 3, 2], [3, 0, 0, 3, 2]])}
    req = from_paths(TraversabilityConfig(), [(0, 0, 1, {}), (1, 1, 2, {})], paths)
    graph = network_vectors(req, mesh.vertices, mesh.faces)
    assert all(e["cavity_witness"]["status"] == "verified" for e in graph["edges"])
    junction = graph["nodes"][1]
    assert junction["cavity_witness_status"] == "unresolved"
    assert junction["endpoint_links"][1]["endpoint_gap_m"] == pytest.approx(.2)


def test_closed_multilayer_ramp_verifies_all_edges_and_junctions(tmp_path):
    # Independent closed floor/ramp/floor fixture, reloaded from its GLB export.
    # Keep production regressions independent of optional publication tooling.
    xs, zs = [-6, -3, 3, 6], [0, 0, 2, 2]
    vertices = [[x, y, z+dz] for x, z in zip(xs, zs)
                for y, dz in [(-.3, -.6), (.3, -.6), (.3, .6), (-.3, .6)]]
    faces = [[0, 2, 1], [0, 3, 2], [12, 13, 14], [12, 14, 15]]
    for ring in range(3):
        for j in range(4):
            a, b = 4*ring+j, 4*ring+(j+1)%4
            faces.extend([[a, b, b+4], [a, b+4, a+4]])
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()
    assert mesh.is_watertight
    target = tmp_path / "ramp.glb"
    mesh.export(target)
    mesh = trimesh.load(target, force="mesh", process=False)
    paths = {0: np.array([[-5, 0, 0, .6, 1.2], [-3, 0, 0, .6, 1.2]]),
             1: np.array([[-3, 0, 0, .6, 1.2], [3, 0, 2, .6, 1.2]]),
             2: np.array([[3, 0, 2, .6, 1.2], [5, 0, 2, .6, 1.2]])}
    segments = [(i, i, i+1, dict(regional_start_layer=a, regional_end_layer=b))
                for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 1)])]
    req = from_paths(TraversabilityConfig(), segments, paths, layered=True, layer_count=2)
    before = mesh.vertices.copy()
    graph = network_vectors(req, mesh.vertices, mesh.faces)
    assert graph["summary"]["verified_edge_witnesses"] == 3
    assert graph["summary"]["unresolved_junctions"] == 0
    assert graph["edges"][1]["chart_id"] == "ramp_1"
    np.testing.assert_array_equal(mesh.vertices, before)


def test_duplicate_segment_id_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        from_paths(TraversabilityConfig(), [(0, 0, 1, {}), (0, 1, 2, {})],
                   {0: np.array([[0, 0, 0, 2, 2], [1, 0, 0, 2, 2]])})


def test_open_boundary_is_not_a_verified_cavity_even_with_floor_and_roof():
    mesh = trimesh.creation.box(extents=[8, 3, 2])
    faces = mesh.faces[mesh.face_normals[:, 0] < .5]  # remove one end wall
    graph = network_vectors(request(), mesh.vertices, faces)
    assert not graph["closed_cave_boundary"]
    assert graph["summary"]["verified_edge_witnesses"] == 0
    assert graph["edges"][0]["cavity_witness"]["connected"] is None


def test_exact_duplicate_vertices_at_visual_seams_do_not_break_closure():
    mesh = trimesh.creation.box(extents=[8, 3, 2])
    vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    graph = network_vectors(request(), vertices, faces)
    assert graph["closed_cave_boundary"]
    assert graph["summary"]["verified_edge_witnesses"] == 1


def test_prop_outlines_use_conservative_mask_on_the_selected_layer():
    from plume_advanced.traversability.raster import obstacle_cells
    from plume_advanced.traversability.vector import chart_vectors

    rock = trimesh.creation.box(extents=[.1, .1, .3])
    rock.apply_translation([.6, .6, .15])
    floor = np.zeros((8, 8))
    for height, expected in [(0, True), (5, False)]:
        blocked, _ = obstacle_cells([(rock.vertices, rock.faces)], floor+height,
                                    floor+height+2, np.zeros(2), .25)
        fields = dict(origin_xy_m=np.zeros(2), resolution_m=np.array(.25),
                      sampled_cavity=np.ones((8, 8), bool), chart_domain=np.ones((8, 8), bool),
                      obstacle=blocked)
        vectors = chart_vectors(dict(id='layer_test', kind='layer'), fields)
        assert bool(vectors['obstacle_rings']) == expected
        assert sum(r['signed_area_m2'] for r in vectors['obstacle_rings']) == blocked.sum()*.25**2
