from dataclasses import replace

import numpy as np
import pytest
import trimesh

from plume_advanced.exporters import collision
from plume_advanced.stages.geometry_types import GeometryConfig


def simplify(mesh, **kwargs):
    report = {}
    result = collision.simplify_collision(mesh.vertices, mesh.faces,
        GeometryConfig(collision_max_error_m=.04, **kwargs), dict(points=((0., 0, 0),)), report=report)
    return result, report


def test_long_thin_corridor_decimates_without_collapsing_cross_section():
    mesh = trimesh.creation.box(extents=(1000, .5, .5))
    for _ in range(3):
        mesh = mesh.subdivide()
    (vertices, faces), report = simplify(mesh)
    assert not report["used_raw_fallback"]
    assert len(faces) < .9*len(mesh.faces)
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [1000, .5, .5], atol=1e-6)
    assert report["attempts"][-1]["surface_deviation"]["passed"]


def test_actual_decimator_replays_exactly_without_modifying_source():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=2)
    before = mesh.vertices.copy()
    (v1, f1), r1 = simplify(mesh)
    (v2, f2), r2 = simplify(mesh)
    assert r1 == r2
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(before, mesh.vertices)


def test_error_guard_rejects_topologically_valid_but_shifted_surface(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=2)
    smaller = trimesh.creation.icosphere(subdivisions=1, radius=2)
    monkeypatch.setattr(collision.fast_simplification, "simplify",
                        lambda *a, **k: (smaller.vertices+[.5, 0, 0], smaller.faces))
    (_, faces), report = simplify(mesh)
    assert report["used_raw_fallback"]
    assert len(report["attempts"]) == 4
    assert len(faces) == len(mesh.faces)


def test_progressive_retry_accepts_gentler_reduction(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=2)
    real = collision.fast_simplification.simplify
    calls = []
    def candidate(vertices, faces, **kwargs):
        calls.append(kwargs["target_reduction"])
        v, f = real(vertices, faces, **kwargs)
        return (v*2, f) if len(calls) == 1 else (v, f)
    monkeypatch.setattr(collision.fast_simplification, "simplify", candidate)
    _, report = simplify(mesh)
    assert not report["attempts"][0]["accepted"]
    assert report["attempts"][-1]["accepted"]
    assert calls[1] == calls[0]/2


def test_topology_change_cannot_be_hidden_by_distance_tolerance(monkeypatch):
    mesh = trimesh.creation.torus(major_radius=2, minor_radius=.5)
    sphere = trimesh.creation.icosphere(subdivisions=1)
    monkeypatch.setattr(collision.fast_simplification, "simplify", lambda *a, **k: (sphere.vertices, sphere.faces))
    report = {}
    collision.simplify_collision(mesh.vertices, mesh.faces, replace(GeometryConfig(), collision_max_error_m=100),
                                dict(points=((2, 0, 0),), expected_genus=1), report=report)
    assert report["used_raw_fallback"]


def test_native_programming_error_does_not_turn_into_successful_fallback(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    def broken(*a, **k):
        raise TypeError("bug")
    monkeypatch.setattr(collision.fast_simplification, "simplify", broken)
    with pytest.raises(TypeError, match="bug"):
        simplify(mesh)


def test_real_float32_collider_damage_is_locally_repaired_without_deleting_faces():
    from pathlib import Path
    with np.load(Path(__file__).parent/'fixtures/geometry/collider_float32_seed1.npz') as saved:
        vertices, faces = saved['vertices'], saved['faces']
    original = vertices.copy()
    triangle = vertices[faces]
    normal = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    naive = vertices.astype(np.float32).astype(float)[faces]
    naive_normal = np.cross(naive[:, 1]-naive[:, 0], naive[:, 2]-naive[:, 0])
    assert np.any(np.sum(normal*naive_normal,axis=1) <= 0)
    repaired, report = collision.stabilize_collision_precision(vertices, faces, .03)
    assert report['passed'] and report['relaxed_vertices'] > 0
    assert report['maximum_vertex_change_m'] < .001
    assert len(report['attempts']) <= 4
    for scale in [1, 100]:
        triangle = (repaired*scale).astype(np.float32).astype(float)[faces]
        corrected = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
        assert np.all(np.sum(normal*corrected,axis=1) > 0)
    again, repeat = collision.stabilize_collision_precision(vertices, faces, .03)
    np.testing.assert_array_equal(repaired, again)
    np.testing.assert_array_equal(vertices, original)
    assert report == repeat


def test_raw_fallback_cannot_bypass_unrepresentable_engine_precision():
    from plume_advanced.stages.mesh_inspection import MeshInspectionError
    mesh = trimesh.creation.box(extents=(.001, .001, .001))
    mesh.apply_translation((1e8, 1e8, 1e8))
    with pytest.raises(MeshInspectionError, match='float32 precision repair'):
        collision.simplify_collision(mesh.vertices, mesh.faces,
            GeometryConfig(collision_repair_attempts=0), dict(points=((1e8,1e8,1e8),)))
