"""Shading performance changes must preserve reference winding and handedness."""

import math
import time

import numpy as np
import pytest

from plume_advanced.stages import surface_frames


def reference_normals(vertices, faces):
    result = np.zeros_like(vertices)
    for face in faces:
        p = vertices[face]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            continue
        for i in range(3):
            a, b = p[(i + 1) % 3] - p[i], p[(i + 2) % 3] - p[i]
            if min(np.linalg.norm(a), np.linalg.norm(b)) <= 1e-12:
                continue
            angle = math.acos(
                float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
            )
            result[face[i]] += normal / length * angle
    length = np.linalg.norm(result, axis=1)
    result[length > 1e-12] /= length[length > 1e-12, None]
    result[length <= 1e-12] = (0, 0, 1)
    return result


def reference_tangents(vertices, faces, uv, normals):
    u, v = np.zeros_like(vertices), np.zeros_like(vertices)
    for face in faces:
        p0, p1, p2 = vertices[face]
        t0, t1, t2 = uv[face]
        d1, d2 = t1 - t0, t2 - t0
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(det) <= 1e-12:
            continue
        s = ((p1 - p0) * d2[1] - (p2 - p0) * d1[1]) / det
        t = ((p2 - p0) * d1[0] - (p1 - p0) * d2[0]) / det
        for index in face:
            u[index] += s
            v[index] += t
    output = []
    for i, n in enumerate(normals):
        tangent = u[i] - n * (n @ u[i])
        if np.linalg.norm(tangent) <= 1e-12:
            tangent = np.cross((0, 0, 1) if abs(n[2]) < 0.9 else (1, 0, 0), n)
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        output.append([*tangent, -1 if np.cross(n, tangent) @ v[i] < 0 else 1])
    return np.asarray(output)


@pytest.mark.parametrize("seed", [0, 1, 3, 17, 42, 123456789])
@pytest.mark.parametrize("batch", [1, 7, 65536])
def test_batched_frames_match_scalar_reference(seed, batch, monkeypatch):
    rng = np.random.default_rng(seed)
    vertices = rng.normal(size=(71, 3))
    faces = rng.integers(0, 70, size=(101, 3))  # shared, duplicate and degenerate corners
    uv = rng.random((71, 2))
    uv[:6] = 0  # UV degeneracies and an isolated vertex
    monkeypatch.setattr(surface_frames, "FACE_BATCH_SIZE", batch)
    normals = surface_frames.angle_weighted_vertex_normals(vertices, faces)
    np.testing.assert_allclose(normals, reference_normals(vertices, faces), atol=2e-12)
    tangents = surface_frames.mesh_tangents(vertices, faces, uv, normals)
    np.testing.assert_allclose(
        tangents, reference_tangents(vertices, faces, uv, normals), atol=2e-11
    )
    assert np.isfinite(tangents).all()
    np.testing.assert_allclose(np.sum(normals * tangents[:, :3], axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(tangents[:, :3], axis=1), 1)


@pytest.mark.performance
def test_large_frame_generation_is_bounded_and_fast():
    rng = np.random.default_rng(42)
    vertices = rng.normal(size=(50000, 3))
    faces = rng.integers(0, 50000, size=(200000, 3))
    started = time.perf_counter()
    normals = surface_frames.angle_weighted_vertex_normals(vertices, faces)
    tangents = surface_frames.mesh_tangents(vertices, faces, rng.random((50000, 2)), normals)
    assert np.isfinite(tangents).all()
    assert time.perf_counter() - started < 10.0  # broad CI budget
    # Scalar version takes tens of seconds.
