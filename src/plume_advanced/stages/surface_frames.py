"""Bounded vectorized normal and tangent accumulation for simulation meshes."""

from __future__ import annotations

import numpy as np

from plume_advanced.progress import report_progress

FACE_BATCH_SIZE = 65_536


def angle_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    positions = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(positions)
    for offset in range(0, len(triangles), FACE_BATCH_SIZE):
        batch = triangles[offset : offset + FACE_BATCH_SIZE]
        p = positions[batch]
        edge_a = np.roll(p, -1, axis=1) - p
        edge_b = np.roll(p, -2, axis=1) - p
        face_normals = np.cross(edge_a[:, 0], edge_b[:, 0])
        lengths = np.linalg.norm(face_normals, axis=1)
        valid = lengths > 1e-12
        face_normals /= np.maximum(lengths[:, None], 1e-12)
        a = np.linalg.norm(edge_a, axis=2)
        b = np.linalg.norm(edge_b, axis=2)
        cosine = np.sum(edge_a * edge_b, axis=2) / np.maximum(a * b, 1e-24)
        angles = np.arccos(np.clip(cosine, -1, 1))
        angles[~(valid[:, None] & (a > 1e-12) & (b > 1e-12))] = 0
        values = face_normals[:, None, :] * angles[:, :, None]
        # Triangle-major order matches serial accumulation, including shared vertices.
        np.add.at(normals, batch.ravel(), values.reshape(-1, 3))
        report_progress(
            "Normal triangles", min(offset + FACE_BATCH_SIZE, len(triangles)), len(triangles)
        )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0.0, 0.0, 1.0)
    return normals


def mesh_tangents(
    vertices: np.ndarray, faces: np.ndarray, texcoords: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    positions = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(texcoords, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    tangent_u, tangent_v = np.zeros_like(positions), np.zeros_like(positions)
    for offset in range(0, len(triangles), FACE_BATCH_SIZE):
        batch = triangles[offset : offset + FACE_BATCH_SIZE]
        p, t = positions[batch], uv[batch]
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        determinant = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        reciprocal = np.zeros_like(determinant)
        np.divide(1.0, determinant, out=reciprocal, where=np.abs(determinant) > 1e-12)
        u = (e1 * d2[:, 1, None] - e2 * d1[:, 1, None]) * reciprocal[:, None]
        v = (e2 * d1[:, 0, None] - e1 * d2[:, 0, None]) * reciprocal[:, None]
        np.add.at(tangent_u, batch.ravel(), np.repeat(u, 3, axis=0))
        np.add.at(tangent_v, batch.ravel(), np.repeat(v, 3, axis=0))
        report_progress(
            "Tangent triangles", min(offset + FACE_BATCH_SIZE, len(triangles)), len(triangles)
        )
    tangent = tangent_u - normals * np.sum(normals * tangent_u, axis=1)[:, None]
    lengths = np.linalg.norm(tangent, axis=1)
    missing = lengths <= 1e-12
    reference = np.zeros((int(missing.sum()), 3))
    z_axis = np.abs(normals[missing, 2]) < 0.9
    reference[z_axis, 2] = 1
    reference[~z_axis, 0] = 1
    tangent[missing] = np.cross(reference, normals[missing])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    handedness = np.where(np.sum(np.cross(normals, tangent) * tangent_v, axis=1) < 0, -1.0, 1.0)
    return np.column_stack((tangent, handedness))
