from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_mesh(phi: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from skimage.measure import marching_cubes

        vertices, faces, _, _ = marching_cubes(phi, level=0.0, spacing=(voxel_size, voxel_size, voxel_size))
        return vertices.astype(float), faces.astype(int), "marching_cubes"
    except Exception:
        vertices, faces = _voxel_surface_mesh(phi < 0.0, voxel_size)
        return vertices, faces, "voxel_surface"


def _voxel_surface_mesh(void_mask: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    index_map: dict[tuple[float, float, float], int] = {}

    face_specs = [
        ((-1, 0, 0), [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
        ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
        ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
        ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
        ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
    ]

    nz, ny, nx = void_mask.shape

    def vertex_index(point: tuple[float, float, float]) -> int:
        if point not in index_map:
            index_map[point] = len(vertices)
            vertices.append(point)
        return index_map[point]

    for z, y, x in np.argwhere(void_mask):
        for (dz, dy, dx), offsets in face_specs:
            nz_idx, ny_idx, nx_idx = z + dz, y + dy, x + dx
            neighbor_is_void = (
                0 <= nz_idx < nz
                and 0 <= ny_idx < ny
                and 0 <= nx_idx < nx
                and void_mask[nz_idx, ny_idx, nx_idx]
            )
            if neighbor_is_void:
                continue

            quad = []
            for ox, oy, oz in offsets:
                point = (
                    (z + ox) * voxel_size,
                    (y + oy) * voxel_size,
                    (x + oz) * voxel_size,
                )
                quad.append(vertex_index(point))
            faces.append((quad[0], quad[1], quad[2]))
            faces.append((quad[0], quad[2], quad[3]))

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def export_obj(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
