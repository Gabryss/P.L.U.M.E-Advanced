"""Bounded spatial diagnostics of rejected meshes, without retaining mesh copies.

Open surface patches with manifold boundary loops have a measurable genus.
Overlapping slabs locate small handles, including handles crossing one partition.
For a cyclic network these regions are suspects, not proof that a particular
handle is unwanted. The full graph and route gates decide every repair.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from plume_advanced.stages.surface_topology import component_count


def _region(vertices, ids, kind, **measurements):
    positions = vertices[np.unique(ids)]
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    return dict(
        kind=kind,
        lower_m=lower.tolist(),
        upper_m=upper.tolist(),
        center_m=((lower + upper) * 0.5).tolist(),
        **measurements,
    )


def localize_surface_defects(
    vertices, faces, *, window_m: float, max_regions: int = 24
) -> list[dict]:
    """Locate bad edges, detached shells and handles; never modify the mesh.

    Work is linear in mesh size apart from sorted edge inventories. Slab width
    is increased for long scenes to cap the number of diagnostic patches at 128.
    A missing region means localization was inconclusive, not that the mesh passed.
    """
    positions = np.asarray(vertices, dtype=float).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles) or not np.isfinite(positions).all():
        return []
    used, inverse = np.unique(triangles, return_inverse=True)
    indexed = inverse.reshape(-1, 3)
    graph = coo_matrix(
        (np.ones(indexed.size, dtype=np.int8), (indexed.ravel(), indexed[:, [1, 2, 0]].ravel())),
        shape=(len(used), len(used)),
    ).tocsr()
    count, labels = connected_components(graph, directed=False)
    regions = []
    if count > 1:
        sizes = np.bincount(labels)
        main = int(np.argmax(sizes))
        for label in sorted((i for i in range(count) if i != main), key=lambda i: (-sizes[i], i))[
            :max_regions
        ]:
            regions.append(_region(positions, used[labels == label], "detached_surface"))
    axis = int(np.argmax(np.ptp(positions, axis=0)))
    lower, upper = positions[:, axis].min(), positions[:, axis].max()
    width = max(float(window_m), float(upper - lower) / 64, 1e-6)
    coordinates = positions[triangles, axis]
    face_min, face_max = coordinates.min(axis=1), coordinates.max(axis=1)
    del coordinates, graph, indexed, inverse, labels
    for start in np.arange(lower - 0.5 * width, upper, 0.5 * width):
        patch = triangles[(face_min >= start) & (face_max < start + width)]
        if not len(patch):
            continue
        edges, counts = np.unique(
            np.sort(np.concatenate((patch[:, [0, 1]], patch[:, [1, 2]], patch[:, [2, 0]])), axis=1),
            axis=0,
            return_counts=True,
        )
        if np.any(counts > 2):
            regions.append(_region(positions, edges[counts > 2], "nonmanifold_edges"))
        else:
            boundary = edges[counts == 1]
            _, degrees = np.unique(boundary, return_counts=True)
            # Cut patches can meet at a single vertex: do not apply the
            # manifold-with-boundary Euler formula to such patches.
            if len(degrees) and np.any(degrees != 2):
                continue
            loops = (
                component_count(np.column_stack((boundary, boundary[:, 0]))) if len(boundary) else 0
            )
            components = component_count(patch)
            euler = len(np.unique(patch)) - len(edges) + len(patch)
            genus = (2 * components - loops - euler) / 2
            if genus > 0 and genus == int(genus):
                regions.append(
                    _region(
                        positions, patch, "handle_patch", genus=int(genus), boundary_loops=loops
                    )
                )
        if len(regions) >= max_regions:
            break
    return regions[:max_regions]
