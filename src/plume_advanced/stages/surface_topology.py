"""Surface connectivity checks at the volume-to-mesh boundary.

Closedness alone does not exclude detached shells or spurious handles. For an
event-free connected tube graph, the surface genus must equal E - V + 1.
This is a necessary topological check, not a geological realism certificate.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class SurfaceTopologyError(ValueError):
    """The generated surface does not realize the accepted tube graph."""

    def __init__(self, message: str, *, report: dict | None = None):
        if report is not None:
            self.report = report
        super().__init__(message)


def component_count(faces) -> int:
    """Count used-vertex components without Python objects per mesh corner."""
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return 0
    used, inverse = np.unique(triangles, return_inverse=True)
    indexed = inverse.reshape(-1, 3)
    start = indexed.ravel()
    end = indexed[:, [1, 2, 0]].ravel()
    graph = coo_matrix((np.ones(len(start), dtype=np.int8), (start, end)),
                       shape=(len(used), len(used))).tocsr()
    return int(connected_components(graph, directed=False, return_labels=False))


def check_closed_surface_topology(vertices, faces, components: int, expected_genus: int) -> dict:
    """Check an already verified closed triangular manifold, before export.

    E = 3F/2 only holds after the caller's manifold check. Counting referenced
    vertices also prevents unused coordinates from changing the Euler result.
    """
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    used = np.unique(triangles)
    euler = int(len(used) - len(triangles) // 2)
    genus = (2 * components - euler) / 2
    report = dict(components=components, euler=euler, genus=genus,
                  expected_genus=expected_genus, triangles=len(triangles))
    if not len(triangles) or not np.isfinite(np.asarray(vertices)).all():
        raise SurfaceTopologyError(f"Empty or nonfinite cave surface: {report}", report=report)
    if components != 1 or genus != expected_genus:
        raise SurfaceTopologyError(f"Surface differs from accepted network: {report}", report=report)
    return report
