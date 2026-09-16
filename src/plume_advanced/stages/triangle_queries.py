"""Bounded spatial queries on actual triangles, without an optional ray backend.

Radius buckets prevent one long triangle from making every local query global.
Candidate selection is conservative; distance tests use the triangles themselves.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point

from plume_advanced.progress import report_progress


def vertical_clearances(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> list[dict]:
    """Measure the vertical cavity enclosing each point on actual triangles."""
    if not len(points):
        return []
    index = VerticalTriangleIndex(vertices, faces)
    rows = []
    for number, point in enumerate(points):
        if number % 100 == 0:
            report_progress(
                "Mesh passage inspection",
                number,
                len(points),
                "measuring floor/roof intersections on actual triangles",
            )
        rows.append(index.measure(point, number))
    report_progress(
        "Mesh passage inspection", len(points), len(points), "passage samples inspected"
    )
    return rows


class VerticalTriangleIndex:
    """Reusable XY radius buckets for floor support and cavity queries."""

    def __init__(self, vertices, faces, *, triangles=None):
        self.triangles = np.asarray(vertices, float)[faces] if triangles is None else triangles
        xy = self.triangles[:, :, :2]
        centers = xy.mean(axis=1)
        radii = np.linalg.norm(xy - centers[:, None], axis=2).max(axis=1)
        bins = np.ceil(np.log2(np.maximum(radii, 1e-6))).astype(int)
        self.groups = []
        for key in np.unique(bins):
            ids = np.flatnonzero(bins == key)
            self.groups.append((ids, cKDTree(centers[ids]), float(radii[ids].max())))
        self.tolerance = max(float(np.ptp(vertices, axis=0).max()) * 1e-10, 1e-9)

    def measure(self, point, sample_index=0):
        point = np.asarray(point, float)
        tolerance = self.tolerance
        groups = [ids[tree.query_ball_point(point[:2], radius + tolerance)]
                  for ids, tree, radius in self.groups]
        ids = np.sort(np.concatenate(groups)) if groups else np.empty(0, dtype=int)
        tri = self.triangles[ids]
        # Project barycentric coordinates in XY. Vertical triangles do not cross
        # a vertical ray; roof/floor triangles supply its actual intersections.
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        v0, v1, v2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], point[:2] - a[:, :2]
        determinant = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        usable = np.abs(determinant) > 1e-16
        u = np.divide(
            v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1],
            determinant,
            out=np.zeros(len(tri)),
            where=usable,
        )
        v = np.divide(
            v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1],
            determinant,
            out=np.zeros(len(tri)),
            where=usable,
        )
        inside_triangle = usable & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        z = np.sort((a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2]))[inside_triangle])
        z = z[np.r_[True, np.diff(z) > tolerance]] if len(z) else z
        lower, upper = z[z < point[2] - tolerance], z[z > point[2] + tolerance]
        on_surface = bool(np.any(np.abs(z - point[2]) <= tolerance))
        inside = bool(len(lower) % 2 and len(upper) % 2 and not on_surface)
        floor = float(point[2] - lower[-1]) if len(lower) else None
        roof = float(upper[0] - point[2]) if len(upper) else None
        return dict(
                sample_index=sample_index,
                point_m=point.tolist(),
                inside=inside,
                floor_distance_m=floor,
                roof_distance_m=roof,
                clearance_m=floor + roof
                if inside and floor is not None and roof is not None
                else None,
        )


def segment_distances(a, b, c, d):
    """Pairwise distances between segments AB and CD, including zero lengths."""
    a, b, c, d = np.broadcast_arrays(*(np.asarray(p, float) for p in (a, b, c, d)))
    u, v, w = b - a, d - c, a - c
    uu, vv = np.sum(u*u, axis=-1), np.sum(v*v, axis=-1)
    uv, uw, vw = (np.sum(x*y, axis=-1) for x, y in ((u, v), (u, w), (v, w)))
    denominator = uu*vv - uv*uv
    s = np.divide(uv*vw-vv*uw, denominator, out=np.zeros_like(uu),
                  where=denominator > 1e-24)
    s = np.clip(s, 0, 1)
    t = np.divide(uv*s+vw, vv, out=np.zeros_like(vv), where=vv > 1e-24)
    t = np.clip(t, 0, 1)
    s = np.divide(uv*t-uw, uu, out=np.zeros_like(uu), where=uu > 1e-24)
    s = np.clip(s, 0, 1)
    return np.linalg.norm(w+s[..., None]*u-t[..., None]*v, axis=-1)


def segment_triangle_distances(a, b, triangles):
    """Distance from segment AB to each triangle, including interior crossings."""
    triangles = np.asarray(triangles, float)
    if not len(triangles):
        return np.empty(0)
    a, b = np.broadcast_to(a, (len(triangles), 3)), np.broadcast_to(b, (len(triangles), 3))
    result = np.minimum(np.linalg.norm(closest_point(triangles, a)-a, axis=1),
                        np.linalg.norm(closest_point(triangles, b)-b, axis=1))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        result = np.minimum(result, segment_distances(a, b, triangles[:, i], triangles[:, j]))
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    direction = b-a
    h = np.cross(direction, e2)
    determinant = np.sum(e1*h, axis=1)
    inverse = np.divide(1., determinant, out=np.zeros(len(triangles)),
                        where=np.abs(determinant) > 1e-14)
    s = a-triangles[:, 0]
    u = inverse*np.sum(s*h, axis=1)
    q = np.cross(s, e1)
    v = inverse*np.sum(direction*q, axis=1)
    t = inverse*np.sum(e2*q, axis=1)
    crossing = ((np.abs(determinant) > 1e-14) & (u >= -1e-12) & (v >= -1e-12)
                & (u+v <= 1+1e-12) & (t >= 0) & (t <= 1))
    result[crossing] = 0.
    return result


class TriangleIndex:
    """An immutable triangle index with bounded batch working memory."""

    def __init__(self, vertices, faces):
        self.triangles = np.asarray(vertices, float)[np.asarray(faces, np.int64)]
        self.centers = self.triangles.mean(axis=1)
        radii = np.linalg.norm(self.triangles-self.centers[:, None], axis=2).max(axis=1)
        bins = np.ceil(np.log2(np.maximum(radii, 1e-6))).astype(int)
        self.groups = []
        for key in np.unique(bins):
            ids = np.flatnonzero(bins == key)
            self.groups.append((ids, cKDTree(self.centers[ids]), float(radii[ids].max())))

    def candidates(self, point, radius):
        ids = [indices[tree.query_ball_point(point, radius+extent+1e-10)]
               for indices, tree, extent in self.groups]
        return np.sort(np.concatenate(ids)) if ids else np.empty(0, dtype=np.int64)

    def distances(self, points, *, batch_size=2048):
        """Exact nearest-triangle distance at supplied points, in bounded batches."""
        points = np.asarray(points, float).reshape(-1, 3)
        result = np.full(len(points), np.inf)
        for start in range(0, len(points), batch_size):
            query = points[start:start+batch_size]
            upper = np.full(len(query), np.inf)
            for indices, tree, _ in self.groups:
                _, ids = tree.query(query)
                near = closest_point(self.triangles[indices[ids]], query)
                upper = np.minimum(upper, np.linalg.norm(near-query, axis=1))
            for indices, tree, extent in self.groups:
                lists = tree.query_ball_point(query, upper+extent+1e-10)
                counts = np.fromiter(map(len, lists), dtype=int, count=len(query))
                if not counts.sum():
                    continue
                rows = np.repeat(np.arange(len(query)), counts)
                ids = indices[np.concatenate(lists).astype(np.int64)]
                near = closest_point(self.triangles[ids], query[rows])
                np.minimum.at(upper, rows, np.linalg.norm(near-query[rows], axis=1))
            result[start:start+len(query)] = upper
        return result

    def swept_capsule_distance(self, start, end, axis_height, search_radius):
        """Distance to the axis ribbon swept by an upright capsule.

        Its Minkowski sum with a sphere is the complete continuous swept volume,
        so a thin obstacle between two path stations cannot evade this check.
        The caller also checks that the initial station lies inside the cavity.
        """
        start, end = np.asarray(start), np.asarray(end)
        vertical = np.array([0., 0., axis_height/2])
        radius = np.linalg.norm(end-start)/2 + axis_height/2 + search_radius
        ids = self.candidates((start+end)/2, radius)
        if not len(ids):
            return float("inf")
        triangles = self.triangles[ids]
        if axis_height < 1e-12:
            return float(segment_triangle_distances(start, end, triangles).min())
        quad = np.array([start-vertical, end-vertical, end+vertical, start+vertical])
        distances = np.full(len(triangles), np.inf)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            distances = np.minimum(distances, segment_triangle_distances(quad[a], quad[b], triangles))
        for face in (quad[[0, 1, 2]], quad[[0, 2, 3]]):
            if np.linalg.norm(np.cross(face[1]-face[0], face[2]-face[0])) < 1e-14:
                continue
            plane = np.broadcast_to(face, triangles.shape)
            for a, b in ((0, 1), (1, 2), (2, 0)):
                distances = np.minimum(distances, segment_triangle_distances(
                    triangles[:, a], triangles[:, b], plane))
        return float(distances.min())
