"""Conservative continuous oriented-box queries against indexed triangles.

Translation uses the separating-axis theorem over the complete time interval.
Rotation is enclosed by inflated, fixed-orientation boxes and subdivided only
when that enclosure meets a triangle. Exhausting the subdivision budget rejects
the edge; it never converts an uncertain result to a clear route.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def translating_box_hits(triangles, start, end, basis, half_extents):
    """Triangle/box overlap anywhere along a translation, including endpoints."""
    tri = (np.asarray(triangles, float) - start) @ basis
    if not len(tri):
        return False
    velocity = (np.asarray(end) - start) @ basis
    edges = np.roll(tri, -1, axis=1) - tri
    axes = [np.broadcast_to(axis, (len(tri), 3)) for axis in np.eye(3)]
    axes.append(np.cross(edges[:, 0], edges[:, 1]))
    axes.extend(np.cross(edges[:, edge], axis) for edge in range(3) for axis in np.eye(3))
    enter, leave = np.zeros(len(tri)), np.ones(len(tri))
    possible = np.ones(len(tri), dtype=bool)
    for axis in axes:
        projected = np.einsum("nvc,nc->nv", tri, axis)
        radius = np.abs(axis) @ np.asarray(half_extents)
        lower, upper = projected.min(axis=1) - radius, projected.max(axis=1) + radius
        speed = axis @ velocity
        stationary = np.abs(speed) < 1e-14
        possible &= ~stationary | ((lower <= 1e-10) & (upper >= -1e-10))
        a = np.divide(lower, speed, out=np.full(len(tri), -np.inf), where=~stationary)
        b = np.divide(upper, speed, out=np.full(len(tri), np.inf), where=~stationary)
        enter = np.maximum(enter, np.minimum(a, b))
        leave = np.minimum(leave, np.maximum(a, b))
        possible &= enter <= leave + 1e-10
        if not possible.any():
            return False
    return bool(possible.any())


def sweep_box(index, start, end, start_basis, end_basis, half_extents, *, debit=None):
    """Return clear fixed-orientation enclosures, or None on collision/uncertainty.

    Bases have columns (forward, left, up). The returned enclosures are also the
    native-engine test plan, so import checks cover exactly the accepted motion.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    half = np.asarray(half_extents, float)
    rotations = Rotation.from_matrix(np.stack([start_basis, end_basis]))
    slerp = Slerp([0., 1.], rotations)
    angle = float((rotations[0].inv() * rotations[1]).magnitude())
    result = []

    def visit(a, b, depth):
        if debit is not None:
            debit()
        first, last = start + a * (end-start), start + b * (end-start)
        basis = slerp((a+b)/2).as_matrix()
        # Every rotated corner stays within this Euclidean distance of its
        # middle orientation; inflating each axis encloses that sphere.
        inflation = 2 * np.linalg.norm(half) * np.sin(angle * (b-a) / 4)
        padded = half + inflation
        ids = index.candidates((first+last)/2, np.linalg.norm(last-first)/2 + np.linalg.norm(padded))
        if not translating_box_hits(index.triangles[ids], first, last, basis, padded):
            result.append(dict(start_m=first.tolist(), end_m=last.tolist(),
                               forward=basis[:, 0].tolist(), up=basis[:, 2].tolist(),
                               half_extents_m=padded.tolist()))
            return True
        if depth >= 8 or inflation <= 0.0005:
            return False
        middle = (a+b)/2
        return visit(a, middle, depth+1) and visit(middle, b, depth+1)

    return result if visit(0., 1., 0) else None
