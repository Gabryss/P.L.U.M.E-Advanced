"""Continuous point-path witnesses on the delivered cave, without robot limits."""

import numpy as np

from plume_advanced.progress import report_progress
from plume_advanced.stages.triangle_queries import TriangleIndex, VerticalTriangleIndex


class ConnectionInspector:
    """Share triangle indexes across paths; the cavity boundary must be closed.

    Failure means unresolved along this witness, not absence of another path.
    Placed props are separate map obstacles, not part of the cavity boundary.
    """

    def __init__(self, vertices, faces):
        report_progress("Vector spatial index", detail=f"indexing {len(faces):,} cave triangles")
        self.index = TriangleIndex(vertices, faces)
        self.vertical = VerticalTriangleIndex(vertices, faces, triangles=self.index.triangles)

    def connection(self, points):
        points = np.asarray(points, float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all()):
            raise ValueError("Connection witness needs at least two finite XYZ stations")
        index, vertical = self.index, self.vertical
        tolerance = max(vertical.tolerance, 1e-6)
        if not np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance):
            raise ValueError("Connection witness must have nonzero length")
        outside = []
        headroom = []
        for i, point in enumerate(points):
            if i % 100 == 0:
                report_progress("Vector cavity stations", i, len(points), "robot-independent interior checks")
            measured = vertical.measure(point)
            if not measured["inside"]:
                outside.append(i)
            elif measured["clearance_m"] is not None:
                headroom.append(measured["clearance_m"])
        blocked = []
        uncertain = []
        for i, (a, b) in enumerate(zip(points, points[1:])):
            if i % 100 == 0:
                report_progress("Vector cavity segments", i, len(points)-1, "continuous line-to-triangle checks")
            distance = index.swept_capsule_distance(a, b, 0., tolerance)
            if np.isnan(distance):
                uncertain.append(i)
            elif distance <= tolerance:
                blocked.append(i)
        verified = not outside and not blocked and not uncertain
        report_progress("Vector cavity segments", len(points)-1, len(points)-1,
                        "Witness verified" if verified else "Connection unresolved along supplied witness")
        return dict(status="verified" if verified else "unresolved",
                    connected=True if verified else None, stations=len(points),
                    outside_stations=outside, surface_contact_intervals=blocked,
                    unresolved_distance_intervals=uncertain,
                    numerical_tolerance_m=tolerance,
                    minimum_sampled_vertical_clearance_m=min(headroom) if headroom else None,
                    scope="Continuous supplied centreline inside the closed cave boundary; no robot or minimum passage-size claim. An unresolved witness is not proof of disconnection.")


def centreline_connection(vertices, faces, points):
    """Check a supplied nonzero polyline with no robot size, slope or floor rule."""
    return ConnectionInspector(vertices, faces).connection(points)
