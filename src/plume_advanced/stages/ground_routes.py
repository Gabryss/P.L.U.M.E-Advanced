"""Deterministic, bounded ground-route inspection on actual mesh triangles.

This reference chassis envelope is a geometric test, not a wheel/track dynamics
model. Floor support is sampled at the reported spacing; narrower gaps can be
missed. Body motion is conservatively enclosed continuously between stations.
No floor is flattened and no robot limit is relaxed to obtain a passing route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from plume_advanced.progress import report_progress
from plume_advanced.stages.box_queries import sweep_box
from plume_advanced.stages.route_clearance import resample_route_polyline
from plume_advanced.stages.triangle_queries import TriangleIndex, VerticalTriangleIndex


@dataclass(frozen=True)
class GroundRobot:
    length_m: float = .7
    width_m: float = .5
    height_m: float = .5
    margin_m: float = .02
    max_slope_deg: float = 20.
    max_step_m: float = .1
    support_spacing_m: float = .1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"Ground robot {name} must be finite and nonnegative")
        if min(self.length_m, self.width_m, self.height_m, self.support_spacing_m) <= 0:
            raise ValueError("Ground body dimensions and support spacing must be positive")
        if not 0 < self.max_slope_deg < 90:
            raise ValueError("Ground maximum slope must be between 0 and 90 degrees")
        if self.support_spacing_m > min(self.length_m, self.width_m)/2:
            raise ValueError("Ground support spacing must resolve each footprint axis at least twice")


class GroundQueryBudget(Exception):
    pass


class GroundInspector:
    """One shared index/budget for every required path and repair candidate."""

    def __init__(self, vertices, faces, robot, max_queries):
        self.robot = robot
        self.index = TriangleIndex(vertices, faces)
        self.vertical = VerticalTriangleIndex(vertices, faces, triangles=self.index.triangles)
        self.queries = 0
        self.max_queries = max_queries
        # The half extents include the declared safety margin, on every side.
        self.half = np.array([robot.length_m, robot.width_m, robot.height_m])/2 + robot.margin_m
        axes = [np.linspace(-size/2, size/2, int(np.ceil(size/robot.support_spacing_m))+1)
                for size in (robot.length_m+2*robot.margin_m, robot.width_m+2*robot.margin_m)]
        self.shape = tuple(map(len, axes))
        self.offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        self.fit = np.linalg.pinv(np.column_stack([self.offsets, np.ones(len(self.offsets))]))

    def debit(self):
        if self.queries >= self.max_queries:
            raise GroundQueryBudget
        self.queries += 1
        if self.queries % 250 == 0:
            report_progress("Ground robot inspection", self.queries, self.max_queries,
                            "checking floor support, chassis poses and continuous box motion")

    def pose(self, point, heading):
        self.debit()
        left = np.array([-heading[1], heading[0], 0.])
        probes = point + self.offsets[:, :1]*heading + self.offsets[:, 1:]*left
        floor_values = []
        for probe in probes:
            row = self.vertical.measure(probe)
            if not row["inside"]:
                return dict(passed=False, failure="missing_floor_or_wrong_cavity")
            floor_values.append(probe[2] - row["floor_distance_m"])
        floors = np.asarray(floor_values)
        coeff = self.fit @ floors
        slope = float(np.degrees(np.arctan(np.linalg.norm(coeff[:2]))))
        residual = floors - (self.offsets @ coeff[:2] + coeff[2])
        # Remove a fitted ramp before measuring abrupt height changes. Also
        # bound the full residual span, so a pit cannot hide between neighbours.
        step = float(np.ptp(residual))
        local = residual.reshape(self.shape)
        for axis in (0, 1):
            step = max(step, float(np.abs(np.diff(local, axis=axis)).max(initial=0)))
        row = dict(passed=False, slope_deg=slope, step_m=step)
        if slope > self.robot.max_slope_deg + 1e-7:
            return row | dict(failure="slope_limit")
        if step > self.robot.max_step_m + 1e-7:
            return row | dict(failure="step_or_support_gap_limit")
        normal = np.array([0., 0., 1.]) - coeff[0]*heading - coeff[1]*left
        normal /= np.linalg.norm(normal)
        forward = heading + np.array([0., 0., coeff[0]])
        forward /= np.linalg.norm(forward)
        # A 3 mm numerical skin prevents coincident floor/body contact. It is
        # reported and added above (never deducted from) the requested margin.
        center = np.array([point[0], point[1], coeff[2]]) + normal * (
            residual.max()*normal[2] + self.half[2] + .003)
        witnesses = np.column_stack([probes[:, :2], floors])
        return row | dict(passed=True, center_m=center.tolist(), forward=forward.tolist(),
                          up=normal.tolist(), floor_points_m=witnesses.tolist(),
                          probe_height_m=float(point[2]))

    def path(self, points, begin=0, end=None, *, fallback_heading=None):
        end = len(points)-1 if end is None else end
        tangent = np.gradient(points, axis=0)
        tangent[:, 2] = 0
        lengths = np.linalg.norm(tangent, axis=1)
        if fallback_heading is not None and np.all(lengths < 1e-9):
            tangent[:] = fallback_heading
            lengths[:] = 1.
        if np.any(lengths < 1e-9):
            return dict(passed=False, failed_stations=list(range(begin, end+1)),
                        failed_edges=[], poses=[], sweeps=[], failure="undefined_plan_heading")
        tangent /= lengths[:, None]
        poses = [self.pose(points[i], tangent[i]) for i in range(begin, end+1)]
        return self.check_poses(poses, begin)

    def check_poses(self, poses, begin=0):
        bad = [i+begin for i, pose in enumerate(poses) if not pose["passed"]]
        edges, sweeps = [], []
        for i, (a, b) in enumerate(zip(poses, poses[1:])):
            if not a["passed"] or not b["passed"]:
                edges.append(i+begin)
                continue
            def basis(p):
                return np.column_stack([p["forward"], np.cross(p["up"], p["forward"]), p["up"]])
            motion = sweep_box(self.index, a["center_m"], b["center_m"], basis(a), basis(b),
                               self.half, debit=self.debit)
            if motion is None:
                edges.append(i+begin)
            else:
                sweeps.extend(motion)
        return dict(passed=not bad and not edges, failed_stations=bad, failed_edges=edges,
                    poses=poses, sweeps=sweeps)

    def junction(self, point, first, last):
        # Front/back are interchangeable for this symmetric chassis envelope.
        # The maneuver may use reverse motion; no steering-radius claim is made.
        a, b = np.arctan2(first[1], first[0]), np.arctan2(last[1], last[0])
        angle = (b-a+np.pi/2) % np.pi - np.pi/2
        count = max(2, int(np.ceil(np.linalg.norm(self.half[:2])*abs(angle)/self.robot.support_spacing_m))+1)
        poses = [self.pose(point, np.array([np.cos(yaw), np.sin(yaw), 0.]))
                 for yaw in np.linspace(a, a+angle, count)]
        return self.check_poses(poses)


def _repair_path(inspector, points, initial, attempts):
    """Try local smooth lateral detours with fixed window/graph endpoints."""
    if initial.get("failure") == "undefined_plan_heading":
        return points, dict(passed=False, failure="No valid horizontal route heading", windows=[])
    if np.linalg.norm(np.ptp(points[:, :2], axis=0)) < 1e-9:
        # Incident section centres can differ only in height. Their shared
        # anchor cannot be relocated by a lateral detour, even when the borrowed
        # incident heading produced a valid (but blocked) floor pose.
        return points, dict(passed=False, failure="Anchored vertical connector has no lateral detour", windows=[])
    failed = np.unique([*initial["failed_stations"], *initial["failed_edges"],
                        *(i+1 for i in initial["failed_edges"])]).astype(int)
    groups = np.split(failed, np.flatnonzero(np.diff(failed) > 1)+1)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    padding = max(3., inspector.robot.length_m*6)
    windows: list[list[int]] = []
    for group in groups:
        begin = max(0, int(np.searchsorted(arc, arc[group[0]]-padding))-1)
        end = min(len(points)-1, int(np.searchsorted(arc, arc[group[-1]]+padding)))
        if windows and begin <= windows[-1][1]:
            windows[-1][1] = max(end, windows[-1][1])
        else:
            windows.append([begin, end])
    repaired = points.copy()
    records = []
    tangent = np.gradient(points, axis=0)[:, :2]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    left = np.column_stack([-tangent[:, 1], tangent[:, 0], np.zeros(len(points))])
    for begin, end in windows:
        record: dict = dict(window=[begin, end], candidates=[], passed=False)
        records.append(record)
        if end <= begin:
            break
        t = (arc[begin:end+1]-arc[begin])/(arc[end]-arc[begin])
        weight = np.sin(np.pi*t)**2
        for attempt in range(attempts):
            for sign in (1, -1):
                offset = sign * inspector.robot.width_m * .5 * (2**attempt)
                trial = repaired.copy()
                trial[begin:end+1] += left[begin:end+1] * (offset*weight[:, None])
                # Preserve graph anchors exactly, including floating point bits.
                trial[[begin, end]] = repaired[[begin, end]]
                checked = inspector.path(trial, max(0, begin-1), min(len(points)-1, end+1))
                record["candidates"].append(dict(offset_m=offset, passed=checked["passed"]))
                if checked["passed"]:
                    repaired, record["passed"] = trial, True
                    break
            if record["passed"]:
                break
        if not record["passed"]:
            return points, dict(passed=False, windows=records)
    return repaired, dict(passed=all(r["passed"] for r in records), windows=records)


def inspect_ground_routes(vertices, faces, paths, segment_ids, *, robot=None,
                          repair_attempts=3, max_queries=200000):
    """Inspect and, where possible, relocate routes without editing the surface."""
    robot = GroundRobot() if robot is None else robot
    if type(repair_attempts) is not int or not 0 <= repair_attempts <= 4:
        raise ValueError("Ground repair attempts must be an integer from 0 to 4")
    if type(max_queries) is not int or not 0 <= max_queries <= 2_000_000:
        raise ValueError("Ground query budget must be an integer from 0 to 2000000")
    report: dict = dict(enabled=True, passed=False, robot=asdict(robot), paths=[], junctions=[], failures=[],
                  defect_regions=[], query_budget=max_queries, queries=0,
                  numerical_skin_m=.003, repair_attempt_budget=repair_attempts,
                  scope="Sampled floor support and continuous conservative chassis clearance; "
                        "not wheel/track dynamics, friction or controller certification")
    if not paths or len(paths) != len(segment_ids):
        report["failures"] = ["Ground routes are missing or segment IDs do not match"]
        return report
    inspector = GroundInspector(vertices, faces, robot, max_queries)
    report["support_offsets_m"] = inspector.offsets.tolist()
    report["support_fit_weights"] = inspector.fit.tolist()
    endpoints: dict[tuple, list] = {}
    try:
        for number, (path, sid) in enumerate(zip(paths, segment_ids, strict=True)):
            points = np.asarray(resample_route_polyline(path, robot.support_spacing_m), float)
            fallback = None
            if sid < 0 and np.linalg.norm(np.ptp(points[:, :2], axis=0)) < 1e-9:
                # Stage-C section centres can differ only in Z at one graph
                # node. Such a connector is a floor pose/turn, not vertical
                # driving. Borrow an incident path's horizontal heading.
                for neighbour in paths:
                    neighbour = np.asarray(neighbour, float)
                    for i, j in ((0, 1), (-1, -2)):
                        direction = neighbour[j]-neighbour[i]
                        direction[2] = 0.
                        norm = np.linalg.norm(direction)
                        if norm > 1e-9 and np.linalg.norm(neighbour[i, :2]-points[0, :2]) < 1e-8:
                            fallback = direction/norm
                            break
                    if fallback is not None:
                        break
            original = inspector.path(points, fallback_heading=fallback)
            checked = original
            repair = dict(passed=original["passed"], attempted=False)
            if not original["passed"] and repair_attempts:
                moved, repair = _repair_path(inspector, points, original, repair_attempts)
                repair["attempted"] = True
                if repair["passed"]:
                    # Fresh evaluation of the entire final route, not search cache.
                    checked = inspector.path(moved)
                    repair["verified"] = checked["passed"]
                    repair["passed"] = checked["passed"]
                    repair["maximum_lateral_change_m"] = float(np.linalg.norm(moved-points, axis=1).max())
                    if checked["passed"]:
                        points = moved
            record = dict(path_index=number, segment_id=sid, samples=len(points),
                          plan_path_m=points.tolist(), placement_repair=repair,
                          initial_failed_stations=original["failed_stations"],
                          initial_failed_edges=original["failed_edges"], **checked)
            report["paths"].append(record)
            if checked["passed"]:
                for i in (0, -1):
                    endpoints.setdefault(tuple(np.asarray(path)[i]), []).append(
                        (sid, checked["poses"][i]["forward"]))
            if not checked["passed"]:
                report["failures"].append(f"Required ground route {sid} fails reference robot limits")
                # Keep this separate from morphologic defects: floor checks must
                # not trigger a relief-erasing local topology repair.
                report["defect_regions"].append(dict(kind="ground_route", segment_ids=[sid],
                    lower_m=points.min(axis=0).tolist(), upper_m=points.max(axis=0).tolist()))
        if not report["failures"]:
            for point, ends in endpoints.items():
                for sid, heading in ends[1:]:
                    joined = inspector.junction(np.asarray(point), ends[0][1], heading)
                    report["junctions"].append(dict(point_m=list(point), segment_ids=[ends[0][0], sid],
                                                    samples=len(joined['poses']), **joined))
                    if not joined["passed"]:
                        report["failures"].append(f"Ground robot cannot turn between routes {ends[0][0]} and {sid}")
    except GroundQueryBudget:
        report["failures"].append("Ground inspection/repair query budget exhausted")
    report["queries"] = inspector.queries
    report["passed"] = not report["failures"] and len(report["paths"]) == len(paths)
    report_progress("Ground robot inspection", inspector.queries, inspector.queries,
                    "reference route accepted" if report["passed"] else "reference route rejected")
    return report
