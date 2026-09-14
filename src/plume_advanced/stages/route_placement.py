"""Bounded vertical corridor search without modifying the cave or body size."""

from __future__ import annotations

import numpy as np

from plume_advanced.progress import report_progress


def repair_vertical_path(
    index, points, lower, upper, distances, *, height, width, margin, attempts=3, max_queries=20000
):
    """Search local height bands around blocked edges; preserve XY and anchors.

    Each edge is accepted only after a continuous triangle/capsule distance
    check. Windows widen deterministically and restart from the original path.
    Failed/budget-exhausted searches return the unchanged path. This is a
    geometric corridor search, not ground-contact or robot dynamics planning.
    """
    points = np.asarray(points, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    radius = width / 2 + margin
    blocked = np.flatnonzero(np.asarray(distances) < radius - 1e-7)
    report = dict(
        method="local_vertical_bands",
        passed=False,
        queries=0,
        query_budget=max_queries,
        attempt_budget=attempts,
        attempts=[],
        original_blocked_edges=blocked.tolist(),
    )
    if not len(blocked):
        report.update(passed=True, changed_stations=0, maximum_vertical_change_m=0.0)
        return points, report
    if np.any(lower > upper) or not np.isfinite([lower, upper]).all():
        report["failure"] = "No finite vertical interval for the requested body"
        return points, report
    fractions = np.array([0.5, 0.25, 0.75, 0.1, 0.9, 0.05, 0.95])
    columns = np.repeat(points[:, None, :], len(fractions), axis=1)
    columns[:, :, 2] = lower[:, None] + (upper - lower)[:, None] * fractions
    columns[:, 0] = points
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    groups = np.split(blocked, np.flatnonzero(np.diff(blocked) > 1) + 1)
    cache = {(i, 0, 0): float(value) for i, value in enumerate(distances)}

    def clear(edge, previous, following):
        key = (edge, previous, following)
        if key not in cache:
            if report["queries"] >= max_queries:
                raise _QueryBudgetReached
            report["queries"] += 1
            cache[key] = index.swept_capsule_distance(
                columns[edge, previous], columns[edge + 1, following], height - width, radius
            )
            if report["queries"] % 250 == 0:
                report_progress(
                    "Route placement repair",
                    report["queries"],
                    max_queries,
                    "checking continuous capsule edges between local height bands",
                )
        return cache[key] >= radius - 1e-7

    for attempt in range(attempts):
        padding = max(1.0, 2 * height) * (2**attempt)
        windows: list[list[int]] = []
        for group in groups:
            begin = max(0, int(np.searchsorted(arc, arc[group[0]] - padding)) - 1)
            end = min(len(points) - 1, int(np.searchsorted(arc, arc[group[-1] + 1] + padding)))
            if windows and begin <= windows[-1][1]:
                windows[-1][1] = max(end, windows[-1][1])
            else:
                windows.append([begin, end])
        record = dict(padding_m=padding, windows=windows, passed=False)
        report["attempts"].append(record)
        trial = points.copy()
        try:
            for begin, end in windows:
                cost = np.full((end - begin + 1, len(fractions)), np.inf)
                parent = np.full(cost.shape, -1, dtype=np.int8)
                cost[0, 0] = 0.0
                for step in range(1, len(cost)):
                    station = begin + step
                    for following in [0] if station == end else range(len(fractions)):
                        point = columns[station, following]
                        if not lower[station] - 1e-9 <= point[2] <= upper[station] + 1e-9:
                            continue
                        ranked = sorted(
                            (
                                cost[step - 1, previous]
                                + float(np.sum((point - columns[station - 1, previous]) ** 2))
                                + float((point[2] - points[station, 2]) ** 2),
                                previous,
                            )
                            for previous in range(len(fractions))
                            if np.isfinite(cost[step - 1, previous])
                        )
                        for score, previous in ranked:
                            if clear(station - 1, previous, following):
                                cost[step, following] = score
                                parent[step, following] = previous
                                break
                    if not np.isfinite(cost[step]).any():
                        break
                if not np.isfinite(cost[-1, 0]):
                    record["failed_window"] = [begin, end]
                    break
                following = 0
                for step in range(len(cost) - 1, -1, -1):
                    trial[begin + step] = columns[begin + step, following]
                    following = int(parent[step, following])
            else:
                change = np.abs(trial[:, 2] - points[:, 2])
                record["passed"] = True
                report.update(
                    passed=True,
                    changed_stations=int(np.count_nonzero(change)),
                    maximum_vertical_change_m=float(change.max(initial=0)),
                )
                return trial, report
        except _QueryBudgetReached:
            report["failure"] = "Vertical placement query budget exhausted"
            return points, report
    report["failure"] = "No corridor found within the local height-band search budget"
    return points, report


class _QueryBudgetReached(Exception):
    pass
