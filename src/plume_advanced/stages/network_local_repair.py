"""Bounded position/heading search in a failed regional bend's local corridor.

The search proposes geometry only. It cannot relax acceptance thresholds, delete
branches or claim success before the complete network has been inspected again.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from plume_advanced.progress import report_progress
from plume_advanced.stages.network_layers import segment_xyz
from plume_advanced.stages.network_quality import _turn_metrics, assess_network


@dataclass(frozen=True)
class CurveSearch:
    points: np.ndarray | None
    expanded: int
    reason: str


@dataclass
class LocalRepairBudget:
    """Shared by successful and rejected proposals in one candidate."""
    limit: int
    expanded: int = 0

    @property
    def remaining(self):
        return max(0, self.limit - self.expanded)

    def provenance(self):
        return dict(local_search_expanded_states=self.expanded,
                    local_search_state_limit=self.limit,
                    local_search_budget_exhausted=self.remaining == 0)


def search_curve(start, end, first_heading, last_heading, radius, valid, cost,
                 *, maximum_expansions=1200):
    """Hybrid A*: exact arc primitives, quantized keys and exact end anchors.

    The finite work limit includes rejected states. A cubic terminal connector
    is allowed only when its sampled curvature and environment are acceptable.
    This is a geometric routing heuristic, not a lava flow solver.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    first_heading = np.asarray(first_heading, float)
    last_heading = np.asarray(last_heading, float)
    if (not all(v.shape == (2,) and np.isfinite(v).all()
                for v in (start, end, first_heading, last_heading))
            or min(np.linalg.norm(first_heading), np.linalg.norm(last_heading)) < 1e-9
            or not np.isfinite(radius) or radius <= 0
            or type(maximum_expansions) is not int or maximum_expansions < 0):
        raise ValueError("Local routing requires finite XY anchors, headings, positive radius and nonnegative work limit")
    step = max(.5, min(3., radius * .6))
    theta = float(np.arctan2(first_heading[1], first_heading[0]))
    spacing = max(.1, min(.5, radius / 6))
    goal_heading = last_heading / np.linalg.norm(last_heading)

    def connector(point, angle):
        length = float(np.linalg.norm(end - point))
        if length < 1e-8:
            return None
        direction = np.array([np.cos(angle), np.sin(angle)])
        for scale in (1., .65, 1.4):
            curve = CubicHermiteSpline([0., length], [point, end],
                                      [scale * direction, scale * goal_heading], axis=0)
            xy = curve(np.linspace(0, length, min(4096, max(5, int(np.ceil(length / spacing)) + 1))))
            xy[0], xy[-1] = point, end
            angles, radii = _turn_metrics(xy, np.ones(len(xy)))
            if (len(radii) and min(radii) >= radius and max(angles) < 30
                    and valid(xy)):
                return xy
        return None

    direct = connector(start, theta)
    if direct is not None:
        return CurveSearch(direct, 0, "anchored_curve")

    def key(point, angle):
        return (*np.rint((point - start) / (step * .5)).astype(int),
                int(round(angle / (np.pi / 16))) % 32)

    initial = key(start, theta)
    # Unique integer ids keep parent paths stable if a quantized state improves.
    states: list[tuple[np.ndarray, float, int, np.ndarray | None]] = [(start, theta, -1, None)]
    pending = [(float(np.linalg.norm(end-start)), 0., 0)]
    best = {initial: 0.}
    expanded = 0
    while pending and expanded < maximum_expansions:
        _, travel, index = heapq.heappop(pending)
        point, angle, _, _ = states[index]
        if travel > best.get(key(point, angle), float("inf")) + 1e-9:
            continue
        expanded += 1
        if expanded % 200 == 0:
            report_progress("Local bend search", expanded, maximum_expansions,
                            "checking headings, host and neighboring passages")
        distance = float(np.linalg.norm(end - point))
        if distance < 4 * step:
            tail = connector(point, angle)
            if tail is not None:
                parts = [tail]
                while states[index][2] >= 0:
                    piece = states[index][3]
                    assert piece is not None
                    parts.append(piece[:-1])
                    index = states[index][2]
                return CurveSearch(np.concatenate(parts[::-1]), expanded, "local_heading_search")
        for curvature in (0., -.5 / radius, .5 / radius, -1 / radius, 1 / radius):
            t = np.linspace(0., step, max(3, int(np.ceil(step / spacing)) + 1))
            if curvature == 0:
                xy = point + t[:, None] * [np.cos(angle), np.sin(angle)]
            else:
                angles = angle + curvature * t
                xy = point + np.column_stack((np.sin(angles) - np.sin(angle),
                                              np.cos(angle) - np.cos(angles))) / curvature
            final_angle = angle + curvature * step
            if not valid(xy):
                continue
            new_cost = travel + step * (1 + cost(xy) + .15 * abs(curvature * radius))
            state_key = key(xy[-1], final_angle)
            if new_cost >= best.get(state_key, float("inf")) - 1e-9:
                continue
            best[state_key] = new_cost
            states.append((xy[-1], final_angle, index, xy))
            heapq.heappush(pending, (new_cost + np.linalg.norm(end-xy[-1]),
                                    new_cost, len(states)-1))
    return CurveSearch(None, expanded, "work_limit" if pending else "no_local_route")


def _sample(host, points, field):
    coords = [(points[:, 1] - host.y_coords[0]) / np.diff(host.y_coords[:2])[0],
              (points[:, 0] - host.x_coords[0]) / np.diff(host.x_coords[:2])[0]]
    return map_coordinates(getattr(host, field), coords, order=1, mode="nearest")


class _Corridor:
    """Fast local proposal screen; full geometry screening remains authoritative."""

    def __init__(self, network, host, segment, reference, first, last, bound):
        from plume_advanced.stages.network_neighborhoods import PassageNeighborhoods

        self.host, self.segment, self.config = host, segment, network.config
        self.reference, self.bound = reference, bound
        self.tree = cKDTree(reference)
        self.low = reference[first:last+1].min(axis=0) - bound
        self.high = reference[first:last+1].max(axis=0) + bound
        self.low = np.maximum(self.low, [host.x_coords[0], host.y_coords[0]])
        self.high = np.minimum(self.high, [host.x_coords[-1], host.y_coords[-1]])
        self.xyz = segment_xyz(segment, network.config.layers)
        self.depths = np.array([p.elevation for p in segment.points]) - self.xyz[:, 2]
        self.widths = np.array([p.width for p in segment.points])
        arrays = [(s, segment_xyz(s, network.config.layers)) for s in network.segments]
        self.neighborhoods = PassageNeighborhoods(arrays)
        self.index = next(i for i, (s, _) in enumerate(arrays) if s.segment_id == segment.segment_id)
        self.others = []
        for i, (other, xyz) in enumerate(arrays):
            if other.segment_id == segment.segment_id:
                continue
            widths = np.array([p.width for p in other.points])
            margin = .7 * (self.widths.max() + widths.max())
            if (np.any(xyz[:, :2].max(axis=0) + margin < self.low)
                    or np.any(xyz[:, :2].min(axis=0) - margin > self.high)):
                continue
            self.others.append((i, widths, xyz, cKDTree(xyz[:, :2])))

    def valid(self, xy):
        if not np.isfinite(xy).all() or np.any(xy < self.low) or np.any(xy > self.high):
            return False
        distance, nearest = self.tree.query(xy)
        if max(distance) > self.bound:
            return False
        surface = _sample(self.host, xy, "elevation")
        for field in ("cover_thickness", "roof_competence"):
            values = _sample(self.host, xy, field)
            if not np.isfinite(values).all() or np.any(values <= 0):
                return False
        if not np.isfinite(_sample(self.host, xy, "growth_cost")).all():
            return False
        lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if (not np.isfinite(surface).all() or np.any(lengths < 1e-9)
                or np.any(np.diff(surface) / lengths > self.config.quality.maximum_uphill_grade)):
            return False
        return not self.conflicts(xy, surface, nearest).any()

    def conflicts(self, xy, surface=None, nearest=None):
        if surface is None:
            surface = _sample(self.host, xy, "elevation")
        if nearest is None:
            nearest = self.tree.query(xy)[1]
        z = surface - self.depths[nearest]
        width = self.widths[nearest]
        conflicts = np.zeros(len(xy), dtype=bool)
        for i, other_widths, xyz, tree in self.others:
            dist, close = tree.query(xy)
            widths = width + other_widths[close]
            conflict = dist < .7 * widths
            if self.config.layers.enabled:
                conflict &= abs(z - xyz[close, 2]) < (
                    self.config.layers.passage_height_m + self.config.layers.minimum_rock_m)
            conflict &= ~self.neighborhoods.local(self.index, nearest, i, close)
            conflicts |= conflict
        return conflicts

    def cost(self, xy):
        distance = self.tree.query(xy)[0]
        growth = np.maximum(_sample(self.host, xy, "growth_cost"), 0)
        return float(np.mean(.2 * (distance / self.bound)**2 + .1 * growth))


def repair_local_bends(generator, host, network, repair_pass, failed_checks, *, work_budget=None):
    """Search at most four failed bend windows; never change unrelated routes.

    Improvement means removing failed checks without introducing any new failed
    check. Exact limits and the complete host/flow/clearance inspection remain in
    force. Unsuccessful proposals leave the original geometry untouched.
    """
    cfg = generator.config
    if work_budget is None:
        work_budget = LocalRepairBudget(2400 * cfg.quality.repair_passes,
                                       network.backend_provenance.get("local_search_expanded_states", 0))
    targets = {sid for check in failed_checks
               if check["name"] in {"bend_radius_relative_to_width", "local_turn_angle",
                                     "regional_passage_clearance", "layer_passage_separation"}
               for sid in check.get("segment_ids", [])}
    history = list(network.backend_provenance.get("local_repair_history", []))
    current = network
    for sid in sorted(targets)[:4]:
        segment = next(s for s in current.segments if s.segment_id == sid)
        raw = np.array([[p.x, p.y] for p in segment.points])
        widths = np.array([p.width for p in segment.points])
        angles, radii = _turn_metrics(raw, widths)
        bad = np.flatnonzero((radii < cfg.quality.minimum_bend_radius_widths)
                             | (angles > cfg.quality.maximum_turn_degrees)) + 1
        if len(bad):
            center = int(bad[np.argmin(radii[bad-1])])
        else:
            probe = _Corridor(current, host, segment, raw, 0, len(raw)-1, widths.max())
            bad = np.flatnonzero(probe.conflicts(raw))
            if not len(bad):
                continue
            center = int(bad[len(bad)//2])
        arc = np.array([p.arc_length for p in segment.points])
        reach = (4 + repair_pass * 2) * widths[center]
        first = max(0, int(np.searchsorted(arc, arc[center] - reach)) - 1)
        last = min(len(raw)-1, int(np.searchsorted(arc, arc[center] + reach)))
        first_heading = raw[first+1] - raw[first] if first == 0 else raw[first]-raw[first-1]
        last_heading = raw[last]-raw[last-1] if last == len(raw)-1 else raw[last+1]-raw[last]
        first_heading /= np.linalg.norm(first_heading)
        last_heading /= np.linalg.norm(last_heading)
        corridor = _Corridor(current, host, segment, raw, first, last, (2+repair_pass)*widths.max())
        search = search_curve(raw[first], raw[last], first_heading, last_heading,
                              1.2 * widths[first:last+1].max() * cfg.quality.minimum_bend_radius_widths,
                              corridor.valid, corridor.cost,
                              maximum_expansions=min(1200, work_budget.remaining))
        work_budget.expanded += search.expanded
        record = dict(segment_id=sid, repair_pass=repair_pass+1, expanded_states=search.expanded,
                      strategy=search.reason, window_m=[float(arc[first]), float(arc[last])],
                      accepted=False)
        if search.points is not None:
            xy = np.concatenate((raw[:first], search.points, raw[last+1:]))
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            # Preserve values exactly outside the window; map its width envelope
            # monotonically onto the new local arclength.
            part_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(search.points, axis=0), axis=1))]
            part_widths = np.interp(arc[first] + part_arc / part_arc[-1] * (arc[last]-arc[first]),
                                    arc, widths)
            new_widths = np.r_[widths[:first], part_widths, widths[last+1:]]
            points = []
            for index, (position, station, width) in enumerate(zip(xy, length, new_widths)):
                sample = host.sample(*position)
                points.append(replace(segment.points[0], index=index, x=float(position[0]),
                                      y=float(position[1]), arc_length=float(station), width=float(width),
                                      elevation=sample.elevation, growth_cost=sample.growth_cost,
                                      cover_thickness=sample.cover_thickness,
                                      roof_competence=sample.roof_competence,
                                      slope_degrees=sample.slope_degrees))
            changed = replace(segment, points=tuple(points), metadata=dict(segment.metadata,
                               quality_repair_kind="local_heading_search"))
            segments = [changed if s.segment_id == sid else s for s in current.segments]
            candidate = generator._finish_network(
                host, generator._build_flow_geometry(host), list(current.nodes), segments,
                backend_provenance=current.backend_provenance,
                skeleton_mask=np.zeros_like(host.growth_cost, dtype=bool),
                total_flux=np.zeros_like(host.growth_cost),
                preserved_segments={s.segment_id for s in segments})
            before, after = assess_network(current, host), assess_network(candidate, host)
            old = {c["name"] for c in before["checks"] if not c["passed"]}
            new = {c["name"] for c in after["checks"] if not c["passed"]}
            # If several segments share a failure, removing one witness counts
            # as progress without demanding a whole-network repair in one move.
            old_witnesses = {(c["name"], s) for c in before["checks"] if not c["passed"]
                             for s in c.get("segment_ids", [])}
            new_witnesses = {(c["name"], s) for c in after["checks"] if not c["passed"]
                             for s in c.get("segment_ids", [])}
            record.update(remaining_failures=sorted(new),
                          accepted=new <= old and new_witnesses < old_witnesses)
            if record["accepted"]:
                current = candidate
        history.append(record)
    return replace(current, backend_provenance=dict(current.backend_provenance,
                                                    local_repair_history=history,
                                                    **work_budget.provenance()))
