"""Persistent, interacting routes and spatial acceptance for a shared host.

These are procedural constraints, not calibrated thermofluid equations. Route
preferences continue after a merge; transported source lineage is maintained by
the common flow ledger separately from the identity of each routing preference.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.ndimage import binary_fill_holes, gaussian_filter1d, label, map_coordinates

from plume_advanced.procedural import procedural_rng


@dataclass(frozen=True)
class InterconnectionConfig:
    lookahead_widths: float = 6.0
    maximum_junction_angle_degrees: float = 45.0
    minimum_parallel_fraction: float = 0.40
    minimum_window_parallel_fraction: float = 0.20
    minimum_parallel_run_widths: float = 8.0
    maximum_single_run_fraction: float = 0.40
    minimum_source_independent_length_widths: float = 5.0
    minimum_clearance_widths: float = 0.25
    interaction_window_m: float = 600.0
    minimum_interactions_per_km: float = 2.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"network.interconnection.{name} must be finite and positive")
            if name.endswith("fraction") and value >= 1:
                raise ValueError(f"network.interconnection.{name} must be below one")
        if self.maximum_junction_angle_degrees >= 90:
            raise ValueError("interconnection junction angle must be below 90 degrees")


def route_preferences(generator, host, geometry):
    """Grow seeded, persistent headings through the unmodified shared host.

    Each front scores a fan of forward trajectories, including their intermediate
    substrate samples and signed uphill grade. Long-range independent targets
    remain centred on their source offsets; there is no decay to a shared trunk.
    The final lateral-order projection prevents implicit source swaps. Explicit
    capture/release planning subsequently turns contact into graph connections.
    """
    from plume_advanced.stages.network_systems import _cross_bounds, enforce_source_order

    cfg, controls = generator.config, generator.config.systems
    width = 2 * cfg.base_passage_radius
    step = max(0.5, min(width / 3, geometry.cell_scale))
    along = np.linspace(0, geometry.along_extent, max(8, int(geometry.along_extent / step) + 1))
    step = float(along[1] - along[0])
    lower, upper = _cross_bounds(host, geometry, along, 2 * width)
    sources = (
        (np.arange(controls.count) - (controls.count - 1) / 2)
        * controls.source_spacing_widths
        * width
    )
    if sources[0] < lower[0] or sources[-1] > upper[0]:
        raise ValueError("Interconnected sources do not fit inside the supplied host")
    correlation = controls.correlation_length_widths * width
    knots = np.linspace(0, along[-1], max(4, int(np.ceil(along[-1] / correlation)) + 1))
    horizon = cfg.interconnection.lookahead_widths * width

    def sample(field, a, c):
        x = geometry.seed_x + a * geometry.flow_x + c * geometry.cross_x
        y = geometry.seed_y + a * geometry.flow_y + c * geometry.cross_y
        ix = (x - host.x_coords[0]) / (host.x_coords[1] - host.x_coords[0])
        iy = (y - host.y_coords[0]) / (host.y_coords[1] - host.y_coords[0])
        return map_coordinates(field, [iy, ix], order=1, mode="nearest")

    tracks = []
    for system_id, source in enumerate(sources):
        rng = procedural_rng(cfg.random_seed, "network-system", system_id)
        offsets = rng.normal(0, controls.lateral_variation_widths * width, len(knots))
        offsets[0] = 0
        target = source + CubicSpline(knots, offsets, bc_type="natural")(along)
        target = np.clip(target, lower, upper)
        track = np.empty_like(along)
        track[0], heading = source, 0.0
        for i in range(1, len(along)):
            reach = min(horizon, along[-1] - along[i - 1])
            # Bound curvature during growth, not just by rejecting sharp turns.
            turn = min(0.10, step / (2 * width))
            headings = np.unique(np.clip(heading + np.linspace(-turn, turn, 11), -0.72, 0.72))
            slopes = np.tan(headings)
            distances = np.linspace(0, reach, 5)
            a = np.broadcast_to(along[i - 1] + distances, (len(slopes), 5))
            c = track[i - 1] + slopes[:, None] * distances
            costs = sample(host.growth_cost, a, c)
            heights = sample(host.elevation, a, c)
            grade = np.diff(heights, axis=1) / np.maximum(
                np.diff(distances)[None, :] * np.sqrt(1 + slopes[:, None] ** 2), 1e-9
            )
            limit = cfg.quality.maximum_uphill_grade
            score = cfg.growth_cost_weight * costs.mean(axis=1)
            score += ((c[:, -1] - np.interp(a[0, -1], along, target)) / (2 * width)) ** 2
            score += 0.8 * ((headings - heading) / max(turn, 1e-9)) ** 2
            score += 20 * np.maximum(0, grade - limit).max(axis=1)
            inside = np.all(
                (c >= np.interp(a, along, lower)) & (c <= np.interp(a, along, upper)), axis=1
            )
            inside &= np.all(grade <= limit, axis=1)
            score[~inside] = np.inf
            if not np.isfinite(score).any():
                raise ValueError("No bounded forward route remains inside the supplied host")
            heading = float(headings[int(np.argmin(score))])
            track[i] = track[i - 1] + math.tan(heading) * step
        tracks.append(track)
    tracks = gaussian_filter1d(np.asarray(tracks), max(1, width / step), axis=1, mode="nearest")
    # Stable pool-adjacent-violators projection; membership is never sorted away.
    enforce_source_order(tracks)
    # Restore the exact source locations after filtering.
    fade = np.clip(along / (2 * width), 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    tracks += (sources - tracks[:, 0])[:, None] * (1 - fade)
    return along, tracks, width


def connection_guard(generator, host, geometry, along, tracks, width):
    """Screen local direction, substrate and viable split allocations.

    This only proposes a connection: final Stage-C grade, continuity, clearance
    and the common discharge ledger still have to pass after endpoint blending.
    """
    cfg = generator.config
    slopes = np.gradient(tracks, along, axis=1)

    def permitted(kind, left, right, index):
        direction = [float(np.arctan(np.mean(slopes[list(g), index]))) for g in (left, right)]
        if (
            abs(np.degrees(direction[0] - direction[1]))
            > cfg.interconnection.maximum_junction_angle_degrees
        ):
            return False
        if (
            kind == "split"
            and min(len(left), len(right)) / (len(left) + len(right))
            < cfg.lobe_growth.minimum_viable_flux_fraction
        ):
            return False
        centres = [float(np.mean(tracks[list(g), index])) for g in (left, right)]
        cross = np.linspace(centres[0], centres[1], 7)
        samples = [
            host.sample(
                geometry.seed_x + along[index] * geometry.flow_x + c * geometry.cross_x,
                geometry.seed_y + along[index] * geometry.flow_y + c * geometry.cross_y,
            )
            for c in cross
        ]
        # Refuse host grades that cannot be bridged within the easing reach.
        relief = max(p.elevation for p in samples) - min(p.elevation for p in samples)
        reach = max(4 * width, abs(centres[1] - centres[0]))
        return relief / reach <= cfg.quality.maximum_uphill_grade and all(
            p.cover_thickness > 0 and p.roof_competence > 0 and np.isfinite(p.growth_cost)
            for p in samples
        )

    return permitted


def smooth_routes(host, segments, axis):
    """Smooth in downstream coordinates with shared junction tangents.

    Unlike a free 2D cubic, this cannot double back in downstream position.
    Reusing it after repair preserves the connection tangents as well as nodes.
    Grade, curvature and nonlocal overlap still undergo the acceptance gate.
    """
    axis = np.asarray(axis)
    cross = np.array([-axis[1], axis[0]])
    prepared = {}
    directions: dict[int, list[float]] = {}
    for s in segments:
        if s.metadata.get("topology_role") == "side_branch":
            continue
        xy = np.array([[p.x, p.y] for p in s.points])
        a, c = xy @ axis, xy @ cross
        if np.any(np.diff(a) <= 0):
            raise ValueError("Interconnected route reversed downstream during construction")
        length = a[-1] - a[0]
        positions = np.linspace(
            a[0], a[-1], max(5, int(np.ceil(length / min(2.0, s.mean_width / 3))) + 1)
        )
        values = np.interp(positions, a, c)
        sigma = min(2 * s.mean_width, 0.12 * length)
        values = gaussian_filter1d(
            values, max(0.5, sigma / (positions[1] - positions[0])), mode="nearest"
        )
        t = (positions - positions[0]) / length
        values += (1 - t) * (c[0] - values[0]) + t * (c[-1] - values[-1])
        curve = PchipInterpolator(positions, values)
        for node, pos in ((s.start_node_id, positions[0]), (s.end_node_id, positions[-1])):
            directions.setdefault(node, []).append(float(curve(pos, 1)))
        prepared[s.segment_id] = (a, positions, curve)
    slopes = {
        node: float(np.clip(np.mean(values), -0.5, 0.5)) for node, values in directions.items()
    }
    result = []
    for s in segments:
        if s.segment_id not in prepared:
            result.append(s)
            continue
        old_a, a, curve = prepared[s.segment_id]
        c = curve(a)
        reach = min(0.45 * (a[-1] - a[0]), 6 * s.mean_width)
        start = CubicHermiteSpline(
            [a[0], a[0] + reach],
            [c[0], curve(a[0] + reach)],
            [slopes[s.start_node_id], curve(a[0] + reach, 1)],
        )
        end = CubicHermiteSpline(
            [a[-1] - reach, a[-1]],
            [curve(a[-1] - reach), c[-1]],
            [curve(a[-1] - reach, 1), slopes[s.end_node_id]],
        )
        c[a <= a[0] + reach] = start(a[a <= a[0] + reach])
        c[a >= a[-1] - reach] = end(a[a >= a[-1] - reach])
        xy = a[:, None] * axis + c[:, None] * cross
        # Restore exact world coordinates to avoid accumulated round-off at nodes.
        xy[0] = [s.points[0].x, s.points[0].y]
        xy[-1] = [s.points[-1].x, s.points[-1].y]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        widths = np.interp(a, old_a, [p.width for p in s.points])
        points = []
        for i, (p, distance, width) in enumerate(zip(xy, arc, widths)):
            sample = host.sample(*p)
            points.append(
                replace(
                    s.points[0],
                    index=i,
                    x=float(p[0]),
                    y=float(p[1]),
                    arc_length=float(distance),
                    width=float(width),
                    elevation=sample.elevation,
                    slope_degrees=sample.slope_degrees,
                    cover_thickness=sample.cover_thickness,
                    roof_competence=sample.roof_competence,
                    growth_cost=sample.growth_cost,
                )
            )
        result.append(replace(s, points=tuple(points)))
    return result


def spatial_metrics(network, sections=None):
    """Count physically separated envelopes, not source labels or edge counts.

    The Stage-B estimate uses widths; Stage C uses the actual world-space profile
    extents. Separate intervals need a minimum rock gap to count as two channels.
    Blind tips are excluded so they cannot satisfy the parallel-route target.
    """
    cfg = network.config
    width = 2 * cfg.base_passage_radius
    axis = np.array(network.backend_provenance.get("flow_direction", [0.0, 1.0]), dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    cross = np.array([-axis[1], axis[0]])
    fields = {} if sections is None else {f.segment_id: f for f in sections.segment_fields}
    rows = []
    for s in network.segments:
        if s.metadata.get("topology_role") == "side_branch":
            continue
        if sections is None:
            xy = np.array([[p.x, p.y] for p in s.points])
            a, c = xy @ axis, xy @ cross
            half = np.array([p.width / 2 for p in s.points])
            rows.append(np.column_stack((a, c - half, c + half)))
        else:
            profile_rows = []
            for p in fields[s.segment_id].samples:
                q = np.array(p.profile_points)
                xy = (
                    np.array([p.x, p.y])
                    + q[:, :1] * np.array(p.normal[:2])
                    + q[:, 1:] * np.array(p.binormal[:2])
                )
                profile_rows.append(
                    [np.dot([p.x, p.y], axis), float((xy @ cross).min()), float((xy @ cross).max())]
                )
            rows.append(np.asarray(profile_rows))
    if not rows or any(len(r) < 2 or not np.isfinite(r).all() for r in rows):
        raise ValueError("Invalid interconnected passage envelopes")
    # Growth is downstream; reject folded routes instead of sorting them into
    # apparently valid transverse intervals.
    if any(np.any(np.diff(r[:, 0]) <= 0) for r in rows):
        raise ValueError("Interconnected passage turns back across downstream stations")
    lo, hi = min(r[0, 0] for r in rows), max(r[-1, 0] for r in rows)
    n = max(16, int(np.ceil((hi - lo) / min(2.0, width / 4))))
    boundaries = np.linspace(lo, hi, n + 1)
    stations = (boundaries[:-1] + boundaries[1:]) / 2
    step = float(boundaries[1] - boundaries[0])
    intervals: list[list[tuple[float, float]]] = [[] for _ in stations]
    for r in rows:
        indices = np.flatnonzero((stations >= r[0, 0]) & (stations < r[-1, 0]))
        low, high = (np.interp(stations[indices], r[:, 0], r[:, i]) for i in (1, 2))
        for i, a, b in zip(indices, low, high):
            intervals[i].append((float(a), float(b)))
    station_counts, gaps = [], []
    clearance = cfg.interconnection.minimum_clearance_widths * width
    for interval in intervals:
        count, end = 0, -np.inf
        for a, b in sorted(interval):
            if a - end >= clearance:
                if count:
                    gaps.append(a - end)
                count += 1
            end = max(end, b)
        station_counts.append(count)
    counts = np.asarray(station_counts)

    def longest(mask):
        longest_run = run = 0
        for value in mask:
            run = run + 1 if value else 0
            longest_run = max(longest_run, run)
        return float(longest_run * step)

    # At least two windows: inlet-only branching can never satisfy this gate.
    windows = np.array_split(
        counts, max(2, int(np.ceil((hi - lo - 1e-8) / cfg.interconnection.interaction_window_m)))
    )
    events = network.backend_provenance.get("interaction_events", [])
    event_stations = sorted(float(e["station_m"]) for e in events)
    extent = float(hi - lo)
    event_gaps = np.diff([0.0, *event_stations, extent])
    independent = [
        sum(s.total_length for s in network.segments if s.metadata.get("system_ids") == [i])
        for i in range(cfg.systems.count)
    ]
    return {
        "parallel_fraction": float(np.mean(counts >= 2)),
        "minimum_window_parallel_fraction": min(float(np.mean(w >= 2)) for w in windows),
        "longest_parallel_run_m": longest(counts >= 2),
        "longest_single_run_fraction": longest(counts == 1) / extent,
        "minimum_source_independent_length_m": min(independent),
        "interactions_per_km": len(events) * 1000 / extent,
        "maximum_interaction_gap_m": float(max(event_gaps)),
        "channel_fractions": {
            str(i): float(np.mean(counts == i)) for i in range(cfg.systems.count + 1)
        },
        "median_rock_gap_m": float(np.median(gaps)) if gaps else 0.0,
        "network_lateral_span_m": float(
            max(r[:, 2].max() for r in rows) - min(r[:, 1].min() for r in rows)
        ),
        "downstream_extent_m": extent,
        "station_m": (stations - lo).tolist(),
        "channel_count": counts.tolist(),
        "scope": "section_envelopes" if sections is not None else "network_width_envelopes",
    }


def assess_interconnected(network, host, sections, check):
    if network.config.topology.style != "interconnected":
        return
    cfg = network.config.interconnection
    width = 2 * network.config.base_passage_radius
    maximum_angle, bad_connections = 0.0, []
    for node in network.nodes:
        directions = []
        for s in network.segments:
            if s.metadata.get("topology_role") == "side_branch":
                continue
            xy = np.array([[p.x, p.y] for p in s.points])
            if s.start_node_id == node.node_id:
                d = xy[min(2, len(xy) - 1)] - xy[0]
            elif s.end_node_id == node.node_id:
                d = xy[-1] - xy[max(0, len(xy) - 3)]
            else:
                continue
            directions.append(d / max(float(np.linalg.norm(d)), 1e-9))
        if len(directions) < 3:
            continue
        angle = max(
            float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
            for a in directions
            for b in directions
        )
        maximum_angle = max(maximum_angle, angle)
        if angle > cfg.maximum_junction_angle_degrees:
            bad_connections.append(node.node_id)
    check(
        "interconnected_junction_angles",
        not bad_connections,
        maximum_angle,
        cfg.maximum_junction_angle_degrees,
    )
    starved: list[int] = []
    minimum_supply = (
        network.config.source_flux * network.config.lobe_growth.minimum_viable_flux_fraction
    )
    for node in network.nodes:
        outgoing = [
            s
            for s in network.segments
            if s.start_node_id == node.node_id and s.metadata.get("topology_role") != "side_branch"
        ]
        if len(outgoing) > 1:
            starved.extend(
                s.segment_id
                for s in outgoing
                if min(s.mean_flux, *s.metadata.get("phase_fluxes", [s.mean_flux])) < minimum_supply
            )
    check("interconnected_split_supply", not starved, len(starved), minimum_supply, starved)
    if host is not None:
        uphill = []
        for s in network.segments:
            if s.metadata.get("topology_role") == "side_branch":
                continue
            xy = np.array([[p.x, p.y] for p in s.points])
            heights = np.array([host.sample(p.x, p.y).elevation for p in s.points])
            grades = np.diff(heights) / np.maximum(
                np.linalg.norm(np.diff(xy, axis=0), axis=1), 1e-9
            )
            if np.any(grades > network.config.quality.maximum_uphill_grade):
                uphill.append(s.segment_id)
        check(
            "interconnected_host_uphill_grade",
            not uphill,
            len(uphill),
            network.config.quality.maximum_uphill_grade,
            uphill,
        )
    try:
        m = spatial_metrics(network, sections)
    except (ValueError, KeyError) as error:
        check("interconnected_envelopes", False, str(error))
        return
    for name, limit, minimum in (
        ("parallel_fraction", cfg.minimum_parallel_fraction, True),
        ("minimum_window_parallel_fraction", cfg.minimum_window_parallel_fraction, True),
        ("longest_parallel_run_m", cfg.minimum_parallel_run_widths * width, True),
        ("longest_single_run_fraction", cfg.maximum_single_run_fraction, False),
        (
            "minimum_source_independent_length_m",
            cfg.minimum_source_independent_length_widths * width,
            True,
        ),
        ("interactions_per_km", cfg.minimum_interactions_per_km, True),
        (
            "maximum_interaction_gap_m",
            min(cfg.interaction_window_m, 0.65 * m["downstream_extent_m"]),
            False,
        ),
    ):
        check(
            "interconnected_" + name,
            m[name] >= limit if minimum else m[name] <= limit,
            m[name],
            limit,
        )
    if sections is None:
        return
    from plume_advanced.stages.network_topology import section_footprint

    mask, _, _, spacing = section_footprint(network, sections)
    components = label(mask)[1]
    holes, _ = label(binary_fill_holes(mask) & ~mask)
    areas = np.bincount(holes.ravel())[1:] * spacing**2
    resolved_holes = int(np.sum(areas >= 0.08 * width**2))
    expected = len(network.segments) - len(network.nodes) + 1
    check("interconnected_section_components", components == 1, components, 1)
    check("interconnected_section_cycles", resolved_holes == expected, resolved_holes, expected)
