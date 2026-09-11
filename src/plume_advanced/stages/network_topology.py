"""A dominant gallery with local island bypasses and short blind branches.

The controls describe a reference-informed procedural style, not a reconstruction
or a fluid simulation. Dimensions still obey the selected body and section model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import copy
from dataclasses import asdict, dataclass, replace

import numpy as np

from plume_advanced.procedural import derive_subseed, procedural_rng


@dataclass(frozen=True)
class NetworkTopologyConfig:
    style: str = "general"
    generation_mode: str = "layout"
    island_count: tuple[int, int] = (1, 3)
    island_length_widths: tuple[float, float] = (4.5, 7.0)
    island_half_span_widths: tuple[float, float] = (0.85, 1.20)
    side_branch_count: tuple[int, int] = (1, 3)
    side_branch_length_widths: tuple[float, float] = (2.5, 4.5)
    lateral_variation_widths: float = 0.75
    correlation_length_widths: float = 10.0
    width_variation: float = 0.36
    minimum_trunk_fraction: float = 0.65
    minimum_single_channel_fraction: float = 0.55
    maximum_bypass_fraction: float = 0.22
    maximum_lateral_span_widths: float = 5.0
    minimum_island_clearance_widths: float = 0.20

    def __post_init__(self):
        if self.style not in {"general", "trunk_dominated", "interconnected"}:
            raise ValueError("network.topology.style must be general, trunk_dominated or interconnected")
        if self.generation_mode not in {"layout", "independent_growth"}:
            raise ValueError("network.topology.generation_mode must be layout or independent_growth")
        if self.generation_mode == "independent_growth" and self.style == "general":
            raise ValueError("independent_growth requires trunk_dominated or interconnected topology")
        if self.style == "interconnected" and self.generation_mode != "independent_growth":
            raise ValueError("interconnected topology requires independent_growth")
        for name in (
            "island_count",
            "side_branch_count",
            "island_length_widths",
            "island_half_span_widths",
            "side_branch_length_widths",
        ):
            values = getattr(self, name)
            integer = name.endswith("count")
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 2
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or v < (0 if integer else 0.01)
                    or (integer and type(v) is not int)
                    for v in values
                )
                or values[0] > values[1]
            ):
                raise ValueError(f"network.topology.{name} requires an ordered pair")
            object.__setattr__(self, name, tuple(values))
        if self.island_count[1] > 8 or self.side_branch_count[1] > 8:
            raise ValueError("network.topology counts cannot exceed eight")
        for name, value in asdict(self).items():
            if name in {"style", "generation_mode"} or isinstance(value, tuple):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"network.topology.{name} must be finite and positive")
        for name in (
            "width_variation",
            "minimum_trunk_fraction",
            "minimum_single_channel_fraction",
            "maximum_bypass_fraction",
        ):
            if getattr(self, name) >= 1:
                raise ValueError(f"network.topology.{name} must be below one")


def generate_trunk_network(generator, host):
    if generator.config.topology.generation_mode == "independent_growth":
        from plume_advanced.stages.network_gallery_growth import generate_gallery_growth
        return generate_gallery_growth(generator, host)
    from plume_advanced.stages.network import CaveNode, CavePoint, CaveSegment
    from plume_advanced.stages.network_systems import preferred_tracks

    cfg, controls = generator.config, generator.config.topology
    if cfg.emplacement_backend != "internal":
        raise ValueError("trunk_dominated topology currently requires internal emplacement")
    geometry = generator._build_flow_geometry(host)
    proposal = copy(generator)
    proposal.config = replace(
        cfg,
        systems=replace(
            cfg.systems,
            count=1,
            lateral_variation_widths=controls.lateral_variation_widths,
            correlation_length_widths=controls.correlation_length_widths,
        ),
    )
    along, tracks, width = preferred_tracks(proposal, host, geometry)
    cross = tracks[0]
    length = float(along[-1])
    rng = procedural_rng(cfg.random_seed, "trunk-topology")
    island_count = int(rng.integers(controls.island_count[0], controls.island_count[1] + 1))
    branch_count = int(
        rng.integers(controls.side_branch_count[0], controls.side_branch_count[1] + 1)
    )
    source_join = (
        min(0.12 * length, max(4 * width, cfg.systems.count * width))
        if cfg.systems.count > 1
        else 0.0
    )
    # Separate feature neighborhoods with intact trunk, instead of tiling loops.
    island_lengths = rng.uniform(*controls.island_length_widths, size=island_count) * width
    lower, upper = source_join + 3 * width, length - 6 * width
    slack = upper - lower - float(sum(island_lengths)) - max(0, island_count - 1) * 3 * width
    if slack < 0:
        raise ValueError("Trunk is too short for the requested islands and intact passage spacing")
    spaces = rng.dirichlet(np.full(island_count + 1, 2.0)) * slack
    islands = []
    cursor = lower + spaces[0]
    for i, span in enumerate(island_lengths):
        islands.append((float(cursor), float(cursor + span), i))
        cursor += span + 3 * width + spaces[i + 1]
    # Side branches are independent events on intact trunk, not extra loops.
    candidates = np.linspace(max(source_join + 3 * width, 0.35 * length), length - 3 * width, 200)
    candidates = [
        float(a)
        for a in candidates
        if all(not (start - 2 * width < a < end + 2 * width) for start, end, _ in islands)
    ]
    branch_sites = []
    for _ in range(branch_count):
        if not candidates:
            raise ValueError("Trunk has insufficient room for separate side branches")
        site = candidates[int(rng.integers(len(candidates)))]
        branch_sites.append(site)
        candidates = [a for a in candidates if abs(a - site) >= 3 * width]
    branch_sites.sort()
    anchors = sorted(
        {
            source_join,
            length,
            *branch_sites,
            *(a for start, end, _ in islands for a in (start, end)),
        }
    )
    nodes, segments = [], []

    def world(a, c):
        return np.column_stack(
            (
                geometry.seed_x + a * geometry.flow_x + c * geometry.cross_x,
                geometry.seed_y + a * geometry.flow_y + c * geometry.cross_y,
            )
        )

    def add_node(a, c, kind):
        x, y = world(np.asarray([a]), np.asarray([c]))[0]
        node_id = len(nodes)
        nodes.append(CaveNode(node_id, float(x), float(y), float(a), float(c), kind))
        return node_id

    node_at = {
        a: add_node(
            a,
            float(np.interp(a, along, cross)),
            "exit" if a == length else "entry" if a == 0 else "junction",
        )
        for a in anchors
    }
    width_phase = float(rng.uniform(0, 2 * np.pi))

    def add_segment(start, end, a, c, role, *, island=None, scale=1.0, source=None, taper=False):
        xy = world(a, c)
        xy[0], xy[-1] = (nodes[start].x, nodes[start].y), (nodes[end].x, nodes[end].y)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        modulation = 1 + controls.width_variation * (
            0.72 * np.sin(2 * np.pi * a / (9 * width) + width_phase)
            + 0.28 * np.sin(2 * np.pi * a / (4.3 * width) - 0.7 * width_phase)
        )
        widths = np.clip(
            width * scale * modulation,
            2 * cfg.minimum_passage_radius,
            1.9 * cfg.maximum_passage_radius,
        )
        if taper:
            t = np.clip((arc / arc[-1] - 0.55) / 0.45, 0, 1)
            widths *= 1 - 0.65 * t * t * (3 - 2 * t)
        points = []
        for i, (p, station, w) in enumerate(zip(xy, arc, widths)):
            sample = host.sample(*p)
            points.append(
                CavePoint(
                    i,
                    float(p[0]),
                    float(p[1]),
                    sample.elevation,
                    sample.slope_degrees,
                    sample.cover_thickness,
                    sample.roof_competence,
                    sample.growth_cost,
                    float(station),
                    float(w),
                )
            )
        kind = "spur" if taper else "island_bypass" if role == "island_arm" else "backbone"
        metadata = generator._build_emplacement_metadata(
            kind=kind,
            phase_count=1,
            birth_phase=0,
            death_phase=0,
            formation_state="trunk_dominated",
        )
        metadata.update(topology_role=role, topology_style="trunk_dominated")
        if island is not None:
            metadata["island_id"] = f"pillar_{island}"
        if source is not None:
            metadata.update(
                source_system_id=source,
                system_seed=derive_subseed(cfg.random_seed, "network-system", source),
            )
        if taper:
            metadata["quality_terminal_taper"] = True
        segments.append(CaveSegment(len(segments), start, end, kind, 0, tuple(points), metadata))

    for first, last in zip(anchors, anchors[1:]):
        island = next((i for a, b, i in islands if a == first and b == last), None)
        a = np.linspace(first, last, max(9, int((last - first) / max(0.15 * width, 0.5)) + 1))
        baseline = np.interp(a, along, cross)
        source = 0 if first == 0 else None
        if island is None:
            add_segment(node_at[first], node_at[last], a, baseline, "trunk", source=source)
        else:
            t = (a - first) / (last - first)
            half_span = float(rng.uniform(*controls.island_half_span_widths)) * width
            # Both walls bend around an island; neither duplicates a straight
            # trunk through the rock, and the two arms differ in width and shape.
            for sign, scale in ((1.0, 0.90), (-1.0, 0.72)):
                skew = float(rng.uniform(-0.20, 0.20))
                u = t + skew * np.sin(np.pi * t)
                lobe = np.sin(np.pi * u) ** float(rng.uniform(1.15, 1.65))
                lobe *= 1 + float(rng.uniform(-0.15, 0.15)) * np.sin(2 * np.pi * t)
                c = baseline + sign * half_span * lobe * float(rng.uniform(0.80, 1.20))
                add_segment(
                    node_at[first], node_at[last], a, c, "island_arm", island=island, scale=scale
                )
    if cfg.systems.count > 1:
        for system in range(cfg.systems.count):
            offset = (system - 0.5 * (cfg.systems.count - 1)) * 1.3 * width
            start = add_node(0.0, float(cross[0] + offset), "entry")
            a = np.linspace(0, source_join, 30)
            t = a / source_join
            c = np.interp(a, along, cross) + offset * (1 - t * t * (3 - 2 * t))
            add_segment(start, node_at[source_join], a, c, "feeder", scale=0.75, source=system)
    # Allocate side-branch lengths within the requested range and the gallery
    # dominance budget. The final quality check still measures actual lengths.
    arms = defaultdict(list)
    for segment in segments:
        if segment.metadata.get("island_id"):
            arms[segment.metadata["island_id"]].append(segment.total_length)
    main_length = sum(
        s.total_length for s in segments if s.metadata.get("topology_role") == "trunk"
    )
    main_length += sum(min(lengths) for lengths in arms.values())
    extra_length = sum(max(lengths) for lengths in arms.values())
    branch_budget = (main_length * (1 / controls.minimum_trunk_fraction - 1) - extra_length) / 1.12
    minimum_span = controls.side_branch_length_widths[0] * width
    spans = rng.uniform(*controls.side_branch_length_widths, size=branch_count) * width
    if branch_count and branch_budget < branch_count * minimum_span:
        raise ValueError("Requested side branches exceed the dominant gallery length budget")
    surplus = float(sum(spans - minimum_span))
    if surplus > 0:
        spans = minimum_span + (spans - minimum_span) * min(
            1.0, (branch_budget - branch_count * minimum_span) / surplus
        )
    for i, site in enumerate(branch_sites):
        span = float(spans[i])
        t = np.linspace(0, 1, 30)
        sign = float(rng.choice([-1, 1]))
        angle = float(rng.uniform(0.55, 0.95))
        a = site + span * math.cos(angle) * t
        bend = float(rng.uniform(0.15, 0.35))
        c = np.interp(a, along, cross) + sign * span * math.sin(angle) * (
            t - bend * np.sin(np.pi * t)
        )
        end = add_node(float(a[-1]), float(c[-1]), "spur_terminal")
        add_segment(node_at[site], end, a, c, "side_branch", scale=0.60, taper=True)
    skeleton = np.zeros_like(host.growth_cost, dtype=bool)
    for segment in segments:
        for point in segment.points:
            skeleton[generator._world_to_cell(host, point.x, point.y)] = True
    return generator._finish_network(
        host,
        geometry,
        nodes,
        segments,
        backend_provenance={
            "backend": "internal",
            "version": "trunk_topology_v1",
            "topology_style": controls.style,
            "system_count": cfg.systems.count,
        },
        skeleton_mask=skeleton,
        total_flux=np.zeros_like(host.growth_cost),
    )


def topology_metrics(network):
    nodes = {n.node_id: n for n in network.nodes}
    first, last = [
        nodes[i] for i in (network.dominant_route_node_ids[0], network.dominant_route_node_ids[-1])
    ]
    origin = np.array([first.x, first.y])
    axis = np.array([last.x, last.y]) - origin
    extent = float(np.linalg.norm(axis))
    axis /= max(extent, 1e-9)
    cross = np.array([-axis[1], axis[0]])
    intervals, island_lengths = [], defaultdict(list)
    min_cross, max_cross = np.inf, -np.inf
    for s in network.segments:
        xy = np.array([[p.x, p.y] for p in s.points]) - origin
        min_cross, max_cross = (
            min(min_cross, float((xy @ cross).min())),
            max(max_cross, float((xy @ cross).max())),
        )
        if s.metadata.get("topology_role") != "side_branch":
            a = xy @ axis
            intervals.append((float(a.min()), float(a.max())))
        if s.metadata.get("island_id"):
            island_lengths[s.metadata["island_id"]].append(s.total_length)
    breaks = sorted({0.0, extent, *(max(0.0, min(extent, a)) for pair in intervals for a in pair)})
    single = sum(
        b - a
        for a, b in zip(breaks, breaks[1:])
        if sum(lo <= 0.5 * (a + b) < hi for lo, hi in intervals) == 1
    ) / max(extent, 1e-9)
    # Measure the shared gallery after any inlet confluence. Independent
    # upstream feeders are not alternative routes through that gallery.
    feeders = [s for s in network.segments if s.metadata.get("topology_role") == "feeder"]
    route_pairs = set(zip(network.dominant_route_node_ids, network.dominant_route_node_ids[1:]))
    route_feeder_length = sum(
        s.total_length for s in feeders if (s.start_node_id, s.end_node_id) in route_pairs
    )
    total = sum(s.total_length for s in network.segments) - sum(s.total_length for s in feeders)
    return {
        "topology_source_count": sum(n.kind == "entry" for n in network.nodes),
        "mixed_source_passage_length_m": sum(
            s.total_length
            for s in network.segments
            if len(s.metadata.get("contributing_system_ids", [])) > 1
        ),
        "trunk_length_fraction": (network.dominant_route_length - route_feeder_length)
        / max(total, 1e-9),
        "single_channel_fraction": single,
        "island_count": len(island_lengths),
        "maximum_bypass_fraction": max(
            (max(v) / max(extent, 1e-9) for v in island_lengths.values()), default=0.0
        ),
        "lateral_span_widths": (max_cross - min_cross) / (2 * network.config.base_passage_radius),
        "side_branch_count": sum(
            s.metadata.get("topology_role") == "side_branch" for s in network.segments
        ),
    }


def assess_topology(network, sections, check):
    controls = network.config.topology
    if controls.style != "trunk_dominated":
        return
    try:
        metrics = topology_metrics(network)
    except (KeyError, IndexError, ValueError):
        check("trunk_topology", False, "invalid dominant route")
        return
    for name, key, minimum in (
        ("trunk_length_fraction", "minimum_trunk_fraction", True),
        ("single_channel_fraction", "minimum_single_channel_fraction", True),
        ("maximum_bypass_fraction", "maximum_bypass_fraction", False),
        ("lateral_span_widths", "maximum_lateral_span_widths", False),
    ):
        value, limit = metrics[name], getattr(controls, key)
        check(name, value >= limit if minimum else value <= limit, value, limit)
    for name in ("island_count", "side_branch_count"):
        low, high = getattr(controls, name)
        check("topology_" + name, low <= metrics[name] <= high, metrics[name], [low, high])
    islands = defaultdict(list)
    for segment in network.segments:
        if segment.metadata.get("island_id"):
            islands[segment.metadata["island_id"]].append(segment)
    invalid = [
        s.segment_id
        for group in islands.values()
        for s in group
        if len(group) != 2 or len({(s.start_node_id, s.end_node_id) for s in group}) != 1
    ]
    check("island_split_rejoin_topology", not invalid, len(invalid), 0, invalid)
    sources = [
        s.metadata.get("source_system_id")
        for s in network.segments
        if any(n.node_id == s.start_node_id and n.kind == "entry" for n in network.nodes)
    ]
    check(
        "trunk_source_identity",
        all(type(i) is int for i in sources)
        and sorted(sources) == list(range(network.config.systems.count)),
        len(sources),
        network.config.systems.count,
    )
    from plume_advanced.stages.network import CaveNetworkGenerator
    from plume_advanced.stages.network_systems import annotate_source_lineage

    try:
        order = CaveNetworkGenerator._topological_node_ids(
            list(network.nodes), list(network.segments)
        )
        expected_lineage = annotate_source_lineage(network.nodes, network.segments, order)
        invalid_lineage = [
            s.segment_id
            for s, expected_segment in zip(network.segments, expected_lineage)
            if s.metadata.get("contributing_system_ids")
            != expected_segment.metadata["contributing_system_ids"]
        ]
    except (ValueError, KeyError, TypeError):
        invalid_lineage = [s.segment_id for s in network.segments]
    check("trunk_source_lineage", not invalid_lineage, len(invalid_lineage), 0, invalid_lineage)
    incoming = {s.end_node_id for s in network.segments}
    outgoing = {s.start_node_id for s in network.segments}
    leaves = incoming - outgoing
    supply = sum(s.mean_flux for s in network.segments if s.start_node_id not in incoming)
    drained = sum(s.mean_flux for s in network.segments if s.end_node_id in leaves)
    expected = network.config.source_flux * network.config.systems.count
    error = max(abs(supply - expected), abs(drained - expected)) / max(expected, 1e-9)
    check("trunk_total_discharge", error < 1e-8, error, 1e-8)
    if sections is None:
        return
    invalid_profiles = []
    for field in sections.segment_fields:
        for sample in field.samples:
            profile = np.asarray(sample.profile_points)
            if (
                profile.ndim != 2
                or profile.shape[1] != 2
                or len(profile) < 3
                or not np.isfinite(profile).all()
                or not np.isfinite([sample.x, sample.y, *sample.normal, *sample.binormal]).all()
            ):
                invalid_profiles.append(field.segment_id)
    if invalid_profiles:
        # General section checks report the invalid data. Do not rasterize it.
        check("section_footprint_connected", False, "invalid profiles", 1, invalid_profiles)
        check(
            "section_footprint_islands", False, "invalid profiles", len(islands), invalid_profiles
        )
        check("section_island_clearance", False, "invalid profiles", None, invalid_profiles)
        return
    fields = {f.segment_id: f for f in sections.segment_fields}
    minimum_gap, failed = np.inf, []
    for group in islands.values():
        if len(group) != 2:
            continue
        s = group[0]
        origin = np.array([s.points[0].x, s.points[0].y])
        axis = np.array([s.points[-1].x, s.points[-1].y]) - origin
        length = np.linalg.norm(axis)
        axis /= max(length, 1e-9)
        cross = np.array([-axis[1], axis[0]])
        positions = np.linspace(0.38 * length, 0.62 * length, 11)
        bounds = []
        for arm in group:
            field = fields.get(arm.segment_id)
            if field is None or not field.samples:
                failed.extend(x.segment_id for x in group)
                continue
            rows = []
            for sample in field.samples:
                profile = np.array(sample.profile_points)
                xy = (
                    np.array([sample.x, sample.y])
                    - origin
                    + profile[:, :1] * np.array(sample.normal[:2])
                    + profile[:, 1:2] * np.array(sample.binormal[:2])
                )
                rows.append(
                    [
                        np.dot(np.array([sample.x, sample.y]) - origin, axis),
                        float((xy @ cross).min()),
                        float((xy @ cross).max()),
                    ]
                )
            rows = np.array(sorted(rows))
            bounds.append(np.array([np.interp(positions, rows[:, 0], rows[:, i]) for i in (1, 2)]))
        if len(bounds) == 2:
            below, above = sorted(bounds, key=lambda b: float(b.mean()))
            gap = float(np.min(above[0] - below[1]))
            minimum_gap = min(minimum_gap, gap)
            if (
                gap
                < controls.minimum_island_clearance_widths * 2 * network.config.base_passage_radius
            ):
                failed.extend(s.segment_id for s in group)
    check(
        "section_island_clearance",
        not failed,
        float(minimum_gap) if np.isfinite(minimum_gap) else None,
        controls.minimum_island_clearance_widths * 2 * network.config.base_passage_radius,
        failed,
    )
    mask, _, _, spacing = section_footprint(network, sections)
    from scipy.ndimage import binary_fill_holes, label

    components = label(mask)[1]
    hole_labels, _ = label(binary_fill_holes(mask) & ~mask)
    areas = np.bincount(hole_labels.ravel())[1:] * spacing**2
    holes = int(np.sum(areas >= 0.08 * (2 * network.config.base_passage_radius) ** 2))
    check("section_footprint_connected", components == 1, components, 1)
    check("section_footprint_islands", holes == len(islands), holes, len(islands))


def section_footprint(network, sections):
    """Rasterize the union of actual section envelopes in a flow-aligned plan.

    This checks a section-level projection, not the eventual triangle mesh.
    A bounded raster resolution makes hole/component tests deterministic.
    """
    from skimage.draw import polygon

    nodes = {n.node_id: n for n in network.nodes}
    first, last = [
        nodes[i] for i in (network.dominant_route_node_ids[0], network.dominant_route_node_ids[-1])
    ]
    origin = np.array([first.x, first.y])
    axis = np.array([last.x, last.y]) - origin
    axis /= max(np.linalg.norm(axis), 1e-9)
    cross = np.array([-axis[1], axis[0]])
    polygons = []
    for field in sections.segment_fields:
        left, right = [], []
        for sample in field.samples:
            profile = np.array(sample.profile_points)
            xy = (
                np.array([sample.x, sample.y])
                - origin
                + profile[:, :1] * np.array(sample.normal[:2])
                + profile[:, 1:2] * np.array(sample.binormal[:2])
            )
            points = np.column_stack((xy @ axis, xy @ cross))
            left.append(points[int(np.argmin(points[:, 1]))])
            right.append(points[int(np.argmax(points[:, 1]))])
        if len(left) > 1:
            polygons.append(np.array(left + right[::-1]))
    if not polygons:
        return np.zeros((3, 3), bool), np.arange(3), np.arange(3), 1.0
    vertices = np.concatenate(polygons)
    spacing = max(0.08, 0.025 * 2 * network.config.base_passage_radius)
    lo, hi = vertices.min(axis=0) - 2 * spacing, vertices.max(axis=0) + 2 * spacing
    spacing = max(spacing, float(np.sqrt(np.prod(hi - lo) / 4_000_000)))
    xs, ys = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(2)]
    mask = np.zeros((len(ys), len(xs)), bool)
    for poly in polygons:
        rr, cc = polygon((poly[:, 1] - ys[0]) / spacing, (poly[:, 0] - xs[0]) / spacing, mask.shape)
        mask[rr, cc] = True
    return mask, xs, ys, spacing
