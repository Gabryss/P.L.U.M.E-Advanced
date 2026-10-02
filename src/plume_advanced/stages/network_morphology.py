"""Survey-inspired regional morphology; heuristics, not an eruption solver.

Separate reproducible construction provenance from geological interpretation.
No pillar, collapse event or eruption time is inferred from a graph loop.
"""

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

from plume_advanced.procedural import procedural_rng


def branch_zones(planner):
    """Local opportunities in physical coordinates, independent of graph sampling."""
    controls, g = planner.config.regional, planner.geometry
    if not controls.branch_localization:
        return []
    zones = []
    count = int(
        min(
            controls.maximum_branches,
            max(1, np.ceil(g.along_extent / (3 * controls.correlation_length_m))),
        )
    )
    layers = planner.config.layers.count if planner.config.layers.enabled else 1
    for layer in range(layers):
        rng = procedural_rng(planner.config.random_seed, "regional-branch-zones", layer)
        extent = planner.extents[layer] if hasattr(planner, "extents") else g.along_extent
        for i in range(count):
            centre = extent * (i + rng.uniform(0.25, 0.75)) / count
            zones.append(
                dict(
                    layer=layer,
                    along_m=float(centre),
                    scale_m=float(extent / count * rng.uniform(0.12, 0.22)),
                )
            )
    return zones


def branch_weights(planner, candidates, zones):
    if not zones:
        return None
    g = planner.geometry
    along = (planner.xy[candidates] - [g.seed_x, g.seed_y]) @ [g.flow_x, g.flow_y]
    density = np.zeros(len(candidates))
    for z in zones:
        density += (planner.layer_ids[candidates] == z["layer"]) * np.exp(
            -0.5 * ((along - z["along_m"]) / z["scale_m"]) ** 2
        )
    strength = planner.config.regional.branch_localization
    # A candidate set can lie entirely outside every zone; keep normalization
    # finite even at full localization and for very small spatial scales.
    weights = (1 - strength) + strength * np.maximum(density, 1e-12) ** 2
    return weights / weights.sum()


def route_capacity(config, owner, *, retained=False, blind=False):
    """A stable route preference survives subdivision at subsequent junctions."""
    if owner < 0 or not config.regional.hierarchy_strength:
        return 1.0
    rng = procedural_rng(config.random_seed, "regional-route-capacity", owner)
    value = float(rng.uniform(0.35, 0.75))
    if retained:
        value = float(rng.uniform(0.75, 1.15))
    if blind:
        value *= 0.4
    return 1 + config.regional.hierarchy_strength * (value - 1)


def width_profile(config, segment, flux):
    """Bounded supply response and long spatial variations; no hydraulic claim."""
    strength = config.regional.hierarchy_strength
    base = float(
        np.clip(
            1.6 * config.base_passage_radius,
            2 * config.minimum_passage_radius,
            1.9 * config.maximum_passage_radius,
        )
    )
    xy = np.array([(p.x, p.y) for p in segment.points])
    response = np.clip((max(flux, 1e-9) / max(config.source_flux, 1e-9)) ** 0.22, 0.60, 1.30)
    if config.regional.width_log_sigma:
        from plume_advanced.stages.network_width_field import log_width_field

        # This statistical profile replaces the two-wave modulation. It does
        # not turn routing cost or source flux into measured hydraulic head.
        layer = int(segment.metadata.get("regional_start_layer", 0))
        modulation = np.exp(config.regional.width_log_sigma * log_width_field(
            xy, config.random_seed, config.regional.width_correlation_m, layer=layer))
        raw_widths = base * (1 + strength * (response - 1)) * modulation
    else:
        phase = procedural_rng(config.random_seed, "regional-width-field").uniform(0, 2 * np.pi, 2)
        scale = config.regional.correlation_length_m
        field = 0.10 * np.sin(xy @ np.array([0.8, 1.0]) / scale + phase[0])
        field += 0.06 * np.sin(xy @ np.array([-0.6, 1.0]) / (0.4 * scale) + phase[1])
        raw_widths = base * (1 + strength * (response * (1 + field) - 1))
    widths = np.clip(
        raw_widths,
        2 * config.minimum_passage_radius,
        1.9 * config.maximum_passage_radius,
    )
    if (segment.metadata.get("regional_route_type") == "blind_branch"
            and segment.metadata.get("regional_blind_terminal", True)):
        t = np.array([p.arc_length for p in segment.points]) / max(segment.total_length, 1e-9)
        u = np.clip((t - 0.55) / 0.45, 0, 1)
        widths *= 1 - (1 - 0.8 * config.quality.terminal_width_ratio) * u * u * (3 - 2 * u)
    from plume_advanced.stages.network_acceptance import limit_width_gradient

    return limit_width_gradient(
        widths,
        np.array([p.arc_length for p in segment.points]),
        0.9 * config.quality.maximum_width_gradient,
    )


def annotate_regional_geometry(segments, controls):
    """Correct grade diagnostics to use explicit XYZ, not the substrate elevation."""
    from plume_advanced.stages.network_layers import segment_xyz

    result = []
    for s in segments:
        z = (
            segment_xyz(s, controls)[:, 2]
            if controls.enabled
            else np.array([p.elevation for p in s.points])
        )
        lengths = np.linalg.norm(np.diff([(p.x, p.y) for p in s.points], axis=0), axis=1)
        uphill = np.diff(z) > 0
        result.append(
            replace(
                s,
                metadata=dict(
                    s.metadata,
                    uphill_distance_m=float(lengths[uphill].sum()),
                    sustained_uphill_step_count=int(np.sum(uphill[1:] & uphill[:-1])),
                    grade_profile="local_uphill" if uphill.any() else "downhill",
                    reference_flow_model="conserved allocation; not an active eruption simulation",
                ),
            )
        )
    return result


def morphology_metrics(network):
    """Measured graph descriptors, without invented geological acceptance limits."""
    incoming = Counter(s.end_node_id for s in network.segments)
    outgoing = Counter(s.start_node_id for s in network.segments)
    junctions = [n for n in network.nodes if incoming[n.node_id] > 1 or outgoing[n.node_id] > 1]
    extent = max((n.along_position for n in network.nodes), default=0)
    bins = np.arange(0, max(extent, 1) + 100, 100)
    widths, lengths, sinuosity = [], [], []
    roles: Counter[str] = Counter()
    layer_lengths: defaultdict[int, float] = defaultdict(float)
    branch_lengths: defaultdict[int, float] = defaultdict(float)
    aligned_length = 0.
    longest_straight = 0.
    for s in network.segments:
        xy = np.array([(p.x, p.y) for p in s.points])
        lengths.append(float(s.total_length))
        widths.append(float(s.mean_width))
        sinuosity.append(float(s.total_length / max(np.linalg.norm(xy[-1] - xy[0]), 1e-9)))
        roles[s.metadata.get("regional_route_type", "unclassified")] += 1
        layer_lengths[int(s.metadata.get("regional_start_layer", 0))] += s.total_length
        if "regional_branch_id" in s.metadata:
            branch_lengths[int(s.metadata["regional_branch_id"])] += s.total_length
        delta = np.diff(xy, axis=0)
        distance = np.linalg.norm(delta, axis=1)
        heading = np.arctan2(delta[:, 1], delta[:, 0])
        axis_offset = abs((heading + np.pi/8) % (np.pi/4) - np.pi/8)
        aligned_length += float(distance[axis_offset <= np.radians(3)].sum())
        # Explicit diagnostic, not a rejection threshold. Starting a new run
        # when heading spans five degrees exposes long grid-locked reaches.
        run, low, high = 0., 0., 0.
        for angle, step in zip(np.unwrap(heading), distance):
            if not run or max(high, angle) - min(low, angle) > np.radians(5):
                run, low, high = 0., angle, angle
            run += step
            low, high = min(low, angle), max(high, angle)
            longest_straight = max(longest_straight, run)
    return dict(
        schema="plume.network-morphology.v1",
        calibration="uncalibrated_descriptors",
        width_model=("spatial_log_width_v1" if network.config.regional.width_log_sigma
                     else "regional_supply_profile"),
        width_log_sigma=network.config.regional.width_log_sigma,
        width_correlation_m=network.config.regional.width_correlation_m,
        junction_bin_edges_m=bins.tolist(),
        junction_counts=np.histogram([n.along_position for n in junctions], bins)[0].tolist(),
        segment_lengths_m=lengths,
        segment_mean_widths_m=widths,
        segment_sinuosity=sinuosity,
        split_nodes=sum(v > 1 for v in outgoing.values()),
        merge_nodes=sum(v > 1 for v in incoming.values()),
        blind_terminals=sum(n.kind == "terminal" for n in network.nodes),
        loop_rank=network._loop_rank(),
        route_segment_counts=dict(sorted(roles.items())),
        branch_lengths_m={str(k): float(v) for k, v in sorted(branch_lengths.items())},
        grid_aligned_length_fraction=aligned_length / max(sum(lengths), 1e-9),
        longest_near_straight_reach_m=float(longest_straight),
        layer_lengths_m={str(k): float(v) for k, v in sorted(layer_lengths.items())},
    )
