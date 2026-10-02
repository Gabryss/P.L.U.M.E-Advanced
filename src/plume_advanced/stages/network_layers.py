"""Optional Stage-B layer geometry and conservative reference-envelope checks.

Host elevation remains the substrate reference in CavePoint. Explicit network
centreline elevations are derived here and retained by the layered section fitter.
"""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class NetworkLayersConfig:
    enabled: bool = False
    preserve_layer_trunks: bool = False
    count: int = 2
    spacing_m: float = 12.0
    passage_height_m: float = 3.0
    minimum_rock_m: float = 3.0
    maximum_connection_grade: float = 0.25
    connection_opportunities_per_km: float = 6.0
    connection_variation: float = 0.0
    minimum_extent_fraction: float = 1.0
    spacing_variation: float = 0.0

    def __post_init__(self):
        for name in ("enabled", "preserve_layer_trunks"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"network.layers.{name} must be boolean")
        if type(self.count) is not int or not 2 <= self.count <= 4:
            raise ValueError("network.layers.count must be an integer from 2 to 4")
        for name, value in asdict(self).items():
            if name in {"enabled", "preserve_layer_trunks", "count"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (
                    value < 0
                    if name in {"connection_variation", "spacing_variation"}
                    else value <= 0
                )
            ):
                raise ValueError(f"network.layers.{name} must be finite and positive")
        if self.spacing_m < self.passage_height_m + self.minimum_rock_m:
            raise ValueError(
                "network.layers.spacing_m must retain passage height plus minimum rock"
            )
        if self.maximum_connection_grade > 1:
            raise ValueError("network.layers.maximum_connection_grade must be at most 1")
        if self.connection_opportunities_per_km > 32:
            raise ValueError("network.layers.connection_opportunities_per_km must be at most 32")
        if self.connection_variation > 1:
            raise ValueError("network.layers.connection_variation must be between 0 and 1")
        if not 0.5 <= self.minimum_extent_fraction <= 1:
            raise ValueError("network.layers.minimum_extent_fraction must be between 0.5 and 1")
        if self.spacing_variation > 1:
            raise ValueError("network.layers.spacing_variation must be between 0 and 1")

    def depth(self, layer):
        return self.minimum_rock_m + 0.5 * self.passage_height_m + layer * self.spacing_m


def segment_depths(segment, controls):
    start = segment.metadata.get("regional_start_layer")
    end = segment.metadata.get("regional_end_layer")
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < controls.count:
        return np.full(len(segment.points), np.nan)
    t = np.array([p.arc_length for p in segment.points]) / max(segment.total_length, 1e-9)
    depths = segment.metadata.get("regional_layer_depths_m")
    if depths is not None:
        if (
            not isinstance(depths, list)
            or len(depths) != controls.count
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in depths)
            or abs(depths[0] - controls.depth(0)) > 1e-8
            or np.any(np.diff(depths) < controls.passage_height_m + controls.minimum_rock_m - 1e-8)
            or np.any(
                np.diff(depths) > controls.spacing_m * (1 + controls.spacing_variation) + 1e-8
            )
        ):
            return np.full(len(segment.points), np.nan)
        first, last = depths[start], depths[end]
    elif controls.spacing_variation:
        return np.full(len(segment.points), np.nan)
    else:
        first, last = controls.depth(start), controls.depth(end)
    # Zero offset derivative at either end makes ramps tangent to their layers.
    return first + (last - first) * t * t * (3 - 2 * t)


def segment_xyz(segment, controls):
    xyz = np.array([(p.x, p.y, p.elevation) for p in segment.points])
    if controls.enabled:
        xyz[:, 2] -= segment_depths(segment, controls)
    offsets = segment.metadata.get("network_detail_vertical_offsets_m")
    if offsets is not None:
        if (not controls.enabled or not isinstance(offsets, list) or len(offsets) != len(xyz)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in offsets)):
            return np.full_like(xyz, np.nan)
        xyz[:, 2] += offsets
    return xyz


def crossing_clearance(hit, segments, controls):
    heights = []
    for label, sid in zip(("first", "second"), hit["segments"]):
        z = segment_xyz(segments[sid], controls)[:, 2]
        i, fraction = hit[f"{label}_piece"], hit[f"{label}_fraction"]
        heights.append(z[i] * (1 - fraction) + z[i + 1] * fraction)
    return abs(heights[0] - heights[1]) - controls.passage_height_m


def assess_layers(network, host, check):
    """Inspect actual elevations, including ramps; layer labels alone never suffice."""
    controls = network.config.layers
    invalid, steep, substrate = [], [], []
    endpoints = defaultdict(list)
    used: set[int] = set()
    links: set[tuple[int, int]] = set()
    arrays = []
    for s in network.segments:
        start, end = (s.metadata.get(f"regional_{key}_layer") for key in ("start", "end"))
        if (
            type(start) is not int
            or type(end) is not int
            or not (0 <= start <= end < controls.count)
            or end - start > 1
        ):
            invalid.append(s.segment_id)
            continue
        used.update((start, end))
        if end != start:
            links.add((start, end))
        xyz = segment_xyz(s, controls)
        if not np.isfinite(xyz).all():
            invalid.append(s.segment_id)
            continue
        endpoints[s.start_node_id].append(xyz[0, 2])
        endpoints[s.end_node_id].append(xyz[-1, 2])
        distance = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        grade = np.diff(xyz[:, 2]) / np.maximum(distance, 1e-9)
        if np.any(abs(grade) > controls.maximum_connection_grade + 1e-9):
            steep.append(s.segment_id)
        # Sub-metre samples screen the interior of long ramp edges as well.
        arc = np.r_[0.0, np.cumsum(distance)]
        stations = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1] / 0.5)) + 1))
        dense = np.column_stack([np.interp(stations, arc, xyz[:, k]) for k in range(3)])
        arrays.append((s, dense))
        if host is not None:
            coords = [
                (dense[:, 1] - host.y_coords[0]) / np.diff(host.y_coords[:2])[0],
                (dense[:, 0] - host.x_coords[0]) / np.diff(host.x_coords[:2])[0],
            ]
            surface, thickness, cover = [
                map_coordinates(getattr(host, name), coords, order=1, mode="nearest")
                for name in ("elevation", "emplacement_thickness", "cover_thickness")
            ]
            depth = surface - dense[:, 2]
            roof = depth - controls.passage_height_m / 2
            bottom = thickness - depth - controls.passage_height_m / 2
            if (
                not np.isfinite([surface, thickness, cover]).all()
                or np.any(roof < controls.minimum_rock_m - 0.02)
                or np.any(bottom < controls.minimum_rock_m)
                or np.any(cover < controls.minimum_rock_m)
            ):
                substrate.append(s.segment_id)
    check("layer_metadata_and_elevations", not invalid, len(invalid), 0, invalid)
    layouts = {
        tuple(s.metadata.get("regional_layer_depths_m", []))
        for s in network.segments
        if isinstance(s.metadata.get("regional_layer_depths_m", []), list)
        and all(type(v) in (int, float) for v in s.metadata.get("regional_layer_depths_m", []))
    }
    check("layer_shared_depth_layout", len(layouts) == 1, len(layouts), 1)
    check("layer_count", len(used) == controls.count, len(used), controls.count)
    check("layer_connections", len(links) == controls.count - 1, len(links), controls.count - 1)
    if controls.preserve_layer_trunks:
        exits = {n.node_id for n in network.nodes if n.kind == "exit"}
        outlet_layers = {
            s.metadata.get("regional_end_layer") for s in network.segments if s.end_node_id in exits
        }
        check(
            "layer_retained_outlets",
            outlet_layers == set(range(controls.count)),
            len(outlet_layers),
            controls.count,
        )
        missing_trunks = []
        for layer in range(controls.count):
            reached = {n.node_id for n in network.nodes if n.kind == "entry"}
            edges = [(s.start_node_id, s.end_node_id) for s in network.segments
                     if s.metadata.get("regional_start_layer") == layer
                     and s.metadata.get("regional_end_layer") == layer]
            while True:
                enlarged = reached | {b for a, b in edges if a in reached}
                if enlarged == reached:
                    break
                reached = enlarged
            if not reached & exits:
                missing_trunks.append(layer)
        check("layer_source_to_exit_trunks", not missing_trunks, missing_trunks, [])
    discontinuous = [n for n, heights in endpoints.items() if np.ptp(heights) > 1e-5]
    check("layer_junction_elevation_continuity", not discontinuous, len(discontinuous), 0)
    check("layer_connection_grade", not steep, len(steep), controls.maximum_connection_grade, steep)
    check("layer_host_thickness", not substrate, len(substrate), controls.minimum_rock_m, substrate)
    extents = network.backend_provenance.get("layer_extents_m")
    if extents is not None or controls.minimum_extent_fraction < 1:
        valid = (
            isinstance(extents, list)
            and len(extents) == controls.count
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in extents)
        )
        outside = []
        if valid and host is not None:
            direction = network.backend_provenance["flow_direction"]
            origin = np.array(host.config.seed_point)
            tolerance = float(network.backend_provenance["planning_step_m"]) * 1.1
            for s, xyz in arrays:
                a, b = (s.metadata[f"regional_{key}_layer"] for key in ("start", "end"))
                if a == b and np.any((xyz[:, :2] - origin) @ direction > extents[a] + tolerance):
                    outside.append(s.segment_id)
        check("layer_extent_bounds", valid and not outside, len(outside), 0, outside)

    # Broad-phase plan bounds, then actual local vertical separation. A nearby
    # XY point on another layer is harmless only when the rock gap is retained.
    from plume_advanced.stages.network_neighborhoods import passage_conflicts

    conflicts = passage_conflicts(
        arrays, vertical_clearance=controls.passage_height_m + controls.minimum_rock_m
    )
    check("layer_passage_separation", not conflicts, len(set(conflicts)), 0, conflicts)
