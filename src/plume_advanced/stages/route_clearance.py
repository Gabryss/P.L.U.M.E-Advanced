"""Required-route envelopes and continuous capsule checks on exported triangles.

The corridor test is geometric: it does not certify traction, slope handling or
ground contact for a particular robot. Optional branches are left unconstrained.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from plume_advanced.progress import report_progress
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.stages.surface_topology import SurfaceTopologyError
from plume_advanced.stages.triangle_queries import TriangleIndex, vertical_clearances


def resample_route_polyline(points, spacing, trim_start=0., trim_end=0.):
    """Sample a route at bounded intervals while preserving its original corners."""
    points = np.asarray(points, float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    keep = np.r_[True, np.diff(arc) > 1e-10]
    points, arc = points[keep], arc[keep]
    if len(points) < 2 or arc[-1] <= trim_start+trim_end:
        raise SurfaceTopologyError("Required route is too short for its endpoint clearance")
    stations = np.unique(np.r_[
        np.linspace(trim_start, arc[-1]-trim_end,
                    max(2, int(np.ceil((arc[-1]-trim_start-trim_end)/spacing))+1)),
        arc[(arc > trim_start) & (arc < arc[-1]-trim_end)],
    ])
    return tuple(map(tuple, np.column_stack([np.interp(stations, arc, points[:, a]) for a in range(3)])))


def required_paths(network, sections, config):
    """Preserve branch boundaries and include connectors at shared graph nodes."""
    if not config.required_route_height_m:
        return (), ()
    fields = {f.segment_id: f for f in sections.segment_fields}
    segments = {s.segment_id: s for s in network.segments}
    degree: dict[int, int] = {}
    for s in network.segments:
        for node in (s.start_node_id, s.end_node_id):
            degree[node] = degree.get(node, 0)+1
    paths, ids = [], []
    endpoints: dict[int, list] = {}
    for sid in sections.dominant_route_segment_ids:
        field, segment = fields[sid], segments[sid]
        points = [(s.x, s.y, s.z) for s in field.samples]
        # The sealed cap itself cannot contain a finite body. Stop one body
        # width from degree-one ends; keep every junction fully inspected.
        trim = config.required_route_width_m+2*config.route_clearance_margin_m
        paths.append(resample_route_polyline(points, config.route_inspection_spacing_m,
                               trim if degree[segment.start_node_id] == 1 else 0,
                               trim if degree[segment.end_node_id] == 1 else 0))
        ids.append(sid)
        for node, p in ((segment.start_node_id, paths[-1][0]), (segment.end_node_id, paths[-1][-1])):
            endpoints.setdefault(node, []).append(p)
    for node, points in sorted(endpoints.items()):
        for end in points[1:]:
            if np.linalg.norm(np.asarray(end)-points[0]) > 1e-8:
                paths.append(resample_route_polyline([points[0], end], config.route_inspection_spacing_m))
                ids.append(-node-1)
    if not paths:
        raise SurfaceTopologyError("A clearance requirement needs a nonempty dominant route")
    return tuple(paths), tuple(ids)


def fit_required_sections(network, sections, config, host=None, *, extra_margin_m=0., affected=None):
    """Enlarge only deficient route envelopes, respecting frame and roof limits.

    Centers, graph and random state stay unchanged. The original shape scales
    continuously about its center; final cavity and capsule gates remain mandatory.
    A buffer reserves space for discretization/accretion, not extra robot size.
    """
    if not config.required_route_height_m:
        return sections, dict(enabled=False, changed_samples=0)
    generator = SectionFieldGenerator(sections.config)
    selected = set(sections.dominant_route_segment_ids)
    fields = []
    changes = []
    failed = []
    for field in sections.segment_fields:
        samples = []
        for sample in field.samples:
            if field.segment_id not in selected:
                samples.append(sample)
                continue
            padding = 2*config.voxel_size+2*config.route_clearance_margin_m
            if affected is None or field.segment_id in affected:
                padding += extra_margin_m
            target_h = config.required_route_height_m+padding
            target_w = config.required_route_width_m+padding
            profile = np.asarray(sample.profile_points, float)
            width, height = np.ptp(profile, axis=0)
            # Vertical clearance is measured in world Z, not a tilted profile axis.
            target_h /= max(abs(sample.binormal[2]), 1e-6)
            new_h = max(height, target_h)
            new_w = max(width, target_w, new_h/sections.config.maximum_height_ratio
                        if new_h > height*(1+1e-10) else width)
            if new_h <= height*(1+1e-10) and new_w <= width*(1+1e-10):
                samples.append(sample)
                continue
            if new_w > max(width, sections.config.maximum_tube_width)*(1+1e-10):
                failed.append(dict(segment_id=field.segment_id, sample_index=sample.index,
                                   reason="Required capsule exceeds configured width/aspect limits"))
                samples.append(sample)
                continue
            scaled = profile*np.array([new_w/width, new_h/height])
            changed = generator._assess_roof(replace(
                sample, profile_points=tuple(map(tuple, scaled)),
                tube_width=sample.tube_width*new_w/width, tube_height=sample.tube_height*new_h/height))
            world = (np.array([sample.x, sample.y, sample.z]) + scaled[:, 0, None]*sample.normal
                     + scaled[:, 1, None]*sample.binormal)
            in_host = host is None or all(host.contains(float(p[0]), float(p[1])) for p in world)
            if (changed.collapse_required or changed.roof_thickness < sections.config.minimum_roof_thickness
                    or not in_host):
                failed.append(dict(segment_id=field.segment_id, sample_index=sample.index,
                                   reason="Enlargement violates host bounds or roof stability/cover"))
                samples.append(sample)
                continue
            samples.append(changed)
            changes.append(dict(segment_id=field.segment_id, sample_index=sample.index,
                                before_width_m=float(width), before_height_m=float(height),
                                after_width_m=float(new_w), after_height_m=float(new_h)))
        fields.append(replace(field, samples=tuple(samples)))
    report = dict(enabled=True, changed_samples=len(changes), changes=changes,
                  extra_margin_m=extra_margin_m, rejected_enlargements=failed)
    if failed:
        raise SurfaceTopologyError("Required route cannot be enlarged within physical constraints",
                                   report=dict(route_section_repair=report))
    return replace(sections, segment_fields=tuple(fields)), report


def inspect_capsule_routes(vertices, faces, paths, segment_ids, *, height, width, margin,
                           placement_attempts=3, placement_max_sweeps=20000):
    from plume_advanced.stages.route_placement import repair_vertical_path

    report = dict(enabled=True, passed=False, height_m=height, width_m=width, margin_m=margin,
                  paths=[], failures=[], defect_regions=[], placement_queries=0,
                  placement_query_budget=placement_max_sweeps,
                  scope="Continuous upright-capsule sweep through the measured corridor; not ground-contact, traction or dynamics validation")
    if not paths:
        report["failures"].append("Required route paths are missing")
        return report
    index = TriangleIndex(vertices, faces)
    # Build the projected triangle index once for all paths, including tiny
    # junction connectors; rebuilding it per segment dominates large meshes.
    all_measurements = vertical_clearances(np.asarray(vertices), np.asarray(faces),
                                           np.concatenate(paths))
    measurement_offset = 0
    total = sum(len(p)-1 for p in paths)
    completed = 0
    radius = width/2+margin
    for number, (path, sid) in enumerate(zip(paths, segment_ids, strict=True)):
        points = np.asarray(path, float)
        measurements = all_measurements[measurement_offset:measurement_offset+len(points)]
        measurement_offset += len(points)
        centers = points.copy()
        invalid = []
        for i, row in enumerate(measurements):
            if not row["inside"] or row["clearance_m"] < height+2*margin:
                invalid.append(i)
            if row["inside"]:
                centers[i, 2] += (row["roof_distance_m"]-row["floor_distance_m"])/2
        bad_edges = []
        minimum = float("inf")
        distances = []
        for i, (a, b) in enumerate(zip(centers, centers[1:])):
            distance = index.swept_capsule_distance(a, b, height-width, radius)
            distances.append(distance)
            minimum = min(minimum, distance)
            if distance < radius-1e-7 or i in invalid or i+1 in invalid:
                bad_edges.append(i)
            completed += 1
            if completed % 100 == 0:
                report_progress("Required route clearance", completed, total,
                                "sweeping the capsule between stations on actual triangles")
        placement = None
        if bad_edges and not invalid and placement_attempts:
            lower = np.array([p[2]-r['floor_distance_m']+height/2+margin
                              for p,r in zip(points, measurements, strict=True)])
            upper = np.array([p[2]+r['roof_distance_m']-height/2-margin
                              for p,r in zip(points, measurements, strict=True)])
            repaired, placement = repair_vertical_path(
                index, centers, lower, upper, distances, height=height, width=width,
                margin=margin, attempts=placement_attempts,
                max_queries=max(0, placement_max_sweeps-report['placement_queries']))
            report['placement_queries'] += placement['queries']
            if placement['passed']:
                # Recheck the complete path independently of the search cache.
                checked = []
                for i,(a,b) in enumerate(zip(repaired,repaired[1:])):
                    checked.append(index.swept_capsule_distance(a,b,height-width,radius))
                    if i % 100 == 0:
                        report_progress('Repaired route verification',i,len(repaired)-1,
                                        'rechecking every capsule edge after path placement')
                new_bad = np.flatnonzero(np.asarray(checked)<radius-1e-7).tolist()
                placement['verified'] = not new_bad
                if new_bad:
                    placement.update(passed=False, failure='Repaired path failed its independent full sweep')
                else:
                    centers, distances, bad_edges = repaired, checked, new_bad
                    minimum = min(checked, default=float('inf'))
        record = dict(path_index=number, segment_id=sid, samples=len(points),
                      passed=not invalid and not bad_edges, invalid_samples=invalid,
                      blocked_edges=bad_edges, center_path_m=centers.tolist(),
                      minimum_axis_distance_m=minimum if np.isfinite(minimum) else None)
        if placement is not None:
            record['placement_repair'] = placement
        report["paths"].append(record)
        if not record["passed"]:
            report["failures"].append(f"Required route {sid} fails capsule clearance")
            selected = np.unique(np.r_[invalid, bad_edges, np.asarray(bad_edges)+1]).astype(int)
            # Separate disconnected failures on one long route. One bounding
            # box around all failures would erase detail in healthy passages.
            for group in np.split(selected, np.flatnonzero(np.diff(selected) > 1)+1):
                affected = centers[group]
                report["defect_regions"].append(dict(kind="route_clearance", segment_ids=[sid],
                    lower_m=(affected.min(axis=0)-radius).tolist(), upper_m=(affected.max(axis=0)+radius).tolist(),
                    center_m=affected.mean(axis=0).tolist()))
    report["passed"] = not report["failures"]
    report_progress("Required route clearance", total, total,
                    "capsule route accepted" if report["passed"] else "capsule route requires repair")
    return report
