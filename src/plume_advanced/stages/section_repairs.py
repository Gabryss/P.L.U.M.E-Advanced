"""Measured local profile edits; all candidates still need full mesh acceptance.

Floor grading is an explicit simulation intervention, not a lava-flow model.
The host, section centres, frames, graph endpoints and random streams are fixed.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import diags, eye, hstack, vstack

from plume_advanced.progress import report_progress
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.stages.surface_topology import SurfaceTopologyError


def grade_floor(arc, floor, lower, upper, *, max_grade, max_curvature):
    """Minimum absolute change with bounded grade, curvature and fixed regions.

    Constraints are linear on irregularly spaced stations. Returned floors are
    only a design proposal; they do not certify terrain between profile samples.
    """
    arc, floor, lower, upper = (np.asarray(a, float) for a in (arc, floor, lower, upper))
    n = len(arc)
    if (n < 3 or any(a.shape != (n,) for a in (floor, lower, upper))
            or not all(np.isfinite(a).all() for a in (arc, floor, lower, upper))
            or np.any(np.diff(arc) <= 1e-8) or np.any(lower > upper)):
        return None
    # Work relative to the first elevation for translation-invariant numerics.
    origin = floor[0]
    base, lo, hi = floor-origin, lower-origin, upper-origin
    ds = np.diff(arc)
    slope = diags([-1/ds, 1/ds], [0, 1], shape=(n-1, n), format="csr")
    span = (ds[:-1]+ds[1:])/2
    curvature = diags([1/span], [0]) @ (slope[1:]-slope[:-1])
    constraints = vstack([slope, -slope, curvature, -curvature], format="csr")
    limits = np.r_[np.full(2*(n-1), max_grade), np.full(2*(n-2), max_curvature)]
    identity = eye(n, format="csr")
    matrix = vstack([hstack([constraints, constraints*0]),
                     hstack([identity, -identity]), hstack([-identity, -identity])], format="csr")
    result = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=matrix,
                     b_ub=np.r_[limits, base, -base],
                     bounds=[*zip(lo, hi, strict=True), *[(0, None)]*n], method="highs")
    if not result.success:
        return None
    candidate = result.x[:n]
    if (np.any(candidate < lo-1e-7) or np.any(candidate > hi+1e-7)
            or np.any(constraints @ candidate > limits+1e-7)):
        return None
    # Pinned stations remain bit-for-bit unchanged, including graph junctions.
    candidate = candidate+origin
    candidate[lower == upper] = floor[lower == upper]
    return candidate


def _checked_sample(generator, sample, profile, host):
    """Keep the same physical guards as ordinary section-envelope fitting."""
    width, height = np.ptp(profile, axis=0)
    changed = generator._assess_roof(replace(sample,
        profile_points=tuple(map(tuple, profile)), tube_width=float(width), tube_height=float(height)))
    world = (np.array([sample.x, sample.y, sample.z])
             + profile[:, 0, None]*sample.normal + profile[:, 1, None]*sample.binormal)
    config = generator.config
    if (width > config.maximum_tube_width+1e-7
            or height < config.minimum_tube_height-1e-7
            or height > width*config.maximum_height_ratio+1e-7
            or changed.collapse_required or changed.roof_thickness < config.minimum_roof_thickness
            or changed.floor_world_z < sample.surface_z-sample.cover_thickness+config.minimum_vertical_clearance
            or (host is not None and not all(host.contains(float(p[0]), float(p[1])) for p in world))):
        raise SurfaceTopologyError("Local section repair violates host, aspect, height or roof stability constraints")
    return changed


def repair_obstructed_sections(sections, controls, host, points, *, attempt):
    """Expand near measured blocked probes; preserve remote profiles and endpoints.

    Unlike a bounding-box clearance fit, this expands around the protected
    station itself. It can therefore address off-centre relief intrusions even
    when the total section height was already sufficient.
    """
    generator = SectionFieldGenerator(sections.config)
    points = np.asarray(points, float)
    fields, changes = [], []
    bound = min(.5, max(2*controls.voxel_size, .25)*attempt)
    blend = max(4., 4*controls.surface_feature_scale_m)
    for field in sections.segment_fields:
        samples = []
        for i, sample in enumerate(field.samples):
            distance = float(np.linalg.norm(points-[sample.x, sample.y, sample.z], axis=1).min())
            u = max(0., 1-distance/blend)
            if not u or i in (0, len(field.samples)-1):
                samples.append(sample)
                continue
            profile = np.asarray(sample.profile_points)
            delta = bound*u*u*(3-2*u)
            # Axis offsets expand a floor/roof even when it nearly touches the
            # protected origin. Uniform scaling barely moves such a boundary.
            # Each axis moves at most delta/sqrt(2), so the vector edit stays
            # inside the recorded bound. Full section/mesh guards still apply.
            expanded = profile + np.sign(profile)*delta/np.sqrt(2.)
            changed = _checked_sample(generator, sample, expanded, host)
            samples.append(changed)
            changes.append(dict(segment_id=field.segment_id, sample_index=sample.index,
                maximum_profile_change_m=float(np.linalg.norm(expanded-profile, axis=1).max())))
        fields.append(replace(field, samples=tuple(samples)))
    return replace(sections, segment_fields=tuple(fields)), dict(
        method="local_profile_expansion", changed_samples=len(changes), changes=changes,
        maximum_change_m=bound, blend_m=blend, endpoints_preserved=True)


def repair_ground_sections(sections, controls, host, ground_report, *, attempt=1, network=None):
    """Soften longitudinal floor transitions only near measured slope/step failures.

    Roof contours and junction endpoints are pinned. Measured cross-slope is
    graded near the route as well. Fine relief, absent support and arbitrary
    body collisions are not guaranteed repairable by this operation.
    """
    targets: dict[int, list] = {}
    gradients: dict[int, list] = {}
    degree: dict[int, int] = {}
    endpoints = {}
    if network is not None:
        for segment in network.segments:
            endpoints[segment.segment_id] = (segment.start_node_id, segment.end_node_id)
            for node in endpoints[segment.segment_id]:
                degree[node] = degree.get(node, 0)+1
    required = set(sections.dominant_route_segment_ids)
    for path in ground_report.get("paths", []):
        sid = path["segment_id"]
        if sid not in required:
            continue
        for point, pose in zip(path.get("plan_path_m", []), path.get("poses", []), strict=True):
            if pose.get("failure") in {"slope_limit", "step_or_support_gap_limit"}:
                targets.setdefault(sid, []).append(point)
                probes = np.asarray(pose.get("floor_points_m", []), float)
                if len(probes) >= 3:
                    matrix = np.c_[probes[:, :2]-np.mean(probes[:, :2], axis=0), np.ones(len(probes))]
                    fit, _, rank, _ = np.linalg.lstsq(matrix, probes[:, 2], rcond=None)
                    gradients.setdefault(sid, []).append(fit[:2] if rank == 3 else np.zeros(2))
                else:
                    gradients.setdefault(sid, []).append(np.zeros(2))
    generator = SectionFieldGenerator(sections.config)
    bound = controls.ground_ramp_max_change_m
    grade = .8*np.tan(np.radians(controls.ground_max_slope_deg))
    curvature = .5*controls.ground_max_step_m/max(controls.ground_robot_length_m, .01)**2
    reach = attempt*max(4., 4*controls.ground_robot_length_m, 2*bound/max(grade, 1e-8))
    fields, changes, rejected = [], [], []
    for field in sections.segment_fields:
        if field.segment_id not in targets or len(field.samples) < 3:
            fields.append(field)
            continue
        report_progress("Ground ramp design", detail=f"segment {field.segment_id}; {reach:g} m local reach; {bound:g} m edit cap")
        samples = field.samples
        xy = np.array([(s.x, s.y) for s in samples])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        floor = np.array([s.floor_world_z for s in samples])
        indices = sorted({int(np.linalg.norm(xy-np.asarray(p[:2]), axis=1).argmin())
                          for p in targets[field.segment_id]})
        active = np.zeros(len(samples), bool)
        for i in indices:
            active |= np.abs(arc-arc[i]) < reach
        pinned = [0, len(samples)-1]
        if field.segment_id in endpoints:
            pinned = [i for i, node in zip((0, len(samples)-1), endpoints[field.segment_id], strict=True)
                      if degree[node] > 1]
        active[pinned] = False
        delta = np.where(active, bound, 0.)
        lower, upper = floor-delta, floor+delta
        reserve = (controls.required_route_height_m+2*controls.route_clearance_margin_m
                   + controls.surface_floor_relief_m+controls.surface_roof_relief_m
                   + 2*controls.surface_crust_relief_m+2*controls.voxel_size)
        # Never take away the headroom needed by the unchanged robot/relief.
        upper[active] = np.minimum(upper[active],
            np.array([s.roof_world_z for s in samples])[active]-reserve)
        proposed = grade_floor(arc, floor, lower, upper, max_grade=grade, max_curvature=curvature)
        if proposed is None:
            rejected.append(dict(segment_id=field.segment_id, reason="No bounded floor profile satisfies grade, transitions and headroom"))
            fields.append(field)
            continue
        updated, local_changes = [], []
        for i, (sample, old, new) in enumerate(zip(samples, floor, proposed, strict=True)):
            nearest = int(np.linalg.norm(np.asarray(targets[field.segment_id])[:, :2]-xy[i], axis=1).argmin())
            gradient = np.asarray(gradients[field.segment_id][nearest])
            normal_xy, tangent_xy = np.asarray(sample.normal[:2]), np.asarray(sample.tangent[:2])
            cross = float(gradient @ normal_xy)
            along = float(gradient @ tangent_xy)
            allowable_cross = np.sqrt(max(0., grade**2-min(along**2, grade**2)))
            excess = cross-float(np.clip(cross, -allowable_cross, allowable_cross))
            if not active[i]:
                excess = 0.
            if abs(new-old) <= 1e-8 and abs(excess) <= 1e-8:
                updated.append(sample)
                continue
            profile = np.array(sample.profile_points)
            up = np.array([sample.normal[2], sample.binormal[2]])
            if up @ up < 1e-8:
                raise SurfaceTopologyError("Ground repair cannot grade a vertical section plane")
            elevations = sample.z + profile @ up
            weight = (elevations.max()-elevations)/max(float(np.ptp(elevations)), 1e-8)
            # Taper cross-floor changes toward the walls. Include the nearest
            # bottom vertices so a coarse contour can represent the correction.
            lower_u = profile[weight > .5, 0]
            nearest_sides = [float(np.min(np.abs(lower_u[lower_u*sign > 0])))
                             if np.any(lower_u*sign > 0) else 0. for sign in (-1, 1)]
            corridor = max(controls.required_route_width_m+2*controls.route_clearance_margin_m,
                           3*max(nearest_sides))
            lateral_weight = np.exp(-(profile[:, 0]/max(corridor, 1e-8))**4)
            shift = (new-old)-excess*profile[:, 0]*lateral_weight
            shift = np.clip(shift, -bound, bound)
            adjusted = profile + weight[:, None]*shift[:, None]*up/(up @ up)
            updated.append(_checked_sample(generator, sample, adjusted, host))
            local_changes.append(dict(segment_id=field.segment_id, sample_index=sample.index,
                before_floor_m=float(old), after_floor_m=updated[-1].floor_world_z,
                measured_cross_grade=cross, cross_grade_correction=excess,
                maximum_vertical_edit_m=float(np.max(np.abs(weight*shift)))))
        changes.extend(local_changes)
        fields.append(replace(field, samples=tuple(updated)))
    return replace(sections, segment_fields=tuple(fields)), dict(
        method="bounded_floor_grading", changed_samples=len(changes), changes=changes,
        rejected_segments=rejected, maximum_change_m=bound, reach_m=reach,
        design_slope_deg=float(np.degrees(np.arctan(grade))),
        design_curvature_per_m=curvature, roof_and_junctions_preserved=True,
        scope="Engineered simulation floor; final visual/collision mesh inspection remains mandatory")
