"""Place section profiles inside an accepted regional layer envelope.

The network owns centreline XYZ. Morphology may change the cross section inside
that envelope, but cannot move a ramp, change a layer depth or create a new join.
"""
from collections import defaultdict
from dataclasses import replace

import numpy as np

from plume_advanced.progress import report_progress
from plume_advanced.stages.network_layers import segment_xyz


def layer_section_fields(network, fields, build_frame):
    controls = network.config.layers
    segments = {segment.segment_id: segment for segment in network.segments}
    prepared = {}
    endpoints = defaultdict(list)
    for field in fields:
        segment = segments[field.segment_id]
        arc = np.array([s.segment_arc_length for s in field.samples])
        stations = np.array([p.arc_length for p in segment.points])
        network_xyz = segment_xyz(segment, controls)
        if len(arc) < 2 or not np.isfinite(network_xyz).all() or np.any(np.diff(arc) <= 0):
            raise ValueError('Layer sections require finite accepted XYZ and increasing stations')
        xyz = np.column_stack([np.interp(arc, stations, network_xyz[:, k]) for k in range(3)])
        tangents = np.gradient(xyz, arc, axis=0)
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1e-12)
        widths = np.interp(arc, stations, [p.width for p in segment.points])
        records = []
        for sample, center, tangent, width in zip(field.samples, xyz, tangents, widths):
            normal, binormal = map(np.asarray, build_frame(tuple(tangent), None))
            profile = np.array(sample.profile_points)
            if not np.isfinite(profile).all() or np.any(np.ptp(profile, axis=0) <= 0):
                raise ValueError('Layer sections require finite positive morphology profiles')
            profile[:, 1] -= (profile[:, 1].min() + profile[:, 1].max()) / 2
            halfwidth = min(float(abs(profile[:, 0]).max()), width / 2)
            height = min(float(np.ptp(profile[:, 1])), controls.passage_height_m)
            # Leave lateral room even for a steep, narrow ramp. This constraint
            # is applied before choosing the shared floor at a graph node.
            height = min(height, .9 * width * binormal[2] / max(np.linalg.norm(binormal[:2]), 1e-12))
            shape = profile / [max(float(abs(profile[:, 0]).max()), 1e-12), np.ptp(profile[:, 1]) / 2]
            records.append(dict(center=center, tangent=tangent, normal=normal, binormal=binormal,
                                width=width, halfwidth=halfwidth, height=height, shape=shape))
        prepared[field.segment_id] = records
        for node, index in ((segment.start_node_id, 0), (segment.end_node_id, -1)):
            endpoints[node].append((field.segment_id, index))
    targets = {}
    for group in endpoints.values():
        if len(group) < 2:
            continue
        values = [prepared[sid][index] for sid, index in group]
        if np.max(np.ptp([value['center'] for value in values], axis=0)) > 1e-5:
            raise ValueError('Layer junctions must share a single accepted XYZ position')
        reference = min(values, key=lambda value: value['halfwidth'])
        shared = dict(height=min(value['height'] for value in values),
                      halfwidth=min(value['halfwidth'] for value in values), shape=reference['shape'])
        for key in group:
            targets[key] = shared
    result = []
    for number, field in enumerate(fields):
        report_progress('Layer cross sections', number, len(fields), f'segment {field.segment_id}; preserve network XYZ')
        records = prepared[field.segment_id]
        length = field.samples[-1].segment_arc_length
        samples = []
        for sample, value in zip(field.samples, records):
            height, halfwidth, shape = value['height'], value['halfwidth'], value['shape'].copy()
            for index in (0, -1):
                target = targets.get((field.segment_id, index))
                if target is None:
                    continue
                reach = min(2 * records[index]['width'], .45 * length)
                distance = abs(sample.segment_arc_length - field.samples[index].segment_arc_length)
                u = max(0., 1 - distance / max(reach, 1e-12))
                weight = u*u*(3-2*u)
                height += weight * (target['height'] - height)
                halfwidth += weight * (target['halfwidth'] - halfwidth)
                shape += weight * (target['shape'] - shape)
            # Convex blending can reduce extrema; restore a centred height so
            # incident floors agree exactly, without changing centreline XYZ.
            shape[:, 1] -= (shape[:, 1].min() + shape[:, 1].max()) / 2
            shape[:, 1] /= np.ptp(shape[:, 1]) / 2
            v = shape[:, 1] * height / (2 * value['binormal'][2])
            # Orthogonal horizontal axes give a conservative circular plan
            # envelope matching Stage B, including the ramp's tilted profile.
            drift = float(abs(v).max()) * np.linalg.norm(value['binormal'][:2])
            lateral_limit = np.sqrt(max((value['width']/2)**2 - drift**2, 0.))
            halfwidth = min(halfwidth, lateral_limit)
            profile = np.column_stack((shape[:, 0] * halfwidth, v))
            x, y, z = value['center']
            samples.append(replace(sample, x=float(x), y=float(y), z=float(z),
                centerline_depth=sample.surface_z-float(z),
                tangent=tuple(value['tangent']), normal=tuple(value['normal']), binormal=tuple(value['binormal']),
                tube_width=float(np.ptp(profile[:, 0])), tube_height=float(np.ptp(profile[:, 1])),
                profile_points=tuple(map(tuple, profile))))
        result.append(replace(field, samples=tuple(samples)))
    report_progress('Layer cross sections', len(fields), len(fields), 'layer envelopes and shared junction floors fitted')
    return result


def assess_layer_sections(network, sections):
    """Fail closed if a later profile edit leaves the inspected layer envelope."""
    lookup = {segment.segment_id: segment for segment in network.segments}
    controls = network.config.layers
    failures: dict[str, list[int]] = {
        name: [] for name in ('layer_section_centerline', 'layer_section_envelope', 'layer_section_roof')}
    for field in sections.segment_fields:
        segment = lookup[field.segment_id]
        xyz = segment_xyz(segment, controls)
        stations = [p.arc_length for p in segment.points]
        for sample in field.samples:
            expected = np.array([np.interp(sample.segment_arc_length, stations, xyz[:, k]) for k in range(3)])
            if not np.allclose([sample.x, sample.y, sample.z], expected, rtol=0, atol=1e-6):
                failures['layer_section_centerline'].append(field.segment_id)
            width = np.interp(sample.segment_arc_length, stations, [p.width for p in segment.points])
            profile = np.asarray(sample.profile_points)
            offset = profile[:, :1]*sample.normal + profile[:, 1:]*sample.binormal
            if (not np.isfinite(offset).all() or np.max(abs(offset[:, 2])) > controls.passage_height_m/2 + 1e-6
                    or np.max(np.linalg.norm(offset[:, :2], axis=1)) > width/2 + 1e-6):
                failures['layer_section_envelope'].append(field.segment_id)
            minimum_roof = max(controls.minimum_rock_m, sections.config.minimum_roof_thickness)
            if sample.roof_thickness < minimum_roof - .02:
                failures['layer_section_roof'].append(field.segment_id)
    return [dict(name=name, passed=not ids, severity='error', value=len(set(ids)), limit=0,
                 segment_ids=sorted(set(ids))) for name, ids in failures.items()]
