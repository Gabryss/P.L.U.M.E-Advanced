"""Optional refinement of an accepted regional graph, with transactional edits.

Sparse host-conditioned events add bends, constrictions and wider pockets.
Host, grade and clearance checks bound each edit; compatible encounters become
explicit junctions. This is geometric refinement, not a lava-flow solver.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from plume_advanced.progress import report_progress
from plume_advanced.stages.network_detail_features import plan_features
from plume_advanced.stages.network_layers import segment_xyz
from plume_advanced.stages.network_quality import assess_network, shape_hash


@dataclass(frozen=True)
class NetworkDetailConfig:
    enabled: bool = False
    strength: float = 0.5
    feature_scale_m: float = 24.0
    sampling_error_m: float = 0.025
    maximum_samples: int = 50_000

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("network.detail.enabled must be boolean")
        for name in ("strength", "feature_scale_m", "sampling_error_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"network.detail.{name} must be finite")
        if not 0 <= self.strength <= 1:
            raise ValueError("network.detail.strength must be between zero and one")
        if not 4 <= self.feature_scale_m <= 200:
            raise ValueError("network.detail.feature_scale_m must be between 4 and 200 m")
        if not .001 <= self.sampling_error_m <= .1:
            raise ValueError("network.detail.sampling_error_m must be between .001 and .1 m")
        if type(self.maximum_samples) is not int or not 100 <= self.maximum_samples <= 1_000_000:
            raise ValueError("network.detail.maximum_samples must be an integer from 100 to 1000000")


def _arc(xy):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def _interpolate(stations, arc, values):
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(stations, arc, values)
    return np.column_stack([np.interp(stations, arc, col) for col in values.T])


def adaptive_indices(stations, values, tolerance, maximum_step, mandatory=()):
    """Bound every reference-sample interpolation error, plus physical spacing.

    Iterative subdivision avoids recursion limits. XYZ and width are all in
    metres; each component is bounded by tolerance. The reference itself is a
    densely sampled polyline, so this is not a continuous-spline error proof.
    """
    keep = {0, len(stations)-1, *map(int, mandatory)}
    anchors = sorted(keep)
    pending = list(zip(anchors[:-1], anchors[1:]))
    while pending:
        a, b = pending.pop()
        if b <= a+1:
            continue
        t = (stations[a+1:b] - stations[a]) / (stations[b] - stations[a])
        error = np.max(abs(values[a+1:b] - ((1-t[:, None])*values[a] + t[:, None]*values[b])), axis=1)
        i = a + 1 + int(np.argmax(error))
        if error[i-a-1] <= tolerance:
            if stations[b] - stations[a] <= maximum_step:
                continue
            i = (a+b)//2
        keep.add(i)
        pending.extend(((a, i), (i, b)))
    return np.array(sorted(keep), dtype=int)


def _proposal(segment, config, host):
    controls = config.detail
    xy = np.array([(p.x, p.y) for p in segment.points])
    arc = _arc(xy)
    original_widths = np.array([p.width for p in segment.points])
    # Physical averaging keeps feature planning independent of sample density.
    mean_width = float(np.sum(.5*(original_widths[:-1]+original_widths[1:])*np.diff(arc))/arc[-1])
    guard = max(2*mean_width, arc[1], arc[-1]-arc[-2])
    if arc[-1] <= 2*guard + controls.feature_scale_m:
        return None, dict(reason="protected_approaches")
    step = min(.5, controls.feature_scale_m/20)
    count = int(np.ceil(arc[-1]/step)) + 1
    if count + len(arc) > controls.maximum_samples:
        return None, dict(reason="reference_sample_budget")
    features, scale = plan_features(segment, config, host, arc, xy, mean_width, guard)
    audit: dict[str, Any] = dict(method="localized_host_features", effective_feature_scale_m=scale,
                                 features=[asdict(f) for f in features])
    if not features:
        return None, dict(audit, reason="no_feature_fits")
    anchors = np.array([v for f in features for v in (f.start_m, f.peak_m, f.end_m)])
    stations = np.unique(np.r_[arc, np.linspace(0, arc[-1], count), anchors])
    if len(stations) > controls.maximum_samples:
        return None, dict(audit, reason="reference_sample_budget")
    reference = _interpolate(stations, arc, xy)
    xyz = _interpolate(stations, arc, segment_xyz(segment, config.layers))
    widths = _interpolate(stations, arc, original_widths)
    delta = np.zeros((len(stations), 4))
    for feature in features:
        profile = feature.profile(stations)
        before = _interpolate([max(feature.peak_m-scale/3, 0)], arc, xy)[0]
        after = _interpolate([min(feature.peak_m+scale/3, arc[-1])], arc, xy)[0]
        direction = after-before
        direction /= max(np.linalg.norm(direction), 1e-9)
        # A single event direction avoids translating fine changes in the
        # original polyline tangent into extra waviness.
        normal = np.array([-direction[1], direction[0]])
        delta[:, :2] += normal*(feature.lateral_m*profile)[:, None]
        delta[:, 3] += widths*feature.width_fraction*profile
        if config.layers.enabled:
            delta[:, 2] -= feature.burial_m*profile
    delta[:, 3] = np.clip(widths+delta[:, 3], np.minimum(widths, 2*config.minimum_passage_radius),
                          np.maximum(widths, 1.9*config.maximum_passage_radius)) - widths
    # Keep feature boundaries/peaks and all original vertices outside active
    # supports exact. Adaptive interpolation must not leak into quiet reaches.
    active = np.any(delta != 0, axis=1)
    mandatory = np.unique(np.r_[np.searchsorted(stations, arc)[~active[np.searchsorted(stations, arc)]],
                                np.searchsorted(stations, anchors)])
    return (stations, reference, xyz[:, 2], widths, delta, mandatory), audit


def _materialize(segment, config, host, proposal, factor):
    stations, reference, z, widths, delta, mandatory = proposal
    xy = reference + factor*delta[:, :2]
    width = widths + factor*delta[:, 3]
    samples = [host.sample(float(x), float(y)) for x, y in xy]
    elevation = np.array([p.elevation for p in samples])
    # Moving XY carries the original host-relative depth. Vertical refinement
    # changes that depth explicitly, without overwriting substrate elevation.
    original_elevation = _interpolate(stations, _arc(np.array([(p.x, p.y) for p in segment.points])),
                                      [p.elevation for p in segment.points])
    target_z = z + elevation-original_elevation + factor*delta[:, 2]
    values = np.column_stack((xy, target_z, width))
    chosen = adaptive_indices(stations, values, config.detail.sampling_error_m,
                              min(2., config.detail.feature_scale_m/4), mandatory)
    new_arc = _arc(xy[chosen])
    points = tuple(replace(segment.points[0], index=i, x=float(xy[j, 0]), y=float(xy[j, 1]),
                           elevation=float(elevation[j]), width=float(width[j]), arc_length=float(new_arc[i]),
                           slope_degrees=samples[j].slope_degrees, cover_thickness=samples[j].cover_thickness,
                           roof_competence=samples[j].roof_competence, growth_cost=samples[j].growth_cost)
                   for i, j in enumerate(chosen))
    result = replace(segment, points=points, metadata=dict(segment.metadata))
    if config.layers.enabled:
        # Remove the previous detail offset before evaluating the new baseline.
        result.metadata.pop("network_detail_vertical_offsets_m", None)
        offsets = target_z[chosen] - segment_xyz(result, config.layers)[:, 2]
        result.metadata["network_detail_vertical_offsets_m"] = offsets.tolist()
    error = float(np.max(abs(values-_interpolate(stations, stations[chosen], values[chosen]))))
    return result, dict(samples=len(chosen), reference_samples=len(stations), sampling_error_m=error,
                        maximum_xy_displacement_m=float(np.linalg.norm(factor*delta[:, :2], axis=1).max()),
                        maximum_z_adjustment_m=float(abs(factor*delta[:, 2]).max()),
                        maximum_width_adjustment_m=float(abs(factor*delta[:, 3]).max()))


def refine_network(generator, host, network):
    """Refine once; all rejected proposals roll back without changing the seed."""
    controls = network.config.detail
    if not controls.enabled:
        return network
    if network.config.topology.generation_mode != "regional_growth" or not network.config.quality.enabled:
        raise ValueError("network.detail requires inspected regional_growth (network-only)")
    if "detail" in network.backend_provenance:
        if network.backend_provenance["detail"].get("controls") != asdict(controls):
            raise ValueError("Changed detail settings require the original coarse network")
        return network
    baseline = assess_network(network, host)
    if not baseline["accepted"]:
        raise ValueError("Detail refinement requires an accepted coarse network")
    report: dict[str, Any] = dict(schema="plume.network-detail.v3", controls=asdict(controls),
                  baseline_shape_sha256=shape_hash(network), edits=[], accepted_edits=0,
                  baseline_samples=sum(len(s.points) for s in network.segments),
                  limitation="Bounded morphology and local junction heuristic; no geological validation.")
    if not controls.strength:
        report["skip_reason"] = "zero_strength"
    elif report["baseline_samples"] > controls.maximum_samples:
        report["skip_reason"] = "baseline_exceeds_sample_budget"
    current = network
    if controls.strength and report["baseline_samples"] <= controls.maximum_samples:
        total = len(network.segments)
        for index, original in enumerate(network.segments):
            sid = original.segment_id
            report_progress("Network detail", index, total, f"passage {sid}: host features, coupled geometry and full inspection")
            segment = next(s for s in current.segments if s.segment_id == sid)
            proposal, audit = _proposal(segment, network.config, host)
            edit = dict(segment_id=sid, kind="localized_features", accepted=False, proposal=audit, trials=[])
            report["edits"].append(edit)
            if proposal is None:
                continue
            for factor in (1., .5, .25):
                replacement, measurements = _materialize(segment, network.config, host, proposal, factor)
                samples = sum(len(s.points) for s in current.segments) - len(segment.points) + len(replacement.points)
                trial = dict(factor=factor, **measurements)
                edit["trials"].append(trial)
                if samples > controls.maximum_samples:
                    trial.update(accepted=False, failures=["sample_budget"])
                    continue
                candidate = replace(current, segments=tuple(replacement if s.segment_id == sid else s for s in current.segments))
                assessment = assess_network(candidate, host)
                overlap_checks = {"unmodeled_plan_crossings", "regional_passage_clearance", "layer_passage_separation"}
                if any(c["name"] in overlap_checks and not c["passed"] for c in assessment["checks"]):
                    from plume_advanced.stages.network_detail_junctions import connect_overlaps

                    candidate, junction_audit = connect_overlaps(generator, host, candidate, {sid})
                    trial["connections"] = junction_audit
                    assessment = assess_network(candidate, host)
                    if sum(len(s.points) for s in candidate.segments) > controls.maximum_samples:
                        assessment["accepted"] = False
                        assessment["checks"].append(dict(name="sample_budget", passed=False))
                trial.update(accepted=assessment["accepted"],
                             failures=[c["name"] for c in assessment["checks"] if not c["passed"]])
                if assessment["accepted"]:
                    current = candidate
                    edit["accepted"] = True
                    report["accepted_edits"] += 1
                    break
        report_progress("Network detail", total, total, "rebuilding flow, junctions, occupancy and final inspection")
    # Rebuild derived state only once, after the transactional geometry trials.
    if report["accepted_edits"]:
        from plume_advanced.stages.network_acceptance import rebuild_network_geometry

        current = rebuild_network_geometry(generator, host, current, list(current.segments))
    final = assess_network(current, host)
    graph_preserved = current.nodes == network.nodes and [
        (s.segment_id, s.start_node_id, s.end_node_id, s.kind, s.z_level) for s in current.segments
    ] == [(s.segment_id, s.start_node_id, s.end_node_id, s.kind, s.z_level) for s in network.segments]
    from plume_advanced.stages.network_detail_junctions import original_routes_preserved

    routes_preserved = original_routes_preserved(network, current)
    if not final["accepted"] or not routes_preserved:
        current = network
        failures = [c["name"] for c in final["checks"] if not c["passed"]]
        if not routes_preserved:
            failures.append("detail_original_route_preservation")
        report.update(status="rolled_back", final_failures=failures)
        final = baseline
    else:
        report["status"] = "refined" if report["accepted_edits"] else "unchanged"
    report.update(final_shape_sha256=shape_hash(current), final_samples=sum(len(s.points) for s in current.segments),
                  graph_preserved=graph_preserved if report["status"] != "rolled_back" else True,
                  original_routes_preserved=original_routes_preserved(network, current),
                  added_junctions=len(current.nodes)-len(network.nodes), retained_edits=report["accepted_edits"] if report["status"] == "refined" else 0)
    quality = dict(current.quality_report, detail=report, final_assessment=final,
                   selected_shape_sha256=shape_hash(current))
    return replace(current, backend_provenance=dict(current.backend_provenance, detail=report),
                   quality_report=quality)
