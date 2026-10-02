"""Sparse, host-conditioned morphology events on accepted passage routes.

These are bounded geometric heuristics, not predictions of erosion or collapse.
The event catalogue is sampled in metres before geometry is changed; plan, width
and burial share each event's support instead of using independent noise fields.
"""

from dataclasses import dataclass

import numpy as np

from plume_advanced.procedural import procedural_rng


@dataclass(frozen=True)
class PassageFeature:
    start_m: float
    peak_m: float
    end_m: float
    lateral_m: float
    width_fraction: float
    burial_m: float
    host_contrast: float
    roof_anomaly: float
    kind: str

    def profile(self, stations):
        """One asymmetric, compact C2 bump, exactly zero outside its support."""
        stations = np.asarray(stations)
        t = np.where(stations <= self.peak_m,
                     (stations-self.start_m)/(self.peak_m-self.start_m),
                     (self.end_m-stations)/(self.end_m-self.peak_m))
        t = np.clip(t, 0, 1)
        return t*t*t*(t*(6*t-15)+10)


def passage_scale(segment, config, mean_width):
    """High-supply trunks have broader features than smaller side passages."""
    supply = np.clip((segment.mean_flux/max(config.source_flux, 1e-9))**.2, .7, 1.4)
    role = segment.metadata.get("regional_route_type")
    hierarchy = .85 if role in {"bypass", "blind_branch"} else 1.15
    return float(max(config.detail.feature_scale_m * supply * hierarchy, 6*mean_width))


def plan_features(segment, config, host, arc, xy, mean_width, guard):
    """Plan nonoverlapping events with irregular gaps and locally selected peaks.

    Stable route IDs and seed define the catalogue. Queries and dense sampling
    introduce no extra random draws. Splitting the original graph can change the
    catalogue; resampling the same route does not introduce per-vertex noise.
    """
    rng = procedural_rng(config.random_seed, "network-detail-locality-v3", segment.segment_id)
    scale = passage_scale(segment, config, mean_width)
    cursor = guard
    features = []

    def at(station):
        return np.array([np.interp(station, arc, col) for col in xy.T])

    while cursor < arc[-1]-guard:
        start = cursor + np.clip(rng.lognormal(np.log(.9), .65), .45, 3.)*scale
        span = np.clip(rng.lognormal(np.log(1.4), .5), .85, 3.)*scale
        end = start + span
        if end > arc[-1]-guard:
            break
        # Each possible peak probes both the accepted corridor and its sides.
        # Contrast biases placement, while a finite floor avoids identical
        # choices at every broad host extremum.
        candidates = []
        for fraction in rng.uniform(.28, .72, 5):
            peak = start + fraction*span
            center = at(peak)
            before, after = at(max(0, peak-scale/3)), at(min(arc[-1], peak+scale/3))
            tangent = after-before
            tangent /= max(np.linalg.norm(tangent), 1e-9)
            normal = np.array([-tangent[1], tangent[0]])
            c, a, b, left, right = [host.sample(float(p[0]), float(p[1])) for p in
                                   (center, before, after, center+mean_width*normal, center-mean_width*normal)]
            cross_cost = right.growth_cost-left.growth_cost
            axial_cost = c.growth_cost-.5*(a.growth_cost+b.growth_cost)
            roof = c.roof_competence-.5*(a.roof_competence+b.roof_competence)
            contrast = abs(cross_cost)+abs(axial_cost)+abs(roof)
            candidates.append((peak, cross_cost, axial_cost, roof, contrast))
        weights = .02+np.array([c[-1] for c in candidates])
        peak, cross_cost, axial_cost, roof, contrast = candidates[int(rng.choice(len(candidates), p=weights/weights.sum()))]
        amplitude = config.detail.strength * rng.uniform(.4, 1.)
        # Prefer the less costly side. Homogeneous hosts receive only a small
        # seeded deflection, never a continuous wandering background field.
        direction = np.sign(cross_cost) if abs(cross_cost) > .005 else rng.choice([-1., 1.])
        lateral = direction * mean_width * amplitude * (.25+.75*min(abs(cross_cost)/.08, 1.))
        suitability = float(np.clip(roof/.12-axial_cost/.12, -1., 1.))
        if abs(suitability) < .15:
            suitability = float(rng.choice([-1., 1.]) * rng.uniform(.2, .5))
        width_fraction = .4 * amplitude * suitability
        # Only local roof weakening requests extra burial. No arbitrary
        # elevation oscillation is added to an otherwise uniform layer.
        burial = min(1.5, .35*mean_width)*amplitude*min(max(-roof/.12, 0.), 1.)
        kind = "constriction" if width_fraction < 0 else "pocket"
        features.append(PassageFeature(float(start), float(peak), float(end), float(lateral),
                                       float(width_fraction), float(burial), float(contrast), float(roof), kind))
        cursor = end
    return features, scale
