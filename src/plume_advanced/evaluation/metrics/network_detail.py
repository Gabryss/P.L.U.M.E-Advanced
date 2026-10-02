"""Descriptive locality diagnostics for paired saved network artifacts.

These measurements diagnose distributed waviness; they are not geological
acceptance criteria. Projection follows each original route, including its
junction-split descendants, rather than matching unrelated nearby passages.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


def _route(payload, original):
    pieces = {s['source_node_id']: s for s in payload['segments']
              if s.get('metadata', {}).get('detail_parent_segment_id', s['segment_id']) == original['segment_id']}
    samples: list[np.ndarray] = []
    node = original['source_node_id']
    while node in pieces:
        segment = pieces.pop(node)
        xy = [(p['x'], p['y']) for p in segment['centerline']]
        samples.extend(np.c_[xy, [p['width'] for p in segment['centerline']]][1 if samples else 0:])
        node = segment['target_node_id']
    if pieces or node != original['target_node_id'] or len(samples) < 2:
        raise ValueError('Locality measurements require preserved original routes')
    return np.array(samples)


def _sample(values, step=.5):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1))]
    stations = np.linspace(0, arc[-1], max(3, int(np.ceil(arc[-1]/step))+1))
    return stations, np.column_stack([np.interp(stations, arc, col) for col in values.T])


def route_residuals(coarse, detailed, original):
    """Project onto a dense original route and measure lateral/width residuals."""
    stations, baseline = _sample(_route(coarse, original))
    _, target = _sample(_route(detailed, original))
    a, chord = baseline[:-1, :2], np.diff(baseline[:, :2], axis=0)
    _, nearby = cKDTree(a+.5*chord).query(target[:, :2], k=min(4, len(chord)))
    if nearby.ndim == 1:
        nearby = nearby[:, None]
    d = chord[nearby]
    fraction = np.clip(np.sum((target[:, None, :2]-a[nearby])*d, axis=2)/np.sum(d*d, axis=2), 0, 1)
    residual = target[:, None, :2] - a[nearby]-fraction[:, :, None]*d
    best = np.argmin(np.sum(residual**2, axis=2), axis=1)
    rows = np.arange(len(target))
    j, f, r = nearby[rows, best], fraction[rows, best], residual[rows, best]
    along = stations[j]+f*np.diff(stations)[j]
    lateral = (chord[j, 0]*r[:, 1]-chord[j, 1]*r[:, 0])/np.linalg.norm(chord[j], axis=1)
    width = target[:, 2]-((1-f)*baseline[j, 2]+f*baseline[j+1, 2])
    # Projection can tie at an endpoint. Keep one deterministic value per station.
    u, selected = np.unique(along, return_index=True)
    return stations, np.interp(stations, u, lateral[selected]), np.interp(stations, u, width[selected])


def _runs(mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(changes[::2], changes[1::2]))


def locality_diagnostics(coarse, detailed):
    """Lengths are measured on the original XY route, with a 0.5 m grid.

    Quiet means <=5 cm lateral and <=3 cm width change. Profile coherence is
    absolute bend/width correlation within changed reaches of at least 5 m.
    Peak spacing CV is reported only with >=3 spacings on one route; null means
    insufficient evidence, not perfect irregularity. None are pass/fail gates.
    """
    total = quiet = 0.
    quiet_runs: list[float] = []
    active_runs: list[float] = []
    coherence, spacing_cv = [], []
    for original in coarse['segments']:
        station, lateral, width = route_residuals(coarse, detailed, original)
        step = station[1]-station[0]
        active = (abs(lateral) > .05) | (abs(width) > .03)
        total += station[-1]
        # Trapezoid endpoint weights avoid counting an extra station as length.
        quiet += float(np.sum(.5*((~active[:-1]).astype(float)+(~active[1:]).astype(float)))*step)
        for flag, collection in ((~active, quiet_runs), (active, active_runs)):
            collection.extend(min(station[-1], (b-a)*step) for a, b in _runs(flag))
        for a, b in _runs(active):
            bend, envelope = abs(lateral[a:b]), abs(width[a:b])
            if (b-a)*step >= 5 and np.std(bend) > 1e-5 and np.std(envelope) > 1e-5:
                coherence.append(float(np.corrcoef(bend, envelope)[0, 1]))
        peaks, _ = find_peaks(abs(width), prominence=.03, distance=max(1, int(10/step)))
        gaps = np.diff(station[peaks])
        if len(gaps) >= 3:
            spacing_cv.append(float(np.std(gaps)/np.mean(gaps)))
    def percentile(values, p):
        return float(np.percentile(values, p)) if values else None
    return dict(quiet_length_fraction=float(quiet/max(total, 1e-9)),
                quiet_reach_median_m=percentile(quiet_runs, 50), quiet_reach_p90_m=percentile(quiet_runs, 90),
                changed_reach_count=len(active_runs), changed_reach_median_m=percentile(active_runs, 50),
                changed_reach_span_cv=float(np.std(active_runs)/np.mean(active_runs)) if len(active_runs) > 1 else None,
                bend_width_profile_correlation_median=percentile(coherence, 50),
                width_peak_spacing_cv_median=percentile(spacing_cv, 50),
                routes_with_enough_peaks=len(spacing_cv),
                limitation='Geometric locality diagnostics, not proof of geological realism; ignores vertical changes.')
