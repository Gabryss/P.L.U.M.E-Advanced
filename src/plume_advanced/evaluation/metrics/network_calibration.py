"""Reference statistics for partial Earth network calibration.

Widths from cross-sections and chords through a projected LiDAR footprint are
different observables. Only the latter supplies a spatial scale in metres.
No branching or layer-frequency distribution is inferred from either dataset.
"""

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_closing, label


def weighted_quantiles(values, weights, probabilities):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("Quantiles require matching finite values and positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return values[order[np.minimum(np.searchsorted(cumulative, probabilities), len(values)-1)]]


def width_targets(rows):
    """Equal cave weights; separate between-cave size from within-cave variation.

    Each section is divided by its cave median. Pool these relative widths with
    equal cave weights, then scale by the median of cave medians. This describes
    a representative size scenario, not the pooled global size distribution.
    """
    grouped = defaultdict(list)
    for row in rows:
        value = float(row["width_m"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Survey widths must be finite and positive")
        grouped[row["cave_id"]].append(value)
    if not grouped:
        raise ValueError("No surveyed widths")
    medians: list[float] = []
    relative: list[float] = []
    weights: list[float] = []
    for cave in sorted(grouped):
        values = np.array(grouped[cave])
        median = float(np.median(values))
        medians.append(median)
        relative.extend(values / median)
        weights.extend(np.full(len(values), 1 / len(values)))
    median_width = float(np.median(medians))
    probabilities = [.05, .25, .5, .75, .95]
    relative_quantiles = weighted_quantiles(relative, weights, probabilities)
    return dict(caves=len(grouped), sections=len(rows), probabilities=probabilities,
                median_cave_width_m=median_width,
                cave_median_width_quantiles_m=np.quantile(medians, probabilities).tolist(),
                relative_width_quantiles=relative_quantiles.tolist(),
                representative_width_quantiles_m=(median_width*relative_quantiles).tolist())


def projected_width_series(xy, *, resolution_m=.25, closing_radius_m=.5,
                           minimum_chord_m=1., trim_fraction=.05):
    """Axis-normal chords, preserving gaps and multiple passages.

    PCA is fit to the measured coordinates, not to a hand-traced centreline.
    Small holes (<1 m²) may be sampling shadows; larger holes are kept. A station
    with two occupied intervals is NEVER bridged into one passage width.
    End 5% margins and multi-interval stations do not enter the scale fit.
    """
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 10 or not np.isfinite(xy).all():
        raise ValueError("Footprint requires at least ten finite XY points")
    if (not np.isfinite(resolution_m) or resolution_m <= 0
            or not np.isfinite(closing_radius_m) or closing_radius_m < 0
            or not np.isfinite(minimum_chord_m) or minimum_chord_m <= 0
            or not 0 <= trim_fraction < .5):
        raise ValueError("Invalid footprint measurement controls")
    centered = xy-np.median(xy, axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(centered.T))
    axis = vectors[:, np.argmax(eigenvalues)]
    if axis[np.argmax(abs(axis))] < 0:
        axis = -axis
    aligned = centered @ np.column_stack((axis, [-axis[1], axis[0]]))
    padding = int(np.ceil(closing_radius_m/resolution_m))+2
    origin = aligned.min(axis=0)-padding*resolution_m
    cells = np.floor((aligned-origin)/resolution_m).astype(int)
    shape = cells.max(axis=0)+padding+1
    if int(shape[0])*int(shape[1]) > 5_000_000:
        raise ValueError("Footprint raster exceeds five million cells; increase resolution_m")
    mask = np.zeros(tuple(shape), bool)
    mask[cells[:, 0], cells[:, 1]] = True
    radius = int(np.ceil(closing_radius_m/resolution_m))
    grid = np.arange(-radius, radius+1)*resolution_m
    structure = grid[:, None]**2 + grid[None, :]**2 <= closing_radius_m**2 + 1e-12
    mask = binary_closing(mask, structure=structure)
    holes, _ = label(~mask)
    boundary_labels = np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]])
    sizes = np.bincount(holes.ravel())
    fill = (sizes*resolution_m**2 < 1.)
    fill[boundary_labels] = False
    fill[0] = False
    mask |= fill[holes]
    widths = np.full(len(mask), np.nan)
    interval_count = np.zeros(len(mask), int)
    stations = origin[0] + (np.arange(len(mask))+.5)*resolution_m
    low, high = np.quantile(aligned[:, 0], [0, 1])
    margin = (high-low)*trim_fraction
    for i, row in enumerate(mask):
        intervals, _ = label(row)
        lengths = np.bincount(intervals)[1:]*resolution_m
        lengths = lengths[lengths >= minimum_chord_m]
        interval_count[i] = len(lengths)
        if len(lengths) == 1 and low+margin <= stations[i] <= high-margin:
            widths[i] = lengths[0]
    return dict(stations=stations, widths=widths, mask=mask, origin=origin,
                resolution_m=resolution_m, intervals=interval_count, aligned=aligned)


def common_footprint_support(series, *, step_m=1.):
    """Compare raster resolutions on the same observed physical reaches.

    Resolution-dependent one-pixel gaps otherwise change which long-lag pairs
    enter a variogram. Intersect validity after interpolation (NaNs propagate);
    never interpolate through a missing station to improve coverage.
    """
    if not series or not np.isfinite(step_m) or step_m <= 0:
        raise ValueError("Common footprint support needs series and a positive step")
    start = max(s["stations"][0] for s in series)
    end = min(s["stations"][-1] for s in series)
    stations = np.arange(np.ceil(start/step_m), np.floor(end/step_m)+1)*step_m
    widths = np.stack([np.interp(stations, s["stations"], s["widths"]) for s in series])
    common = np.isfinite(widths).all(axis=0)
    widths[:, ~common] = np.nan
    return [dict(stations=stations, widths=w, resolution_m=step_m) for w in widths]


def spatial_scale(series, *, lags_m=(1., 2., 4., 6., 8., 12.)):
    """Fit Gaussian covariance length to single-interval width increments.

    Pairs must be in the SAME uninterrupted observed reach. Missing stations
    and branches are not interpolated. This is one footprint's descriptive
    scale, not a globally calibrated geological correlation length.
    """
    widths = np.asarray(series["widths"])
    valid = np.isfinite(widths) & (widths > 0)
    groups, _ = label(valid)
    if valid.sum() < 20:
        raise ValueError("Insufficient single-interval footprint coverage")
    logged = np.zeros(len(widths))
    logged[valid] = np.log(widths[valid])
    variance = float(np.var(logged[valid]))
    if variance < 1e-6:
        raise ValueError("Footprint has no identifiable width variation")
    rows = []
    for lag in lags_m:
        offset = max(1, int(round(lag/series["resolution_m"])))
        pairs = (groups[:-offset] > 0) & (groups[:-offset] == groups[offset:])
        if pairs.sum() >= 20:
            gamma = float(.5*np.mean((logged[offset:][pairs]-logged[:-offset][pairs])**2))
            rows.append(dict(lag_m=offset*series["resolution_m"], pairs=int(pairs.sum()),
                             semivariance=gamma, normalized_semivariance=gamma/variance))
    if len(rows) < 3:
        raise ValueError("Insufficient continuous reach lengths for a spatial-scale fit")
    lags = np.array([r["lag_m"] for r in rows])
    observed = np.array([r["normalized_semivariance"] for r in rows])
    candidates = np.linspace(1., 40., 391)
    loss = np.mean((1-np.exp(-.5*(lags[None, :]/candidates[:, None])**2)-observed)**2, axis=1)
    best = int(np.argmin(loss))
    return dict(correlation_m=float(candidates[best]), fit_loss=float(loss[best]),
                at_search_boundary=best in {0, len(candidates)-1}, variogram=rows,
                valid_length_m=float(valid.sum()*series["resolution_m"]),
                width_quantiles_m=np.quantile(widths[valid], [.05, .25, .5, .75, .95]).tolist(),
                log_width_std=float(np.sqrt(variance)))


def fit_width_parameters(targets, fields, arc, *, minimum_m, maximum_m, gradient):
    """Bounded deterministic grid fit of central width quantiles, not extremes.

    The physical width and gradient caps are applied during fitting. Tail
    discrepancies remain visible in the report; they are not silently clipped
    out of the observations. Fields are cached across candidate evaluations.
    """
    from plume_advanced.stages.network_acceptance import limit_width_gradient

    target = np.array(targets["representative_width_quantiles_m"])[1:4]
    median = targets["median_cave_width_m"]
    candidates = []
    for sigma in np.arange(.1, .851, .025):
        modulation = np.exp(sigma*fields)
        for scale in np.arange(.85, 1.301, .025):
            base = median*scale
            widths = np.clip(base*modulation, minimum_m, maximum_m)
            for index in range(len(widths)):
                widths[index] = limit_width_gradient(widths[index], arc, gradient)
            q = np.quantile(widths, [.05, .25, .5, .75, .95])
            loss = float(np.mean(np.log(q[1:4]/target)**2))
            candidates.append(dict(loss=loss, base_passage_radius=float(base/1.6),
                                   width_log_sigma=float(sigma), predicted_quantiles_m=q.tolist()))
    best = min(candidates, key=lambda c: c["loss"])
    return dict(best=best, candidates=candidates, objective="mean squared log error of quartiles",
                at_search_boundary=(np.isclose(best["width_log_sigma"], [.1, .85]).any()
                                    or np.isclose(best["base_passage_radius"]*1.6/median, [.85, 1.3]).any()).item(),
                limits_m=[minimum_m, maximum_m], maximum_width_gradient=gradient)
