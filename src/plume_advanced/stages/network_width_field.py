"""Stationary, query-independent log-width field in physical XY coordinates.

128 Gaussian spectral modes approximate covariance exp(-r²/(2 L²)). Their
incommensurate frequencies have no imposed repeating wavelength. Coefficients
are seeded once per layer; segment IDs, sample count and query order play no
part. This is a statistical prior, not a model of lava erosion or hydraulics.
"""

import numpy as np

from plume_advanced.procedural import procedural_rng


def log_width_field(xy, seed, correlation_m, *, layer=0):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Width field requires finite XY coordinates")
    if not np.isfinite(correlation_m) or correlation_m <= 0:
        raise ValueError("Width correlation length must be finite and positive")
    modes = 128
    rng = procedural_rng(seed, "network-log-width-v1", layer)
    frequency = rng.normal(size=(modes, 2)) / correlation_m
    coefficients = rng.normal(size=(2, modes)) / np.sqrt(modes)
    values = np.empty(len(xy))
    for start in range(0, len(xy), 1024):
        points = xy[start:start+1024]
        phase = points[:, :1]*frequency[:, 0] + points[:, 1:]*frequency[:, 1]
        values[start:start+1024] = np.sum(
            np.cos(phase)*coefficients[0] + np.sin(phase)*coefficients[1], axis=1)
    return values
