"""Physical clearance coordinates for bounded regional connection searches."""

import numpy as np
from scipy.spatial import cKDTree


def connection_space(planner, paths):
    """Return search positions, passage distances and per-cell host costs.

    Layer labels alone cannot screen ramps crossing between levels. Sample the
    swept ramp centreline and scale vertical distance by the declared passage
    height plus rock gap. This is a coarse search filter; the candidate's actual
    smoothed XYZ and widths still require full metric inspection.
    """
    p = planner
    occupied = sorted({cell for path in paths for cell in path})
    layered = bool(np.any(p.layer_ids))
    if not layered:
        return p.xy, cKDTree(p.xy[occupied]).query(p.xy)[0], p.cost.ravel()
    controls = p.config.layers
    scale = np.array([1., 1., 1.1 * p.width / (controls.passage_height_m + controls.minimum_rock_m)])
    positions = np.column_stack((p.xy, p.sample_host(p.xy)[:, 0] - p.depths[p.layer_ids])) * scale
    samples = []
    for a, b in sorted({edge for path in paths for edge in zip(path, path[1:])}):
        length = np.linalg.norm(p.xy[b] - p.xy[a])
        t = np.linspace(0., 1., max(2, int(np.ceil(length / min(2., p.width / 3))) + 1))
        xy = p.xy[a] * (1 - t[:, None]) + p.xy[b] * t[:, None]
        depth = p.depths[p.layer_ids[a]] + (p.depths[p.layer_ids[b]] - p.depths[p.layer_ids[a]]) * t*t*(3 - 2*t)
        samples.append(np.column_stack((xy, p.sample_host(xy)[:, 0] - depth)) * scale)
    distance = cKDTree(np.concatenate(samples)).query(positions)[0]
    return positions, distance, p.cost_fields.ravel()
