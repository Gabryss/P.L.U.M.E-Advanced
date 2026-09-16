"""Select reproducible passage views using the imported surface's sight lines."""

from __future__ import annotations

import numpy as np


def select_passage_target(centers, arc, index, eye, ray_distance):
    """Score a small viewing cone, rather than accepting one unobstructed ray.

    ``ray_distance(direction, reach)`` returns the first hit distance or reach
    if no hit. Callers supply queries against their actual imported mesh.
    Only candidates with at least two metres of central visibility qualify.
    """
    centers, arc, eye = np.asarray(centers, float), np.asarray(arc, float), np.asarray(eye, float)
    if centers.shape != (len(arc), 3) or not 0 <= index < len(arc):
        raise ValueError("Invalid passage view samples")
    if not np.isfinite(np.r_[centers.ravel(), arc, eye]).all():
        raise ValueError("Passage view coordinates must be finite")
    best = None
    seen = set()
    for sign in (1., -1.):
        for reach in (10., 6., 4., 2.):
            candidate = int(np.argmin(abs(arc - (arc[index] + sign * reach))))
            if candidate == index or candidate in seen:
                continue
            seen.add(candidate)
            target = centers[candidate].copy()
            direction = target-eye
            distance = np.linalg.norm(direction)
            if distance < 2. or np.linalg.norm(direction[:2]) < .5:
                continue
            forward = direction/distance
            right = np.cross(forward, [0., 0., 1.])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            rays = [forward, forward+.35*right, forward-.35*right,
                    forward+.2*up, forward-.2*up]
            hits = [float(ray_distance(ray/np.linalg.norm(ray), 12.)) for ray in rays]
            if not np.isfinite(hits).all() or min(hits) < 0 or hits[0] < 2.:
                continue
            score = float(np.percentile(hits, 25) + .25*hits[0])
            if best is None or score > best["score"]:
                best = dict(target_m=target.tolist(), score=score, sight_distances_m=hits,
                            target_sample_index=candidate)
    return best
