"""Deterministic branch-wide floor/roof views for native material inspection."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def plan_material_views(centers, segment_ids, arc_lengths, tangents, measurements,
                        *, spacing_m=30., max_views=256):
    if not np.isfinite(spacing_m) or spacing_m <= 0 or type(max_views) is not int or max_views < 2:
        raise ValueError("View spacing must be positive and the view budget at least two")
    centers, tangents = np.asarray(centers, float), np.asarray(tangents, float)
    segment_ids, arc_lengths = np.asarray(segment_ids), np.asarray(arc_lengths, float)
    if centers.shape != tangents.shape or centers.shape != (len(segment_ids), 3) or arc_lengths.shape != segment_ids.shape:
        raise ValueError("Material view inputs have inconsistent shapes")
    if not len(centers) or not np.isfinite(np.r_[centers.ravel(), tangents.ravel(), arc_lengths]).all():
        raise ValueError("Material view inputs must be finite and nonempty")
    air = np.array([m["point_m"] for m in measurements], float)
    if not len(air):
        raise ValueError("Material views require actual-mesh passage measurements")
    tree = cKDTree(air)
    views: list[dict] = []
    coverage: list[dict] = []
    for sid in sorted(set(segment_ids.tolist())):
        ids = np.flatnonzero(segment_ids == sid)
        ids = ids[np.argsort(arc_lengths[ids], kind="stable")]
        arc = arc_lengths[ids]
        length = float(arc[-1]-arc[0])
        if length <= 0:
            raise ValueError(f"Segment {sid} has no inspectable length")
        inset = min(2., length*.25)
        targets = np.unique(np.r_[arc[0]+inset, arc[-1]-inset,
                                  np.arange(arc[0]+inset, arc[-1]-inset, spacing_m)])
        chosen = sorted({int(ids[np.argmin(abs(arc-t))]) for t in targets})
        stations = sorted(float(arc_lengths[i]) for i in chosen)
        sampling_gap = float(np.diff(arc).max(initial=0))
        actual_gap = max(np.diff(stations), default=0.)
        # Nearest measured sections can shift a target by half a section gap.
        # Record achieved coverage and reject inputs too sparse for the request.
        if sampling_gap > spacing_m or actual_gap > spacing_m+sampling_gap+1e-9:
            raise ValueError(f"Segment {sid} sections are too sparse for the requested view spacing")
        first = len(views)
        for i in chosen:
            distance, sample_index = tree.query(centers[i])
            if distance > .5:
                raise ValueError(f"Segment {sid} has no nearby inspected mesh sample")
            row = measurements[int(sample_index)]
            if not row["inside"] or row["clearance_m"] is None:
                raise ValueError(f"Segment {sid} view lies outside the accepted cavity")
            origin = np.array(row["point_m"], float)
            origin[2] += (row["roof_distance_m"]-row["floor_distance_m"])/2
            direction = tangents[i].copy()
            norm = np.linalg.norm(direction[:2])
            if norm < 1e-6:
                raise ValueError(f"Segment {sid} has no finite horizontal view direction")
            direction /= norm
            # Opposite directions and vertical bias expose both floor and roof.
            for angle, sign in (("floor_forward", 1), ("roof_backward", -1)):
                look = origin+sign*direction*3
                look[2] += -.6 if sign == 1 else .6
                views.append(dict(index=len(views)+1, segment_id=int(sid),
                    sample_index=int(sample_index), arc_length_m=float(arc_lengths[i]),
                    angle=angle, point_m=origin.tolist(), look_m=look.tolist()))
        coverage.append(dict(segment_id=int(sid), views=len(views)-first, length_m=length,
                             max_station_gap_m=actual_gap, source_sampling_gap_m=sampling_gap))
    if len(views) > max_views:
        raise ValueError(f"Full branch coverage needs {len(views)} views, exceeding budget {max_views}; increase the explicit view budget")
    return dict(schema="plume.native-view-plan.v1", spacing_m=spacing_m, max_views=max_views,
                view_count=len(views), views=views, coverage=coverage,
                scope="Every segment at intervals and near both ends, with opposing floor/roof angles; finite visual coverage, not every surface texel")
