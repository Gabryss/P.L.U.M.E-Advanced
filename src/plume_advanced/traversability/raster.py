"""Bounded vertical triangle rasterization and sampled terrain classification.

Rays are evaluated at cell centres. All intersections are kept, so a chart can
select its own cavity rather than projecting the highest floor onto every layer.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from plume_advanced.progress import report_progress


def vertical_hits(vertices, faces, origin, shape, resolution):
    """Return unique (pixel, Z) ray hits; duplicate triangle edges count once."""
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    hits = []
    # Dynamic batches bound expanded triangle bounding boxes, even for big faces.
    for start in range(0, len(faces), 8192):
        report_progress(
            "Traversability triangle rays",
            start,
            len(faces),
            "all vertical intersections; retain stacked cavities",
        )
        tri = vertices[faces[start : start + 8192]]
        lo = np.maximum(
            np.ceil((tri[:, :, :2].min(axis=1) - origin) / resolution - 0.5).astype(int), 0
        )
        hi = np.minimum(
            np.floor((tri[:, :, :2].max(axis=1) - origin) / resolution - 0.5).astype(int),
            np.array(shape[::-1]) - 1,
        )
        size = np.maximum(hi - lo + 1, 0)
        counts = size.prod(axis=1)
        offsets = np.r_[0, counts.cumsum()]
        for begin in range(0, int(offsets[-1]), 262144):
            flat = np.arange(begin, min(begin + 262144, offsets[-1]))
            ids = np.searchsorted(offsets[1:], flat, side="right")
            local = flat - offsets[ids]
            x = lo[ids, 0] + local % size[ids, 0]
            y = lo[ids, 1] + local // size[ids, 0]
            xy = origin + np.column_stack((x + 0.5, y + 0.5)) * resolution
            a, b, c = tri[ids, 0], tri[ids, 1], tri[ids, 2]
            ab, ac, delta = b - a, c - a, xy - a[:, :2]
            det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
            usable = abs(det) > 1e-16
            u = np.divide(
                delta[:, 0] * ac[:, 1] - delta[:, 1] * ac[:, 0],
                det,
                out=np.zeros(len(ids)),
                where=usable,
            )
            v = np.divide(
                ab[:, 0] * delta[:, 1] - ab[:, 1] * delta[:, 0],
                det,
                out=np.zeros(len(ids)),
                where=usable,
            )
            valid = usable & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            z = a[:, 2] + u * ab[:, 2] + v * ac[:, 2]
            hits.append((y[valid] * shape[1] + x[valid], z[valid]))
    report_progress(
        "Traversability triangle rays", len(faces), len(faces), "intersection raster complete"
    )
    if not hits:
        return np.empty(0, dtype=np.int64), np.empty(0)
    pixels = np.concatenate([p for p, z in hits])
    heights = np.concatenate([z for p, z in hits])
    order = np.lexsort((heights, pixels))
    pixels, heights = pixels[order], heights[order]
    keep = (
        np.r_[True, (np.diff(pixels) != 0) | (np.diff(heights) > 1e-7)]
        if len(pixels)
        else np.empty(0, bool)
    )
    return pixels[keep], heights[keep]


def chart_reference(paths, origin, shape, resolution):
    """Interpolate within each passage only; never bridge separate path records."""
    station_paths = []
    for path in paths:
        path = np.asarray(path, float)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1))]
        unique = np.r_[True, np.diff(arc) > 1e-9]
        path, arc = path[unique], arc[unique]
        along = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1] / resolution)) + 1))
        station_paths.append(np.column_stack([np.interp(along, arc, path[:, i]) for i in range(5)]))
    stations = np.concatenate(station_paths)
    tree = cKDTree(stations[:, :2])
    hint = np.full(np.prod(shape), np.nan)
    for start in range(0, len(hint), 100000):
        ids = np.arange(start, min(start + 100000, len(hint)))
        xy = origin + np.column_stack((ids % shape[1] + 0.5, ids // shape[1] + 0.5)) * resolution
        distance, nearest = tree.query(xy)
        selected = stations[nearest]
        valid = distance <= selected[:, 3] / 2 + 2 * resolution
        hint[ids[valid]] = selected[valid, 2]
    return hint.reshape(shape)


def cavity_fields(pixels, heights, hint):
    """Choose the vertical air interval containing the chart's reference Z."""
    ref = hint.ravel()
    below, above = heights < ref[pixels] - 1e-7, heights > ref[pixels] + 1e-7
    floor, roof = np.full(len(ref), -np.inf), np.full(len(ref), np.inf)
    np.maximum.at(floor, pixels[below], heights[below])
    np.minimum.at(roof, pixels[above], heights[above])
    lower = np.bincount(pixels[below], minlength=len(ref))
    upper = np.bincount(pixels[above], minlength=len(ref))
    on = np.zeros(len(ref), bool)
    on[pixels[abs(heights - ref[pixels]) <= 1e-7]] = True
    valid = (lower % 2 == 1) & (upper % 2 == 1) & ~on & np.isfinite(ref)
    floor[~valid], roof[~valid] = np.nan, np.nan
    return floor.reshape(hint.shape), roof.reshape(hint.shape), on.reshape(hint.shape)


def obstacle_cells(meshes, floor, roof, origin, resolution):
    """Conservative triangle-bbox supercover, including subpixel event props.

    Props are obstacles, not climbable terrain. Z filtering prevents a rock on
    one layer from blocking another. The supercover can overestimate boundaries.
    """
    blocked = np.zeros(floor.shape, bool)
    height = np.zeros(floor.shape, np.float32)
    for number, (vertices, faces) in enumerate(meshes):
        report_progress(
            "Traversability obstacles",
            number,
            len(meshes),
            "project placed props without mixing layers",
        )
        tri = np.asarray(vertices, float)[np.asarray(faces, int)]
        for t in tri:
            low, high = t.min(axis=0), t.max(axis=0)
            lo = np.maximum(np.floor((low[:2] - origin) / resolution).astype(int), 0)
            hi = np.minimum(
                np.floor((high[:2] - origin) / resolution).astype(int),
                np.array(floor.shape[::-1]) - 1,
            )
            if np.any(hi < lo):
                continue
            sl = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            intersects = (
                np.isfinite(floor[sl]) & (high[2] > floor[sl] + 1e-7) & (low[2] < roof[sl] - 1e-7)
            )
            blocked[sl] |= intersects
            height[sl] = np.maximum(height[sl], np.where(intersects, high[2] - floor[sl], 0))
    return blocked, height


def classify(floor, roof, uncertain, obstacles, config):
    """Heading-independent reference footprint, with a detrended step metric."""
    valid = np.isfinite(floor) & np.isfinite(roof)
    r = config.resolution_m
    radius = config.radius_m + np.sqrt(2) * r / 2
    n = int(np.ceil(radius / r))
    y, x = np.mgrid[-n : n + 1, -n : n + 1] * r
    disk = x * x + y * y <= radius * radius + 1e-12
    support = ndimage.minimum_filter(
        valid.astype(np.uint8), footprint=disk, mode="constant", cval=0
    ).astype(bool)
    values = np.where(valid, floor, 0.0)
    mean = ndimage.correlate(values, disk.astype(float) / disk.sum(), mode="constant")
    bx = ndimage.correlate(values, np.where(disk, x, 0) / np.sum(x[disk] ** 2), mode="constant")
    by = ndimage.correlate(values, np.where(disk, y, 0) / np.sum(y[disk] ** 2), mode="constant")
    low, high = np.full(floor.shape, np.inf), np.full(floor.shape, -np.inf)
    padded = np.pad(values, n, mode="constant")
    for iy, ix in np.argwhere(disk):
        residual = (
            padded[iy : iy + floor.shape[0], ix : ix + floor.shape[1]]
            - mean
            - bx * x[iy, ix]
            - by * y[iy, ix]
        )
        np.minimum(low, residual, out=low)
        np.maximum(high, residual, out=high)
    step = high - low
    slope = np.degrees(np.arctan(np.hypot(bx, by)))
    clearance = ndimage.minimum_filter(
        np.where(valid, roof, np.inf), footprint=disk, mode="constant", cval=-np.inf
    ) - ndimage.maximum_filter(
        np.where(valid, floor, -np.inf), footprint=disk, mode="constant", cval=np.inf
    )
    prop = ndimage.maximum_filter(obstacles, footprint=disk, mode="constant", cval=False)
    unknown = ndimage.maximum_filter(uncertain, footprint=disk, mode="constant", cval=False)
    height = config.robot_height_m + 2 * config.margin_m
    passed = (
        support
        & ~prop
        & ~unknown
        & (slope <= config.max_slope_deg + 1e-7)
        & (step <= config.max_step_m + 1e-7)
        & (clearance >= height)
    )
    status = np.zeros(floor.shape, np.uint8)
    status[valid] = 2
    status[passed] = 1
    status[uncertain | (valid & unknown)] = 255
    reason = np.zeros(floor.shape, np.uint8)
    for mask, bit in (
        (valid & ~support, 1),
        (support & (slope > config.max_slope_deg), 2),
        (support & (step > config.max_step_m), 4),
        (support & (clearance < height), 8),
        (valid & prop, 16),
        (status == 255, 32),
    ):
        reason[mask] |= bit
    for field in (step, slope, clearance):
        field[~support] = np.nan
    return dict(
        status=status,
        reason_bits=reason,
        slope_deg=slope.astype(np.float32),
        step_m=step.astype(np.float32),
        body_clearance_m=clearance.astype(np.float32),
        component_id=ndimage.label(passed)[0].astype(np.int32),
    )
