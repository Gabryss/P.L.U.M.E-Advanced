"""Continuous, metre-scale accretion on the assembled cave boundary.

This is a morphological approximation, not a cooling or fluid simulation.
Adding solid material inside the existing void cannot enlarge its roof span.
World-space fields are evaluated after unions, so branches and tile boundaries
do not restart the pattern or stamp a ring at every section sample.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import ndimage

from plume_advanced.stages.geometry_types import GeometryConfig, TiledVoxelGrid, VoxelGrid


def _hash(points: np.ndarray, seed: int) -> np.ndarray:
    """Stateless lattice hash, including negative coordinates."""
    p = points.astype(np.int64).astype(np.uint64)
    h = p[..., 0]*np.uint64(0x9E3779B185EBCA87)
    h ^= p[..., 1]*np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= p[..., 2]*np.uint64(0x165667B19E3779F9)
    h ^= np.uint64(seed % (1 << 64))
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(float) / float(1 << 53)


def value_noise(points: np.ndarray, seed: int) -> np.ndarray:
    """Bounded C2 value noise; no repeating sinusoidal bands."""
    cell = np.floor(points).astype(np.int64)
    t = points-cell
    w = t*t*t*(t*(t*6.-15.)+10.)
    result = np.zeros(len(points))
    for corner in np.ndindex(2, 2, 2):
        weight = np.prod(np.where(np.asarray(corner), w, 1.-w), axis=1)
        result += weight*_hash(cell+corner, seed)
    return result


def relief_depth(points: np.ndarray, upward: np.ndarray, config: GeometryConfig) -> np.ndarray:
    """Return bounded inward offsets with distinct roof, wall and floor forms."""
    seed = config.random_seed or 0
    # Do not introduce an octave narrower than four grid intervals.
    scale = max(config.surface_feature_scale_m, 4.*config.voxel_size)
    p = points/scale
    # Rotate noise coordinates so its lattice does not align with voxel axes.
    q = p @ np.array([[.36, .48, -.8], [-.8, .6, 0.], [.48, .64, .6]])
    zone = value_noise(q/9., seed+101)
    zone = np.clip((zone-.25)/.5, 0., 1.)
    zone = .15+.85*zone*zone*(3.-2.*zone)
    broad = value_noise(q/3.4, seed+211)
    warp = value_noise(q/2.2, seed+307)-.5
    small = value_noise(q + np.column_stack((warp, .6*warp, .4*warp)), seed+401)

    # Broken, near-horizontal accreted layers. Slow horizontal warping and
    # a separate patch mask keep these from becoming continuous level shelves.
    layered = p*np.array([.15, .15, 1.3])
    layered[:, 2] += .65*warp
    ledge = np.clip((value_noise(layered, seed+503)-.45)*3., 0., 1.)
    ledge *= np.clip((value_noise(p/4.7, seed+601)-.25)*2., 0., 1.)
    wall = config.surface_wall_relief_m*zone*(.35*broad + .30*small + .35*ledge)

    # Uneven lobate floor crust with localized ridges; smoother patches survive.
    floor_ridge = (1.-np.abs(2.*small-1.))**4
    floor = config.surface_floor_relief_m*zone*(.4*broad + .6*floor_ridge)

    # Scattered thick, tapering roof pendants. Their roots are attached to the
    # continuous roof; no separate rock objects or periodic ceiling ridges.
    xy = p.copy()
    xy[:, 2] = 0.
    cells = np.floor(xy/2.4).astype(np.int64)
    drips = np.zeros(len(p))
    for dx, dy in np.ndindex(3, 3):
        cell = cells+np.array([dx-1, dy-1, 0])
        r = _hash(cell, seed+701)
        center_x = (cell[:, 0]+.15+.7*r)*2.4
        center_y = (cell[:, 1]+.15+.7*_hash(cell, seed+709))*2.4
        radius = .5+.35*_hash(cell, seed+719)
        distance = np.hypot(p[:, 0]-center_x, p[:, 1]-center_y)/radius
        taper = np.maximum(1.-distance, 0.)**1.4
        drips = np.maximum(drips, taper*(.5+.5*r)*(r > .30))
    roof = config.surface_roof_relief_m*(.35+.65*zone)*(.25*broad + .15*small + .6*drips)

    roof_weight = np.clip(upward, 0., 1.)**2
    floor_weight = np.clip(-upward, 0., 1.)**2
    depth = roof*roof_weight + floor*floor_weight + wall*(1.-roof_weight-floor_weight)
    crust_scale = max(.4*config.surface_feature_scale_m, 4.*config.voxel_size)
    grain = value_noise(q*(scale/crust_scale), seed+811)
    # Ridged patches provide tighter folds/creases beneath the broad lobes.
    # Attenuate an unresolved octave rather than enlarging it on coarse grids.
    resolved_gain = min(1., .4*config.surface_feature_scale_m/crust_scale)
    crust = (1.-np.abs(2.*grain-1.))**3
    depth += config.surface_crust_relief_m*resolved_gain*zone*crust*(1.-.7*floor_weight)
    return depth


def apply_surface_relief(
    grid: VoxelGrid | TiledVoxelGrid,
    config: GeometryConfig,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Modify density once, using immutable neighbors for seam agreement."""
    maximum = max(config.surface_wall_relief_m, config.surface_roof_relief_m,
                  config.surface_floor_relief_m) + config.surface_crust_relief_m
    if maximum <= 0.:
        return
    size = grid.voxel_size
    # Nearby peak distance limits accretion in shallow passages. This is a
    # local clearance safeguard, not a proof of route traversability.
    reach = max(1, math.ceil(.75/size))
    halo = max(reach, 4)

    def modify(density: np.ndarray, start: np.ndarray, core: tuple[slice, ...]) -> np.ndarray:
        source = density[core]
        active = (source >= grid.iso_level-2.) & (source <= grid.iso_level+maximum/size+2.)
        indices = np.column_stack(np.nonzero(active))
        if not len(indices):
            return source.copy()
        smoothed = ndimage.gaussian_filter(density, .65, truncate=3., mode="nearest")
        gradients = np.stack([np.gradient(smoothed, axis=a)[core][active] for a in range(3)], axis=1)
        upward = -gradients[:, 2]/np.maximum(np.linalg.norm(gradients, axis=1), 1e-9)
        points = np.asarray(grid.origin)+(start+indices)*size
        offset = np.empty(len(points))
        for begin in range(0, len(points), 65536):
            end = begin+65536
            offset[begin:end] = relief_depth(points[begin:end], upward[begin:end], config)
        peak = ndimage.maximum_filter(density-grid.iso_level, size=2*reach+1, mode="nearest")[core][active]
        offset = np.minimum(offset, .4*size*np.maximum(peak, 0.))
        result = source.copy()
        result[active] -= (offset/size).astype(result.dtype)
        return result

    if isinstance(grid, VoxelGrid):
        grid.density[...] = modify(grid.density, np.zeros(3, dtype=int), (slice(None),)*3)
        return
    grid.synchronize_halos()
    # Keep the source immutable until every tile has been evaluated.
    result_tiles = {}
    for index, (key, tile) in enumerate(sorted(grid.tiles.items()), 1):
        start = np.asarray(key)*grid.tile_size
        padded_start = start-halo
        shape = np.asarray(tile.shape)+2*halo
        padded = np.full(tuple(shape), grid.iso_level-8., dtype=np.float32)
        neighbor_reach = math.ceil(halo/grid.tile_size)
        for delta in np.ndindex(*((2*neighbor_reach+1,)*3)):
            other_key = (key[0]+delta[0]-neighbor_reach, key[1]+delta[1]-neighbor_reach,
                         key[2]+delta[2]-neighbor_reach)
            other = grid.tiles.get(other_key)
            if other is None:
                continue
            other_start = np.asarray(other_key)*grid.tile_size
            low = np.maximum(padded_start, other_start)
            high = np.minimum(padded_start+shape, other_start+other.shape)
            if np.any(high <= low):
                continue
            dest = tuple(slice(int(a), int(b)) for a, b in zip(low-padded_start, high-padded_start))
            src = tuple(slice(int(a), int(b)) for a, b in zip(low-other_start, high-other_start))
            padded[dest] = other[src]
        core = tuple(slice(halo, halo+n) for n in tile.shape)
        result_tiles[key] = modify(padded, start, core)
        if progress:
            progress("surface-relief", index, len(grid.tiles), "shaping wall, roof and floor accretion")
    grid.tiles = result_tiles
    grid.synchronize_halos()
