"""Topology-aware 2D floor atlas derived from the generated cave volume.

The intrinsic atlas is the authoritative representation: every traversable
floor cell is addressed by segment id, distance along the segment, and lateral
offset.  World-space plan coordinates are retained for previews and robotics
maps, but are not used as a unique key because vertically separated passages
can overlap in a top-down projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from stages.network import CaveNetwork
from stages.section_field import SectionField, SectionSample


@dataclass(frozen=True)
class FloorMapConfig:
    """Controls floor sampling and conventional plan-map rasterization."""

    lateral_spacing_m: float = 2.5
    maximum_lateral_fraction: float = 0.72
    plan_resolution_m: float = 2.0
    minimum_clearance_m: float = 1.0


@dataclass(frozen=True)
class FloorCell:
    """One liftable cell in the intrinsic cave-floor atlas."""

    cell_id: int
    segment_id: int
    sample_index: int
    z_level: int
    distance_along_m: float
    lateral_offset_m: float
    atlas_x_m: float
    atlas_y_m: float
    x: float
    y: float
    z: float
    normal_x: float
    normal_y: float
    normal_z: float
    clearance_m: float
    tube_width_m: float
    grounded: bool

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def normal(self) -> tuple[float, float, float]:
        return (self.normal_x, self.normal_y, self.normal_z)


@dataclass(frozen=True)
class FloorAtlas:
    """Machine-readable cave floor with intrinsic and world coordinates."""

    config: FloorMapConfig
    cells: tuple[FloorCell, ...]
    segment_band_offsets_m: tuple[tuple[int, float], ...]

    def summary(self) -> dict[str, float]:
        if not self.cells:
            return {
                "cell_count": 0.0,
                "grounded_cell_count": 0.0,
                "segment_count": 0.0,
                "mean_clearance_m": 0.0,
                "overlap_cell_count": 0.0,
            }
        pixels: dict[tuple[int, int], set[int]] = {}
        resolution = max(self.config.plan_resolution_m, 1e-6)
        for cell in self.cells:
            key = (int(round(cell.x / resolution)), int(round(cell.y / resolution)))
            pixels.setdefault(key, set()).add(cell.z_level)
        return {
            "cell_count": float(len(self.cells)),
            "grounded_cell_count": float(sum(cell.grounded for cell in self.cells)),
            "segment_count": float(len({cell.segment_id for cell in self.cells})),
            "mean_clearance_m": float(
                np.mean([cell.clearance_m for cell in self.cells])
            ),
            "vertical_overlap_pixel_count": float(
                sum(len(levels) > 1 for levels in pixels.values())
            ),
        }

    def sample_lookup(self) -> dict[tuple[int, int], tuple[FloorCell, ...]]:
        grouped: dict[tuple[int, int], list[FloorCell]] = {}
        for cell in self.cells:
            grouped.setdefault((cell.segment_id, cell.sample_index), []).append(cell)
        return {
            key: tuple(sorted(values, key=lambda cell: cell.lateral_offset_m))
            for key, values in grouped.items()
        }


class FloorMapGenerator:
    """Raycast a topology-aware floor atlas from the Stage-D base volume."""

    def __init__(self, config: FloorMapConfig | None = None) -> None:
        self.config = config or FloorMapConfig()

    def generate(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        base_geometry: Any,
    ) -> FloorAtlas:
        voxel_grid = getattr(base_geometry, "voxel_grid", None)
        if voxel_grid is None:
            raise ValueError("Floor-map generation requires base geometry with a voxel grid")

        level_lookup = {
            segment.segment_id: segment.z_level for segment in cave_network.segments
        }
        band_offsets = self._segment_band_offsets(section_field)
        cells: list[FloorCell] = []
        for segment_field in section_field.segment_fields:
            band_offset = band_offsets.get(segment_field.segment_id, 0.0)
            z_level = level_lookup.get(segment_field.segment_id, 0)
            for sample in segment_field.samples:
                for lateral in self._lateral_offsets(sample):
                    cell = self._lift_cell(
                        cell_id=len(cells),
                        sample=sample,
                        lateral_offset=lateral,
                        z_level=z_level,
                        atlas_band_offset=band_offset,
                        voxel_grid=voxel_grid,
                    )
                    if cell is not None:
                        cells.append(cell)
        return FloorAtlas(
            config=self.config,
            cells=tuple(cells),
            segment_band_offsets_m=tuple(sorted(band_offsets.items())),
        )

    def _lateral_offsets(self, sample: SectionSample) -> tuple[float, ...]:
        half_span = (
            0.5
            * sample.tube_width
            * float(np.clip(self.config.maximum_lateral_fraction, 0.0, 0.98))
        )
        spacing = max(self.config.lateral_spacing_m, 0.25)
        lane_count = max(1, int(math.floor((2.0 * half_span) / spacing)) + 1)
        if lane_count == 1:
            return (0.0,)
        return tuple(float(value) for value in np.linspace(-half_span, half_span, lane_count))

    def _lift_cell(
        self,
        *,
        cell_id: int,
        sample: SectionSample,
        lateral_offset: float,
        z_level: int,
        atlas_band_offset: float,
        voxel_grid: Any,
    ) -> FloorCell | None:
        center = np.asarray((sample.x, sample.y, sample.z), dtype=float)
        normal = self._unit(sample.normal, fallback=(1.0, 0.0, 0.0))
        vertical = self._unit(sample.binormal, fallback=(0.0, 0.0, 1.0))
        origin = center + normal * lateral_offset
        max_distance = max(
            sample.tube_height * 1.4,
            float(voxel_grid.voxel_size) * 4.0,
        )
        floor_hit = voxel_grid.raycast_isosurface(origin, -vertical, max_distance)
        roof_hit = voxel_grid.raycast_isosurface(origin, vertical, max_distance)
        if floor_hit is None:
            return None

        contact = np.asarray(floor_hit.position, dtype=float)
        contact_normal = self._unit(floor_hit.normal, fallback=tuple(vertical))
        if float(np.dot(contact_normal, vertical)) < 0.0:
            contact_normal = -contact_normal
        clearance = (
            float(floor_hit.distance + roof_hit.distance)
            if roof_hit is not None
            else float(sample.tube_height)
        )
        if clearance < self.config.minimum_clearance_m:
            return None

        return FloorCell(
            cell_id=cell_id,
            segment_id=sample.segment_id,
            sample_index=sample.index,
            z_level=z_level,
            distance_along_m=float(sample.segment_arc_length),
            lateral_offset_m=float(lateral_offset),
            atlas_x_m=float(sample.segment_arc_length),
            atlas_y_m=float(atlas_band_offset + lateral_offset),
            x=float(contact[0]),
            y=float(contact[1]),
            z=float(contact[2]),
            normal_x=float(contact_normal[0]),
            normal_y=float(contact_normal[1]),
            normal_z=float(contact_normal[2]),
            clearance_m=clearance,
            tube_width_m=float(sample.tube_width),
            grounded=True,
        )

    def _segment_band_offsets(self, section_field: SectionField) -> dict[int, float]:
        offsets: dict[int, float] = {}
        cursor = 0.0
        gap = max(4.0 * self.config.lateral_spacing_m, 5.0)
        for segment_field in sorted(
            section_field.segment_fields, key=lambda value: value.segment_id
        ):
            maximum_width = max(
                (sample.tube_width for sample in segment_field.samples),
                default=gap,
            )
            offsets[segment_field.segment_id] = cursor + 0.5 * maximum_width
            cursor += maximum_width + gap
        return offsets

    @staticmethod
    def _unit(
        values: tuple[float, float, float],
        *,
        fallback: tuple[float, float, float],
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            return np.asarray(fallback, dtype=float)
        return vector / length


def export_floor_atlas(
    floor_atlas: FloorAtlas,
    output_base: str | Path,
) -> tuple[Path, Path]:
    """Write compact NPZ arrays plus a descriptive JSON sidecar."""

    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    cells = floor_atlas.cells
    plan = _build_plan_raster(floor_atlas)
    np.savez_compressed(
        npz_path,
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int64),
        segment_id=np.asarray([cell.segment_id for cell in cells], dtype=np.int64),
        sample_index=np.asarray([cell.sample_index for cell in cells], dtype=np.int64),
        z_level=np.asarray([cell.z_level for cell in cells], dtype=np.int32),
        distance_along_m=np.asarray([cell.distance_along_m for cell in cells]),
        lateral_offset_m=np.asarray([cell.lateral_offset_m for cell in cells]),
        atlas_xy_m=np.asarray(
            [(cell.atlas_x_m, cell.atlas_y_m) for cell in cells], dtype=float
        ).reshape((-1, 2)),
        world_xyz_m=np.asarray(
            [cell.position for cell in cells], dtype=float
        ).reshape((-1, 3)),
        normal_xyz=np.asarray(
            [cell.normal for cell in cells], dtype=float
        ).reshape((-1, 3)),
        clearance_m=np.asarray([cell.clearance_m for cell in cells]),
        tube_width_m=np.asarray([cell.tube_width_m for cell in cells]),
        grounded=np.asarray([cell.grounded for cell in cells], dtype=bool),
        plan_occupancy=plan["occupancy"],
        plan_floor_z_min_m=plan["floor_z_min"],
        plan_floor_z_max_m=plan["floor_z_max"],
        plan_clearance_max_m=plan["clearance_max"],
        plan_level_count=plan["level_count"],
        plan_origin_xy_m=plan["origin_xy"],
        plan_resolution_m=np.asarray(floor_atlas.config.plan_resolution_m),
    )
    metadata = {
        "schema": "plume.floor-atlas.v1",
        "coordinate_system": {
            "world": "right-handed, metres, Z-up",
            "intrinsic": [
                "segment_id",
                "distance_along_m",
                "lateral_offset_m",
            ],
            "note": (
                "World XY is a preview projection and is not unique where "
                "passages overlap vertically."
            ),
        },
        "config": asdict(floor_atlas.config),
        "summary": floor_atlas.summary(),
        "segment_band_offsets_m": dict(floor_atlas.segment_band_offsets_m),
        "npz_file": npz_path.name,
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, json_path


def _build_plan_raster(floor_atlas: FloorAtlas) -> dict[str, np.ndarray]:
    """Rasterize a conventional XY map while preserving overlap diagnostics."""

    cells = floor_atlas.cells
    resolution = max(floor_atlas.config.plan_resolution_m, 1e-6)
    if not cells:
        empty = np.zeros((0, 0), dtype=float)
        return {
            "occupancy": np.zeros((0, 0), dtype=bool),
            "floor_z_min": empty,
            "floor_z_max": empty.copy(),
            "clearance_max": empty.copy(),
            "level_count": np.zeros((0, 0), dtype=np.int16),
            "origin_xy": np.zeros(2, dtype=float),
        }

    minimum = np.floor(
        np.min(np.asarray([(cell.x, cell.y) for cell in cells]), axis=0)
        / resolution
    ) * resolution
    maximum = np.ceil(
        np.max(np.asarray([(cell.x, cell.y) for cell in cells]), axis=0)
        / resolution
    ) * resolution
    width, height = (
        np.maximum(np.round((maximum - minimum) / resolution).astype(int) + 1, 1)
    )
    shape = (int(height), int(width))
    occupancy = np.zeros(shape, dtype=bool)
    floor_z_min = np.full(shape, np.nan, dtype=np.float32)
    floor_z_max = np.full(shape, np.nan, dtype=np.float32)
    clearance_max = np.full(shape, np.nan, dtype=np.float32)
    levels: dict[tuple[int, int], set[int]] = {}
    for cell in cells:
        x_index = int(round((cell.x - minimum[0]) / resolution))
        y_index = int(round((cell.y - minimum[1]) / resolution))
        key = (y_index, x_index)
        occupancy[key] = True
        floor_z_min[key] = (
            cell.z
            if np.isnan(floor_z_min[key])
            else min(float(floor_z_min[key]), cell.z)
        )
        floor_z_max[key] = (
            cell.z
            if np.isnan(floor_z_max[key])
            else max(float(floor_z_max[key]), cell.z)
        )
        clearance_max[key] = (
            cell.clearance_m
            if np.isnan(clearance_max[key])
            else max(float(clearance_max[key]), cell.clearance_m)
        )
        levels.setdefault(key, set()).add(cell.z_level)
    level_count = np.zeros(shape, dtype=np.int16)
    for key, values in levels.items():
        level_count[key] = len(values)
    return {
        "occupancy": occupancy,
        "floor_z_min": floor_z_min,
        "floor_z_max": floor_z_max,
        "clearance_max": clearance_max,
        "level_count": level_count,
        "origin_xy": minimum.astype(float),
    }
