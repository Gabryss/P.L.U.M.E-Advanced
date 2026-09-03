"""Topology-aware 2D floor atlas derived from the generated cave volume.

The intrinsic atlas is the authoritative representation: every traversable
floor cell is addressed by segment id, distance along the segment, and lateral
offset.  World-space plan coordinates are retained for previews and geological
maps, but are not used as a unique key because vertically separated passages
can overlap in a top-down projection.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.stages.network import CaveNetwork
from plume_advanced.stages.section_field import SectionField, SectionSample


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
    surface_slope_degrees: float = 0.0
    geology_class: str = "bare_basalt"
    event_influence: float = 0.0
    sediment_thickness_m: float = 0.0
    debris_density: float = 0.0
    nearest_event_kind: str = ""
    is_chamber: bool = False
    is_terminus: bool = False

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
    generation_stage: str = "base"
    source_cell_count: int = 0
    invalidated_cell_ids: tuple[int, ...] = ()

    def summary(self) -> dict[str, float]:
        if not self.cells:
            return {
                "cell_count": 0.0,
                "grounded_cell_count": 0.0,
                "segment_count": 0.0,
                "mean_clearance_m": 0.0,
                "vertical_overlap_pixel_count": 0.0,
                "invalidated_cell_count": float(len(self.invalidated_cell_ids)),
                "geologically_influenced_cell_count": 0.0,
                "sediment_cell_count": 0.0,
                "breakdown_cell_count": 0.0,
                "debris_cell_count": 0.0,
                "constriction_cell_count": 0.0,
                "bare_basalt_cell_count": 0.0,
                "chamber_cell_count": 0.0,
                "terminus_cell_count": 0.0,
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
            "invalidated_cell_count": float(len(self.invalidated_cell_ids)),
            "geologically_influenced_cell_count": float(
                sum(cell.event_influence > 0.0 for cell in self.cells)
            ),
            "sediment_cell_count": float(
                sum(cell.geology_class == "sediment" for cell in self.cells)
            ),
            "breakdown_cell_count": float(
                sum(cell.geology_class == "breakdown" for cell in self.cells)
            ),
            "debris_cell_count": float(
                sum(cell.geology_class == "debris" for cell in self.cells)
            ),
            "constriction_cell_count": float(
                sum(cell.geology_class == "constriction" for cell in self.cells)
            ),
            "bare_basalt_cell_count": float(
                sum(cell.geology_class == "bare_basalt" for cell in self.cells)
            ),
            "chamber_cell_count": float(sum(cell.is_chamber for cell in self.cells)),
            "terminus_cell_count": float(sum(cell.is_terminus for cell in self.cells)),
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
            generation_stage="base",
            source_cell_count=len(cells),
        )

    def revalidate(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        final_geometry: Any,
        base_atlas: FloorAtlas,
        event_field: Any | None = None,
    ) -> FloorAtlas:
        """Relift stable atlas addresses against the post-event cave volume."""

        voxel_grid = getattr(final_geometry, "voxel_grid", None)
        if voxel_grid is None:
            raise ValueError("Floor-map revalidation requires final voxel geometry")
        sample_lookup = {
            (sample.segment_id, sample.index): sample
            for segment_field in section_field.segment_fields
            for sample in segment_field.samples
        }
        band_offsets = dict(base_atlas.segment_band_offsets_m)
        applied_structural_ids = set(
            getattr(final_geometry, "structural_event_ids", ())
        )
        node_degree: dict[int, int] = {}
        for segment in cave_network.segments:
            node_degree[segment.start_node_id] = node_degree.get(segment.start_node_id, 0) + 1
            node_degree[segment.end_node_id] = node_degree.get(segment.end_node_id, 0) + 1
        terminal_ranges: dict[int, tuple[float, ...]] = {}
        for segment in cave_network.segments:
            distances: list[float] = []
            if node_degree.get(segment.start_node_id, 0) == 1:
                distances.append(0.0)
            if node_degree.get(segment.end_node_id, 0) == 1:
                distances.append(segment.total_length)
            terminal_ranges[segment.segment_id] = tuple(distances)

        cells: list[FloorCell] = []
        invalidated: list[int] = []
        for source_cell in base_atlas.cells:
            sample = sample_lookup.get(
                (source_cell.segment_id, source_cell.sample_index)
            )
            if sample is None:
                invalidated.append(source_cell.cell_id)
                continue
            lifted = self._lift_cell(
                cell_id=source_cell.cell_id,
                sample=sample,
                lateral_offset=source_cell.lateral_offset_m,
                z_level=source_cell.z_level,
                atlas_band_offset=band_offsets.get(source_cell.segment_id, 0.0),
                voxel_grid=voxel_grid,
            )
            if lifted is None:
                invalidated.append(source_cell.cell_id)
                continue
            is_terminus = any(
                abs(lifted.distance_along_m - terminal_distance)
                <= max(self.config.lateral_spacing_m * 2.0, 5.0)
                for terminal_distance in terminal_ranges.get(lifted.segment_id, ())
            )
            cells.append(
                self._classify_cell(
                    lifted,
                    event_field=event_field,
                    applied_structural_ids=applied_structural_ids,
                    is_chamber=any(
                        influence.kind == "chamber"
                        and influence.weight >= 0.75
                        for influence in sample.junction_influences
                    ),
                    is_terminus=is_terminus,
                )
            )
        cells = self._ensure_structural_event_coverage(
            cells,
            event_field=event_field,
            applied_structural_ids=applied_structural_ids,
        )
        return FloorAtlas(
            config=self.config,
            cells=tuple(cells),
            segment_band_offsets_m=base_atlas.segment_band_offsets_m,
            generation_stage="final",
            source_cell_count=len(base_atlas.cells),
            invalidated_cell_ids=tuple(invalidated),
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
        if voxel_grid.sample_density(origin) < voxel_grid.iso_level:
            return None
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
            surface_slope_degrees=float(
                math.degrees(
                    math.acos(float(np.clip(contact_normal[2], -1.0, 1.0)))
                )
            ),
        )

    @staticmethod
    def _classify_cell(
        cell: FloorCell,
        *,
        event_field: Any | None,
        applied_structural_ids: set[int],
        is_chamber: bool,
        is_terminus: bool,
    ) -> FloorCell:
        events = tuple(getattr(event_field, "events", ()))
        nearest_kind = ""
        strongest_influence = 0.0
        sediment = 0.0
        debris = 0.0
        class_scores = {
            "sediment": 0.0,
            "breakdown": 0.0,
            "debris": 0.0,
            "constriction": 0.0,
        }
        position = np.asarray(cell.position, dtype=float)
        for event in events:
            # Collapse talus remains a real floor deposit even if the optional
            # volume cut is rejected to preserve cave connectivity.  Choke and
            # infill classifications, by contrast, require an applied volume
            # modifier.
            if (
                event.kind in {"choke", "infill"}
                and event.event_id not in applied_structural_ids
            ):
                continue
            event_position = np.asarray(event.position, dtype=float)
            distance = float(np.linalg.norm(position - event_position))
            influence_radius = max(2.5 * event.max_radius, 1.0)
            influence = max(0.0, 1.0 - distance / influence_radius)
            if influence <= 0.0:
                continue
            if influence > strongest_influence:
                strongest_influence = influence
                nearest_kind = event.kind
            if event.kind == "infill":
                class_scores["sediment"] = max(
                    class_scores["sediment"], influence
                )
                sediment = max(sediment, influence * event.radius_z)
            elif event.kind == "collapse":
                class_scores["breakdown"] = max(
                    class_scores["breakdown"], influence
                )
                debris = max(debris, 0.65 * influence)
            elif event.kind in {"rock", "boulder"}:
                class_scores["debris"] = max(class_scores["debris"], influence)
                debris = max(
                    debris,
                    influence * (0.65 if event.kind == "rock" else 1.0),
                )
            elif event.kind == "choke":
                class_scores["constriction"] = max(
                    class_scores["constriction"], influence
                )
        structural_class = max(
            ("sediment", "breakdown", "constriction"),
            key=lambda name: class_scores[name],
        )
        if class_scores[structural_class] > 0.0:
            geology_class = structural_class
        elif class_scores["debris"] > 0.0:
            geology_class = "debris"
        else:
            geology_class = "bare_basalt"
        return replace(
            cell,
            geology_class=geology_class,
            event_influence=float(strongest_influence),
            sediment_thickness_m=float(sediment),
            debris_density=float(np.clip(debris, 0.0, 1.0)),
            nearest_event_kind=nearest_kind,
            is_chamber=is_chamber,
            is_terminus=is_terminus,
        )

    @staticmethod
    def _ensure_structural_event_coverage(
        cells: list[FloorCell],
        *,
        event_field: Any | None,
        applied_structural_ids: set[int],
    ) -> list[FloorCell]:
        """Project structural evidence onto the surviving floor.

        Structural carving can remove every atlas sample inside an event's
        ordinary influence radius.  In that case the nearest surviving floor
        cell on the same segment still represents the accessible margin of the
        feature. Collapse talus is retained even when its volume cut was
        rejected by the connectivity safeguard.
        """

        if not cells:
            return cells
        kind_to_class = {
            "collapse": "breakdown",
            "choke": "constriction",
            "infill": "sediment",
        }
        events = tuple(
            event
            for event in getattr(event_field, "events", ())
            if event.kind in kind_to_class
            and (
                event.kind == "collapse"
                or event.event_id in applied_structural_ids
            )
        )
        claimed_cell_ids: set[int] = set()
        for event_kind, geology_class in kind_to_class.items():
            if any(cell.geology_class == geology_class for cell in cells):
                continue
            matching_events = tuple(event for event in events if event.kind == event_kind)
            if not matching_events:
                continue
            best: tuple[float, int, Any] | None = None
            for event in matching_events:
                same_segment = tuple(
                    (index, cell)
                    for index, cell in enumerate(cells)
                    if cell.segment_id == event.segment_id
                    and cell.cell_id not in claimed_cell_ids
                )
                candidates = same_segment or tuple(
                    (index, cell)
                    for index, cell in enumerate(cells)
                    if cell.cell_id not in claimed_cell_ids
                )
                event_position = np.asarray(event.position, dtype=float)
                for index, cell in candidates:
                    distance = float(
                        np.linalg.norm(np.asarray(cell.position) - event_position)
                    )
                    candidate = (distance, index, event)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
            if best is None:
                continue
            distance, index, event = best
            cell = cells[index]
            influence = max(
                0.05,
                float(event.max_radius) / max(float(event.max_radius) + distance, 1e-6),
            )
            replacements: dict[str, Any] = {
                "geology_class": geology_class,
                "event_influence": max(cell.event_influence, influence),
                "nearest_event_kind": event_kind,
            }
            if event_kind == "collapse":
                replacements["debris_density"] = max(
                    cell.debris_density, 0.65 * influence
                )
            elif event_kind == "infill":
                replacements["sediment_thickness_m"] = max(
                    cell.sediment_thickness_m, influence * event.radius_z
                )
            cells[index] = replace(cell, **replacements)
            claimed_cell_ids.add(cell.cell_id)
        return cells

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
        surface_slope_degrees=np.asarray(
            [cell.surface_slope_degrees for cell in cells]
        ),
        geology_class=np.asarray([cell.geology_class for cell in cells], dtype="U16"),
        event_influence=np.asarray([cell.event_influence for cell in cells]),
        sediment_thickness_m=np.asarray(
            [cell.sediment_thickness_m for cell in cells]
        ),
        debris_density=np.asarray([cell.debris_density for cell in cells]),
        nearest_event_kind=np.asarray(
            [cell.nearest_event_kind for cell in cells], dtype="U16"
        ),
        is_chamber=np.asarray([cell.is_chamber for cell in cells], dtype=bool),
        is_terminus=np.asarray([cell.is_terminus for cell in cells], dtype=bool),
        plan_occupancy=plan["occupancy"],
        plan_floor_z_min_m=plan["floor_z_min"],
        plan_floor_z_max_m=plan["floor_z_max"],
        plan_clearance_max_m=plan["clearance_max"],
        plan_level_count=plan["level_count"],
        plan_geology_class=plan["geology_class"],
        plan_event_influence=plan["event_influence"],
        plan_debris_density=plan["debris_density"],
        plan_origin_xy_m=plan["origin_xy"],
        plan_resolution_m=np.asarray(floor_atlas.config.plan_resolution_m),
    )
    metadata = {
        "schema": "plume.floor-atlas.v2",
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
        "generation_stage": floor_atlas.generation_stage,
        "source_cell_count": floor_atlas.source_cell_count,
        "invalidated_cell_ids": list(floor_atlas.invalidated_cell_ids),
        "geology_class_codes": {
            "bare_basalt": 0,
            "sediment": 1,
            "breakdown": 2,
            "debris": 3,
            "constriction": 4,
        },
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
            "geology_class": np.zeros((0, 0), dtype=np.uint8),
            "event_influence": empty.copy(),
            "debris_density": empty.copy(),
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
    geology_class = np.zeros(shape, dtype=np.uint8)
    event_influence = np.zeros(shape, dtype=np.float32)
    debris_density = np.zeros(shape, dtype=np.float32)
    geology_codes = {
        "bare_basalt": 0,
        "sediment": 1,
        "breakdown": 2,
        "debris": 3,
        "constriction": 4,
    }
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
        if cell.event_influence >= float(event_influence[key]):
            event_influence[key] = cell.event_influence
            geology_class[key] = geology_codes.get(cell.geology_class, 0)
        debris_density[key] = max(float(debris_density[key]), cell.debris_density)
    level_count = np.zeros(shape, dtype=np.int16)
    for key, values in levels.items():
        level_count[key] = len(values)
    return {
        "occupancy": occupancy,
        "floor_z_min": floor_z_min,
        "floor_z_max": floor_z_max,
        "clearance_max": clearance_max,
        "level_count": level_count,
        "geology_class": geology_class,
        "event_influence": event_influence,
        "debris_density": debris_density,
        "origin_xy": minimum.astype(float),
    }
