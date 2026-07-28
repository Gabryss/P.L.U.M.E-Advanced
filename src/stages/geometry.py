"""Stage D voxel stamping and marching-cubes isosurface generation."""

from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from skimage import measure
from scipy import ndimage

from stages.events import GeologicalEvent, GeologicalEventField
from stages.geometry_types import (
    CaveGeometry,
    GeometryChunkMesh,
    GeometryConfig,
    TiledVoxelGrid,
    VoxelGrid,
)
from stages.network import CaveNetwork
from stages.section_field import SectionField, SectionSample

GeometryProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class _JunctionStamp:
    center: np.ndarray
    radius_long: float
    radius_short: float
    radius_z: float
    angle: float
    phase: tuple[float, float, float]
    kind: str


class GeometryGenerator:
    """Build cave geometry by stamping a density grid and polygonizing it."""

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._roughness_phase = tuple(float(value) for value in self._rng.uniform(0.0, 2.0 * math.pi, size=3))

    def generate(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        event_field: GeologicalEventField | None = None,
        progress: GeometryProgressCallback | None = None,
    ) -> CaveGeometry:
        """Compatibility one-pass API built from the two-pass workflow."""

        base_geometry = self.build_base_volume(
            cave_network,
            section_field,
            progress=progress,
        )
        return self.finalize(
            base_geometry,
            event_field,
            progress=progress,
        )

    def build_base_volume(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        progress: GeometryProgressCallback | None = None,
    ) -> CaveGeometry:
        """Build the cave density field without polygonizing it."""

        self._emit_progress(progress, "prepare", 0, 1, "collecting section samples")
        samples_by_segment = {
            segment_field.segment_id: segment_field.samples
            for segment_field in section_field.segment_fields
            if segment_field.samples
        }
        stamp_samples = [
            sample
            for samples in samples_by_segment.values()
            for sample in samples
        ]
        self._emit_progress(
            progress,
            "prepare",
            1,
            1,
            f"collected {len(stamp_samples)} samples from {len(samples_by_segment)} segments",
        )
        if not stamp_samples:
            voxel_grid = VoxelGrid(
                origin=(0.0, 0.0, 0.0),
                voxel_size=self.config.voxel_size,
                density=np.full((2, 2, 2), -1.0, dtype=np.float32),
                iso_level=self.config.iso_level,
            )
            return CaveGeometry(
                config=self.config,
                voxel_grid=voxel_grid,
                chunk_meshes=(),
                assembled_vertices=(),
                assembled_faces=(),
                component_count=0,
                stamped_sample_count=0,
                stamped_segment_ids=(),
            )

        voxel_grid = self._build_voxel_grid(
            samples_by_segment,
            cave_network,
            progress,
        )
        return CaveGeometry(
            config=self.config,
            voxel_grid=voxel_grid,
            chunk_meshes=(),
            assembled_vertices=(),
            assembled_faces=(),
            component_count=0,
            stamped_sample_count=len(stamp_samples),
            stamped_segment_ids=tuple(sorted(samples_by_segment)),
            minimum_section_width_m=min(
                (
                    float(sample.tube_width)
                    for sample in stamp_samples
                    if sample.tube_width > 0.0
                ),
                default=0.0,
            ),
            protected_route_points=tuple(
                (float(sample.x), float(sample.y), float(sample.z))
                for segment_id in section_field.dominant_route_segment_ids
                for sample in samples_by_segment.get(segment_id, ())
            ),
        )

    def finalize(
        self,
        base_geometry: CaveGeometry,
        event_field: GeologicalEventField | None = None,
        progress: GeometryProgressCallback | None = None,
    ) -> CaveGeometry:
        """Apply structural modifiers, polygonize, and attach prop meshes."""

        voxel_grid = base_geometry.voxel_grid
        structural_event_ids: tuple[int, ...] = ()
        event_meshes = ()
        if event_field is not None:
            voxel_grid, structural_event_ids = self._apply_structural_events_to_grid(
                voxel_grid,
                event_field,
                progress,
                protected_points=base_geometry.protected_route_points,
            )
            event_meshes = event_field.meshes

        if (
            not structural_event_ids
            and base_geometry.chunk_meshes
            and base_geometry.assembled_faces
        ):
            return CaveGeometry(
                config=base_geometry.config,
                voxel_grid=base_geometry.voxel_grid,
                chunk_meshes=base_geometry.chunk_meshes,
                assembled_vertices=base_geometry.assembled_vertices,
                assembled_faces=base_geometry.assembled_faces,
                component_count=base_geometry.component_count,
                stamped_sample_count=base_geometry.stamped_sample_count,
                stamped_segment_ids=base_geometry.stamped_segment_ids,
                minimum_section_width_m=base_geometry.minimum_section_width_m,
                protected_route_points=base_geometry.protected_route_points,
                event_meshes=event_meshes,
                structural_event_ids=(),
            )

        chunk_meshes = self._march_chunks(voxel_grid, progress)
        self._emit_progress(
            progress,
            "assemble",
            0,
            2,
            f"welding {sum(mesh.vertex_count for mesh in chunk_meshes)} chunk vertices",
        )
        assembled_vertices, assembled_faces = self._assemble_chunks(chunk_meshes)
        self._emit_progress(
            progress,
            "assemble",
            1,
            2,
            f"assembled {len(assembled_vertices)} vertices and {len(assembled_faces)} faces",
        )
        component_count = self._count_components(assembled_faces)
        self._emit_progress(
            progress,
            "assemble",
            2,
            2,
            f"mesh components: {component_count}",
        )
        return CaveGeometry(
            config=self.config,
            voxel_grid=voxel_grid,
            chunk_meshes=tuple(chunk_meshes),
            assembled_vertices=assembled_vertices,
            assembled_faces=assembled_faces,
            component_count=component_count,
            stamped_sample_count=base_geometry.stamped_sample_count,
            stamped_segment_ids=base_geometry.stamped_segment_ids,
            minimum_section_width_m=base_geometry.minimum_section_width_m,
            protected_route_points=base_geometry.protected_route_points,
            event_meshes=event_meshes,
            structural_event_ids=structural_event_ids,
        )

    def apply_events(
        self,
        base_geometry: CaveGeometry,
        event_field: GeologicalEventField,
        progress: GeometryProgressCallback | None = None,
    ) -> CaveGeometry:
        """Apply grounded structural events and remesh the base cave volume."""

        return self.finalize(
            base_geometry,
            event_field,
            progress=progress,
        )

    def _apply_structural_events_to_grid(
        self,
        voxel_grid: VoxelGrid | TiledVoxelGrid,
        event_field: GeologicalEventField,
        progress: GeometryProgressCallback | None,
        *,
        protected_points: tuple[tuple[float, float, float], ...] = (),
    ) -> tuple[VoxelGrid | TiledVoxelGrid, tuple[int, ...]]:
        modifiers = tuple(
            event
            for event in event_field.events
            if event.kind in {"collapse", "choke", "infill"}
        )
        if not modifiers:
            return voxel_grid, ()
        if isinstance(voxel_grid, TiledVoxelGrid):
            return self._apply_structural_events_to_tiled_grid(
                voxel_grid,
                modifiers,
                progress,
                protected_points=protected_points,
            )

        density = np.array(voxel_grid.density, dtype=np.float32, copy=True)
        self._emit_progress(
            progress,
            "events",
            0,
            len(modifiers),
            f"applying {len(modifiers)} structural modifiers",
        )
        applied_ids: list[int] = []
        component_count = self._carved_component_count(
            density,
            voxel_grid.iso_level,
        )
        for index, event in enumerate(modifiers, start=1):
            previous_density = np.array(density, copy=True)
            applied = self._stamp_structural_event(
                density=density,
                voxel_grid=voxel_grid,
                event=event,
            )
            updated_component_count = self._carved_component_count(
                density,
                voxel_grid.iso_level,
            )
            candidate_grid = VoxelGrid(
                origin=voxel_grid.origin,
                voxel_size=voxel_grid.voxel_size,
                density=density,
                iso_level=voxel_grid.iso_level,
            )
            route_open = self._protected_route_is_open(
                candidate_grid,
                protected_points,
            )
            if (
                applied
                and route_open
                and 0 < updated_component_count <= component_count
            ):
                applied_ids.append(event.event_id)
                component_count = updated_component_count
            elif applied:
                density[...] = previous_density
                applied = self._stamp_structural_event(
                    density=density,
                    voxel_grid=voxel_grid,
                    event=event,
                    radius_scale=0.55,
                )
                updated_component_count = self._carved_component_count(
                    density,
                    voxel_grid.iso_level,
                )
                route_open = self._protected_route_is_open(
                    candidate_grid,
                    protected_points,
                )
                if (
                    applied
                    and route_open
                    and 0 < updated_component_count <= component_count
                ):
                    applied_ids.append(event.event_id)
                    component_count = updated_component_count
                else:
                    density[...] = previous_density
            self._emit_progress(
                progress,
                "events",
                index,
                len(modifiers),
                f"applied {event.kind} {index}/{len(modifiers)}",
            )

        return (
            VoxelGrid(
                origin=voxel_grid.origin,
                voxel_size=voxel_grid.voxel_size,
                density=density,
                iso_level=voxel_grid.iso_level,
            ),
            tuple(applied_ids),
        )

    def _apply_structural_events_to_tiled_grid(
        self,
        voxel_grid: TiledVoxelGrid,
        modifiers: tuple[GeologicalEvent, ...],
        progress: GeometryProgressCallback | None,
        *,
        protected_points: tuple[tuple[float, float, float], ...] = (),
    ) -> tuple[TiledVoxelGrid, tuple[int, ...]]:
        tiles = {key: np.array(tile, copy=True) for key, tile in voxel_grid.tiles.items()}
        result = TiledVoxelGrid(
            origin=voxel_grid.origin,
            voxel_size=voxel_grid.voxel_size,
            global_shape=voxel_grid.global_shape,
            iso_level=voxel_grid.iso_level,
            tile_size=voxel_grid.tile_size,
            tiles=tiles,
        )
        component_count = result.component_count
        applied_ids: list[int] = []
        for index, event in enumerate(modifiers, start=1):
            affected = self._event_tile_keys(result, event)
            backups = {key: np.array(result.tiles[key], copy=True) for key in affected}
            applied = self._stamp_event_into_tiles(result, event, affected)
            updated_components = result.component_count
            if (
                applied
                and self._protected_route_is_open(result, protected_points)
                and 0 < updated_components <= component_count
            ):
                applied_ids.append(event.event_id)
                component_count = updated_components
            else:
                for key, backup in backups.items():
                    result.tiles[key][...] = backup
                applied = self._stamp_event_into_tiles(
                    result,
                    event,
                    affected,
                    radius_scale=0.55,
                )
                updated_components = result.component_count
                if (
                    applied
                    and self._protected_route_is_open(result, protected_points)
                    and 0 < updated_components <= component_count
                ):
                    applied_ids.append(event.event_id)
                    component_count = updated_components
                else:
                    for key, backup in backups.items():
                        result.tiles[key][...] = backup
            self._emit_progress(
                progress,
                "events",
                index,
                len(modifiers),
                f"applied {event.kind} {index}/{len(modifiers)}",
            )
        return result, tuple(applied_ids)

    @staticmethod
    def _protected_route_is_open(
        voxel_grid: VoxelGrid | TiledVoxelGrid,
        protected_points: tuple[tuple[float, float, float], ...],
    ) -> bool:
        return all(
            voxel_grid.sample_density(point) >= voxel_grid.iso_level
            for point in protected_points
        )

    def _event_tile_keys(
        self,
        voxel_grid: TiledVoxelGrid,
        event: GeologicalEvent,
    ) -> tuple[tuple[int, int, int], ...]:
        center = np.asarray(event.position, dtype=float)
        radius = max(event.max_radius, voxel_grid.voxel_size)
        origin = np.asarray(voxel_grid.origin, dtype=float)
        low = np.floor((center - radius - origin) / voxel_grid.voxel_size).astype(int)
        high = np.ceil((center + radius - origin) / voxel_grid.voxel_size).astype(int)
        low_key = np.maximum(low // voxel_grid.tile_size, 0)
        high_key = np.maximum(high // voxel_grid.tile_size, 0)
        return tuple(
            key
            for key in voxel_grid.tiles
            if all(
                low_key[axis] <= key[axis] <= high_key[axis]
                for axis in range(3)
            )
        )

    def _stamp_event_into_tiles(
        self,
        voxel_grid: TiledVoxelGrid,
        event: GeologicalEvent,
        keys: tuple[tuple[int, int, int], ...],
        *,
        radius_scale: float = 1.0,
    ) -> bool:
        applied = False
        for key in keys:
            bounds = voxel_grid.tile_bounds(key)
            start = np.asarray((bounds[0], bounds[2], bounds[4]), dtype=float)
            tile_grid = VoxelGrid(
                origin=tuple(
                    np.asarray(voxel_grid.origin)
                    + start * voxel_grid.voxel_size
                ),
                voxel_size=voxel_grid.voxel_size,
                density=voxel_grid.tiles[key],
                iso_level=voxel_grid.iso_level,
            )
            applied = (
                self._stamp_structural_event(
                    density=tile_grid.density,
                    voxel_grid=tile_grid,
                    event=event,
                    radius_scale=radius_scale,
                )
                or applied
            )
        return applied

    @staticmethod
    def _carved_component_count(density: np.ndarray, iso_level: float) -> int:
        carved = density >= iso_level
        if not bool(np.any(carved)):
            return 0
        _labels, count = ndimage.label(
            carved,
            structure=ndimage.generate_binary_structure(rank=3, connectivity=1),
        )
        return int(count)

    def _stamp_structural_event(
        self,
        *,
        density: np.ndarray,
        voxel_grid: VoxelGrid,
        event: GeologicalEvent,
        radius_scale: float = 1.0,
    ) -> bool:
        center = np.asarray(event.position, dtype=float)
        radius_x = max(
            float(event.radius_x) * radius_scale,
            0.25 * voxel_grid.voxel_size,
        )
        radius_y = max(
            float(event.radius_y) * radius_scale,
            0.25 * voxel_grid.voxel_size,
        )
        radius_z = max(
            float(event.radius_z) * radius_scale,
            0.25 * voxel_grid.voxel_size,
        )
        horizontal_radius = max(radius_x, radius_y)
        padding = voxel_grid.voxel_size + max(
            float(self.config.structural_event_blend),
            0.0,
        )
        lower_world = center - np.array(
            (horizontal_radius + padding, horizontal_radius + padding, radius_z + padding)
        )
        upper_world = center + np.array(
            (horizontal_radius + padding, horizontal_radius + padding, radius_z + padding)
        )
        origin = np.asarray(voxel_grid.origin, dtype=float)
        lower = np.floor((lower_world - origin) / voxel_grid.voxel_size).astype(int)
        upper = np.ceil((upper_world - origin) / voxel_grid.voxel_size).astype(int)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.asarray(density.shape) - 1)
        if np.any(lower > upper):
            return False

        x_slice = slice(lower[0], upper[0] + 1)
        y_slice = slice(lower[1], upper[1] + 1)
        z_slice = slice(lower[2], upper[2] + 1)
        x_coords = origin[0] + np.arange(lower[0], upper[0] + 1) * voxel_grid.voxel_size
        y_coords = origin[1] + np.arange(lower[1], upper[1] + 1) * voxel_grid.voxel_size
        z_coords = origin[2] + np.arange(lower[2], upper[2] + 1) * voxel_grid.voxel_size
        grid_x, grid_y, grid_z = np.meshgrid(
            x_coords,
            y_coords,
            z_coords,
            indexing="ij",
        )
        dx = grid_x - center[0]
        dy = grid_y - center[1]
        dz = grid_z - center[2]
        cos_angle = math.cos(event.angle)
        sin_angle = math.sin(event.angle)
        local_x = cos_angle * dx + sin_angle * dy
        local_y = -sin_angle * dx + cos_angle * dy
        normalized_distance = np.sqrt(
            (local_x / radius_x) ** 2
            + (local_y / radius_y) ** 2
            + (dz / radius_z) ** 2
        )
        obstacle_exterior = (normalized_distance - 1.0) * min(
            radius_x,
            radius_y,
            radius_z,
        )
        target = density[x_slice, y_slice, z_slice]
        blend = max(float(self.config.structural_event_blend), 0.0)
        if blend <= 1e-9:
            target[...] = np.minimum(target, obstacle_exterior).astype(np.float32)
        else:
            difference = np.abs(target - obstacle_exterior)
            smoothing = np.maximum(blend - difference, 0.0)
            smooth_min = (
                np.minimum(target, obstacle_exterior)
                - smoothing * smoothing * 0.25 / blend
            )
            target[...] = smooth_min.astype(np.float32)
        return True

    def _build_voxel_grid(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        cave_network: CaveNetwork,
        progress: GeometryProgressCallback | None,
    ) -> VoxelGrid:
        self._emit_progress(progress, "voxel", 0, 4, "building stamp bounds")
        stamp_points = self._stamp_bounds_points(samples_by_segment)
        junction_stamp_points = self._junction_stamp_points(samples_by_segment, cave_network)
        stamp_points.extend(
            (
                stamp.center,
                max(stamp.radius_long, stamp.radius_short),
                stamp.radius_z,
            )
            for stamp in junction_stamp_points
        )
        margin = max(
            self.config.density_margin,
            max(max(radius_xy, radius_z) for _, radius_xy, radius_z in stamp_points) + self.config.voxel_size,
        )
        positions = np.array([position for position, _, _ in stamp_points], dtype=float)
        lower = positions.min(axis=0) - margin
        upper = positions.max(axis=0) + margin
        voxel_size = self.config.voxel_size
        shape = tuple(int(math.ceil((upper[axis] - lower[axis]) / voxel_size)) + 1 for axis in range(3))
        use_tiled_storage = (
            self.config.storage_mode == "tiled"
            or (
                self.config.storage_mode == "auto"
                and int(np.prod(shape)) > self.config.max_dense_voxels
            )
        )
        if use_tiled_storage:
            return self._build_tiled_voxel_grid(
                lower=lower,
                shape=shape,
                samples_by_segment=samples_by_segment,
                junction_stamps=junction_stamp_points,
                stamp_points=stamp_points,
                progress=progress,
            )
        density = np.full(shape, -1.0, dtype=np.float32)
        self._emit_progress(
            progress,
            "voxel",
            1,
            4,
            f"allocated density grid {shape[0]}x{shape[1]}x{shape[2]}",
        )

        segment_items = list(samples_by_segment.items())
        for index, (_segment_id, samples) in enumerate(segment_items, start=1):
            self._stamp_sample_chain(
                density=density,
                origin=lower,
                samples=samples,
            )
            self._emit_progress(
                progress,
                "voxel",
                index,
                len(segment_items) + len(junction_stamp_points),
                f"stamped segment {index}/{len(segment_items)}",
            )

        for index, stamp in enumerate(junction_stamp_points, start=1):
            self._stamp_junction_volume(
                density=density,
                origin=lower,
                stamp=stamp,
            )
            self._emit_progress(
                progress,
                "voxel",
                len(segment_items) + index,
                len(segment_items) + len(junction_stamp_points),
                f"stamped junction volume {index}/{len(junction_stamp_points)}",
            )

        removed_components, removed_voxels = self._remove_small_carved_components(density)
        carved_count = int(np.count_nonzero(density >= self.config.iso_level))
        self._emit_progress(
            progress,
            "voxel",
            4,
            4,
            f"carved {carved_count} voxels; removed {removed_voxels} island voxels from {removed_components} components",
        )

        return VoxelGrid(
            origin=tuple(float(value) for value in lower),
            voxel_size=voxel_size,
            density=density,
            iso_level=self.config.iso_level,
        )

    def _build_tiled_voxel_grid(
        self,
        *,
        lower: np.ndarray,
        shape: tuple[int, int, int],
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        junction_stamps: list[_JunctionStamp],
        stamp_points: list[tuple[np.ndarray, float, float]],
        progress: GeometryProgressCallback | None,
    ) -> TiledVoxelGrid:
        tile_size = max(int(self.config.chunk_size), 4)
        maximum_key = np.maximum(
            np.ceil((np.asarray(shape) - 1) / tile_size).astype(int) - 1,
            0,
        )
        active_keys: set[tuple[int, int, int]] = set()
        for position, radius_xy, radius_z in stamp_points:
            radius = max(radius_xy, radius_z) + self.config.voxel_size
            low_index = np.maximum(
                np.floor((position - radius - lower) / self.config.voxel_size).astype(int),
                0,
            )
            high_index = np.minimum(
                np.ceil((position + radius - lower) / self.config.voxel_size).astype(int),
                np.asarray(shape) - 1,
            )
            low_key = np.minimum(low_index // tile_size, maximum_key)
            high_key = np.minimum(high_index // tile_size, maximum_key)
            for x_key in range(int(low_key[0]), int(high_key[0]) + 1):
                for y_key in range(int(low_key[1]), int(high_key[1]) + 1):
                    for z_key in range(int(low_key[2]), int(high_key[2]) + 1):
                        active_keys.add((x_key, y_key, z_key))

        tiles: dict[tuple[int, int, int], np.ndarray] = {}
        ordered_keys = sorted(active_keys)
        self._emit_progress(
            progress,
            "voxel",
            0,
            len(ordered_keys),
            f"stamping {len(ordered_keys)} sparse tiles",
        )
        for index, key in enumerate(ordered_keys, start=1):
            start = np.asarray(key, dtype=int) * tile_size
            end = np.minimum(start + tile_size, np.asarray(shape) - 1)
            tile_shape = tuple(int(value) for value in end - start + 1)
            tile = np.full(tile_shape, -1.0, dtype=np.float32)
            tile_origin = lower + start * self.config.voxel_size
            for samples in samples_by_segment.values():
                self._stamp_sample_chain(
                    density=tile,
                    origin=tile_origin,
                    samples=samples,
                )
            for stamp in junction_stamps:
                self._stamp_junction_volume(
                    density=tile,
                    origin=tile_origin,
                    stamp=stamp,
                )
            if np.any(tile >= self.config.iso_level):
                tiles[key] = tile
            self._emit_progress(
                progress,
                "voxel",
                index,
                len(ordered_keys),
                f"tile {index}/{len(ordered_keys)}",
            )
        return TiledVoxelGrid(
            origin=tuple(float(value) for value in lower),
            voxel_size=self.config.voxel_size,
            global_shape=shape,
            iso_level=self.config.iso_level,
            tile_size=tile_size,
            tiles=tiles,
        )

    def _remove_small_carved_components(self, density: np.ndarray) -> tuple[int, int]:
        carved = density >= self.config.iso_level
        if not bool(np.any(carved)):
            return 0, 0
        labels, component_count = ndimage.label(
            carved,
            structure=ndimage.generate_binary_structure(rank=3, connectivity=1),
        )
        if component_count <= 1:
            return 0, 0

        sizes = np.bincount(labels.ravel())
        if sizes.size <= 1:
            return 0, 0
        largest_label = int(np.argmax(sizes[1:]) + 1)
        largest_size = int(sizes[largest_label])
        minimum_component_size = max(8, int(0.002 * largest_size))
        remove_mask = (labels != 0) & (labels != largest_label) & (sizes[labels] < minimum_component_size)
        removed_voxels = int(np.count_nonzero(remove_mask))
        removed_labels = {
            int(label)
            for label in np.unique(labels[remove_mask])
            if label != 0
        }
        density[remove_mask] = np.float32(self.config.iso_level - 1.0)
        return len(removed_labels), removed_voxels

    def _stamp_bounds_points(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
    ) -> list[tuple[np.ndarray, float, float]]:
        stamp_points: list[tuple[np.ndarray, float, float]] = []
        for samples in samples_by_segment.values():
            for sample in samples:
                position = np.array((sample.x, sample.y, sample.z), dtype=float)
                stamp_points.append((position, self._radius_xy(sample), self._radius_z(sample)))
        return stamp_points

    def _junction_stamp_points(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        cave_network: CaveNetwork,
    ) -> list[_JunctionStamp]:
        samples_by_junction: dict[int, list[SectionSample]] = defaultdict(list)
        for samples in samples_by_segment.values():
            for sample in samples:
                for influence in sample.junction_influences:
                    samples_by_junction[influence.junction_id].append(sample)

        stamp_points: list[_JunctionStamp] = []
        for junction in cave_network.junctions:
            influenced_samples = samples_by_junction.get(junction.junction_id, [])
            if not influenced_samples:
                continue
            mean_z = float(np.mean([sample.z for sample in influenced_samples]))
            mean_height = float(np.mean([sample.tube_height for sample in influenced_samples]))
            sample_radius = max(self._radius_xy(sample) for sample in influenced_samples)
            blend_radius = max(junction.blend_length * 0.28, sample_radius)
            if junction.kind == "chamber":
                blend_radius *= self.config.chamber_radius_scale
            position = np.array((junction.center_x, junction.center_y, mean_z), dtype=float)
            angle = self._junction_orientation(position, influenced_samples)
            radius_long = max(blend_radius, self.config.minimum_radius)
            short_scale = 0.68 if junction.kind == "chamber" else 0.58
            radius_short = max(radius_long * short_scale, sample_radius, self.config.minimum_radius)
            phase = tuple(
                float(value)
                for value in self._rng.uniform(0.0, 2.0 * math.pi, size=3)
            )
            stamp_points.append(
                _JunctionStamp(
                    center=position,
                    radius_long=radius_long,
                    radius_short=radius_short,
                    radius_z=max(mean_height * 0.70, self.config.minimum_radius),
                    angle=angle,
                    phase=phase,
                    kind=junction.kind,
                )
            )
        return stamp_points

    @staticmethod
    def _junction_orientation(
        center: np.ndarray,
        samples: list[SectionSample],
    ) -> float:
        offsets = np.array(
            [
                (sample.x - center[0], sample.y - center[1])
                for sample in samples
            ],
            dtype=float,
        )
        if offsets.shape[0] < 2 or np.allclose(offsets, 0.0):
            mean_tangent = np.mean(
                np.array([sample.tangent[:2] for sample in samples], dtype=float),
                axis=0,
            )
            return float(math.atan2(mean_tangent[1], mean_tangent[0]))

        covariance = offsets.T @ offsets
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        return float(math.atan2(direction[1], direction[0]))

    def _radius_xy(self, sample: SectionSample) -> float:
        scale = self.config.tunnel_radius_scale
        if any(influence.kind == "chamber" for influence in sample.junction_influences):
            scale = max(scale, self.config.chamber_radius_scale)
        elif sample.junction_blend_weight > 0.0:
            scale = max(scale, self.config.junction_radius_scale)
        return max(sample.tube_width * 0.5 * scale, self.config.minimum_radius)

    def _radius_z(self, sample: SectionSample) -> float:
        return max(sample.tube_height * 0.5 * self.config.tunnel_radius_scale, self.config.minimum_radius)

    def _stamp_sample_chain(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        samples: tuple[SectionSample, ...],
    ) -> None:
        if self.config.use_section_profiles:
            for start, end in zip(samples, samples[1:]):
                self._stamp_profile_segment(
                    density=density,
                    origin=origin,
                    start=start,
                    end=end,
                )
                self._stamp_capsule(
                    density=density,
                    origin=origin,
                    start=np.array((start.x, start.y, start.z), dtype=float),
                    end=np.array((end.x, end.y, end.z), dtype=float),
                    start_radius_xy=0.45 * self._radius_xy(start),
                    end_radius_xy=0.45 * self._radius_xy(end),
                    start_radius_z=0.45 * self._radius_z(start),
                    end_radius_z=0.45 * self._radius_z(end),
                )
            for sample in (samples[0], samples[-1]):
                self._stamp_profile_cap(
                    density=density,
                    origin=origin,
                    sample=sample,
                )
            return

        for start, end in zip(samples, samples[1:]):
            self._stamp_capsule(
                density=density,
                origin=origin,
                start=np.array((start.x, start.y, start.z), dtype=float),
                end=np.array((end.x, end.y, end.z), dtype=float),
                start_radius_xy=self._radius_xy(start),
                end_radius_xy=self._radius_xy(end),
                start_radius_z=self._radius_z(start),
                end_radius_z=self._radius_z(end),
            )
        for sample in (samples[0], samples[-1]):
            self._stamp_ellipsoid(
                density=density,
                origin=origin,
                center=np.array((sample.x, sample.y, sample.z), dtype=float),
                radius_xy=self._radius_xy(sample),
                radius_z=self._radius_z(sample),
            )

    def _stamp_profile_segment(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        start: SectionSample,
        end: SectionSample,
    ) -> None:
        start_position = np.array((start.x, start.y, start.z), dtype=float)
        end_position = np.array((end.x, end.y, end.z), dtype=float)
        segment = end_position - start_position
        segment_length_squared = float(np.dot(segment, segment))
        if segment_length_squared < 1e-9:
            self._stamp_profile_cap(density=density, origin=origin, sample=start)
            return

        radius = max(
            self._profile_bounds_radius(start),
            self._profile_bounds_radius(end),
            self.config.minimum_radius,
        )
        voxel_size = self.config.voxel_size
        lower = np.maximum(
            np.floor((np.minimum(start_position, end_position) - radius - origin) / voxel_size).astype(int),
            0,
        )
        upper = np.minimum(
            np.ceil((np.maximum(start_position, end_position) + radius - origin) / voxel_size).astype(int) + 1,
            np.array(density.shape, dtype=int),
        )
        if np.any(upper <= lower):
            return

        x_values = origin[0] + np.arange(lower[0], upper[0]) * voxel_size
        y_values = origin[1] + np.arange(lower[1], upper[1]) * voxel_size
        z_values = origin[2] + np.arange(lower[2], upper[2]) * voxel_size
        x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing="ij")

        point_x = x_grid - start_position[0]
        point_y = y_grid - start_position[1]
        point_z = z_grid - start_position[2]
        projection = (
            point_x * segment[0] + point_y * segment[1] + point_z * segment[2]
        ) / segment_length_squared
        projection = np.clip(projection, 0.0, 1.0)

        closest_x = start_position[0] + projection * segment[0]
        closest_y = start_position[1] + projection * segment[1]
        closest_z = start_position[2] + projection * segment[2]
        local_x = x_grid - closest_x
        local_y = y_grid - closest_y
        local_z = z_grid - closest_z

        start_normal = np.array(start.normal, dtype=float)
        start_binormal = np.array(start.binormal, dtype=float)
        end_normal = np.array(end.normal, dtype=float)
        end_binormal = np.array(end.binormal, dtype=float)
        normal = self._normalize_vector(
            (1.0 - projection)[..., None] * start_normal + projection[..., None] * end_normal,
            fallback=start_normal,
        )
        binormal = self._normalize_vector(
            (1.0 - projection)[..., None] * start_binormal + projection[..., None] * end_binormal,
            fallback=start_binormal,
        )

        section_x = local_x * normal[..., 0] + local_y * normal[..., 1] + local_z * normal[..., 2]
        section_z = local_x * binormal[..., 0] + local_y * binormal[..., 1] + local_z * binormal[..., 2]
        start_profile = np.array(start.profile_points, dtype=float) * self._profile_scale(start)
        end_profile = np.array(end.profile_points, dtype=float) * self._profile_scale(end)
        t_values = np.clip(projection.reshape(-1), 0.0, 1.0)
        signed_distance = self._interpolated_profile_signed_distance(
            section_x.reshape(-1),
            section_z.reshape(-1),
            t_values,
            start_profile,
            end_profile,
        ).reshape(section_x.shape)
        density_values = -signed_distance / max(voxel_size, 1e-6)
        density_values += self._wall_roughness(
            x_grid,
            y_grid,
            z_grid,
            signed_distance,
        )

        region = density[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        np.maximum(region, density_values.astype(np.float32), out=region)

    def _stamp_profile_cap(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        sample: SectionSample,
    ) -> None:
        position = np.array((sample.x, sample.y, sample.z), dtype=float)
        profile_radius = max(self._profile_bounds_radius(sample), self.config.minimum_radius)
        cap_length = max(1.5 * self.config.voxel_size, 0.35 * profile_radius)
        radius = max(profile_radius, cap_length)
        voxel_size = self.config.voxel_size
        lower = np.maximum(np.floor((position - radius - origin) / voxel_size).astype(int), 0)
        upper = np.minimum(
            np.ceil((position + radius - origin) / voxel_size).astype(int) + 1,
            np.array(density.shape, dtype=int),
        )
        if np.any(upper <= lower):
            return

        x_values = origin[0] + np.arange(lower[0], upper[0]) * voxel_size
        y_values = origin[1] + np.arange(lower[1], upper[1]) * voxel_size
        z_values = origin[2] + np.arange(lower[2], upper[2]) * voxel_size
        x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing="ij")
        dx = x_grid - position[0]
        dy = y_grid - position[1]
        dz = z_grid - position[2]
        normal = np.array(sample.normal, dtype=float)
        binormal = np.array(sample.binormal, dtype=float)
        tangent = np.array(sample.tangent, dtype=float)
        section_x = dx * normal[0] + dy * normal[1] + dz * normal[2]
        section_z = dx * binormal[0] + dy * binormal[1] + dz * binormal[2]
        along = dx * tangent[0] + dy * tangent[1] + dz * tangent[2]
        profile = np.array(sample.profile_points, dtype=float) * self._profile_scale(sample)
        normalized_along = np.clip(np.abs(along) / max(cap_length, 1e-6), 0.0, 1.0)
        taper = 1.0 - 0.45 * normalized_along * normalized_along
        cap_phase = (
            0.45 * np.sin(0.37 * section_x + self._roughness_phase[0])
            + 0.28 * np.cos(0.31 * section_z + self._roughness_phase[1])
        )
        taper = np.clip(taper * (1.0 + 0.04 * cap_phase), 0.48, 1.08)
        tapered_distance = self._profile_signed_distance(
            (section_x / taper).reshape(-1),
            (section_z / taper).reshape(-1),
            profile,
        ).reshape(section_x.shape) * taper
        cap_distance = np.maximum(tapered_distance, np.abs(along) - cap_length)
        density_values = -cap_distance / max(voxel_size, 1e-6)
        density_values += self._wall_roughness(
            x_grid,
            y_grid,
            z_grid,
            cap_distance,
        )
        region = density[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        np.maximum(region, density_values.astype(np.float32), out=region)

    def _profile_scale(self, sample: SectionSample) -> float:
        scale = self.config.tunnel_radius_scale
        if any(influence.kind == "chamber" for influence in sample.junction_influences):
            scale = max(scale, self.config.chamber_radius_scale)
        elif sample.junction_blend_weight > 0.0:
            scale = max(scale, self.config.junction_radius_scale)
        return scale

    def _profile_bounds_radius(self, sample: SectionSample) -> float:
        profile = np.array(sample.profile_points, dtype=float) * self._profile_scale(sample)
        if profile.size == 0:
            return max(self._radius_xy(sample), self._radius_z(sample))
        return max(float(np.max(np.linalg.norm(profile, axis=1))), self.config.minimum_radius)

    def _interpolated_profile_signed_distance(
        self,
        x_values: np.ndarray,
        z_values: np.ndarray,
        t_values: np.ndarray,
        start_profile: np.ndarray,
        end_profile: np.ndarray,
    ) -> np.ndarray:
        if start_profile.shape != end_profile.shape:
            return np.minimum(
                self._profile_signed_distance(x_values, z_values, start_profile),
                self._profile_signed_distance(x_values, z_values, end_profile),
            )

        distances = np.empty_like(x_values, dtype=float)
        rounded_t = np.round(t_values, 2)
        for t_value in np.unique(rounded_t):
            mask = rounded_t == t_value
            profile = (1.0 - t_value) * start_profile + t_value * end_profile
            distances[mask] = self._profile_signed_distance(
                x_values[mask],
                z_values[mask],
                profile,
            )
        return distances

    @staticmethod
    def _profile_signed_distance(
        x_values: np.ndarray,
        z_values: np.ndarray,
        profile: np.ndarray,
    ) -> np.ndarray:
        if profile.shape[0] < 3:
            return np.full_like(x_values, math.inf, dtype=float)

        vertices = profile[:-1] if np.allclose(profile[0], profile[-1]) else profile
        next_vertices = np.roll(vertices, -1, axis=0)
        px = x_values[:, None]
        pz = z_values[:, None]
        ax = vertices[None, :, 0]
        az = vertices[None, :, 1]
        bx = next_vertices[None, :, 0]
        bz = next_vertices[None, :, 1]
        edge_x = bx - ax
        edge_z = bz - az
        edge_length_squared = np.maximum(edge_x * edge_x + edge_z * edge_z, 1e-9)
        t = np.clip(((px - ax) * edge_x + (pz - az) * edge_z) / edge_length_squared, 0.0, 1.0)
        closest_x = ax + t * edge_x
        closest_z = az + t * edge_z
        distances = np.sqrt(np.square(px - closest_x) + np.square(pz - closest_z))
        unsigned_distance = np.min(distances, axis=1)

        crosses = ((az > pz) != (bz > pz)) & (
            px < (edge_x * (pz - az) / np.where(np.abs(edge_z) < 1e-9, 1e-9, edge_z) + ax)
        )
        inside = np.count_nonzero(crosses, axis=1) % 2 == 1
        return np.where(inside, -unsigned_distance, unsigned_distance)

    def _wall_roughness(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        signed_distance: np.ndarray,
    ) -> np.ndarray:
        amplitude = max(self.config.wall_roughness_amplitude, 0.0)
        if math.isclose(amplitude, 0.0):
            return np.zeros_like(signed_distance, dtype=float)

        frequency = max(self.config.wall_roughness_frequency, 1e-6)
        phase_x, phase_y, phase_z = self._roughness_phase
        roughness = (
            0.50 * np.sin(frequency * x_grid + phase_x)
            + 0.35 * np.sin(frequency * y_grid + phase_y)
            + 0.25 * np.cos(frequency * z_grid + phase_z)
            + 0.20 * np.sin(frequency * 0.53 * (x_grid + y_grid + z_grid))
        )
        blend_distance = max(self.config.wall_roughness_blend * self.config.voxel_size, 1e-6)
        wall_weight = np.exp(-np.abs(signed_distance) / blend_distance)
        return amplitude * roughness * wall_weight

    @staticmethod
    def _normalize_vector(
        vectors: np.ndarray,
        *,
        fallback: np.ndarray,
    ) -> np.ndarray:
        lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
        safe = lengths > 1e-9
        fallback_array = np.broadcast_to(fallback, vectors.shape)
        return np.where(safe, vectors / np.maximum(lengths, 1e-9), fallback_array)

    def _stamp_capsule(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        start_radius_xy: float,
        end_radius_xy: float,
        start_radius_z: float,
        end_radius_z: float,
    ) -> None:
        segment = end - start
        segment_length_squared = float(np.dot(segment, segment))
        if segment_length_squared < 1e-9:
            self._stamp_ellipsoid(
                density=density,
                origin=origin,
                center=start,
                radius_xy=max(start_radius_xy, end_radius_xy),
                radius_z=max(start_radius_z, end_radius_z),
            )
            return

        radius = max(start_radius_xy, end_radius_xy, start_radius_z, end_radius_z)
        voxel_size = self.config.voxel_size
        lower = np.maximum(
            np.floor((np.minimum(start, end) - radius - origin) / voxel_size).astype(int),
            0,
        )
        upper = np.minimum(
            np.ceil((np.maximum(start, end) + radius - origin) / voxel_size).astype(int) + 1,
            np.array(density.shape, dtype=int),
        )
        if np.any(upper <= lower):
            return

        x_values = origin[0] + np.arange(lower[0], upper[0]) * voxel_size
        y_values = origin[1] + np.arange(lower[1], upper[1]) * voxel_size
        z_values = origin[2] + np.arange(lower[2], upper[2]) * voxel_size
        x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing="ij")

        point_x = x_grid - start[0]
        point_y = y_grid - start[1]
        point_z = z_grid - start[2]
        projection = (
            point_x * segment[0] + point_y * segment[1] + point_z * segment[2]
        ) / segment_length_squared
        projection = np.clip(projection, 0.0, 1.0)

        closest_x = start[0] + projection * segment[0]
        closest_y = start[1] + projection * segment[1]
        closest_z = start[2] + projection * segment[2]
        radius_xy = (1.0 - projection) * start_radius_xy + projection * end_radius_xy
        radius_z = (1.0 - projection) * start_radius_z + projection * end_radius_z
        dx = (x_grid - closest_x) / np.maximum(radius_xy, 1e-6)
        dy = (y_grid - closest_y) / np.maximum(radius_xy, 1e-6)
        dz = (z_grid - closest_z) / np.maximum(radius_z, 1e-6)
        stamp_density = 1.0 - np.sqrt(dx * dx + dy * dy + dz * dz)
        region = density[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        np.maximum(region, stamp_density.astype(np.float32), out=region)

    def _stamp_ellipsoid(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        center: np.ndarray,
        radius_xy: float,
        radius_z: float,
    ) -> None:
        radius = max(radius_xy, radius_z)
        voxel_size = self.config.voxel_size
        lower = np.maximum(np.floor((center - radius - origin) / voxel_size).astype(int), 0)
        upper = np.minimum(
            np.ceil((center + radius - origin) / voxel_size).astype(int) + 1,
            np.array(density.shape, dtype=int),
        )
        if np.any(upper <= lower):
            return
        x_values = origin[0] + np.arange(lower[0], upper[0]) * voxel_size
        y_values = origin[1] + np.arange(lower[1], upper[1]) * voxel_size
        z_values = origin[2] + np.arange(lower[2], upper[2]) * voxel_size
        dx = (x_values[:, None, None] - center[0]) / max(radius_xy, 1e-6)
        dy = (y_values[None, :, None] - center[1]) / max(radius_xy, 1e-6)
        dz = (z_values[None, None, :] - center[2]) / max(radius_z, 1e-6)
        stamp_density = 1.0 - np.sqrt(dx * dx + dy * dy + dz * dz)
        region = density[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        np.maximum(region, stamp_density.astype(np.float32), out=region)

    def _stamp_junction_volume(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        stamp: _JunctionStamp,
    ) -> None:
        radius = max(stamp.radius_long, stamp.radius_short, stamp.radius_z)
        voxel_size = self.config.voxel_size
        lower = np.maximum(
            np.floor((stamp.center - radius - origin) / voxel_size).astype(int),
            0,
        )
        upper = np.minimum(
            np.ceil((stamp.center + radius - origin) / voxel_size).astype(int) + 1,
            np.array(density.shape, dtype=int),
        )
        if np.any(upper <= lower):
            return

        x_values = origin[0] + np.arange(lower[0], upper[0]) * voxel_size
        y_values = origin[1] + np.arange(lower[1], upper[1]) * voxel_size
        z_values = origin[2] + np.arange(lower[2], upper[2]) * voxel_size
        x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing="ij")

        dx = x_grid - stamp.center[0]
        dy = y_grid - stamp.center[1]
        dz = z_grid - stamp.center[2]
        cos_angle = math.cos(stamp.angle)
        sin_angle = math.sin(stamp.angle)
        local_long = dx * cos_angle + dy * sin_angle
        local_short = -dx * sin_angle + dy * cos_angle

        amplitude = max(self.config.junction_irregularity_amplitude, 0.0)
        frequency = max(self.config.junction_irregularity_frequency, 1e-6)
        phase_a, phase_b, phase_c = stamp.phase
        theta = np.arctan2(
            local_short / max(stamp.radius_short, 1e-6),
            local_long / max(stamp.radius_long, 1e-6),
        )
        radius_variation = 1.0 + amplitude * (
            0.38 * np.sin(3.0 * theta + phase_a)
            + 0.26 * np.sin(5.0 * theta + phase_b)
            + 0.18 * np.cos(frequency * (local_long - 0.6 * local_short) + phase_c)
        )
        radius_variation = np.clip(radius_variation, 0.72, 1.22)

        scaled_long = local_long / np.maximum(stamp.radius_long * radius_variation, 1e-6)
        scaled_short = local_short / np.maximum(stamp.radius_short * radius_variation, 1e-6)
        scaled_z = dz / max(stamp.radius_z, 1e-6)
        normalized_distance = np.sqrt(
            scaled_long * scaled_long
            + scaled_short * scaled_short
            + scaled_z * scaled_z
        )
        signed_distance = (normalized_distance - 1.0) * min(
            stamp.radius_long,
            stamp.radius_short,
            stamp.radius_z,
        )
        density_values = -signed_distance / max(voxel_size, 1e-6)
        density_values += self._wall_roughness(
            x_grid,
            y_grid,
            z_grid,
            signed_distance,
        )
        if stamp.kind == "chamber":
            density_values += 0.35 * amplitude * np.sin(
                frequency * 0.7 * (local_long + local_short + dz)
                + phase_b
            )

        region = density[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        np.maximum(region, density_values.astype(np.float32), out=region)

    def _march_chunks(
        self,
        voxel_grid: VoxelGrid | TiledVoxelGrid,
        progress: GeometryProgressCallback | None,
    ) -> list[GeometryChunkMesh]:
        if isinstance(voxel_grid, TiledVoxelGrid):
            return self._march_tiled_chunks(voxel_grid, progress)
        meshes: list[GeometryChunkMesh] = []
        chunk_size = max(int(self.config.chunk_size), 4)
        nx, ny, nz = voxel_grid.shape
        chunk_bounds = [
            (
                x_start,
                min(x_start + chunk_size, nx - 1),
                y_start,
                min(y_start + chunk_size, ny - 1),
                z_start,
                min(z_start + chunk_size, nz - 1),
            )
            for x_start in range(0, nx - 1, chunk_size)
            for y_start in range(0, ny - 1, chunk_size)
            for z_start in range(0, nz - 1, chunk_size)
        ]
        self._emit_progress(
            progress,
            "mesh",
            0,
            len(chunk_bounds),
            f"scanning {len(chunk_bounds)} chunks",
        )
        for index, (x_start, x_end, y_start, y_end, z_start, z_end) in enumerate(
            chunk_bounds,
            start=1,
        ):
            chunk_density = voxel_grid.density[
                x_start : x_end + 1,
                y_start : y_end + 1,
                z_start : z_end + 1,
            ]
            if (
                np.all(chunk_density < voxel_grid.iso_level)
                or np.all(chunk_density >= voxel_grid.iso_level)
            ):
                self._emit_progress(
                    progress,
                    "mesh",
                    index,
                    len(chunk_bounds),
                    f"chunk {index}/{len(chunk_bounds)} empty",
                )
                continue
            mesh = self._march_chunk(
                chunk_id=len(meshes),
                voxel_grid=voxel_grid,
                bounds=(x_start, x_end, y_start, y_end, z_start, z_end),
            )
            if mesh.faces:
                meshes.append(mesh)
            self._emit_progress(
                progress,
                "mesh",
                index,
                len(chunk_bounds),
                f"chunk {index}/{len(chunk_bounds)} -> {len(mesh.faces)} faces",
            )
        self._emit_progress(
            progress,
            "mesh",
            len(chunk_bounds),
            len(chunk_bounds),
            f"meshed {len(meshes)} non-empty chunks",
        )
        return meshes

    def _march_tiled_chunks(
        self,
        voxel_grid: TiledVoxelGrid,
        progress: GeometryProgressCallback | None,
    ) -> list[GeometryChunkMesh]:
        meshes: list[GeometryChunkMesh] = []
        items = sorted(voxel_grid.tiles.items())
        for index, (key, density) in enumerate(items, start=1):
            if (
                np.all(density < voxel_grid.iso_level)
                or np.all(density >= voxel_grid.iso_level)
            ):
                continue
            local_vertices, faces, _normals, _values = measure.marching_cubes(
                density,
                level=voxel_grid.iso_level,
                spacing=(voxel_grid.voxel_size,) * 3,
                allow_degenerate=False,
            )
            bounds = voxel_grid.tile_bounds(key)
            start = np.asarray((bounds[0], bounds[2], bounds[4]), dtype=float)
            chunk_origin = np.asarray(voxel_grid.origin) + start * voxel_grid.voxel_size
            world_vertices = local_vertices + chunk_origin
            meshes.append(
                GeometryChunkMesh(
                    chunk_id=len(meshes),
                    grid_bounds=bounds,
                    vertices=tuple(
                        tuple(float(value) for value in vertex)
                        for vertex in world_vertices
                    ),
                    faces=tuple(
                        tuple(int(value) for value in face)
                        for face in faces
                    ),
                )
            )
            self._emit_progress(
                progress,
                "mesh",
                index,
                len(items),
                f"tile {index}/{len(items)} -> {len(faces)} faces",
            )
        return meshes

    def _march_chunk(
        self,
        *,
        chunk_id: int,
        voxel_grid: VoxelGrid,
        bounds: tuple[int, int, int, int, int, int],
    ) -> GeometryChunkMesh:
        x_start, x_end, y_start, y_end, z_start, z_end = bounds
        chunk_density = voxel_grid.density[
            x_start : x_end + 1,
            y_start : y_end + 1,
            z_start : z_end + 1,
        ]
        local_vertices, faces, _normals, _values = measure.marching_cubes(
            chunk_density,
            level=voxel_grid.iso_level,
            spacing=(
                voxel_grid.voxel_size,
                voxel_grid.voxel_size,
                voxel_grid.voxel_size,
            ),
            allow_degenerate=False,
        )
        chunk_origin = np.array(voxel_grid.origin, dtype=float) + np.array(
            (x_start, y_start, z_start),
            dtype=float,
        ) * voxel_grid.voxel_size
        world_vertices = local_vertices + chunk_origin

        return GeometryChunkMesh(
            chunk_id=chunk_id,
            grid_bounds=bounds,
            vertices=tuple(tuple(float(value) for value in vertex) for vertex in world_vertices),
            faces=tuple(tuple(int(value) for value in face) for face in faces),
        )

    def _assemble_chunks(
        self,
        chunk_meshes: list[GeometryChunkMesh],
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...]]:
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        vertex_lookup: dict[tuple[int, int, int], int] = {}
        quantize = max(int(round(1.0 / max(self.config.weld_tolerance, 1e-12))), 1)
        for mesh in chunk_meshes:
            index_map: dict[int, int] = {}
            for local_index, vertex in enumerate(mesh.vertices):
                key = tuple(int(round(value * quantize)) for value in vertex)
                if key not in vertex_lookup:
                    vertex_lookup[key] = len(vertices)
                    vertices.append(vertex)
                index_map[local_index] = vertex_lookup[key]
            for a, b, c in mesh.faces:
                face = (index_map[a], index_map[b], index_map[c])
                if len(set(face)) == 3:
                    faces.append(face)
        return tuple(vertices), tuple(faces)

    @staticmethod
    def _count_components(faces: tuple[tuple[int, int, int], ...]) -> int:
        if not faces:
            return 0
        vertex_to_faces: dict[int, list[int]] = defaultdict(list)
        for face_index, face in enumerate(faces):
            for vertex_index in face:
                vertex_to_faces[vertex_index].append(face_index)
        visited: set[int] = set()
        component_count = 0
        for start_face in range(len(faces)):
            if start_face in visited:
                continue
            component_count += 1
            stack = [start_face]
            visited.add(start_face)
            while stack:
                face_index = stack.pop()
                for vertex_index in faces[face_index]:
                    for neighbor_face in vertex_to_faces[vertex_index]:
                        if neighbor_face in visited:
                            continue
                        visited.add(neighbor_face)
                        stack.append(neighbor_face)
        return component_count

    @staticmethod
    def _emit_progress(
        progress: GeometryProgressCallback | None,
        phase: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if progress is not None:
            progress(phase, current, max(total, 1), message)


__all__ = [
    "CaveGeometry",
    "GeometryChunkMesh",
    "GeometryConfig",
    "GeometryGenerator",
    "VoxelGrid",
]
