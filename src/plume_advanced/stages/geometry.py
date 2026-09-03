"""Stage D voxel stamping and marching-cubes isosurface generation."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

from plume_advanced.stages.events import (
    GeologicalEvent,
    GeologicalEventField,
    GeologicalEventMesh,
)
from plume_advanced.stages.geometry_types import (
    CaveGeometry,
    GeometryChunkMesh,
    GeometryConfig,
    SurfaceTextureFrame,
    TiledVoxelGrid,
    VoxelGrid,
)
from plume_advanced.stages.network import CaveNetwork
from plume_advanced.stages.section_field import SectionField, SectionSample

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
            empty_voxel_grid = VoxelGrid(
                origin=(0.0, 0.0, 0.0),
                voxel_size=self.config.voxel_size,
                density=np.full((2, 2, 2), -1.0, dtype=np.float32),
                iso_level=self.config.iso_level,
            )
            return CaveGeometry(
                config=self.config,
                voxel_grid=empty_voxel_grid,
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
        self._remove_small_solid_pockets(voxel_grid)
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
            surface_texture_frames=self._surface_texture_frames(
                cave_network,
                section_field,
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
        event_meshes: tuple[GeologicalEventMesh, ...] = ()
        if event_field is not None:
            voxel_grid, structural_event_ids = self._apply_structural_events_to_grid(
                voxel_grid,
                event_field,
                progress,
                protected_points=base_geometry.protected_route_points,
            )
            event_meshes = event_field.meshes
            if structural_event_ids:
                self._remove_small_solid_pockets(voxel_grid)

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
                surface_texture_frames=base_geometry.surface_texture_frames,
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
        if assembled_faces:
            triangles = np.asarray(assembled_faces, dtype=np.int64)
            edges = np.sort(np.concatenate([
                triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
            ]), axis=1)
            _, edge_counts = np.unique(edges, axis=0, return_counts=True)
            if np.any(edge_counts != 2):
                raise ValueError(
                    "Cave mesh is not closed and manifold: "
                    f"{np.count_nonzero(edge_counts == 1)} boundary edges, "
                    f"{np.count_nonzero(edge_counts > 2)} nonmanifold edges"
                )
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
            surface_texture_frames=base_geometry.surface_texture_frames,
            event_meshes=event_meshes,
            structural_event_ids=structural_event_ids,
        )

    @staticmethod
    def _remove_small_solid_pockets(grid: VoxelGrid | TiledVoxelGrid) -> None:
        """Remove unresolved floating rock specks, retaining connected dividers.

        The solid uses face connectivity, matching the traversability grid.
        Eight cells is a resolution criterion, not a passage-size
        filter. A matching halo makes this independent of chunk boundaries.
        """
        limit = 8

        def pockets(density: np.ndarray) -> np.ndarray:
            labels, _ = ndimage.label(
                density < grid.iso_level, structure=ndimage.generate_binary_structure(3, 1)
            )
            sizes = np.bincount(labels.ravel())
            removable = sizes <= limit
            removable[0] = False
            for axis in range(3):
                removable[np.unique(np.take(labels, (0, -1), axis=axis))] = False
            return removable[labels]

        if isinstance(grid, VoxelGrid):
            grid.density[pockets(grid.density)] = grid.iso_level + 1.0
            return
        replacements = []
        for key, tile in sorted(grid.tiles.items()):
            start = np.asarray(key) * grid.tile_size - limit
            shape = np.asarray(tile.shape) + 2 * limit
            neighborhood = np.full(tuple(shape), grid.iso_level - 1.0, dtype=np.float32)
            span = math.ceil(limit / grid.tile_size)
            for delta in np.ndindex(*((2 * span + 1,) * 3)):
                neighbor_key = tuple(key[axis] + delta[axis] - span for axis in range(3))
                neighbor = grid.tiles.get(neighbor_key)
                if neighbor is None:
                    continue
                neighbor_start = np.asarray(neighbor_key) * grid.tile_size
                low = np.maximum(start, neighbor_start)
                high = np.minimum(start + shape, neighbor_start + neighbor.shape)
                if np.any(high <= low):
                    continue
                target = tuple(slice(int(a), int(b)) for a, b in zip(low - start, high - start))
                source = tuple(slice(int(a), int(b)) for a, b in zip(low - neighbor_start, high - neighbor_start))
                neighborhood[target] = neighbor[source]
            interior = tuple(slice(limit, limit + size) for size in tile.shape)
            mask = pockets(neighborhood)[interior]
            if np.any(mask):
                replacements.append((tile, mask))
        for tile, mask in replacements:
            tile[mask] = grid.iso_level + 1.0

    def _surface_texture_frames(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
    ) -> tuple[SurfaceTextureFrame, ...]:
        """Retain transported section frames for route-aware surface mapping."""

        segment_lookup = {
            segment.segment_id: segment
            for segment in cave_network.segments
        }
        node_lookup = {node.node_id: node for node in cave_network.nodes}
        node_longitudinal = self._texture_node_longitudinals(cave_network)
        frames: list[SurfaceTextureFrame] = []
        for segment_field in section_field.segment_fields:
            segment = segment_lookup.get(segment_field.segment_id)
            if segment is None or not segment_field.samples:
                continue
            start_node = node_lookup.get(segment.start_node_id)
            end_node = node_lookup.get(segment.end_node_id)
            total_length = max(float(segment.total_length), 1e-9)
            start_longitudinal = node_longitudinal.get(
                segment.start_node_id,
                0.0,
            )
            end_longitudinal = node_longitudinal.get(
                segment.end_node_id,
                start_longitudinal + total_length,
            )
            forward = (
                start_longitudinal < end_longitudinal
                or (
                    math.isclose(start_longitudinal, end_longitudinal)
                    and (
                        start_node is None
                        or end_node is None
                        or start_node.along_position <= end_node.along_position
                    )
                )
            )
            texture_direction = 1.0 if forward else -1.0
            for sample in segment_field.samples:
                vertical_direction = (
                    -1.0
                    if float(sample.binormal[2]) < 0.0
                    else 1.0
                )
                normal_direction = texture_direction * vertical_direction
                profile = (
                    np.asarray(sample.profile_points, dtype=float)
                    * self._profile_scale(sample)
                )
                if len(profile):
                    profile[:, 0] *= normal_direction
                    profile[:, 1] *= vertical_direction
                if len(profile) >= 2:
                    profile_perimeter = float(
                        np.linalg.norm(np.diff(profile, axis=0), axis=1).sum()
                    )
                    profile_points = tuple(
                        (float(point[0]), float(point[1]))
                        for point in profile
                    )
                else:
                    profile_perimeter = 0.0
                    profile_points = ()
                sample_arc_length = float(
                    np.clip(sample.segment_arc_length, 0.0, total_length)
                )
                longitudinal_m = (
                    start_longitudinal + sample_arc_length
                    if forward
                    else end_longitudinal + total_length - sample_arc_length
                )
                frames.append(
                    SurfaceTextureFrame(
                        segment_id=sample.segment_id,
                        center=(float(sample.x), float(sample.y), float(sample.z)),
                        tangent=(
                            texture_direction * float(sample.tangent[0]),
                            texture_direction * float(sample.tangent[1]),
                            texture_direction * float(sample.tangent[2]),
                        ),
                        normal=(
                            normal_direction * float(sample.normal[0]),
                            normal_direction * float(sample.normal[1]),
                            normal_direction * float(sample.normal[2]),
                        ),
                        binormal=(
                            vertical_direction * float(sample.binormal[0]),
                            vertical_direction * float(sample.binormal[1]),
                            vertical_direction * float(sample.binormal[2]),
                        ),
                        longitudinal_m=longitudinal_m,
                        longitudinal_rate=1.0,
                        profile_points=profile_points,
                        profile_perimeter_m=profile_perimeter,
                    )
                )
        return tuple(frames)

    @staticmethod
    def _texture_node_longitudinals(
        cave_network: CaveNetwork,
    ) -> dict[int, float]:
        """Assign graph-geodesic texture distances from the first entry node."""

        if not cave_network.nodes:
            return {}
        node_lookup = {node.node_id: node for node in cave_network.nodes}
        adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for segment in cave_network.segments:
            length = max(float(segment.total_length), 1e-9)
            adjacency[segment.start_node_id].append(
                (segment.end_node_id, length)
            )
            adjacency[segment.end_node_id].append(
                (segment.start_node_id, length)
            )

        distances = {
            node_id: math.inf
            for node_id in node_lookup
        }
        remaining = set(node_lookup)
        component_offset = 0.0
        while remaining:
            root = min(
                remaining,
                key=lambda node_id: (
                    node_lookup[node_id].along_position,
                    node_id,
                ),
            )
            distances[root] = component_offset
            queue = [(component_offset, root)]
            while queue:
                distance, node_id = heapq.heappop(queue)
                if distance > distances[node_id]:
                    continue
                remaining.discard(node_id)
                for neighbor_id, length in adjacency.get(node_id, ()):
                    candidate = distance + length
                    if candidate >= distances.get(neighbor_id, math.inf):
                        continue
                    distances[neighbor_id] = candidate
                    heapq.heappush(queue, (candidate, neighbor_id))
            finite = [
                value
                for value in distances.values()
                if math.isfinite(value)
            ]
            component_offset = max(finite, default=component_offset) + 1.0
        return distances

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
            event_slices = self._structural_event_slices(
                voxel_grid,
                event,
                density.shape,
            )
            if event_slices is None:
                continue
            previous_density = np.array(density[event_slices], copy=True)
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
                density[event_slices] = previous_density
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
                    density[event_slices] = previous_density
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
        event_slices = self._structural_event_slices(
            voxel_grid,
            event,
            density.shape,
            radius_scale=radius_scale,
        )
        if event_slices is None:
            return False
        x_slice, y_slice, z_slice = event_slices
        lower = np.asarray(
            (x_slice.start, y_slice.start, z_slice.start),
            dtype=int,
        )
        upper = np.asarray(
            (x_slice.stop - 1, y_slice.stop - 1, z_slice.stop - 1),
            dtype=int,
        )
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
        origin = np.asarray(voxel_grid.origin, dtype=float)
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

    def _structural_event_slices(
        self,
        voxel_grid: VoxelGrid,
        event: GeologicalEvent,
        shape: tuple[int, ...],
        *,
        radius_scale: float = 1.0,
    ) -> tuple[slice, slice, slice] | None:
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
        extent = np.asarray(
            (
                horizontal_radius + padding,
                horizontal_radius + padding,
                radius_z + padding,
            )
        )
        origin = np.asarray(voxel_grid.origin, dtype=float)
        lower = np.floor(
            (center - extent - origin) / voxel_grid.voxel_size
        ).astype(int)
        upper = np.ceil(
            (center + extent - origin) / voxel_grid.voxel_size
        ).astype(int)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.asarray(shape) - 1)
        if np.any(lower > upper):
            return None
        return (
            slice(int(lower[0]), int(upper[0]) + 1),
            slice(int(lower[1]), int(upper[1]) + 1),
            slice(int(lower[2]), int(upper[2]) + 1),
        )

    def _build_voxel_grid(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        cave_network: CaveNetwork,
        progress: GeometryProgressCallback | None,
    ) -> VoxelGrid | TiledVoxelGrid:
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
        shape = (
            int(math.ceil((upper[0] - lower[0]) / voxel_size)) + 1,
            int(math.ceil((upper[1] - lower[1]) / voxel_size)) + 1,
            int(math.ceil((upper[2] - lower[2]) / voxel_size)) + 1,
        )
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
                progress=progress,
            )
        density = np.full(shape, -8.0, dtype=np.float32)
        self._emit_progress(
            progress,
            "voxel",
            1,
            4,
            f"allocated density grid {shape[0]}x{shape[1]}x{shape[2]}",
        )

        segment_items = list(samples_by_segment.items())
        for index, (_segment_id, samples) in enumerate(segment_items, start=1):
            self._stamp_network_chain(
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
            origin=(float(lower[0]), float(lower[1]), float(lower[2])),
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
        progress: GeometryProgressCallback | None,
    ) -> TiledVoxelGrid:
        tile_size = max(int(self.config.chunk_size), 4)
        maximum_key = np.maximum(
            np.ceil((np.asarray(shape) - 1) / tile_size).astype(int) - 1,
            0,
        )
        segments_by_key: defaultdict[
            tuple[int, int, int],
            list[tuple[SectionSample, ...]],
        ] = defaultdict(list)
        junctions_by_key: defaultdict[
            tuple[int, int, int],
            list[_JunctionStamp],
        ] = defaultdict(list)
        for samples in samples_by_segment.values():
            segment_keys: set[tuple[int, int, int]] = set()
            for sample in samples:
                position = np.asarray((sample.x, sample.y, sample.z), dtype=float)
                radius = self._sample_stamp_radius(sample)
                segment_keys.update(
                    self._tile_keys_for_world_bounds(
                        position - radius,
                        position + radius,
                        lower=lower,
                        shape=shape,
                        tile_size=tile_size,
                        maximum_key=maximum_key,
                    )
                )
            for start, end in zip(samples, samples[1:]):
                start_position = np.asarray((start.x, start.y, start.z), dtype=float)
                end_position = np.asarray((end.x, end.y, end.z), dtype=float)
                radius = max(
                    self._sample_stamp_radius(start),
                    self._sample_stamp_radius(end),
                )
                segment_keys.update(
                    self._tile_keys_for_world_bounds(
                        np.minimum(start_position, end_position) - radius,
                        np.maximum(start_position, end_position) + radius,
                        lower=lower,
                        shape=shape,
                        tile_size=tile_size,
                        maximum_key=maximum_key,
                    )
                )
            for key in segment_keys:
                segments_by_key[key].append(samples)
        for stamp in junction_stamps:
            radius_xy = 1.22 * max(stamp.radius_long, stamp.radius_short, stamp.radius_z)
            keys = self._tile_keys_for_world_bounds(
                stamp.center - radius_xy - self.config.voxel_size,
                stamp.center + radius_xy + self.config.voxel_size,
                lower=lower,
                shape=shape,
                tile_size=tile_size,
                maximum_key=maximum_key,
            )
            for key in keys:
                junctions_by_key[key].append(stamp)

        tiles: dict[tuple[int, int, int], np.ndarray] = {}
        ordered_keys = sorted(set(segments_by_key) | set(junctions_by_key))
        self._emit_progress(
            progress,
            "voxel",
            0,
            len(ordered_keys),
            f"stamping {len(ordered_keys)} sparse tiles",
        )
        for index, key in enumerate(ordered_keys, start=1):
            tile_start = np.asarray(key, dtype=int) * tile_size
            tile_end = np.minimum(tile_start + tile_size, np.asarray(shape) - 1)
            tile_shape = (
                int(tile_end[0] - tile_start[0] + 1),
                int(tile_end[1] - tile_start[1] + 1),
                int(tile_end[2] - tile_start[2] + 1),
            )
            tile = np.full(tile_shape, -8.0, dtype=np.float32)
            tile_origin = lower + tile_start * self.config.voxel_size
            for samples in segments_by_key.get(key, ()):
                self._stamp_network_chain(
                    density=tile,
                    origin=tile_origin,
                    samples=samples,
                )
            for stamp in junctions_by_key.get(key, ()):
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
            origin=(float(lower[0]), float(lower[1]), float(lower[2])),
            voxel_size=self.config.voxel_size,
            global_shape=shape,
            iso_level=self.config.iso_level,
            tile_size=tile_size,
            tiles=tiles,
        )

    def _tile_keys_for_world_bounds(
        self,
        lower_world: np.ndarray,
        upper_world: np.ndarray,
        *,
        lower: np.ndarray,
        shape: tuple[int, int, int],
        tile_size: int,
        maximum_key: np.ndarray,
    ) -> tuple[tuple[int, int, int], ...]:
        low_index = np.maximum(
            np.floor((lower_world - lower) / self.config.voxel_size).astype(int),
            0,
        )
        high_index = np.minimum(
            np.ceil((upper_world - lower) / self.config.voxel_size).astype(int),
            np.asarray(shape) - 1,
        )
        low_key = np.minimum(low_index // tile_size, maximum_key)
        high_key = np.minimum(high_index // tile_size, maximum_key)
        return tuple(
            (x_key, y_key, z_key)
            for x_key in range(int(low_key[0]), int(high_key[0]) + 1)
            for y_key in range(int(low_key[1]), int(high_key[1]) + 1)
            for z_key in range(int(low_key[2]), int(high_key[2]) + 1)
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
                radius = self._sample_stamp_radius(sample)
                stamp_points.append((position, radius, radius))
        return stamp_points

    def _sample_stamp_radius(self, sample: SectionSample) -> float:
        radius = (
            self._profile_bounds_radius(sample)
            if self.config.use_section_profiles
            else max(self._radius_xy(sample), self._radius_z(sample))
        )
        # Include the signed-distance band consumed by local smooth unions,
        # not just the positive volume. Neighboring tiles need that same band.
        return radius + 3.0 * self.config.voxel_size

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
            # Ordinary confluences are made by the incident sweeps. A generic
            # room at every split obscures the divider and creates swollen hubs.
            if junction.kind != "chamber":
                continue
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
            phase_values = self._rng.uniform(0.0, 2.0 * math.pi, size=3)
            phase = (
                float(phase_values[0]),
                float(phase_values[1]),
                float(phase_values[2]),
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
        return max(
            sample.tube_height * 0.5 * self.config.tunnel_radius_scale, self.config.minimum_radius
        )

    def _stamp_network_chain(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        samples: tuple[SectionSample, ...],
    ) -> None:
        """Blend incident tunnels once per chain, only near actual confluences.

        Per-sample smooth unions would inflate the tunnel when sampling gets
        denser. A bounded scratch region also keeps dense and tiled paths equal.
        Grade-separated crossings deliberately receive no union fillet.
        """
        anchors: dict[int, tuple[float, SectionSample]] = {}
        for sample in samples:
            for influence in sample.junction_influences:
                if influence.kind == "crossing":
                    continue
                if influence.weight > anchors.get(influence.junction_id, (0.0, sample))[0]:
                    anchors[influence.junction_id] = (influence.weight, sample)
        if not anchors:
            self._stamp_sample_chain(density=density, origin=origin, samples=samples)
            return
        positions = np.asarray([(sample.x, sample.y, sample.z) for sample in samples])
        radii = np.asarray([self._sample_stamp_radius(sample) for sample in samples])
        voxel = self.config.voxel_size
        lower = np.maximum(
            np.floor((np.min(positions - radii[:, None], axis=0) - origin) / voxel).astype(int), 0
        )
        upper = np.minimum(
            np.ceil((np.max(positions + radii[:, None], axis=0) - origin) / voxel).astype(int) + 1,
            density.shape,
        )
        if np.any(upper <= lower):
            return
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        region = density[slices]
        # A -1 background lies inside the blend band and can spuriously widen
        # an unrelated tube even when this chain has no support in its tile.
        incoming = np.full(region.shape, -8.0, dtype=np.float32)
        local_origin = origin + lower * voxel
        self._stamp_sample_chain(density=incoming, origin=local_origin, samples=samples)
        coordinates = np.ogrid[tuple(slice(0, size) for size in region.shape)]
        blend = np.zeros(region.shape, dtype=np.float32)
        for weight, sample in anchors.values():
            center = (sample.x, sample.y, sample.z)
            reach = max(sample.tube_width, sample.tube_height) * 1.5
            distance_squared = sum(
                (local_origin[axis] + coordinates[axis] * voxel - center[axis]) ** 2
                for axis in range(3)
            )
            envelope = np.clip(1.0 - distance_squared / (reach * reach), 0.0, 1.0)
            # At most half a voxel of added radius; do not erase rock islands.
            np.maximum(blend, 2.0 * weight * envelope * envelope, out=blend)
        difference = np.abs(region - incoming)
        overlap = np.maximum(blend - difference, 0.0)
        result = np.maximum(region, incoming) + overlap * overlap / np.maximum(4.0 * blend, 1e-12)
        region[...] = result

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
            if len(samples) == 1:
                self._stamp_profile_cap(density=density, origin=origin, sample=samples[0])
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

        radius = (
            max(
                self._profile_bounds_radius(start),
                self._profile_bounds_radius(end),
                self.config.minimum_radius,
            )
            + 3.0 * self.config.voxel_size
        )
        voxel_size = self.config.voxel_size
        lower = np.maximum(
            np.floor(
                (np.minimum(start_position, end_position) - radius - origin) / voxel_size
            ).astype(int),
            0,
        )
        upper = np.minimum(
            np.ceil(
                (np.maximum(start_position, end_position) + radius - origin) / voxel_size
            ).astype(int)
            + 1,
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
        # Retain longitudinal distance before clamping the closest point.
        # A cross-section distance alone extrudes the endpoint indefinitely,
        # leaving the stamp's rectangular allocation bounds as the tube end.
        axial_outside = np.maximum(-projection, projection - 1.0) * math.sqrt(
            segment_length_squared
        )
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
        # Round the finite sweep ends. Adjacent sweeps overlap continuously;
        # true termini close smoothly without a planar clipping surface.
        end_radius = max(
            min(self._profile_bounds_radius(start), self._profile_bounds_radius(end)),
            self.config.minimum_radius,
        )
        rounded_end = (
            np.hypot(
                np.maximum(signed_distance + end_radius, 0.0),
                np.maximum(axial_outside, 0.0),
            )
            - end_radius
        )
        signed_distance = np.where(axial_outside > 0.0, rounded_end, signed_distance)
        density_values = -signed_distance / max(voxel_size, 1e-6)
        density_values += self._wall_roughness(
            x_grid,
            y_grid,
            z_grid,
            signed_distance,
            local_vertical=section_z,
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
        cap_length = profile_radius
        radius = profile_radius + self.config.voxel_size
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
        radial_distance = self._profile_signed_distance(
            section_x.reshape(-1), section_z.reshape(-1), profile,
        ).reshape(section_x.shape)
        cap_distance = np.hypot(
            np.maximum(radial_distance + profile_radius, 0.0),
            along * profile_radius / cap_length,
        ) - profile_radius
        density_values = -cap_distance / max(voxel_size, 1e-6)
        density_values += self._wall_roughness(
            x_grid,
            y_grid,
            z_grid,
            cap_distance,
            local_vertical=section_z,
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
        *,
        local_vertical: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return seeded, zoned multi-scale relief close to the cave boundary.

        The slow field creates coherent smooth and rough lava-flow regions.
        Higher-frequency harmonics add resolvable rock relief without using
        marching-cubes facets as surface detail.  Floors receive a modest
        extra contribution so the traversable terrain is not perfectly flat.
        """

        amplitude = max(self.config.wall_roughness_amplitude, 0.0)
        if math.isclose(amplitude, 0.0):
            return np.zeros_like(signed_distance, dtype=float)

        frequency = max(self.config.wall_roughness_frequency, 1e-6)
        phase_x, phase_y, phase_z = self._roughness_phase
        broad = (
            0.52 * np.sin(frequency * x_grid + phase_x)
            + 0.38 * np.sin(frequency * y_grid + phase_y)
            + 0.30 * np.cos(frequency * z_grid + phase_z)
        )
        medium = (
            0.32
            * np.sin(
                frequency * 2.7 * (0.73 * x_grid - 0.41 * y_grid + 0.29 * z_grid)
                + phase_y
            )
            + 0.24
            * np.cos(
                frequency * 3.9 * (0.31 * x_grid + 0.67 * y_grid - 0.22 * z_grid)
                + phase_z
            )
        )
        fine = 0.14 * np.sin(
            frequency * 6.1 * (x_grid + 0.61 * y_grid + 0.37 * z_grid)
            + phase_x
        )
        roughness = broad + medium + fine

        zone_field = (
            0.58 * np.sin(frequency * 0.11 * x_grid + phase_z)
            + 0.42 * np.cos(frequency * 0.09 * y_grid + phase_x)
            + 0.30
            * np.sin(
                frequency * 0.07 * (x_grid + y_grid + 0.5 * z_grid)
                + phase_y
            )
        )
        zone = np.clip(0.5 + 0.42 * zone_field, 0.0, 1.0)
        zone = zone * zone * (3.0 - 2.0 * zone)
        zone_strength = 0.16 + 0.84 * zone

        terrain_gain: float | np.ndarray = 1.0
        if local_vertical is not None:
            vertical_scale = max(
                0.35 * self.config.characteristic_passage_width_m,
                self.config.voxel_size,
            )
            floor_weight = np.clip(
                0.5 - np.asarray(local_vertical, dtype=float) / vertical_scale,
                0.0,
                1.0,
            )
            floor_weight = floor_weight * floor_weight * (3.0 - 2.0 * floor_weight)
            terrain_gain = 1.0 + 0.55 * floor_weight

        blend_distance = max(self.config.wall_roughness_blend * self.config.voxel_size, 1e-6)
        wall_weight = np.exp(-np.abs(signed_distance) / blend_distance)
        return amplitude * roughness * zone_strength * terrain_gain * wall_weight

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
        radius = (
            1.22 * max(stamp.radius_long, stamp.radius_short, stamp.radius_z)
            + self.config.voxel_size
        )
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
            local_vertical=dz,
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
                        (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                        for vertex in world_vertices
                    ),
                    faces=tuple(
                        (int(face[0]), int(face[1]), int(face[2]))
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
        chunk_origin = (
            np.array(voxel_grid.origin, dtype=float)
            + np.array(
                (x_start, y_start, z_start),
                dtype=float,
            )
            * voxel_grid.voxel_size
        )
        world_vertices = local_vertices + chunk_origin

        return GeometryChunkMesh(
            chunk_id=chunk_id,
            grid_bounds=bounds,
            vertices=tuple(
                (float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in world_vertices
            ),
            faces=tuple((int(face[0]), int(face[1]), int(face[2])) for face in faces),
        )

    def _assemble_chunks(
        self,
        chunk_meshes: list[GeometryChunkMesh],
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...]]:
        if not chunk_meshes:
            return (), ()
        # A rounded coordinate is a bucket, not a distance tolerance: adjacent
        # buckets can contain two copies of the same marching-cubes vertex.
        # Merge by actual distance before any smoothing, UVs or displacement.
        positions = np.concatenate([np.asarray(mesh.vertices) for mesh in chunk_meshes])
        parent = np.arange(len(positions))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        pairs = cKDTree(positions).query_pairs(
            max(self.config.weld_tolerance, 1e-12), output_type="ndarray"
        )
        for a, b in pairs:
            ra, rb = root(int(a)), root(int(b))
            parent[max(ra, rb)] = min(ra, rb)
        roots = np.asarray([root(index) for index in range(len(positions))])
        representatives, inverse = np.unique(roots, return_inverse=True)
        vertices = positions[representatives]
        faces: list[tuple[int, int, int]] = []
        offset = 0
        for mesh in chunk_meshes:
            for a, b, c in mesh.faces:
                face = tuple(int(inverse[offset + index]) for index in (a, b, c))
                if len(set(face)) == 3:
                    faces.append(face)
            offset += len(mesh.vertices)
        return tuple(tuple(float(value) for value in vertex) for vertex in vertices), tuple(faces)

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
