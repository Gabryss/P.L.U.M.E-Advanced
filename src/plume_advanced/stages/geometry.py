"""Stage D voxel stamping and marching-cubes isosurface generation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from skimage import measure

from plume_advanced.procedural import procedural_rng
from plume_advanced.stages.events import (
    GeologicalEvent,
    GeologicalEventField,
    GeologicalEventMesh,
)
from plume_advanced.stages.geometry_types import (
    CaveGeometry,
    GeometryChunkMesh,
    GeometryConfig,
    TiledVoxelGrid,
    VoxelGrid,
)
from plume_advanced.stages.network import CaveNetwork
from plume_advanced.stages.section_field import SectionField, SectionFieldGenerator, SectionSample
from plume_advanced.stages.surface_relief import apply_surface_relief

GeometryProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class _JunctionEnvelope:
    """Incident-section envelope used for roof screening and diagnostics."""
    center: np.ndarray
    radius_long: float
    radius_short: float
    radius_z: float
    angle: float
    kind: str
    junction_id: int = -1
    blend_length_m: float = 0.0
    incident_widths: tuple[float, ...] = ()
    incident_heights: tuple[float, ...] = ()
    incident_fluxes: tuple[float, ...] = ()
    incident_segment_ids: tuple[int, ...] = ()
    floor_span_m: float = 0.0
    chamber_type: str = ""
    process_cause: str = ""
    pool_depth_m: float = 0.0


class GeometryGenerator:
    """Build cave geometry by stamping a density grid and polygonizing it."""

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()
        roughness_rng = procedural_rng(self.config.random_seed, "wall-roughness")
        self._roughness_phase = tuple(
            float(value)
            for value in roughness_rng.uniform(0.0, 2.0 * math.pi, size=3)
        )

    def generate(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        event_field: GeologicalEventField | None = None,
        progress: GeometryProgressCallback | None = None,
    ) -> CaveGeometry:
        """Convenience one-pass API built from the two-pass workflow."""

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

        if cave_network.config.quality.enabled:
            from plume_advanced.stages.network_quality import NetworkQualityError, assess_network
            assessment = assess_network(cave_network, sections=section_field)
            if not assessment["accepted"]:
                raise NetworkQualityError(assessment)
        self._preserved_pillar_columns = 0
        self._emit_progress(progress, "prepare", 0, 1, "collecting section samples")
        samples_by_segment = {
            segment_field.segment_id: self._refine_profile_chain(segment_field.samples)
            for segment_field in section_field.segment_fields
            if segment_field.samples
        }
        section_field = replace(section_field, segment_fields=tuple(
            replace(field, samples=samples_by_segment.get(field.segment_id, ()))
            for field in section_field.segment_fields
        ))
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

        junction_stamp_points = self._junction_stamp_points(samples_by_segment, cave_network)
        voxel_grid = self._build_voxel_grid(
            samples_by_segment,
            cave_network,
            progress,
            junction_stamps=junction_stamp_points,
        )
        self._remove_small_solid_pockets(voxel_grid)
        apply_surface_relief(voxel_grid, self.config, progress)
        if any(getattr(self.config, name) > 0. for name in (
            "surface_wall_relief_m", "surface_roof_relief_m",
            "surface_floor_relief_m", "surface_crust_relief_m",
        )):
            # Accretion can leave a one-voxel solid flake or air speck near a
            # tangential boundary. Repair only unresolved isolated pockets,
            # before roof-collapse screening, never whole disconnected routes.
            self._remove_small_solid_pockets(voxel_grid, include_void=True)
        from plume_advanced.stages.voxel_topology import close_density_fissures
        close_density_fissures(voxel_grid, self.config.density_closing_voxels)
        stability_records = self._enforce_roof_stability(
            voxel_grid, section_field, junction_stamp_points,
        )
        if isinstance(voxel_grid, TiledVoxelGrid):
            voxel_grid.synchronize_halos()
        return CaveGeometry(
            config=self.config,
            voxel_grid=voxel_grid,
            chunk_meshes=(),
            assembled_vertices=(),
            assembled_faces=(),
            component_count=0,
            stamped_sample_count=len(stamp_samples),
            stability_records=stability_records,
            preserved_pillar_columns=self._preserved_pillar_columns,
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
            route_centers=tuple(
                (float(s.x), float(s.y), float(s.z))
                for field in section_field.segment_fields for s in field.samples
            ),
            junction_report=self._junction_report(
                cave_network,
                samples_by_segment,
                junction_stamp_points=junction_stamp_points,
                voxel_grid=voxel_grid,
            ),
            junction_records=self._junction_records(
                cave_network,
                junction_stamp_points=junction_stamp_points,
                voxel_grid=voxel_grid,
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
                route_centers=base_geometry.route_centers,
                event_meshes=event_meshes,
                structural_event_ids=(),
                junction_report=base_geometry.junction_report,
                junction_records=base_geometry.junction_records,
                stability_records=base_geometry.stability_records,
                preserved_pillar_columns=base_geometry.preserved_pillar_columns,
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
        if assembled_faces and not self._mesh_is_closed_manifold(assembled_faces):
            # Independent chunk marches can disagree at an event surface that
            # is nearly tangent to an internal interface.  Re-marching the
            # unchanged dense field in one frame removes that seam ambiguity;
            # the invariant is checked again and never suppressed.
            if isinstance(voxel_grid, VoxelGrid):
                chunk_meshes = self._march_global(voxel_grid)
                assembled_vertices, assembled_faces = self._assemble_chunks(chunk_meshes)
            if assembled_faces and not self._mesh_is_closed_manifold(assembled_faces):
                triangles = np.asarray(assembled_faces, dtype=np.int64)
                edges = np.sort(
                    np.concatenate(
                        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
                        axis=0,
                    ),
                    axis=1,
                )
                _, edge_counts = np.unique(edges, axis=0, return_counts=True)
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
            route_centers=base_geometry.route_centers,
            event_meshes=event_meshes,
            structural_event_ids=structural_event_ids,
            junction_report=base_geometry.junction_report,
            junction_records=base_geometry.junction_records,
            stability_records=base_geometry.stability_records,
            preserved_pillar_columns=base_geometry.preserved_pillar_columns,
        )

    def _enforce_roof_stability(
        self,
        grid: VoxelGrid | TiledVoxelGrid,
        sections: SectionField,
        junctions: list[_JunctionEnvelope],
    ) -> tuple[tuple[tuple[str, object], ...], ...]:
        """Convert unsupported envelopes into mandatory breakdown plugs.

        This conservative end state represents a blocked, collapsed passage,
        not a simulated fracture/debris trajectory or a guaranteed skylight.
        Unlike decorative events it may disconnect a route or close it fully.
        Run after all section/junction unions so a later union cannot reopen it.
        """
        model = sections.config.roof_stability_model
        records: list[tuple[tuple[str, object], ...]] = []
        envelopes: list[tuple[str, np.ndarray, float, float, float, float, float]] = []
        sample_lookup = {}
        for field in sections.segment_fields:
            for index, sample in enumerate(field.samples):
                sample_lookup[(field.segment_id, sample.index)] = sample
                profile = np.asarray(sample.profile_points, dtype=float) * self._profile_scale(sample)
                if not len(profile):
                    continue
                offsets = profile[:, 0, None] * np.asarray(sample.normal)
                offsets += profile[:, 1, None] * np.asarray(sample.binormal)
                low, high = offsets.min(axis=0), offsets.max(axis=0)
                width = float(np.ptp(profile[:, 0]))
                center = np.asarray((sample.x, sample.y, sample.z)) + 0.5 * (low + high)
                height = float(high[2] - low[2])
                floor_depth = sample.surface_z - (sample.z + float(low[2]))
                neighbors = field.samples[max(index - 1, 0):index + 2]
                reach = max((float(np.linalg.norm(
                    np.asarray((n.x - sample.x, n.y - sample.y, n.z - sample.z))
                )) for n in neighbors), default=grid.voxel_size)
                envelopes.append((
                    f"section:{field.segment_id}:{sample.index}", center,
                    width, height, floor_depth,
                    max(reach, width, grid.voxel_size),
                    math.atan2(sample.tangent[1], sample.tangent[0]),
                ))
        for stamp in junctions:
            anchors = [s for s in sample_lookup.values() if any(
                i.junction_id == stamp.junction_id for i in s.junction_influences
            )]
            if not anchors:
                continue
            # Screen the enlarged junction too, including the bounded lobes
            # and sub-voxel stencil used by _stamp_junction. The shorter plan
            # axis is the assumed unsupported roof span (no arch support).
            width = 2.0 * stamp.radius_short * 1.22 + grid.voxel_size
            height = 2.0 * stamp.radius_z + grid.voxel_size
            surface = min(s.surface_z for s in anchors)
            floor_depth = surface - float(stamp.center[2]) + 0.5 * height
            envelopes.append((
                f"junction:{stamp.junction_id}", stamp.center, width, height,
                floor_depth, 2.0 * stamp.radius_long, stamp.angle,
            ))
        for identifier, center, width, height, floor_depth, reach, angle in envelopes:
            assessment = model.assess(
                width_m=width, height_m=height, floor_depth_m=floor_depth,
            )
            record: dict[str, object] = {
                "source": identifier, "width_m": width, "height_m": height,
                "floor_depth_m": floor_depth, **asdict(assessment),
                "outcome": "blocked_by_breakdown" if assessment.failed else "intact",
                "model": "self_weight_roof_beam_v1",
            }
            records.append(tuple(record.items()))
            if not assessment.failed:
                continue
            event = GeologicalEvent(
                event_id=-len(records), kind="collapse", segment_id=-1, sample_index=-1,
                x=float(center[0]), y=float(center[1]), z=float(center[2]),
                surface_z=float(center[2]) - 0.5 * height + floor_depth,
                floor_z=float(center[2]) - 0.5 * height,
                radius_x=max(reach, grid.voxel_size),
                radius_y=max(width, grid.voxel_size),
                radius_z=max(height + 0.25 * reach, grid.voxel_size),
                angle=angle, severity=1.0, material_hint="gravity_breakdown",
            )
            # Mandatory structural failure deliberately bypasses the optional
            # event path's connectivity/rover-route rollback.
            if isinstance(grid, TiledVoxelGrid):
                self._stamp_event_into_tiles(grid, event, self._event_tile_keys(grid, event))
            else:
                self._stamp_structural_event(density=grid.density, voxel_grid=grid, event=event)
        return tuple(records)

    @staticmethod
    def _mesh_is_closed_manifold(
        faces: tuple[tuple[int, int, int], ...],
    ) -> bool:
        if not faces:
            return True
        triangles = np.asarray(faces, dtype=np.int64)
        edges = np.sort(
            np.concatenate(
                [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
                axis=0,
            ),
            axis=1,
        )
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(edge_counts == 2))

    def _march_global(self, voxel_grid: VoxelGrid) -> list[GeometryChunkMesh]:
        """Polygonize a dense field in one shared coordinate frame."""

        density = voxel_grid.density
        if (
            density.ndim != 3
            or min(density.shape) < 2
            or np.all(density < voxel_grid.iso_level)
            or np.all(density >= voxel_grid.iso_level)
        ):
            return []
        local_vertices, faces, _normals, _values = measure.marching_cubes(
            density,
            level=voxel_grid.iso_level,
            spacing=(voxel_grid.voxel_size,) * 3,
            allow_degenerate=False,
        )
        world_vertices = local_vertices + np.asarray(voxel_grid.origin, dtype=float)
        nx, ny, nz = voxel_grid.shape
        return [
            GeometryChunkMesh(
                chunk_id=0,
                grid_bounds=(0, nx - 1, 0, ny - 1, 0, nz - 1),
                vertices=tuple(
                    (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                    for vertex in world_vertices
                ),
                faces=tuple(
                    (int(face[0]), int(face[1]), int(face[2])) for face in faces
                ),
            )
        ]

    @staticmethod
    def _remove_small_solid_pockets(
        grid: VoxelGrid | TiledVoxelGrid, *, include_void: bool = False,
    ) -> None:
        """Remove unresolved floating rock specks, retaining connected dividers.

        The solid uses face connectivity, matching the traversability grid.
        Eight cells is a resolution criterion, not a passage-size
        filter. A matching halo makes this independent of chunk boundaries.
        After accretion, include_void also closes unresolved isolated air specks.
        """
        limit = 8

        def pockets(density: np.ndarray, *, solid: bool = True) -> np.ndarray:
            labels, _ = ndimage.label(
                density < grid.iso_level if solid else density >= grid.iso_level,
                structure=ndimage.generate_binary_structure(3, 1),
            )
            sizes = np.bincount(labels.ravel())
            removable = sizes <= limit
            removable[0] = False
            for axis in range(3):
                removable[np.unique(np.take(labels, (0, -1), axis=axis))] = False
            return removable[labels]

        if isinstance(grid, VoxelGrid):
            if include_void:
                grid.density[pockets(grid.density, solid=False)] = grid.iso_level - 1.0
            # Close air first: flipping both masks simultaneously can turn a
            # tiny air shell and its solid center into a new isolated air cell.
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
            if include_void:
                void_pockets = pockets(neighborhood, solid=False)
                neighborhood[void_pockets] = grid.iso_level - 1.0
                void_mask = void_pockets[interior]
            else:
                void_mask = np.zeros(tile.shape, dtype=bool)
            mask = pockets(neighborhood)[interior]
            if np.any(mask) or np.any(void_mask):
                replacements.append((tile, mask, void_mask))
        for tile, mask, void_mask in replacements:
            tile[mask] = grid.iso_level + 1.0
            tile[void_mask] = grid.iso_level - 1.0

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
        junction_stamps: list[_JunctionEnvelope] | None = None,
    ) -> VoxelGrid | TiledVoxelGrid:
        self._emit_progress(progress, "voxel", 0, 4, "building stamp bounds")
        stamp_points = self._stamp_bounds_points(samples_by_segment)
        junction_stamp_points = (
            junction_stamps
            if junction_stamps is not None
            else self._junction_stamp_points(samples_by_segment, cave_network)
        )
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
                len(segment_items),
                f"stamped segment {index}/{len(segment_items)}",
            )

        pillars = self._solid_pillar_columns(np.any(density >= self.config.iso_level, axis=2))
        self._preserved_pillar_columns = int(np.count_nonzero(pillars))
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
        tiles: dict[tuple[int, int, int], np.ndarray] = {}
        projected_void = np.zeros(shape[:2], dtype=bool)
        ordered_keys = sorted(segments_by_key)
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
            projected_void[
                tile_start[0]:tile_end[0] + 1, tile_start[1]:tile_end[1] + 1
            ] |= np.any(tile >= self.config.iso_level, axis=2)
            if np.any(tile >= self.config.iso_level):
                tiles[key] = tile
            self._emit_progress(progress, "voxel", index, len(ordered_keys),
                                f"sampled passage tile {index}/{len(ordered_keys)}")
        # Global projection, including all vertical levels and tile halos,
        # gives exactly the same remnant mask as the dense implementation.
        pillars = self._solid_pillar_columns(projected_void)
        self._preserved_pillar_columns = int(np.count_nonzero(pillars))
        for index, key in enumerate(sorted(tiles), start=1):
            tile = tiles[key]
            tile_start = np.asarray(key, dtype=int) * tile_size
            tile_origin = lower + tile_start * self.config.voxel_size
            if not np.any(tile >= self.config.iso_level):
                del tiles[key]
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

    @staticmethod
    def _solid_pillar_columns(projected_void: np.ndarray) -> np.ndarray:
        """Keep existing solid columns enclosed by split/rejoin passages.

        A column containing any existing passage, including an underpass,
        cannot be marked solid. This preserves remnants rather than adding
        decorative pillars across paths. Open-ended gaps are not inferred.
        """
        return np.asarray(ndimage.binary_fill_holes(projected_void) & ~projected_void)

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
        radius = self._profile_bounds_radius(sample)
        # Include the signed-distance band consumed by local smooth unions,
        # not just the positive volume. Neighboring tiles need that same band.
        return radius + 3.0 * self.config.voxel_size

    def _junction_stamp_points(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        cave_network: CaveNetwork,
    ) -> list[_JunctionEnvelope]:
        def metadata_dimension(
            metadata: object,
            key: str,
            fallback: float,
        ) -> float:
            try:
                value = float(getattr(metadata, "get", lambda *_: fallback)(key, fallback))
            except (TypeError, ValueError):
                return float(fallback)
            return value if math.isfinite(value) and value > 0.0 else float(fallback)

        # Keep one representative section per incident segment.  Using all
        # exponentially weighted samples makes the room dimensions depend on
        # sampling density and can pull a junction centre down a long branch.
        samples_by_junction: dict[int, dict[int, tuple[float, SectionSample]]] = defaultdict(dict)
        blend_lengths_by_junction: dict[int, list[float]] = defaultdict(list)
        for segment_id, samples in samples_by_segment.items():
            for sample in samples:
                for influence in sample.junction_influences:
                    if influence.kind == "crossing":
                        # XY-overlapping grade-separated passages must remain
                        # separate volumes; no shared transition stamp.
                        continue
                    previous = samples_by_junction[influence.junction_id].get(segment_id)
                    if previous is None or influence.weight > previous[0]:
                        samples_by_junction[influence.junction_id][segment_id] = (
                            float(influence.weight), sample,
                        )
                    # Stage-C's per-sample metadata is the primary finite
                    # transition contract; influence metadata is a fallback
                    # for legacy fixtures that predate the field.
                    metadata_length = float(sample.junction_blend_length_m)
                    if metadata_length <= 0.0:
                        metadata_length = float(influence.blend_length_m)
                    if metadata_length > 0.0:
                        blend_lengths_by_junction[influence.junction_id].append(
                            metadata_length
                        )

        stamp_points: list[_JunctionEnvelope] = []
        segment_lookup = {
            segment.segment_id: segment
            for segment in getattr(cave_network, "segments", ())
        }
        for junction in cave_network.junctions:
            # Crossing junctions intentionally have no finite union volume.
            if junction.kind == "crossing":
                continue
            incident = samples_by_junction.get(junction.junction_id, {})
            if not incident:
                continue
            incident_levels = {
                getattr(segment_lookup.get(segment_id), "z_level", None)
                for segment_id in incident
            }
            incident_levels.discard(None)
            crossing_groups = {
                getattr(segment_lookup.get(segment_id), "metadata", {}).get(
                    "crossing_group_id"
                )
                for segment_id in incident
            }
            crossing_groups.discard(None)
            # Distinct level bands are crossings unless the network explicitly
            # marks a vertical capture/coalescence or chamber transition.
            if (len(incident_levels) > 1 or crossing_groups) and junction.kind not in {
                "chamber",
                "capture",
                "coalescence",
                "vertical_capture",
            }:
                continue
            anchors = [item[1] for item in incident.values()]
            widths = tuple(float(max(sample.tube_width, 0.0)) for sample in anchors)
            heights = tuple(float(max(sample.tube_height, 0.0)) for sample in anchors)
            fluxes = tuple(float(max(sample.lava_flux, 0.0)) for sample in anchors)
            if not widths:
                continue
            median_width = float(np.median(widths))
            median_height = float(np.median(heights))
            median_flux = float(np.median(fluxes)) if fluxes else 0.0
            flux_scale = (
                float(np.clip(np.mean(fluxes) / max(median_flux, 1e-9), 0.80, 1.20))
                if median_flux > 0.0
                else 1.0
            )
            # The network contract supplies a 1–3 diameter blend length. Clamp
            # malformed/legacy values to that local scale before constructing
            # a finite transition volume.
            diameter = max(median_width, 2.0 * self.config.minimum_radius)
            metadata_lengths = blend_lengths_by_junction.get(junction.junction_id, [])
            requested_blend = (
                float(np.median(metadata_lengths))
                if metadata_lengths
                else float(junction.blend_length)
            )
            blend_length = float(max(requested_blend, diameter, 1e-6))
            blend_length = min(blend_length, 3.0 * diameter)
            metadata = getattr(junction, "metadata", {}) or {}
            chamber_type = str(
                getattr(metadata, "get", lambda *_: "")(
                    "chamber_type",
                    "",
                )
            )
            is_drained_pool = (
                junction.kind == "chamber"
                and chamber_type == "drained_lava_pool"
            )
            pool_depth_m = 0.0
            process_cause = ""
            if is_drained_pool:
                requested_width = metadata_dimension(
                    metadata,
                    "pool_width_m",
                    2.0 * median_width,
                )
                pool_width = float(
                    np.clip(
                        requested_width,
                        median_width,
                        5.0 * median_width,
                    )
                )
                pool_aspect = float(
                    np.clip(
                        metadata_dimension(metadata, "pool_aspect_ratio", 2.0),
                        1.0,
                        8.0,
                    )
                )
                requested_length = metadata_dimension(
                    metadata,
                    "pool_length_m",
                    max(blend_length, pool_width * pool_aspect),
                )
                pool_length = max(requested_length, pool_width, blend_length)
                requested_depth = metadata_dimension(
                    metadata,
                    "pool_depth_m",
                    median_height,
                )
                pool_depth_m = float(
                    np.clip(
                        requested_depth,
                        0.65 * median_height,
                        1.50 * median_height,
                    )
                )
                radius_long = max(0.5 * pool_length, self.config.minimum_radius)
                process_cause = str(
                    getattr(metadata, "get", lambda *_: "")(
                        "process_cause",
                        "",
                    )
                )
            elif junction.kind == "chamber":
                # Chambers are broad but still bounded by the incident tube
                # scale; this avoids a Boolean-looking spherical room.
                radius_long = max(0.5 * blend_length, median_width * 0.95)
                radius_long *= min(max(self.config.chamber_radius_scale, 1.0), 1.7)
            else:
                # A split/merge needs a finite saddle spanning the transition,
                # not a tiny point union.  Keep transverse growth modest so
                # natural profile asymmetry remains visible.
                radius_long = max(0.5 * blend_length, median_width * 1.05)
                radius_long *= min(max(self.config.junction_radius_scale, 1.0), 1.35)
            if not is_drained_pool:
                radius_long *= flux_scale
                radius_long = max(radius_long, self.config.minimum_radius)
            floor_values = np.asarray(
                [self._sample_floor(sample) for sample in anchors],
                dtype=float,
            )
            mean_z = float(np.mean([sample.z for sample in anchors]))
            position = np.array((junction.center_x, junction.center_y, mean_z), dtype=float)
            angle = self._junction_orientation(position, anchors)
            contours = [np.asarray(s.profile_points) * self._profile_scale(s) for s in anchors]
            floors = [self._sample_floor(s) for s in anchors]
            roofs = [
                s.z + float(np.max(p[:, 0] * s.normal[2] + p[:, 1] * s.binormal[2]))
                for s, p in zip(anchors, contours, strict=True)
            ]
            position[2] = 0.5 * (min(floors) + max(roofs))
            radius_short = 0.5 * max(float(np.ptp(p[:, 0])) for p in contours)
            radius_z = 0.5 * (max(roofs) - min(floors))
            stamp_points.append(
                _JunctionEnvelope(
                    center=position,
                    radius_long=radius_long,
                    radius_short=radius_short,
                    radius_z=radius_z,
                    angle=angle,
                    kind=junction.kind,
                    junction_id=int(junction.junction_id),
                    blend_length_m=blend_length,
                    incident_widths=widths,
                    incident_heights=heights,
                    incident_fluxes=fluxes,
                    incident_segment_ids=tuple(sorted(incident)),
                    floor_span_m=(
                        float(np.ptp(floor_values)) if floor_values.size else 0.0
                    ),
                    chamber_type=chamber_type,
                    process_cause=process_cause,
                    pool_depth_m=pool_depth_m,
                )
            )
        return stamp_points

    def _junction_report(
        self,
        cave_network: CaveNetwork,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        *,
        junction_stamp_points: list[_JunctionEnvelope],
        voxel_grid: VoxelGrid | TiledVoxelGrid | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Summarize local junction geometry for QA and scientific reports."""

        _ = samples_by_segment
        connected = [stamp for stamp in junction_stamp_points if stamp.incident_widths]
        widths = [width for stamp in connected for width in stamp.incident_widths]
        heights = [height for stamp in connected for height in stamp.incident_heights]
        if not widths:
            return (
                ("junction_count", 0.0),
                ("junction_max_width_m", 0.0),
                ("junction_incident_max_width_m", 0.0),
                ("junction_median_incident_width_m", 0.0),
                ("junction_daughter_parent_area_ratio", 0.0),
                ("junction_floor_continuity_m", 0.0),
                ("minimum_throat_clearance_m", 0.0),
                ("unresolved_sub_two_voxel_features", 0.0),
                ("junction_envelope_count", 0.0),
            )
        area_ratios: list[float] = []
        generated_widths: list[float] = []
        segment_lookup = {
            segment.segment_id: segment
            for segment in getattr(cave_network, "segments", ())
        }
        for stamp in connected:
            generated_widths.append(2.0 * float(stamp.radius_short))
            areas = np.pi * (
                np.asarray(stamp.incident_widths, dtype=float) * 0.5
            ) * (
                np.asarray(stamp.incident_heights, dtype=float) * 0.5
            )
            if areas.size >= 2:
                incoming = []
                outgoing = []
                junction = next(
                    (
                        item
                        for item in getattr(cave_network, "junctions", ())
                        if item.junction_id == stamp.junction_id
                    ),
                    None,
                )
                node_ids = set(junction.node_ids) if junction is not None else set()
                for index, segment_id in enumerate(stamp.incident_segment_ids):
                    segment = segment_lookup.get(segment_id)
                    if segment is None:
                        continue
                    if segment.end_node_id in node_ids:
                        incoming.append(index)
                    if segment.start_node_id in node_ids:
                        outgoing.append(index)
                # A directed ratio is only meaningful when incidence is
                # unambiguous; otherwise expose it as unavailable in records.
                if incoming and outgoing and set(incoming).isdisjoint(outgoing):
                    parent_area = float(np.sum(areas[incoming]))
                    daughter_area = float(np.sum(areas[outgoing]))
                    area_ratios.append(daughter_area / max(parent_area, 1e-9))
        clearance_values: list[float] = []
        if voxel_grid is not None:
            for stamp in connected:
                theta = np.asarray((-math.sin(stamp.angle), math.cos(stamp.angle), 0.0))
                hit_plus = voxel_grid.raycast_isosurface(
                    stamp.center,
                    theta,
                    max(2.5 * stamp.radius_short, self.config.voxel_size),
                )
                hit_minus = voxel_grid.raycast_isosurface(
                    stamp.center,
                    -theta,
                    max(2.5 * stamp.radius_short, self.config.voxel_size),
                )
                if hit_plus is not None and hit_minus is not None:
                    clearance_values.append(float(hit_plus.distance + hit_minus.distance))
        clearance = min(clearance_values) if clearance_values else 0.0
        unresolved = sum(
            1
            for width, height in zip(widths, heights or widths, strict=False)
            if min(width, height) / max(self.config.voxel_size, 1e-9) < 2.0
        )
        return (
            ("junction_count", float(len(connected))),
            ("junction_max_width_m", float(max(generated_widths))),
            ("junction_incident_max_width_m", float(max(widths))),
            ("junction_median_incident_width_m", float(np.median(widths))),
            (
                "junction_daughter_parent_area_ratio",
                float(np.mean(area_ratios)) if area_ratios else 0.0,
            ),
            (
                "junction_floor_continuity_m",
                float(max(stamp.floor_span_m for stamp in connected)),
            ),
            ("minimum_throat_clearance_m", float(max(clearance, 0.0))),
            ("unresolved_sub_two_voxel_features", float(unresolved)),
            (
                "junction_envelope_count",
                float(len(connected)),
            ),
        )

    def _junction_records(
        self,
        cave_network: CaveNetwork,
        *,
        junction_stamp_points: list[_JunctionEnvelope],
        voxel_grid: VoxelGrid | TiledVoxelGrid | None = None,
    ) -> tuple[tuple[tuple[str, object], ...], ...]:
        """Return immutable per-junction records, including unresolved flags."""

        segment_lookup = {
            segment.segment_id: segment
            for segment in getattr(cave_network, "segments", ())
        }
        records: list[tuple[tuple[str, object], ...]] = []
        for stamp in junction_stamp_points:
            junction = next(
                (
                    item
                    for item in getattr(cave_network, "junctions", ())
                    if item.junction_id == stamp.junction_id
                ),
                None,
            )
            node_ids = set(junction.node_ids) if junction is not None else set()
            incoming: list[int] = []
            outgoing: list[int] = []
            for index, segment_id in enumerate(stamp.incident_segment_ids):
                segment = segment_lookup.get(segment_id)
                if segment is None:
                    continue
                if segment.end_node_id in node_ids:
                    incoming.append(index)
                if segment.start_node_id in node_ids:
                    outgoing.append(index)
            ratio: object = None
            if incoming and outgoing and set(incoming).isdisjoint(outgoing):
                widths = np.asarray(stamp.incident_widths, dtype=float)
                heights = np.asarray(stamp.incident_heights, dtype=float)
                areas = np.pi * (0.5 * widths) * (0.5 * heights)
                ratio = float(np.sum(areas[outgoing]) / max(np.sum(areas[incoming]), 1e-9))
            throat = 0.0
            if voxel_grid is not None:
                direction = np.asarray((-math.sin(stamp.angle), math.cos(stamp.angle), 0.0))
                plus = voxel_grid.raycast_isosurface(stamp.center, direction, 2.5 * stamp.radius_short)
                minus = voxel_grid.raycast_isosurface(stamp.center, -direction, 2.5 * stamp.radius_short)
                if plus is not None and minus is not None:
                    throat = float(plus.distance + minus.distance)
            unresolved = min(
                stamp.incident_widths or (0.0,)
            ) / max(self.config.voxel_size, 1e-9) < 2.0
            records.append(
                (
                    ("junction_id", int(stamp.junction_id)),
                    ("kind", str(stamp.kind)),
                    ("chamber_type", str(stamp.chamber_type)),
                    ("process_cause", str(stamp.process_cause)),
                    ("construction", "section_sweeps"),
                    ("generated_width_m", float(2.0 * stamp.radius_short)),
                    ("generated_length_m", float(2.0 * stamp.radius_long)),
                    ("generated_height_m", float(2.0 * stamp.radius_z)),
                    ("pool_depth_m", float(stamp.pool_depth_m)),
                    ("incident_widths_m", tuple(float(value) for value in stamp.incident_widths)),
                    ("daughter_parent_area_ratio", ratio),
                    ("floor_span_m", float(stamp.floor_span_m)),
                    ("blend_length_m", float(stamp.blend_length_m)),
                    ("minimum_throat_clearance_m", throat),
                    ("resolved", not unresolved),
                )
            )
        return tuple(records)

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

    def _sample_floor(self, sample: SectionSample) -> float:
        """Return the lowest point of a geometry-ready profile in world Z."""

        profile = np.asarray(sample.profile_points, dtype=float)
        if profile.size == 0:
            return float(sample.z - 0.5 * sample.tube_height)
        profile = profile * self._profile_scale(sample)
        normal = np.asarray(sample.normal, dtype=float)
        binormal = np.asarray(sample.binormal, dtype=float)
        vertical = profile[:, 0] * normal[2] + profile[:, 1] * binormal[2]
        return float(sample.z + np.min(vertical))

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

    def _refine_profile_chain(
        self, samples: tuple[SectionSample, ...]
    ) -> tuple[SectionSample, ...]:
        """Resolve smooth bends before finite sweeps, without spline overshoot.

        Interpolate the floor independently of section height, so widening a
        chamber cannot lift its floor. Original nodes and contours are retained.
        """
        if len(samples) < 3:
            return samples
        if any(len(s.profile_points) != len(samples[0].profile_points) for s in samples):
            return samples  # retain the existing unequal-contour fallback
        arc = np.asarray([s.segment_arc_length for s in samples])
        if np.any(np.diff(arc) <= 1e-9):
            return samples
        step = max(2.0, 4.0 * self.config.voxel_size)
        positions = np.unique(np.concatenate([
            np.linspace(a, b, max(2, int(math.ceil((b - a) / step)) + 1))
            for a, b in zip(arc[:-1], arc[1:], strict=True)
        ]))
        centers = np.asarray([(s.x, s.y, s.z) for s in samples])
        curve = PchipInterpolator(arc, centers, axis=0)
        points = curve(positions)
        directions = curve.derivative()(positions)
        floors = PchipInterpolator(arc, [self._sample_floor(s) for s in samples])(positions)
        result = []
        for i, (along, point, direction, floor) in enumerate(zip(positions, points, directions, floors, strict=True)):
            index = int(np.clip(np.searchsorted(arc, along, side="right") - 1, 0, len(samples) - 2))
            start, end = samples[index:index + 2]
            weight = float((along - arc[index]) / (arc[index + 1] - arc[index]))
            reference = start if weight < 0.5 else end
            if np.linalg.norm(direction) < 1e-9:
                direction = np.asarray(reference.tangent)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            normal, binormal = SectionFieldGenerator._build_frame(tuple(direction), reference.normal)
            profile = (1.0 - weight) * np.asarray(start.profile_points) + weight * np.asarray(end.profile_points)
            offset = float(np.min(profile[:, 0] * normal[2] + profile[:, 1] * binormal[2]))
            z = float(floor - self._profile_scale(reference) * offset)
            surface = (1.0 - weight) * start.surface_z + weight * end.surface_z
            result.append(replace(
                reference, index=i, segment_arc_length=float(along),
                x=float(point[0]), y=float(point[1]), z=z,
                tangent=(float(direction[0]), float(direction[1]), float(direction[2])),
                normal=normal, binormal=binormal,
                profile_points=tuple((float(p[0]), float(p[1])) for p in profile),
                tube_width=(1.0 - weight) * start.tube_width + weight * end.tube_width,
                tube_height=(1.0 - weight) * start.tube_height + weight * end.tube_height,
                surface_z=surface, centerline_depth=surface - z,
            ))
        return tuple(result)

    def _stamp_sample_chain(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        samples: tuple[SectionSample, ...],
    ) -> None:
        for index, (start, end) in enumerate(zip(samples, samples[1:])):
            self._stamp_profile_segment(
                density=density,
                origin=origin,
                start=start,
                end=end,
                cap_start=index == 0,
                cap_end=index == len(samples) - 2,
            )
        if len(samples) == 1:
            self._stamp_profile_cap(density=density, origin=origin, sample=samples[0])
        return

    def _stamp_profile_segment(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        start: SectionSample,
        end: SectionSample,
        cap_start: bool = True,
        cap_end: bool = True,
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
        chord_projection = (
            point_x * segment[0] + point_y * segment[1] + point_z * segment[2]
        ) / segment_length_squared
        # Use the shared section planes as the loft boundaries. Projection
        # onto each chord independently assigns different profiles to the same
        # boundary point on a bend, making a repeated ridge at every sample.
        start_tangent, end_tangent = np.asarray(start.tangent), np.asarray(end.tangent)
        start_distance = point_x*start_tangent[0] + point_y*start_tangent[1] + point_z*start_tangent[2]
        end_distance = ((point_x-segment[0])*end_tangent[0]
                        + (point_y-segment[1])*end_tangent[1]
                        + (point_z-segment[2])*end_tangent[2])
        denominator = start_distance - end_distance
        projection = np.divide(start_distance, denominator, out=chord_projection.copy(),
                               where=denominator > 1e-9)
        axial_outside = np.maximum(-start_distance, end_distance)
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
        if start_profile.shape != end_profile.shape:
            raise ValueError("Section lofts require profiles with matching vertex counts")
        t_values = np.clip(projection.reshape(-1), 0.0, 1.0)
        # Only the narrow band can affect the isosurface or confluence fillet.
        # Conservative interpolated bounds reject distant queries; keep exact
        # polygon distances throughout the band used by marching cubes.
        lower_profile = ((1.0 - t_values[:, None]) * start_profile.min(axis=0)
                         + t_values[:, None] * end_profile.min(axis=0))
        upper_profile = ((1.0 - t_values[:, None]) * start_profile.max(axis=0)
                         + t_values[:, None] * end_profile.max(axis=0))
        query = np.column_stack((section_x.ravel(), section_z.ravel()))
        outside_box = np.maximum(np.maximum(lower_profile - query, query - upper_profile), 0.0)
        signed_distance = np.linalg.norm(outside_box, axis=1)
        band = voxel_size * (3.0 + abs(self.config.iso_level) + 3.0*self.config.wall_roughness_amplitude)
        active = signed_distance <= band
        signed_distance[active] = self._interpolated_profile_signed_distance(
            query[active, 0], query[active, 1], t_values[active], start_profile, end_profile,
        )
        signed_distance = signed_distance.reshape(section_x.shape)
        # Round the finite sweep ends. Adjacent sweeps overlap continuously;
        # true termini close smoothly without a planar clipping surface.
        terminal_radius = max(
            min(self._profile_bounds_radius(start), self._profile_bounds_radius(end)),
            self.config.minimum_radius,
        )
        # Only real termini have rounded caps. Interior samples need a small
        # overlap at the shared plane, not another full-size bulb per sample.
        terminal = ((start_distance < 0.0) & cap_start) | ((end_distance > 0.0) & cap_end)
        end_radius = np.where(terminal, terminal_radius, 0.5*voxel_size)
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
        return self.config.tunnel_radius_scale

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
            raise ValueError("Section lofts require profiles with matching vertex counts")

        # Evaluate each query against its own interpolated polygon. Rounding
        # the interpolation parameter produces 100 discrete terraces per span.
        closed = np.allclose(start_profile[0], start_profile[-1]) and np.allclose(
            end_profile[0], end_profile[-1]
        )
        first = start_profile[:-1] if closed else start_profile
        last = end_profile[:-1] if closed else end_profile
        if len(first) < 3:
            return np.full_like(x_values, math.inf, dtype=float)
        distances = np.empty_like(x_values, dtype=float)
        for begin in range(0, len(x_values), 4096):
            block = slice(begin, begin + 4096)
            t = t_values[block, None, None]
            a = first[None, :, :] + t * (last - first)[None, :, :]
            b = np.roll(a, -1, axis=1)
            px, pz = x_values[block, None], z_values[block, None]
            ax, az = a[:, :, 0], a[:, :, 1]
            ex, ez = b[:, :, 0] - ax, b[:, :, 1] - az
            along = np.clip(
                ((px - ax) * ex + (pz - az) * ez) / np.maximum(ex * ex + ez * ez, 1e-12),
                0.0, 1.0,
            )
            unsigned = np.sqrt(np.min((px - ax - along * ex)**2 + (pz - az - along * ez)**2, axis=1))
            crosses = ((az > pz) != (b[:, :, 1] > pz)) & (
                px < ax + ex * (pz - az) / np.where(np.abs(ez) < 1e-12, 1e-12, ez)
            )
            inside = np.count_nonzero(crosses, axis=1) % 2 == 1
            distances[block] = np.where(inside, -unsigned, unsigned)
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
            terrain_gain = (
                self.config.roof_roughness_scale * (1.0 - floor_weight)
                + self.config.floor_roughness_scale * floor_weight
            )

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
        voxel_grid.synchronize_halos()
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
        mesh_ids = np.concatenate(
            [
                np.full(len(mesh.vertices), mesh_index, dtype=np.int64)
                for mesh_index, mesh in enumerate(chunk_meshes)
            ]
        )
        parent = np.arange(len(positions))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        # Shared scalar samples already agree. Only reconcile float32 local
        # marching-cubes roundoff; a fraction-of-voxel radius can incorrectly
        # join different grid-edge intersections near a tangential surface.
        largest_extent = max(
            mesh.grid_bounds[axis + 1] - mesh.grid_bounds[axis]
            for mesh in chunk_meshes for axis in (0, 2, 4)
        ) * self.config.voxel_size
        seam_tolerance = max(
            self.config.weld_tolerance,
            4.0 * np.finfo(np.float32).eps * largest_extent,
            1e-12,
        )
        pairs = cKDTree(positions).query_pairs(seam_tolerance, output_type="ndarray")
        # Reconcile the closest copies first. A cross-chunk bridge must never
        # merge two distinct vertices from one original chunk transitively:
        # doing that deletes thin triangles and opens holes at event surfaces.
        distances = np.sum((positions[pairs[:, 0]] - positions[pairs[:, 1]]) ** 2, axis=1)
        pairs = pairs[np.argsort(distances, kind="stable")]
        cluster_chunks: dict[int, set[int]] = {}
        cluster_members: dict[int, list[int]] = {}
        for a, b in pairs:
            # Marching cubes may emit distinct, extremely close vertices for a
            # thin but valid feature.  Collapsing those vertices deletes its
            # triangles and opens a hole.  Welding is only required to join
            # duplicate vertices emitted by adjacent chunks.
            if mesh_ids[int(a)] == mesh_ids[int(b)]:
                continue
            ra, rb = root(int(a)), root(int(b))
            if ra == rb:
                continue
            first_chunks = cluster_chunks.get(ra, {int(mesh_ids[ra])})
            second_chunks = cluster_chunks.get(rb, {int(mesh_ids[rb])})
            shared_chunks = first_chunks & second_chunks
            first_members = cluster_members.get(ra, [ra])
            second_members = cluster_members.get(rb, [rb])
            if shared_chunks:
                # MC can emit several copies of an exact iso-level corner.
                # Those may coincide within the explicit numerical tolerance;
                # only the larger cross-seam search radius is forbidden here.
                duplicate_distances = [
                    float(np.linalg.norm(positions[x] - positions[y]))
                    for x in first_members for y in second_members
                    if mesh_ids[x] == mesh_ids[y]
                ]
                if max(duplicate_distances, default=0.0) > max(self.config.weld_tolerance, 1e-12):
                    continue
            keep, remove = min(ra, rb), max(ra, rb)
            parent[remove] = keep
            cluster_chunks[keep] = first_chunks | second_chunks
            cluster_chunks.pop(remove, None)
            cluster_members[keep] = first_members + second_members
            cluster_members.pop(remove, None)
        roots = np.asarray([root(index) for index in range(len(positions))])
        representatives, inverse = np.unique(roots, return_inverse=True)
        vertices = positions[representatives]
        faces_by_signature: dict[
            tuple[int, int, int],
            tuple[int, int, int],
        ] = {}
        offset = 0
        for mesh in chunk_meshes:
            for a, b, c in mesh.faces:
                face = (
                    int(inverse[offset + a]),
                    int(inverse[offset + b]),
                    int(inverse[offset + c]),
                )
                if len(set(face)) != 3:
                    continue
                sorted_face = sorted(face)
                signature = (sorted_face[0], sorted_face[1], sorted_face[2])
                existing = faces_by_signature.get(signature)
                if existing is None:
                    faces_by_signature[signature] = face
                    continue
                same_winding = face in (
                    existing,
                    (existing[1], existing[2], existing[0]),
                    (existing[2], existing[0], existing[1]),
                )
                if not same_winding:
                    # The chunks marched opposite sides of their overlapping
                    # interface. Both coincident triangles are internal and
                    # must cancel; retaining either leaves a three-face edge.
                    faces_by_signature.pop(signature)
            offset += len(mesh.vertices)
        faces = tuple(faces_by_signature.values())
        return (
            tuple(
                (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                for vertex in vertices
            ),
            faces,
        )

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
