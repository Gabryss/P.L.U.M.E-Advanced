"""Stage E geological event and coarse debris generation."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.stages.floor_map import FloorAtlas, FloorCell
from plume_advanced.stages.section_field import SectionField, SectionSample

EventProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class GeologicalEventConfig:
    """Parameters controlling mesh-stage geological event placement."""

    random_seed: int | None = None
    enabled: bool = True
    include_rock_props: bool = True
    enabled_kinds: tuple[str, ...] = ("rock", "boulder", "collapse", "choke", "infill")
    rock_population_multiplier: float = 10.0
    debris_density_basis: str = "floor_area"
    rock_density_per_100m: float = 2.00
    boulder_density_per_100m: float = 0.18
    rock_density_per_100m2: float = 0.45
    boulder_density_per_100m2: float = 0.04
    geological_event_density_per_100m: float = 0.12
    collapse_event_fraction: float = 0.35
    choke_event_fraction: float = 0.30
    infill_event_fraction: float = 0.35
    minimum_event_spacing: float = 22.0
    minimum_rock_spacing: float = 0.10
    minimum_boulder_spacing: float = 0.35
    rock_radius_range: tuple[float, float] = (0.03, 0.75)
    boulder_radius_range: tuple[float, float] = (0.30, 4.0)
    rock_size_bias: float = 2.8
    boulder_size_bias: float = 2.4
    collapse_radius_range: tuple[float, float] = (4.0, 9.0)
    choke_radius_range: tuple[float, float] = (3.0, 7.0)
    infill_radius_range: tuple[float, float] = (5.0, 12.0)
    max_lateral_floor_fraction: float = 0.62
    mesh_latitude_segments: int = 8
    mesh_longitude_segments: int = 14
    use_rocky_meshes: bool = True
    rocky_source_path: str = ""
    rocky_texture_dir: str = ""
    rocky_output_dir: str = "outputs/rocky_stage_e"
    rocky_resolution_scale: float = 1.0
    rocky_max_subdivisions: int = 5
    strict_optional_provider: bool = True
    gravity_m_s2: float = 9.80665
    rock_density_kg_m3: float = 2_900.0
    effective_tensile_strength_pa: float = 3_000_000.0
    ground_embed_fraction: float = 0.08
    clustered_debris_fraction: float = 0.45
    collapse_cluster_radius_scale: float = 4.0
    collapse_cluster_spacing_scale: float = 0.90
    gallery_width_size_fraction: float = 0.30
    gallery_clearance_size_fraction: float = 0.45
    boulder_max_height_fraction: float = 2.0 / 3.0
    roof_block_size_fraction: float = 0.35
    edge_accumulation_strength: float = 0.70
    placement_jitter_m: float = 0.75
    background_contact_spacing: float = 1.15
    rover_width_m: float = 1.0
    rover_side_margin_m: float = 0.10
    rover_max_lateral_slope: float = 0.50
    preserve_rover_route: bool = True
    enable_debris_families: bool = True
    clean_floor_fraction: float = 0.45
    debris_patch_length_m: float = 30.0
    wall_scree_fraction: float = 0.35
    transported_lag_fraction: float = 0.15
    boulder_satellite_count_range: tuple[int, int] = (25, 55)
    boulder_halo_radius_range_m: tuple[float, float] = (1.2, 4.5)
    collapse_fragment_count_range: tuple[int, int] = (50, 120)
    collapse_talus_radius_range_m: tuple[float, float] = (5.0, 15.0)
    minor_cluster_density_per_1000m2: float = 0.80
    minor_cluster_count_range: tuple[int, int] = (8, 18)
    minor_cluster_radius_range_m: tuple[float, float] = (1.2, 3.0)


@dataclass(frozen=True)
class GeologicalEvent:
    """One Stage-E event that can influence generated mesh geometry."""

    event_id: int
    kind: str
    segment_id: int
    sample_index: int
    x: float
    y: float
    z: float
    surface_z: float
    floor_z: float
    radius_x: float
    radius_y: float
    radius_z: float
    angle: float
    severity: float
    material_hint: str
    contact_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    grounded: bool = False
    floor_cell_id: int = -1
    cluster_parent_event_id: int = -1
    local_size_cap_m: float = 0.0
    lateral_offset_m: float = 0.0
    source_distance_m: float = 0.0
    rover_bypass_m: float = math.inf
    debris_family_id: int = -1
    family_anchor_event_id: int = -1
    debris_role: str = ""

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def max_radius(self) -> float:
        return max(self.radius_x, self.radius_y, self.radius_z)

    @property
    def is_structural_modifier(self) -> bool:
        return self.kind in {"collapse", "choke", "infill"}


@dataclass(frozen=True)
class GeologicalEventMesh:
    """One placed Stage-E mesh in world coordinates."""

    event_id: int
    kind: str
    material_hint: str
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    face_uvs: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = ()
    material_maps: tuple[tuple[str, str], ...] = ()
    source_generator: str = "native"
    source_shape_type: str = ""
    debris_family_id: int = -1
    family_anchor_event_id: int = -1
    debris_role: str = ""

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)


@dataclass(frozen=True)
class GeologicalEventField:
    """Stage-E event output used by visualization, geometry, and later texturing."""

    config: GeologicalEventConfig
    events: tuple[GeologicalEvent, ...]
    meshes: tuple[GeologicalEventMesh, ...] = ()
    rover_route_valid: bool = True
    minimum_rover_bypass_m: float = math.inf

    def summary(self) -> dict[str, float]:
        counts = {kind: 0 for kind in ("rock", "boulder", "collapse", "choke", "infill")}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        props = [event for event in self.events if event.kind in {"rock", "boulder"}]
        footprints = [2.0 * max(event.radius_x, event.radius_y) for event in props]
        family_props = [event for event in props if event.debris_family_id >= 0]
        return {
            "event_count": float(len(self.events)),
            "rock_count": float(counts.get("rock", 0)),
            "boulder_count": float(counts.get("boulder", 0)),
            "collapse_count": float(counts.get("collapse", 0)),
            "choke_count": float(counts.get("choke", 0)),
            "infill_count": float(counts.get("infill", 0)),
            "event_mesh_count": float(len(self.meshes)),
            "event_mesh_vertex_count": float(sum(mesh.vertex_count for mesh in self.meshes)),
            "event_mesh_face_count": float(sum(mesh.face_count for mesh in self.meshes)),
            "prop_count": float(counts.get("rock", 0) + counts.get("boulder", 0)),
            "structural_modifier_count": float(
                counts.get("collapse", 0) + counts.get("choke", 0) + counts.get("infill", 0)
            ),
            "grounded_prop_count": float(
                sum(event.grounded for event in self.events if event.kind in {"rock", "boulder"})
            ),
            "clustered_prop_count": float(
                sum(
                    event.cluster_parent_event_id >= 0
                    for event in self.events
                    if event.kind in {"rock", "boulder"}
                )
            ),
            "debris_family_count": float(len({event.debris_family_id for event in family_props})),
            "family_prop_count": float(len(family_props)),
            "boulder_satellite_count": float(
                sum(event.debris_role.startswith("boulder_") for event in props)
            ),
            "collapse_talus_prop_count": float(
                sum(event.debris_role.startswith("collapse_") for event in props)
            ),
            "minor_rubble_cluster_count": float(
                sum(event.debris_role == "minor_cluster_anchor" for event in props)
            ),
            "minor_rubble_cluster_prop_count": float(
                sum(event.debris_role.startswith("minor_cluster_") for event in props)
            ),
            "wall_scree_prop_count": float(
                sum(event.debris_role == "wall_scree" for event in props)
            ),
            "transported_lag_prop_count": float(
                sum(event.debris_role == "transported_lag" for event in props)
            ),
            "rover_scale_prop_count": float(
                sum(footprint >= self.config.rover_width_m for footprint in footprints)
            ),
            "mean_prop_footprint_m": (float(np.mean(footprints)) if footprints else 0.0),
            "maximum_prop_footprint_m": (float(max(footprints)) if footprints else 0.0),
            "minimum_rover_bypass_m": (
                float(self.minimum_rover_bypass_m)
                if math.isfinite(self.minimum_rover_bypass_m)
                else 0.0
            ),
            "rover_route_valid": float(self.rover_route_valid),
            "mean_severity": float(np.mean([event.severity for event in self.events]))
            if self.events
            else 0.0,
        }


class GeologicalEventGenerator:
    """Place Stage-E mesh events from host-aware section samples."""

    _GROUND_CONTACT_SLOTS_PER_CELL = 32

    def __init__(self, config: GeologicalEventConfig | None = None) -> None:
        self.config = config or GeologicalEventConfig()
        self._ground_contact_cache: dict[
            tuple[int, int, int],
            tuple[np.ndarray, np.ndarray],
        ] = {}
        self._candidate_weights_cache: dict[
            int,
            tuple[list[tuple[float, SectionSample, FloorCell]], np.ndarray],
        ] = {}
        self._runtime_index_active = False
        self._runtime_prop_bins: dict[
            tuple[int, int, int],
            list[GeologicalEvent],
        ] = {}
        self._runtime_props_by_segment: dict[int, list[GeologicalEvent]] = {}
        self._runtime_structural_events: list[GeologicalEvent] = []
        self._runtime_collapse_events: list[GeologicalEvent] = []
        self._runtime_samples_by_segment: dict[int, list[SectionSample]] = {}
        self._runtime_bin_size_m = max(
            2.0,
            2.0 * self.config.boulder_radius_range[1],
        )
        rocky_kinds_enabled = bool({"rock", "boulder"}.intersection(self.config.enabled_kinds))
        should_load_rocky = (
            self.config.enabled
            and self.config.include_rock_props
            and rocky_kinds_enabled
            and self.config.use_rocky_meshes
        )
        self._rocky_api = (
            _load_rocky_api(self.config.rocky_source_path) if should_load_rocky else None
        )
        if should_load_rocky and self._rocky_api is None and self.config.strict_optional_provider:
            raise RuntimeError(
                "Rocky meshes were requested but the optional Rocky provider "
                f"could not be imported from {self.config.rocky_source_path!r}"
            )

    def generate(
        self,
        section_field: SectionField,
        base_geometry: Any | None = None,
        floor_atlas: FloorAtlas | None = None,
        progress: EventProgressCallback | None = None,
    ) -> GeologicalEventField:
        self._ground_contact_cache.clear()
        self._candidate_weights_cache.clear()
        if not self.config.enabled or not self.config.enabled_kinds:
            return GeologicalEventField(config=self.config, events=(), meshes=())

        rng = np.random.default_rng(self.config.random_seed)
        samples = [
            sample
            for segment_field in section_field.segment_fields
            for sample in segment_field.samples
        ]
        if not samples:
            return GeologicalEventField(config=self.config, events=())
        self._reset_runtime_indexes(samples)

        events: list[GeologicalEvent] = []
        occupied_positions: list[tuple[np.ndarray, str]] = []
        voxel_grid = (
            getattr(base_geometry, "voxel_grid", None) if base_geometry is not None else None
        )
        (
            rock_count,
            boulder_count,
            collapse_count,
            choke_count,
            infill_count,
        ) = self._counts_from_density(samples, floor_atlas)
        self._emit_progress(
            progress,
            "planning",
            1,
            1,
            (f"requested {rock_count} background rocks and {boulder_count} boulders"),
        )
        recipes = tuple(
            recipe
            for recipe in (
                ("collapse", collapse_count, "roof_breakdown"),
                ("choke", choke_count, "constriction"),
                ("infill", infill_count, "floor_infill"),
                ("boulder", boulder_count, "large_breakdown"),
                ("rock", rock_count, "floor_debris"),
            )
            if recipe[0] in self.config.enabled_kinds
            and (self.config.include_rock_props or recipe[0] not in {"rock", "boulder"})
        )
        sample_lookup = {(sample.segment_id, sample.index): sample for sample in samples}
        for kind, count, material_hint in recipes:
            prop_candidates = (
                self._rank_floor_candidates(
                    kind,
                    floor_atlas,
                    sample_lookup,
                )
                if kind in {"rock", "boulder"} and floor_atlas is not None
                else None
            )
            if prop_candidates is not None and not (
                kind == "rock" and self.config.enable_debris_families
            ):
                prop_candidates = self._filter_structural_conflicts(
                    prop_candidates,
                    events,
                )
            candidates = self._rank_candidates(kind, samples) if prop_candidates is None else []
            cluster_candidates = (
                self._collapse_cluster_candidates(prop_candidates, events)
                if prop_candidates is not None
                else []
            )
            if (
                kind == "rock"
                and prop_candidates is not None
                and self.config.enable_debris_families
            ):
                placed = self._place_rock_populations(
                    background_count=count,
                    candidates=prop_candidates,
                    events=events,
                    sample_lookup=sample_lookup,
                    voxel_grid=voxel_grid,
                    rng=rng,
                    material_hint=material_hint,
                    progress=progress,
                )
                events.extend(placed)
                occupied_positions.extend(
                    (np.asarray(event.position, dtype=float), event.kind) for event in placed
                )
                continue
            cluster_target = int(round(max(count, 0) * self.config.clustered_debris_fraction))
            accepted_for_kind = 0
            for placement_index in range(max(count, 0)):
                floor_cell = None
                if prop_candidates is not None:
                    use_cluster_pool = placement_index < cluster_target and bool(cluster_candidates)
                    event = self._choose_and_build_prop(
                        kind,
                        rng,
                        (cluster_candidates if use_cluster_pool else prop_candidates),
                        events,
                        sample_lookup,
                        voxel_grid,
                        material_hint,
                    )
                    if event is None and use_cluster_pool:
                        event = self._choose_and_build_prop(
                            kind,
                            rng,
                            prop_candidates,
                            events,
                            sample_lookup,
                            voxel_grid,
                            material_hint,
                        )
                    if event is None:
                        self._emit_progress(
                            progress,
                            kind,
                            count,
                            count,
                            (f"capacity reached after {accepted_for_kind} accepted {kind} anchors"),
                        )
                        break
                    events.append(event)
                    self._register_runtime_event(event)
                    accepted_for_kind += 1
                    occupied_positions.append((np.asarray(event.position, dtype=float), kind))
                    self._emit_progress(
                        progress,
                        kind,
                        placement_index + 1,
                        count,
                        f"{accepted_for_kind} accepted {kind} anchors",
                    )
                    continue
                else:
                    sample = self._choose_candidate(
                        kind,
                        rng,
                        candidates,
                        occupied_positions,
                    )
                if sample is None:
                    self._emit_progress(
                        progress,
                        kind,
                        count,
                        count,
                        (f"capacity reached after {accepted_for_kind} accepted {kind} events"),
                    )
                    break
                event = self._build_event(
                    event_id=len(events),
                    kind=kind,
                    material_hint=material_hint,
                    sample=sample,
                    rng=rng,
                    voxel_grid=voxel_grid,
                    floor_cell=floor_cell,
                    cluster_parent_event_id=(
                        self._nearest_collapse_event_id(floor_cell, events)
                        if floor_cell is not None
                        else -1
                    ),
                )
                events.append(event)
                self._register_runtime_event(event)
                accepted_for_kind += 1
                occupied_positions.append(
                    (np.array((event.x, event.y, event.z), dtype=float), kind)
                )
                self._emit_progress(
                    progress,
                    kind,
                    placement_index + 1,
                    count,
                    f"{accepted_for_kind} accepted {kind} events",
                )

        events.sort(key=lambda event: (event.segment_id, event.sample_index, event.event_id))
        event_id_remap = {event.event_id: index for index, event in enumerate(events)}
        normalized_events = tuple(
            GeologicalEvent(
                event_id=index,
                kind=event.kind,
                segment_id=event.segment_id,
                sample_index=event.sample_index,
                x=event.x,
                y=event.y,
                z=event.z,
                surface_z=event.surface_z,
                floor_z=event.floor_z,
                radius_x=event.radius_x,
                radius_y=event.radius_y,
                radius_z=event.radius_z,
                angle=event.angle,
                severity=event.severity,
                material_hint=event.material_hint,
                contact_point=event.contact_point,
                contact_normal=event.contact_normal,
                grounded=event.grounded,
                floor_cell_id=event.floor_cell_id,
                cluster_parent_event_id=event_id_remap.get(
                    event.cluster_parent_event_id,
                    -1,
                ),
                local_size_cap_m=event.local_size_cap_m,
                lateral_offset_m=event.lateral_offset_m,
                source_distance_m=event.source_distance_m,
                rover_bypass_m=event.rover_bypass_m,
                debris_family_id=event_id_remap.get(
                    event.debris_family_id,
                    -1,
                ),
                family_anchor_event_id=event_id_remap.get(
                    event.family_anchor_event_id,
                    -1,
                ),
                debris_role=event.debris_role,
            )
            for index, event in enumerate(events)
        )
        route_valid, minimum_bypass = self._rover_route_status(
            section_field,
            normalized_events,
        )
        props = [event for event in normalized_events if event.kind in {"rock", "boulder"}]
        meshes: list[GeologicalEventMesh] = []
        for index, event in enumerate(props, start=1):
            meshes.append(self._build_event_mesh(event))
            self._emit_progress(
                progress,
                "meshes",
                index,
                len(props),
                f"built {index} / {len(props)} Rocky meshes",
            )
        return GeologicalEventField(
            config=self.config,
            events=normalized_events,
            meshes=tuple(meshes),
            rover_route_valid=route_valid,
            minimum_rover_bypass_m=minimum_bypass,
        )

    def _counts_from_density(
        self,
        samples: list[SectionSample],
        floor_atlas: FloorAtlas | None = None,
    ) -> tuple[int, int, int, int, int]:
        total_length = self._total_sampled_length(samples)
        length_units = total_length / 100.0
        population_multiplier = (
            self.config.rock_population_multiplier if self.config.include_rock_props else 0.0
        )
        if self.config.debris_density_basis == "floor_area":
            floor_area = self._sampled_floor_area(samples)
            area_units = floor_area / 100.0
            rock_count = max(
                0,
                int(round(self.config.rock_density_per_100m2 * area_units * population_multiplier)),
            )
            boulder_count = max(
                0,
                int(
                    round(
                        self.config.boulder_density_per_100m2 * area_units * population_multiplier
                    )
                ),
            )
        else:
            rock_count = max(
                0,
                int(
                    round(self.config.rock_density_per_100m * length_units * population_multiplier)
                ),
            )
            boulder_count = max(
                0,
                int(
                    round(
                        self.config.boulder_density_per_100m * length_units * population_multiplier
                    )
                ),
            )
        geological_event_count = max(
            0, int(round(self.config.geological_event_density_per_100m * length_units))
        )
        fractions = np.array(
            (
                max(self.config.collapse_event_fraction, 0.0),
                max(self.config.choke_event_fraction, 0.0),
                max(self.config.infill_event_fraction, 0.0),
            ),
            dtype=float,
        )
        fractions /= max(float(fractions.sum()), 1e-9)
        raw_counts = fractions * geological_event_count
        counts = np.floor(raw_counts).astype(int)
        remainder = geological_event_count - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw_counts - counts))
            for index in order[:remainder]:
                counts[index] += 1
        return rock_count, boulder_count, int(counts[0]), int(counts[1]), int(counts[2])

    @staticmethod
    def _sampled_floor_area(samples: list[SectionSample]) -> float:
        """Integrate section width along every segment as usable floor area."""

        samples_by_segment: dict[int, list[SectionSample]] = {}
        for sample in samples:
            samples_by_segment.setdefault(sample.segment_id, []).append(sample)
        total = 0.0
        for segment_samples in samples_by_segment.values():
            ordered = sorted(
                segment_samples,
                key=lambda sample: sample.segment_arc_length,
            )
            for first, second in zip(ordered, ordered[1:]):
                distance = max(
                    second.segment_arc_length - first.segment_arc_length,
                    0.0,
                )
                total += 0.5 * (max(first.tube_width, 0.0) + max(second.tube_width, 0.0)) * distance
        return total

    @staticmethod
    def _total_sampled_length(samples: list[SectionSample]) -> float:
        samples_by_segment: dict[int, list[SectionSample]] = {}
        for sample in samples:
            samples_by_segment.setdefault(sample.segment_id, []).append(sample)
        total = 0.0
        for segment_samples in samples_by_segment.values():
            ordered = sorted(segment_samples, key=lambda sample: sample.segment_arc_length)
            if not ordered:
                continue
            total += max(sample.segment_arc_length for sample in ordered)
        return total

    def _rank_candidates(
        self,
        kind: str,
        samples: list[SectionSample],
    ) -> list[tuple[float, SectionSample]]:
        scored: list[tuple[float, SectionSample]] = []
        for sample in samples:
            if sample.tube_width <= 0.0 or sample.tube_height <= 0.0:
                continue
            weak_roof_score = 1.0 / max(sample.roof_thickness, 1.0)
            junction_score = sample.junction_blend_weight
            width_score = sample.tube_width
            narrow_score = 1.0 / max(sample.tube_width, 1.0)
            stability_demand = self._roof_demand_ratio(sample)
            if kind == "collapse":
                score = (
                    2.0 * weak_roof_score
                    + 1.2 * junction_score
                    + 0.05 * width_score
                    + 2.5 * min(stability_demand, 4.0)
                )
            elif kind == "choke":
                score = 2.0 * narrow_score + 0.8 * junction_score
            elif kind == "infill":
                score = 0.08 * width_score + 0.5 * junction_score + 0.3 * sample.floor_flatness
            elif kind == "boulder":
                score = 0.06 * width_score + 0.7 * junction_score + 0.7 * weak_roof_score
            else:
                score = 0.04 * width_score + 0.25 * junction_score
            scored.append((max(score, 1e-6), sample))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _rank_floor_candidates(
        self,
        kind: str,
        floor_atlas: FloorAtlas,
        sample_lookup: dict[tuple[int, int], SectionSample],
    ) -> list[tuple[float, SectionSample, FloorCell]]:
        """Rank actual floor area instead of section centerlines."""

        scored: list[tuple[float, SectionSample, FloorCell]] = []
        centerline_scores = {
            (sample.segment_id, sample.index): score
            for score, sample in self._rank_candidates(
                kind,
                list(sample_lookup.values()),
            )
        }
        for cell in floor_atlas.cells:
            if not cell.grounded:
                continue
            key = (cell.segment_id, cell.sample_index)
            sample = sample_lookup.get(key)
            if sample is None:
                continue
            edge_fraction = abs(cell.lateral_offset_m) / max(
                0.5 * cell.tube_width_m,
                1e-6,
            )
            edge_fraction = float(np.clip(edge_fraction, 0.0, 1.0))
            edge_affinity = 1.0 + self.config.edge_accumulation_strength * (
                edge_fraction**1.5 - 0.35
            )
            clearance_factor = float(
                np.clip(cell.clearance_m / max(sample.tube_height, 1.0), 0.4, 1.4)
            )
            score = centerline_scores.get(key, 1e-6) * max(edge_affinity, 0.2)
            if kind == "boulder":
                gallery_capacity = math.sqrt(max(cell.tube_width_m * cell.clearance_m, 1.0))
                score *= clearance_factor * gallery_capacity
            scored.append((max(score, 1e-6), sample, cell))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _collapse_cluster_candidates(
        self,
        candidates: list[tuple[float, SectionSample, FloorCell]],
        events: list[GeologicalEvent],
    ) -> list[tuple[float, SectionSample, FloorCell]]:
        collapses = [event for event in events if event.kind == "collapse"]
        if not collapses:
            return []
        clustered: list[tuple[float, SectionSample, FloorCell]] = []
        for score, sample, cell in candidates:
            position = np.asarray(cell.position, dtype=float)
            affinity = 0.0
            for collapse in collapses:
                distance = float(
                    np.linalg.norm(position - np.asarray(collapse.position, dtype=float))
                )
                radius = max(
                    collapse.max_radius * self.config.collapse_cluster_radius_scale,
                    1.0,
                )
                if distance <= radius:
                    affinity = max(affinity, math.exp(-((distance / radius) ** 2)))
            if affinity > 0.0:
                clustered.append((score * (1.0 + 4.0 * affinity), sample, cell))
        clustered.sort(key=lambda item: item[0], reverse=True)
        return clustered

    @staticmethod
    def _filter_structural_conflicts(
        candidates: list[tuple[float, SectionSample, FloorCell]],
        events: list[GeologicalEvent],
    ) -> list[tuple[float, SectionSample, FloorCell]]:
        """Keep props out of the volume that a structural event can replace."""

        structural_events = [
            event for event in events if event.kind in {"collapse", "choke", "infill"}
        ]
        if not structural_events:
            return candidates

        filtered: list[tuple[float, SectionSample, FloorCell]] = []
        for candidate in candidates:
            cell = candidate[2]
            position = np.asarray(cell.position, dtype=float)
            conflicts = any(
                float(np.linalg.norm(position - np.asarray(event.position, dtype=float)))
                < (event.max_radius * (1.50 if event.kind == "collapse" else 1.25))
                for event in structural_events
            )
            if not conflicts:
                filtered.append(candidate)
        return filtered

    def _reset_runtime_indexes(self, samples: list[SectionSample]) -> None:
        """Initialize spatial indexes used only during one generation run."""

        self._runtime_index_active = True
        self._runtime_prop_bins.clear()
        self._runtime_props_by_segment.clear()
        self._runtime_structural_events.clear()
        self._runtime_collapse_events.clear()
        self._runtime_samples_by_segment.clear()
        for sample in samples:
            self._runtime_samples_by_segment.setdefault(
                sample.segment_id,
                [],
            ).append(sample)
        for segment_samples in self._runtime_samples_by_segment.values():
            segment_samples.sort(key=lambda sample: sample.segment_arc_length)

    def _register_runtime_event(self, event: GeologicalEvent) -> None:
        if not self._runtime_index_active:
            return
        if event.kind in {"rock", "boulder"}:
            self._runtime_props_by_segment.setdefault(
                event.segment_id,
                [],
            ).append(event)
            position = np.asarray(event.contact_point, dtype=float)
            key = tuple(
                int(math.floor(float(value) / self._runtime_bin_size_m)) for value in position
            )
            self._runtime_prop_bins.setdefault(key, []).append(event)
        elif event.is_structural_modifier:
            self._runtime_structural_events.append(event)
            if event.kind == "collapse":
                self._runtime_collapse_events.append(event)

    def _nearby_runtime_events(
        self,
        event: GeologicalEvent,
    ) -> list[GeologicalEvent]:
        position = np.asarray(event.contact_point, dtype=float)
        radius = max(event.radius_x, event.radius_y)
        global_radius = max(
            self.config.rock_radius_range[1],
            self.config.boulder_radius_range[1],
        )
        contact_factor = max(
            self.config.background_contact_spacing,
            self.config.collapse_cluster_spacing_scale,
            1.0,
        )
        search_radius = max(
            self.config.minimum_rock_spacing,
            self.config.minimum_boulder_spacing,
            contact_factor * (radius + global_radius),
        )
        center_key = tuple(
            int(math.floor(float(value) / self._runtime_bin_size_m)) for value in position
        )
        reach = max(
            1,
            int(math.ceil(search_radius / self._runtime_bin_size_m)),
        )
        nearby = list(self._runtime_structural_events)
        for x_offset in range(-reach, reach + 1):
            for y_offset in range(-reach, reach + 1):
                for z_offset in range(-reach, reach + 1):
                    key = (
                        center_key[0] + x_offset,
                        center_key[1] + y_offset,
                        center_key[2] + z_offset,
                    )
                    nearby.extend(self._runtime_prop_bins.get(key, ()))
        return nearby

    def _nearest_collapse_event_id(
        self,
        floor_cell: FloorCell,
        events: list[GeologicalEvent],
    ) -> int:
        position = np.asarray(floor_cell.position, dtype=float)
        nearest_id = -1
        nearest_distance = math.inf
        collapse_events = self._runtime_collapse_events if self._runtime_index_active else events
        for event in collapse_events:
            if event.kind != "collapse":
                continue
            distance = float(np.linalg.norm(position - np.asarray(event.position, dtype=float)))
            radius = max(
                event.max_radius * self.config.collapse_cluster_radius_scale,
                1.0,
            )
            if distance <= radius and distance < nearest_distance:
                nearest_id = event.event_id
                nearest_distance = distance
        return nearest_id

    def _roof_demand_ratio(self, sample: SectionSample) -> float:
        """Fast body/material-aware collapse surrogate for candidate ranking."""

        demand = (
            self.config.rock_density_kg_m3
            * self.config.gravity_m_s2
            * sample.tube_width
            * sample.tube_width
            / max(sample.roof_thickness, 0.1)
        )
        return float(demand / max(self.config.effective_tensile_strength_pa, 1.0))

    def _place_rock_populations(
        self,
        *,
        background_count: int,
        candidates: list[tuple[float, SectionSample, FloorCell]],
        events: list[GeologicalEvent],
        sample_lookup: dict[tuple[int, int], SectionSample],
        voxel_grid: Any | None,
        rng: np.random.Generator,
        material_hint: str,
        progress: EventProgressCallback | None = None,
    ) -> list[GeologicalEvent]:
        """Build compact geological families plus patchy background deposits."""

        placed: list[GeologicalEvent] = []
        all_events = list(events)
        completed = 0
        accepted = 0
        anchors = [event for event in events if event.kind in {"boulder", "collapse"}]
        family_plans: list[tuple[GeologicalEvent, list[str]]] = []
        for anchor in anchors:
            if anchor.kind == "boulder":
                family_count = self._boulder_satellite_count(anchor, rng)
                roles = self._role_sequence(
                    family_count,
                    (
                        ("boulder_inner_rubble", 0.45),
                        ("boulder_small_rubble", 0.30),
                        ("boulder_companion", 0.20),
                        ("boulder_runout", 0.05),
                    ),
                )
            else:
                family_count = self._collapse_fragment_count(anchor)
                roles = self._role_sequence(
                    family_count,
                    (
                        ("collapse_rubble", 0.65),
                        ("collapse_companion", 0.25),
                        ("collapse_block", 0.10),
                    ),
                )
            family_plans.append((anchor, roles))
        minor_cluster_count = self._minor_cluster_count(sample_lookup.values())
        progress_total = (
            sum(len(roles) for _anchor, roles in family_plans)
            + minor_cluster_count
            + max(background_count, 0)
        )
        self._emit_progress(
            progress,
            "props",
            0,
            progress_total,
            f"0 accepted; planning {progress_total} initial prop slots",
        )
        for anchor, roles in family_plans:
            role_candidate_cache: dict[
                str,
                list[tuple[float, SectionSample, FloorCell]],
            ] = {}
            for role in roles:
                if role not in role_candidate_cache:
                    role_candidate_cache[role] = self._family_candidates(
                        candidates,
                        anchor,
                        role,
                    )
                family_candidates = role_candidate_cache[role]
                event = self._choose_and_build_prop(
                    "rock",
                    rng,
                    family_candidates,
                    all_events,
                    sample_lookup,
                    voxel_grid,
                    material_hint,
                    family_anchor_event=anchor,
                    debris_family_id=anchor.event_id,
                    debris_role=role,
                    prop_diameter_range=self._role_diameter_range(role, anchor),
                )
                completed += 1
                if event is not None:
                    placed.append(event)
                    all_events.append(event)
                    self._register_runtime_event(event)
                    accepted += 1
                self._emit_progress(
                    progress,
                    "props",
                    completed,
                    progress_total,
                    (f"{accepted} accepted, {completed - accepted} rejected; {role}"),
                )

        background_candidates = self._filter_structural_conflicts(
            candidates,
            events,
        )
        anchor_candidates = self._background_role_candidates(
            background_candidates,
            "minor_cluster_anchor",
        )
        for _cluster_index in range(minor_cluster_count):
            anchor = self._choose_and_build_prop(
                "rock",
                rng,
                anchor_candidates,
                all_events,
                sample_lookup,
                voxel_grid,
                material_hint,
                debris_role="minor_cluster_anchor",
                prop_diameter_range=self._role_diameter_range(
                    "minor_cluster_anchor",
                    None,
                ),
            )
            completed += 1
            if anchor is None:
                self._emit_progress(
                    progress,
                    "props",
                    completed,
                    progress_total,
                    (f"{accepted} accepted, {completed - accepted} rejected; minor cluster anchor"),
                )
                continue
            anchor = replace(
                anchor,
                debris_family_id=anchor.event_id,
                family_anchor_event_id=anchor.event_id,
            )
            placed.append(anchor)
            all_events.append(anchor)
            self._register_runtime_event(anchor)
            accepted += 1
            child_count = int(
                rng.integers(
                    self.config.minor_cluster_count_range[0],
                    self.config.minor_cluster_count_range[1] + 1,
                )
            )
            progress_total += child_count
            self._emit_progress(
                progress,
                "props",
                completed,
                progress_total,
                f"{accepted} accepted; minor cluster anchor",
            )
            roles = self._role_sequence(
                child_count,
                (
                    ("minor_cluster_rubble", 0.60),
                    ("minor_cluster_companion", 0.30),
                    ("minor_cluster_runout", 0.10),
                ),
            )
            role_candidate_cache: dict[
                str,
                list[tuple[float, SectionSample, FloorCell]],
            ] = {}
            for role in roles:
                if role not in role_candidate_cache:
                    role_candidate_cache[role] = self._family_candidates(
                        candidates,
                        anchor,
                        role,
                    )
                family_candidates = role_candidate_cache[role]
                child = self._choose_and_build_prop(
                    "rock",
                    rng,
                    family_candidates,
                    all_events,
                    sample_lookup,
                    voxel_grid,
                    material_hint,
                    family_anchor_event=anchor,
                    debris_family_id=anchor.event_id,
                    debris_role=role,
                    prop_diameter_range=self._role_diameter_range(role, anchor),
                )
                completed += 1
                if child is not None:
                    placed.append(child)
                    all_events.append(child)
                    self._register_runtime_event(child)
                    accepted += 1
                self._emit_progress(
                    progress,
                    "props",
                    completed,
                    progress_total,
                    (f"{accepted} accepted, {completed - accepted} rejected; {role}"),
                )

        background_roles = self._role_sequence(
            max(background_count, 0),
            (
                ("wall_scree", self.config.wall_scree_fraction),
                ("transported_lag", self.config.transported_lag_fraction),
                (
                    "background_scatter",
                    max(
                        0.0,
                        1.0
                        - self.config.wall_scree_fraction
                        - self.config.transported_lag_fraction,
                    ),
                ),
            ),
        )
        role_candidate_cache: dict[
            str,
            list[tuple[float, SectionSample, FloorCell]],
        ] = {}
        for role in background_roles:
            if role not in role_candidate_cache:
                role_candidate_cache[role] = self._background_role_candidates(
                    background_candidates,
                    role,
                )
            role_candidates = role_candidate_cache[role]
            event = self._choose_and_build_prop(
                "rock",
                rng,
                role_candidates,
                all_events,
                sample_lookup,
                voxel_grid,
                material_hint,
                debris_role=role,
                prop_diameter_range=self._role_diameter_range(role, None),
            )
            completed += 1
            if event is not None:
                placed.append(event)
                all_events.append(event)
                self._register_runtime_event(event)
                accepted += 1
            self._emit_progress(
                progress,
                "props",
                completed,
                progress_total,
                (f"{accepted} accepted, {completed - accepted} rejected; {role}"),
            )

        requested_accepted_target = progress_total
        shortfall = max(0, requested_accepted_target - accepted)
        if shortfall > 0:
            recovery_candidates = role_candidate_cache.get(
                "background_scatter",
                background_candidates,
            )
            recovery_budget = int(math.ceil(1.5 * shortfall))
            progress_total += recovery_budget
            for recovery_index in range(recovery_budget):
                if accepted >= requested_accepted_target:
                    completed += recovery_budget - recovery_index
                    self._emit_progress(
                        progress,
                        "props",
                        completed,
                        progress_total,
                        (f"target reached: {accepted} accepted, {completed - accepted} rejected"),
                    )
                    break
                event = self._choose_and_build_prop(
                    "rock",
                    rng,
                    recovery_candidates,
                    all_events,
                    sample_lookup,
                    voxel_grid,
                    material_hint,
                    debris_role="dense_micro_debris",
                    prop_diameter_range=self._role_diameter_range(
                        "dense_micro_debris",
                        None,
                    ),
                )
                completed += 1
                if event is not None:
                    placed.append(event)
                    all_events.append(event)
                    self._register_runtime_event(event)
                    accepted += 1
                self._emit_progress(
                    progress,
                    "props",
                    completed,
                    progress_total,
                    (
                        f"{accepted} / {requested_accepted_target} accepted; "
                        "recovering rejected slots with micro debris"
                    ),
                )
        return placed

    def _minor_cluster_count(
        self,
        samples: Any,
    ) -> int:
        floor_area = self._sampled_floor_area(list(samples))
        return max(
            0,
            int(
                round(
                    self.config.minor_cluster_density_per_1000m2
                    * floor_area
                    / 1_000.0
                    * self.config.rock_population_multiplier
                )
            ),
        )

    def _boulder_satellite_count(
        self,
        anchor: GeologicalEvent,
        rng: np.random.Generator,
    ) -> int:
        minimum, maximum = self.config.boulder_satellite_count_range
        diameter = 2.0 * max(anchor.radius_x, anchor.radius_y)
        size_fraction = float(np.clip(diameter / 2.0, 0.0, 1.0))
        expected = minimum + (maximum - minimum) * size_fraction**1.5
        return int(np.clip(round(rng.normal(expected, 1.5)), minimum, maximum))

    def _collapse_fragment_count(self, collapse: GeologicalEvent) -> int:
        minimum, maximum = self.config.collapse_fragment_count_range
        volume = 4.0 * math.pi * collapse.radius_x * collapse.radius_y * collapse.radius_z / 3.0
        volume_fraction = 1.0 - math.exp(-volume / 500.0)
        base_count = minimum + (maximum - minimum) * volume_fraction
        return int(round(base_count * self.config.rock_population_multiplier))

    @staticmethod
    def _role_sequence(
        count: int,
        weighted_roles: tuple[tuple[str, float], ...],
    ) -> list[str]:
        if count <= 0 or not weighted_roles:
            return []
        weights = np.asarray(
            [max(weight, 0.0) for _role, weight in weighted_roles],
            dtype=float,
        )
        if float(weights.sum()) <= 0.0:
            return []
        raw = weights / float(weights.sum()) * count
        counts = np.floor(raw).astype(int)
        for index in np.argsort(-(raw - counts))[: count - int(counts.sum())]:
            counts[index] += 1
        roles: list[str] = []
        while len(roles) < count:
            for index, (role, _weight) in enumerate(weighted_roles):
                if counts[index] > 0:
                    roles.append(role)
                    counts[index] -= 1
        return roles

    def _background_role_candidates(
        self,
        candidates: list[tuple[float, SectionSample, FloorCell]],
        role: str,
    ) -> list[tuple[float, SectionSample, FloorCell]]:
        selected: list[tuple[float, SectionSample, FloorCell]] = []
        for score, sample, cell in candidates:
            if not self._debris_patch_is_active(sample):
                continue
            edge = abs(cell.lateral_offset_m) / max(
                0.5 * cell.tube_width_m,
                1e-6,
            )
            if role == "wall_scree":
                if edge < 0.42:
                    continue
                role_weight = 0.3 + 4.0 * edge * edge
            elif role == "transported_lag":
                if edge > 0.70:
                    continue
                flatness = 1.0 / (1.0 + max(cell.surface_slope_degrees, 0.0) / 8.0)
                role_weight = (1.25 - edge) * (0.5 + flatness)
            else:
                role_weight = 1.0
            selected.append((score * role_weight, sample, cell))
        selected.sort(key=lambda item: item[0], reverse=True)
        return selected

    def _debris_patch_is_active(self, sample: SectionSample) -> bool:
        patch_length = max(self.config.debris_patch_length_m, 1.0)
        patch_index = int(math.floor(sample.segment_arc_length / patch_length))
        seed = int(self.config.random_seed or 0)
        mixed = (
            (seed + 1) * 2_654_435_761
            + (sample.segment_id + 1) * 2_246_822_519
            + (patch_index + 1) * 3_266_489_917
        ) & 0xFFFFFFFF
        value = mixed / float(0xFFFFFFFF)
        return value >= self.config.clean_floor_fraction

    def _family_candidates(
        self,
        candidates: list[tuple[float, SectionSample, FloorCell]],
        anchor: GeologicalEvent,
        role: str,
    ) -> list[tuple[float, SectionSample, FloorCell]]:
        anchor_position = np.asarray(anchor.contact_point, dtype=float)
        anchor_radius = max(anchor.radius_x, anchor.radius_y)
        if anchor.kind == "boulder":
            minimum_reach, maximum_reach = self.config.boulder_halo_radius_range_m
            reach = float(
                np.clip(
                    3.0 * 2.0 * anchor_radius,
                    minimum_reach,
                    maximum_reach,
                )
            )
        elif anchor.kind == "collapse":
            minimum_reach, maximum_reach = self.config.collapse_talus_radius_range_m
            reach = float(
                np.clip(
                    1.9 * anchor.max_radius,
                    minimum_reach,
                    maximum_reach,
                )
            )
        else:
            minimum_reach, maximum_reach = self.config.minor_cluster_radius_range_m
            anchor_diameter = 2.0 * anchor_radius
            reach = float(
                np.clip(
                    4.0 * anchor_diameter,
                    minimum_reach,
                    maximum_reach,
                )
            )
        heading, side, _up = self._event_basis(anchor)
        selected: list[tuple[float, SectionSample, FloorCell]] = []
        for score, sample, cell in candidates:
            delta = np.asarray(cell.position, dtype=float) - anchor_position
            along = float(np.dot(delta, heading))
            across = float(np.dot(delta, side))
            distance = float(np.linalg.norm(delta))
            if distance < 0.75 * anchor_radius:
                continue
            elliptical_distance = math.sqrt(
                (along / max(reach, 1e-6)) ** 2 + (across / max(0.70 * reach, 1e-6)) ** 2
            )
            if elliptical_distance > 1.0:
                continue
            if role.endswith("inner_rubble") or role.endswith("rubble"):
                target = 0.28
                spread = 0.24
            elif role.endswith("companion") or role.endswith("block"):
                target = 0.52
                spread = 0.28
            else:
                target = 0.75
                spread = 0.28
            kernel = math.exp(-(((elliptical_distance - target) / spread) ** 2))
            runout_bias = 1.0 + 0.8 * max(along, 0.0) / reach if role.endswith("runout") else 1.0
            selected.append((score * (0.2 + 5.0 * kernel) * runout_bias, sample, cell))
        selected.sort(key=lambda item: item[0], reverse=True)
        return selected

    @staticmethod
    def _role_diameter_range(
        role: str,
        anchor: GeologicalEvent | None,
    ) -> tuple[float, float]:
        anchor_diameter = 2.0 * max(anchor.radius_x, anchor.radius_y) if anchor is not None else 1.0
        ranges = {
            "boulder_inner_rubble": (0.04, min(0.16, 0.24 * anchor_diameter)),
            "boulder_small_rubble": (0.10, min(0.30, 0.32 * anchor_diameter)),
            "boulder_companion": (0.20, min(0.65, 0.55 * anchor_diameter)),
            "boulder_runout": (0.05, min(0.25, 0.28 * anchor_diameter)),
            "collapse_rubble": (0.04, 0.20),
            "collapse_companion": (0.14, 0.55),
            "collapse_block": (0.35, 1.20),
            "minor_cluster_anchor": (0.28, 0.70),
            "minor_cluster_rubble": (0.05, 0.16),
            "minor_cluster_companion": (0.14, 0.36),
            "minor_cluster_runout": (0.05, 0.20),
            "wall_scree": (0.04, 0.30),
            "transported_lag": (0.04, 0.18),
            "background_scatter": (0.06, 0.32),
            "dense_micro_debris": (0.03, 0.12),
        }
        minimum, maximum = ranges.get(role, (0.06, 0.55))
        return minimum, max(minimum, maximum)

    def _choose_and_build_prop(
        self,
        kind: str,
        rng: np.random.Generator,
        candidates: list[tuple[float, SectionSample, FloorCell]],
        events: list[GeologicalEvent],
        sample_lookup: dict[tuple[int, int], SectionSample],
        voxel_grid: Any | None,
        material_hint: str,
        *,
        family_anchor_event: GeologicalEvent | None = None,
        debris_family_id: int = -1,
        debris_role: str = "",
        prop_diameter_range: tuple[float, float] | None = None,
    ) -> GeologicalEvent | None:
        """Jointly choose location and size, rejecting unsafe proposals."""

        if not candidates:
            return None
        cache_key = id(candidates)
        cached = self._candidate_weights_cache.get(cache_key)
        if cached is None or cached[0] is not candidates:
            weights = np.asarray(
                [score for score, _sample, _cell in candidates],
                dtype=float,
            )
            weights /= max(float(weights.sum()), 1e-9)
            self._candidate_weights_cache[cache_key] = (candidates, weights)
        else:
            weights = cached[1]
        attempts = min(256, max(64, len(candidates) // 4))
        for _attempt in range(attempts):
            index = int(rng.choice(len(candidates), p=weights))
            _score, sample, floor_cell = candidates[index]
            parent = self._nearest_collapse_event(floor_cell, events)
            event = self._build_event(
                event_id=len(events),
                kind=kind,
                material_hint=material_hint,
                sample=sample,
                rng=rng,
                voxel_grid=voxel_grid,
                floor_cell=floor_cell,
                cluster_parent_event_id=parent.event_id if parent is not None else -1,
                cluster_parent_event=parent,
                family_anchor_event_id=(
                    family_anchor_event.event_id if family_anchor_event is not None else -1
                ),
                debris_anchor_event=family_anchor_event,
                debris_family_id=debris_family_id,
                debris_role=debris_role,
                prop_diameter_range=prop_diameter_range,
            )
            if not self._prop_spacing_is_valid(event, events):
                continue
            if not self._prop_proxy_fits_cave(event, voxel_grid):
                continue
            route_valid, bypass = self._candidate_rover_route(
                event,
                events,
                sample_lookup,
            )
            if self.config.preserve_rover_route and not route_valid:
                continue
            return replace(event, rover_bypass_m=bypass)
        return None

    def _nearest_collapse_event(
        self,
        floor_cell: FloorCell,
        events: list[GeologicalEvent],
    ) -> GeologicalEvent | None:
        event_id = self._nearest_collapse_event_id(floor_cell, events)
        if self._runtime_index_active:
            return next(
                (event for event in self._runtime_collapse_events if event.event_id == event_id),
                None,
            )
        return next(
            (event for event in events if event.event_id == event_id),
            None,
        )

    def _prop_spacing_is_valid(
        self,
        event: GeologicalEvent,
        events: list[GeologicalEvent],
    ) -> bool:
        position = np.asarray(event.contact_point, dtype=float)
        radius = max(event.radius_x, event.radius_y)
        comparison_events = (
            self._nearby_runtime_events(event) if self._runtime_index_active else events
        )
        for other in comparison_events:
            distance = float(
                np.linalg.norm(position - np.asarray(other.contact_point, dtype=float))
            )
            if other.kind not in {"rock", "boulder"}:
                if other.is_structural_modifier:
                    own_collapse = (
                        other.kind == "collapse" and other.event_id == event.family_anchor_event_id
                    )
                    structural_factor = 1.05 if own_collapse else 1.25
                    if distance < structural_factor * other.max_radius + radius:
                        return False
                continue
            other_radius = max(other.radius_x, other.radius_y)
            same_cluster = (
                (event.debris_family_id >= 0 and event.debris_family_id == other.debris_family_id)
                or event.family_anchor_event_id == other.event_id
                or other.family_anchor_event_id == event.event_id
                or (
                    event.cluster_parent_event_id >= 0
                    and event.cluster_parent_event_id == other.cluster_parent_event_id
                )
            )
            dense_micro_contact = (
                event.debris_role == "dense_micro_debris"
                and other.debris_role == "dense_micro_debris"
            )
            contact_factor = (
                self.config.collapse_cluster_spacing_scale
                if same_cluster or dense_micro_contact
                else self.config.background_contact_spacing
            )
            configured_minimum = max(
                self._event_spacing(event.kind),
                self._event_spacing(other.kind),
            )
            required = max(
                configured_minimum,
                contact_factor * (radius + other_radius),
            )
            if distance < required:
                return False
        return True

    def _prop_proxy_fits_cave(
        self,
        event: GeologicalEvent,
        voxel_grid: Any | None,
    ) -> bool:
        """Reject ellipsoid-envelope wall/ceiling intersections before meshing."""

        if voxel_grid is None:
            return True
        heading, side, up = self._event_basis(event)
        contact = np.asarray(event.contact_point, dtype=float)
        embed = float(np.clip(self.config.ground_embed_fraction, 0.0, 0.5))
        height = 2.0 * event.radius_z
        middle = contact + up * height * (0.45 - 0.5 * embed)
        points = [
            contact + up * height * (1.0 - embed),
            middle + heading * event.radius_x,
            middle - heading * event.radius_x,
            middle + side * event.radius_y,
            middle - side * event.radius_y,
        ]
        diagonal_scale = 1.0 / math.sqrt(2.0)
        for heading_sign in (-1.0, 1.0):
            for side_sign in (-1.0, 1.0):
                points.append(
                    middle
                    + heading * event.radius_x * heading_sign * diagonal_scale
                    + side * event.radius_y * side_sign * diagonal_scale
                )
        tolerance = 0.05
        return all(
            float(voxel_grid.sample_density(point)) >= float(voxel_grid.iso_level) - tolerance
            for point in points
        )

    def _candidate_rover_route(
        self,
        event: GeologicalEvent,
        events: list[GeologicalEvent],
        sample_lookup: dict[tuple[int, int], SectionSample],
    ) -> tuple[bool, float]:
        if self._runtime_index_active:
            segment_samples = self._runtime_samples_by_segment.get(
                event.segment_id,
                [],
            )
            props = [
                *self._runtime_props_by_segment.get(event.segment_id, ()),
                event,
            ]
        else:
            segment_samples = sorted(
                (
                    sample
                    for (segment_id, _sample_index), sample in sample_lookup.items()
                    if segment_id == event.segment_id
                ),
                key=lambda sample: sample.segment_arc_length,
            )
            props = [
                other
                for other in (*events, event)
                if other.kind in {"rock", "boulder"} and other.segment_id == event.segment_id
            ]
        return self._segment_rover_route(segment_samples, props)

    def _rover_route_status(
        self,
        section_field: SectionField,
        events: tuple[GeologicalEvent, ...],
    ) -> tuple[bool, float]:
        props = [event for event in events if event.kind in {"rock", "boulder"}]
        valid = True
        minimum_bypass = math.inf
        for segment_field in section_field.segment_fields:
            segment_props = [
                event for event in props if event.segment_id == segment_field.segment_id
            ]
            segment_valid, bypass = self._segment_rover_route(
                list(segment_field.samples),
                segment_props,
            )
            valid = valid and segment_valid
            minimum_bypass = min(minimum_bypass, bypass)
        return valid, minimum_bypass

    def _segment_rover_route(
        self,
        samples: list[SectionSample],
        props: list[GeologicalEvent],
    ) -> tuple[bool, float]:
        """Propagate a rover-width free interval through one tunnel segment."""

        if not samples:
            return True, math.inf
        ordered = sorted(samples, key=lambda sample: sample.segment_arc_length)
        half_rover = 0.5 * (self.config.rover_width_m + 2.0 * self.config.rover_side_margin_m)
        reachable: list[tuple[float, float]] | None = None
        previous_arc = ordered[0].segment_arc_length
        minimum_bypass = math.inf
        for sample in ordered:
            obstacles = self._sample_obstacle_intervals(sample, props)
            wall_low = -0.5 * sample.tube_width
            wall_high = 0.5 * sample.tube_width
            free_intervals = self._subtract_intervals(
                (wall_low, wall_high),
                obstacles,
            )
            widest = max(
                (high - low for low, high in free_intervals),
                default=0.0,
            )
            minimum_bypass = min(minimum_bypass, widest)
            allowed_centers = [
                (low + half_rover, high - half_rover)
                for low, high in free_intervals
                if high - low >= 2.0 * half_rover
            ]
            if not allowed_centers:
                return False, minimum_bypass
            if reachable is None:
                reachable = allowed_centers
            else:
                delta = max(sample.segment_arc_length - previous_arc, 0.0)
                lateral_reach = max(
                    0.10,
                    delta * self.config.rover_max_lateral_slope,
                )
                expanded = [(low - lateral_reach, high + lateral_reach) for low, high in reachable]
                reachable = self._intersect_intervals(expanded, allowed_centers)
                if not reachable:
                    return False, minimum_bypass
            previous_arc = sample.segment_arc_length
        return True, minimum_bypass

    @staticmethod
    def _sample_obstacle_intervals(
        sample: SectionSample,
        props: list[GeologicalEvent],
    ) -> list[tuple[float, float]]:
        center = np.asarray((sample.x, sample.y, sample.z), dtype=float)
        tangent = np.asarray(sample.tangent, dtype=float)
        normal = np.asarray(sample.normal, dtype=float)
        intervals: list[tuple[float, float]] = []
        for event in props:
            if event.segment_id != sample.segment_id:
                continue
            offset = np.asarray(event.contact_point, dtype=float) - center
            radius = max(event.radius_x, event.radius_y)
            if abs(float(np.dot(offset, tangent))) > 1.15 * radius:
                continue
            lateral = float(np.dot(offset, normal))
            intervals.append((lateral - radius, lateral + radius))
        return intervals

    @staticmethod
    def _subtract_intervals(
        bounds: tuple[float, float],
        obstacles: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        free = [bounds]
        for obstacle_low, obstacle_high in sorted(obstacles):
            updated: list[tuple[float, float]] = []
            for free_low, free_high in free:
                if obstacle_high <= free_low or obstacle_low >= free_high:
                    updated.append((free_low, free_high))
                    continue
                if obstacle_low > free_low:
                    updated.append((free_low, min(obstacle_low, free_high)))
                if obstacle_high < free_high:
                    updated.append((max(obstacle_high, free_low), free_high))
            free = updated
        return free

    @staticmethod
    def _intersect_intervals(
        first: list[tuple[float, float]],
        second: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        intersections = [
            (max(first_low, second_low), min(first_high, second_high))
            for first_low, first_high in first
            for second_low, second_high in second
            if min(first_high, second_high) >= max(first_low, second_low)
        ]
        if not intersections:
            return []
        intersections.sort()
        merged = [intersections[0]]
        for low, high in intersections[1:]:
            previous_low, previous_high = merged[-1]
            if low <= previous_high:
                merged[-1] = (previous_low, max(previous_high, high))
            else:
                merged.append((low, high))
        return merged

    def _choose_candidate(
        self,
        kind: str,
        rng: np.random.Generator,
        candidates: list[tuple[float, SectionSample]],
        occupied_positions: list[tuple[np.ndarray, str]],
    ) -> SectionSample | None:
        if not candidates:
            return None
        limit = len(candidates)
        weights = np.array([score for score, _sample in candidates[:limit]], dtype=float)
        weights /= max(float(weights.sum()), 1e-9)
        for _attempt in range(max(80, 2 * limit)):
            index = int(rng.choice(limit, p=weights))
            sample = candidates[index][1]
            position = np.array((sample.x, sample.y, sample.z), dtype=float)
            if all(
                float(np.linalg.norm(position - occupied))
                >= self._pair_spacing(kind, occupied_kind)
                for occupied, occupied_kind in occupied_positions
            ):
                return sample
        return None

    def _choose_floor_candidate(
        self,
        kind: str,
        rng: np.random.Generator,
        candidates: list[tuple[float, SectionSample, FloorCell]],
        occupied_positions: list[tuple[np.ndarray, str]],
        *,
        spacing_scale: float = 1.0,
    ) -> tuple[SectionSample, FloorCell] | None:
        if not candidates:
            return None
        weights = np.asarray([score for score, _sample, _cell in candidates], dtype=float)
        weights /= max(float(weights.sum()), 1e-9)
        for _attempt in range(max(100, 2 * len(candidates))):
            index = int(rng.choice(len(candidates), p=weights))
            _score, sample, cell = candidates[index]
            position = np.asarray(cell.position, dtype=float)
            if all(
                float(np.linalg.norm(position - occupied))
                >= spacing_scale * self._pair_spacing(kind, occupied_kind)
                for occupied, occupied_kind in occupied_positions
            ):
                return sample, cell
        return None

    def _pair_spacing(self, first_kind: str, second_kind: str) -> float:
        pair = {first_kind, second_kind}
        if "collapse" in pair and pair.intersection({"rock", "boulder"}):
            prop_kind = first_kind if first_kind in {"rock", "boulder"} else second_kind
            return 0.60 * self._event_spacing(prop_kind)
        return max(self._event_spacing(first_kind), self._event_spacing(second_kind))

    def _event_spacing(self, kind: str) -> float:
        if kind == "rock":
            return max(float(self.config.minimum_rock_spacing), 0.0)
        if kind == "boulder":
            return max(float(self.config.minimum_boulder_spacing), 0.0)
        return max(float(self.config.minimum_event_spacing), 0.0)

    def _build_event(
        self,
        *,
        event_id: int,
        kind: str,
        material_hint: str,
        sample: SectionSample,
        rng: np.random.Generator,
        voxel_grid: Any | None,
        floor_cell: FloorCell | None = None,
        cluster_parent_event_id: int = -1,
        cluster_parent_event: GeologicalEvent | None = None,
        family_anchor_event_id: int = -1,
        debris_anchor_event: GeologicalEvent | None = None,
        debris_family_id: int = -1,
        debris_role: str = "",
        prop_diameter_range: tuple[float, float] | None = None,
    ) -> GeologicalEvent:
        severity = float(np.clip(rng.normal(0.62, 0.18), 0.22, 1.0))
        normal = np.array(sample.normal, dtype=float)
        binormal = np.array(sample.binormal, dtype=float)
        tangent = np.array(sample.tangent, dtype=float)
        section_center = np.array((sample.x, sample.y, sample.z), dtype=float)
        local_size_cap_m = 0.0
        source_distance_m = 0.0
        lateral_offset_m = 0.0

        if floor_cell is not None and kind in {"rock", "boulder"}:
            contact, contact_normal = self._jitter_floor_contact(
                sample,
                floor_cell,
                rng,
                voxel_grid,
            )
            lateral_offset_m = float(np.dot(contact - section_center, normal))
            local_size_cap_m = self._local_prop_size_cap(
                kind,
                sample,
                floor_cell,
                lateral_offset_m,
            )
            if cluster_parent_event is not None:
                source_distance_m = float(
                    np.linalg.norm(contact - np.asarray(cluster_parent_event.position, dtype=float))
                )
                reach = max(
                    cluster_parent_event.max_radius * self.config.collapse_cluster_radius_scale,
                    1.0,
                )
                distance_fraction = source_distance_m / reach
                local_size_cap_m *= 0.40 + 0.60 * math.exp(
                    -2.0 * distance_fraction * distance_fraction
                )
            if debris_anchor_event is not None:
                source_distance_m = float(
                    np.linalg.norm(contact - np.asarray(debris_anchor_event.position, dtype=float))
                )
            if prop_diameter_range is None:
                radius = self._sample_radius(
                    kind,
                    rng,
                    maximum_radius=0.5 * local_size_cap_m,
                )
            else:
                minimum_diameter, maximum_diameter = prop_diameter_range
                maximum_diameter = min(maximum_diameter, local_size_cap_m)
                minimum_diameter = min(minimum_diameter, maximum_diameter)
                bias = (
                    self.config.rock_size_bias if kind == "rock" else self.config.boulder_size_bias
                )
                fraction = float(rng.random()) ** bias
                radius = 0.5 * (minimum_diameter + fraction * (maximum_diameter - minimum_diameter))
        else:
            radius = self._sample_radius(kind, rng)
            contact = section_center
            contact_normal = normal

        if kind == "choke":
            radius_x = min(radius * 1.35, sample.tube_width * 0.42)
            radius_y = min(radius * 0.85, sample.tube_width * 0.32)
            radius_z = min(radius * 0.95, sample.tube_height * 0.46)
        elif kind == "collapse":
            radius_x = min(radius * 1.35, sample.tube_width * 0.55)
            radius_y = min(radius, sample.tube_width * 0.45)
            radius_z = min(radius * 0.70, sample.tube_height * 0.42)
        elif kind == "infill":
            radius_x = min(radius * 1.8, sample.tube_width * 0.70)
            radius_y = min(radius * 1.25, sample.tube_width * 0.55)
            radius_z = min(radius * 0.38, sample.tube_height * 0.30)
        else:
            radius_x = min(radius * float(rng.uniform(0.75, 1.35)), sample.tube_width * 0.36)
            radius_y = min(radius * float(rng.uniform(0.75, 1.35)), sample.tube_width * 0.36)
            radius_z = min(radius * float(rng.uniform(0.55, 1.05)), sample.tube_height * 0.36)

        minimum_radius = 0.02 if kind in {"rock", "boulder"} else 0.25
        radius_x = max(float(radius_x), minimum_radius)
        radius_y = max(float(radius_y), minimum_radius)
        radius_z = max(float(radius_z), minimum_radius)
        if floor_cell is not None and kind in {"rock", "boulder"}:
            horizontal_radius = max(radius_x, radius_y)
            cap_radius = max(0.5 * local_size_cap_m, minimum_radius)
            if horizontal_radius > cap_radius:
                scale = cap_radius / horizontal_radius
                radius_x *= scale
                radius_y *= scale
            height_fraction = (
                self.config.boulder_max_height_fraction
                if kind == "boulder"
                else self.config.gallery_clearance_size_fraction
            )
            radius_z = min(
                radius_z,
                max(
                    minimum_radius,
                    0.5 * height_fraction * floor_cell.clearance_m,
                ),
            )
        angle = self._event_resting_angle(
            sample,
            contact,
            contact_normal,
            cluster_parent_event,
            rng,
        )

        if kind == "choke":
            lateral = float(rng.uniform(-0.25, 0.25) * sample.tube_width)
            center = section_center + normal * lateral
            contact = center
            contact_normal = normal if lateral <= 0.0 else -normal
            grounded = False
        elif floor_cell is not None and kind in {"rock", "boulder"}:
            embed = float(np.clip(self.config.ground_embed_fraction, 0.0, 0.5))
            center = contact + contact_normal * radius_z * (1.0 - embed)
            grounded = floor_cell.grounded
        else:
            lateral_limit = max(
                0.0,
                0.5 * sample.tube_width * self.config.max_lateral_floor_fraction,
            )
            lateral = float(rng.uniform(-lateral_limit, lateral_limit))
            tangent_offset = float(rng.uniform(-0.35, 0.35) * max(radius_x, radius_y))
            ray_origin = section_center + normal * lateral + tangent * tangent_offset
            contact, contact_normal, grounded = self._find_floor_contact(
                sample=sample,
                ray_origin=ray_origin,
                binormal=binormal,
                voxel_grid=voxel_grid,
            )
            if kind in {"rock", "boulder"}:
                embed = float(np.clip(self.config.ground_embed_fraction, 0.0, 0.5))
                center = contact + contact_normal * radius_z * (1.0 - embed)
            elif kind == "collapse":
                center = contact + contact_normal * radius_z * 0.48
            else:
                center = contact + contact_normal * radius_z * 0.32

        return GeologicalEvent(
            event_id=event_id,
            kind=kind,
            segment_id=sample.segment_id,
            sample_index=sample.index,
            x=float(center[0]),
            y=float(center[1]),
            z=float(center[2]),
            surface_z=float(sample.surface_z),
            floor_z=float(contact[2]),
            radius_x=radius_x,
            radius_y=radius_y,
            radius_z=radius_z,
            angle=angle,
            severity=severity,
            material_hint=material_hint,
            contact_point=tuple(float(value) for value in contact),
            contact_normal=tuple(float(value) for value in contact_normal),
            grounded=grounded,
            floor_cell_id=floor_cell.cell_id if floor_cell is not None else -1,
            cluster_parent_event_id=cluster_parent_event_id,
            local_size_cap_m=local_size_cap_m,
            lateral_offset_m=lateral_offset_m,
            source_distance_m=source_distance_m,
            debris_family_id=debris_family_id,
            family_anchor_event_id=family_anchor_event_id,
            debris_role=debris_role,
        )

    def _local_prop_size_cap(
        self,
        kind: str,
        sample: SectionSample,
        floor_cell: FloorCell,
        lateral_offset_m: float,
    ) -> float:
        """Return a location-conditioned maximum full prop diameter."""

        global_max = 2.0 * (
            self.config.rock_radius_range[1]
            if kind == "rock"
            else self.config.boulder_radius_range[1]
        )
        wall_room = max(
            0.04,
            2.0 * (0.5 * floor_cell.tube_width_m - abs(lateral_offset_m) - 0.05),
        )
        clearance_fraction = (
            self.config.boulder_max_height_fraction
            if kind == "boulder"
            else self.config.gallery_clearance_size_fraction
        )
        clearance_cap = clearance_fraction * floor_cell.clearance_m
        width_cap = self.config.gallery_width_size_fraction * floor_cell.tube_width_m
        if kind == "boulder":
            # In ordinary wide, low lava tubes the shared rock width fraction
            # used to prevent boulders from ever reaching their height limit.
            # Wall room still constrains narrow or strongly lateral placements.
            width_cap = max(width_cap, clearance_cap)
        return max(
            0.04,
            min(
                global_max,
                width_cap,
                clearance_cap,
                self.config.roof_block_size_fraction * max(sample.roof_thickness, 0.1),
                wall_room,
            ),
        )

    def _jitter_floor_contact(
        self,
        sample: SectionSample,
        floor_cell: FloorCell,
        rng: np.random.Generator,
        voxel_grid: Any | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move off the discrete atlas lane, then re-ground on the cave surface."""

        tangent = np.asarray(sample.tangent, dtype=float)
        normal = np.asarray(sample.normal, dtype=float)
        binormal = np.asarray(sample.binormal, dtype=float)
        contact = np.asarray(floor_cell.position, dtype=float)
        jitter = max(self.config.placement_jitter_m, 0.0)
        cache_key: tuple[int, int, int] | None = None
        offset_rng = rng
        if voxel_grid is not None:
            slot = int(rng.integers(self._GROUND_CONTACT_SLOTS_PER_CELL))
            cache_key = (id(voxel_grid), floor_cell.cell_id, slot)
            cached = self._ground_contact_cache.get(cache_key)
            if cached is not None:
                return cached[0].copy(), cached[1].copy()
            seed = (
                (int(self.config.random_seed or 0) + 1) * 2_654_435_761
                + (floor_cell.cell_id + 1) * 2_246_822_519
                + (slot + 1) * 3_266_489_917
            ) & 0xFFFFFFFF
            offset_rng = np.random.default_rng(seed)
        tangent_offset = float(offset_rng.uniform(-jitter, jitter))
        lateral_offset = float(offset_rng.uniform(-jitter, jitter))
        target_lateral = float(
            np.clip(
                floor_cell.lateral_offset_m + lateral_offset,
                -0.5 * sample.tube_width * self.config.max_lateral_floor_fraction,
                0.5 * sample.tube_width * self.config.max_lateral_floor_fraction,
            )
        )
        seed = (
            contact
            + tangent * tangent_offset
            + normal * (target_lateral - floor_cell.lateral_offset_m)
        )
        fallback_normal = np.asarray(floor_cell.normal, dtype=float)
        fallback_normal /= max(float(np.linalg.norm(fallback_normal)), 1e-12)
        if voxel_grid is None:
            return seed, fallback_normal

        lift = max(0.35, 1.5 * float(voxel_grid.voxel_size))
        hit = voxel_grid.raycast_isosurface(
            seed + binormal * lift,
            -binormal,
            lift + 3.0 * float(voxel_grid.voxel_size),
        )
        if hit is None:
            if cache_key is not None:
                self._ground_contact_cache[cache_key] = (
                    contact.copy(),
                    fallback_normal.copy(),
                )
            return contact, fallback_normal
        hit_normal = np.asarray(hit.normal, dtype=float)
        if float(np.dot(hit_normal, binormal)) < 0.0:
            hit_normal = -hit_normal
        hit_normal /= max(float(np.linalg.norm(hit_normal)), 1e-12)
        hit_position = np.asarray(hit.position, dtype=float)
        if cache_key is not None:
            self._ground_contact_cache[cache_key] = (
                hit_position.copy(),
                hit_normal.copy(),
            )
        return hit_position, hit_normal

    @staticmethod
    def _event_resting_angle(
        sample: SectionSample,
        contact: np.ndarray,
        contact_normal: np.ndarray,
        cluster_parent_event: GeologicalEvent | None,
        rng: np.random.Generator,
    ) -> float:
        """Orient long axes along runout/downhill rather than arbitrary yaw."""

        if cluster_parent_event is not None:
            direction = contact[:2] - np.asarray(
                cluster_parent_event.position[:2],
                dtype=float,
            )
        else:
            direction = np.asarray(sample.tangent[:2], dtype=float)
        downhill = -np.asarray(contact_normal[:2], dtype=float)
        if float(np.linalg.norm(downhill)) > 0.08:
            downhill /= float(np.linalg.norm(downhill))
            direction = 0.65 * direction + 0.35 * downhill
        if float(np.linalg.norm(direction)) <= 1e-12:
            direction = np.asarray(sample.tangent[:2], dtype=float)
        return float(math.atan2(direction[1], direction[0]) + rng.uniform(-0.35, 0.35))

    @staticmethod
    def _find_floor_contact(
        *,
        sample: SectionSample,
        ray_origin: np.ndarray,
        binormal: np.ndarray,
        voxel_grid: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        profile_floor = (
            min(point[1] for point in sample.profile_points)
            if sample.profile_points
            else -0.5 * sample.tube_height
        )
        fallback = ray_origin + binormal * profile_floor
        fallback_normal = np.array(binormal, dtype=float)
        normal_length = float(np.linalg.norm(fallback_normal))
        if normal_length > 1e-12:
            fallback_normal /= normal_length
        else:
            fallback_normal = np.array((0.0, 0.0, 1.0), dtype=float)
        if voxel_grid is None:
            return fallback, fallback_normal, False

        max_distance = max(
            sample.tube_height * 1.25,
            float(voxel_grid.voxel_size) * 4.0,
        )
        hit = voxel_grid.raycast_isosurface(
            ray_origin,
            -binormal,
            max_distance,
        )
        if hit is None:
            return fallback, fallback_normal, False
        hit_normal = np.asarray(hit.normal, dtype=float)
        if float(np.dot(hit_normal, binormal)) < 0.0:
            hit_normal = -hit_normal
        return np.asarray(hit.position, dtype=float), hit_normal, True

    def _build_event_mesh(self, event: GeologicalEvent) -> GeologicalEventMesh:
        if self._rocky_api is not None and event.kind in {"rock", "boulder"}:
            rocky_mesh = self._build_rocky_event_mesh(event)
            if rocky_mesh is not None:
                return rocky_mesh

        lat_segments = max(int(self.config.mesh_latitude_segments), 4)
        lon_segments = max(int(self.config.mesh_longitude_segments), 6)
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        center = np.array((event.x, event.y, event.z), dtype=float)
        heading, side, up = self._event_basis(event)

        for lat_index in range(lat_segments + 1):
            phi = math.pi * lat_index / lat_segments
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            for lon_index in range(lon_segments):
                theta = 2.0 * math.pi * lon_index / lon_segments
                roughness = 1.0 + 0.10 * event.severity * (
                    math.sin(3.0 * theta + 0.37 * event.event_id)
                    + 0.55 * math.cos(2.0 * phi - 0.19 * event.event_id)
                )
                local_x = event.radius_x * roughness * sin_phi * math.cos(theta)
                local_y = event.radius_y * roughness * sin_phi * math.sin(theta)
                local_z = event.radius_z * roughness * cos_phi
                world = center + heading * local_x + side * local_y + up * local_z
                vertices.append(tuple(float(value) for value in world))

        for lat_index in range(lat_segments):
            for lon_index in range(lon_segments):
                next_lon = (lon_index + 1) % lon_segments
                a = lat_index * lon_segments + lon_index
                b = lat_index * lon_segments + next_lon
                c = (lat_index + 1) * lon_segments + lon_index
                d = (lat_index + 1) * lon_segments + next_lon
                if lat_index != 0:
                    faces.append((a, c, b))
                if lat_index != lat_segments - 1:
                    faces.append((b, c, d))

        return GeologicalEventMesh(
            event_id=event.event_id,
            kind=event.kind,
            material_hint=event.material_hint,
            vertices=tuple(vertices),
            faces=tuple(faces),
            debris_family_id=event.debris_family_id,
            family_anchor_event_id=event.family_anchor_event_id,
            debris_role=event.debris_role,
        )

    def _build_rocky_event_mesh(self, event: GeologicalEvent) -> GeologicalEventMesh | None:
        if self._rocky_api is None:
            return None

        try:
            BatchConfig = self._rocky_api["BatchConfig"]
            RockGenerator = self._rocky_api["RockGenerator"]
            ParameterRanges = self._rocky_api["ParameterRanges"]
            RockParameters = self._rocky_api["RockParameters"]
            discover_texture_sets = self._rocky_api["discover_texture_sets"]
            choose_texture_set = self._rocky_api["choose_texture_set"]
        except KeyError as error:
            return self._rocky_build_failure(
                event,
                f"provider API is missing {error.args[0]!r}",
                error,
            )

        radius_xy = max(event.radius_x, event.radius_y)
        target_height = max(0.04, event.radius_z * 2.0)
        diameter = max(0.06, radius_xy * 2.0)
        shape_type, archetype, material_type, base_shape = self._rocky_profile(event)
        subdivisions = self._rocky_subdivisions(max(target_height, diameter))
        seed = self._event_seed(event)
        params = RockParameters(
            name=f"stage_e_event_{event.event_id:04d}",
            seed=seed,
            size_class=self._rocky_size_class(max(target_height, diameter)),
            archetype=archetype,
            shape_type=shape_type,
            material_type=material_type,
            placement_role="large_debris" if event.kind != "rock" else "floor_scatter",
            max_height=max(target_height * 1.05, 0.05),
            target_height=target_height,
            diameter=diameter,
            base_shape=base_shape,
            subdivisions=subdivisions,
            radius=max(target_height, diameter) * 0.5,
            roughness=(0.10 + 0.24 * event.severity) * max(target_height, diameter),
            angularity=0.35 + 0.65 * event.severity,
            spike_limit=0.72 + 0.14 * (1.0 - event.severity),
            elongation=max(0.65, min(1.85, event.radius_x / max(event.radius_y, 1e-6))),
            floor_flattening=0.25 if event.kind == "boulder" else 0.45,
            fracture_strength=0.04 + 0.08 * event.severity,
            pitting_intensity=0.45 if material_type == "porous_lava" else 0.12,
            ropy_strength=0.20 if shape_type == "ropy_lava_fragment" else 0.0,
            fracture_count=max(2, int(round(3 + 7 * event.severity))),
            crack_count=max(3, int(round(5 + 11 * event.severity))),
        )
        config = BatchConfig(
            seed=seed,
            count=1,
            output_dir=Path(self.config.rocky_output_dir),
            texture_dir=Path(self.config.rocky_texture_dir),
            export_formats=(),
            max_height=params.max_height,
            resolution_scale=self.config.rocky_resolution_scale,
            ranges=ParameterRanges(),
        )
        try:
            state = RockGenerator(config).generate_one(params)
            if state.mesh is None:
                raise RuntimeError("provider returned no mesh")
            texture_sets = discover_texture_sets(config.texture_dir)
            material_maps = choose_texture_set(
                texture_sets,
                np.random.default_rng(seed),
                material_type,
            )
        except Exception as error:
            return self._rocky_build_failure(
                event,
                f"generation failed: {error}",
                error,
            )
        vertices = self._transform_rocky_vertices(event, state.mesh)
        return GeologicalEventMesh(
            event_id=event.event_id,
            kind=event.kind,
            material_hint=event.material_hint,
            vertices=vertices,
            faces=tuple(tuple(int(index) for index in face) for face in state.mesh.faces),
            face_uvs=tuple(state.mesh.face_uvs),
            material_maps=tuple((name, str(path)) for name, path in sorted(material_maps.items())),
            source_generator="rocky",
            source_shape_type=shape_type,
            debris_family_id=event.debris_family_id,
            family_anchor_event_id=event.family_anchor_event_id,
            debris_role=event.debris_role,
        )

    def _rocky_build_failure(
        self,
        event: GeologicalEvent,
        reason: str,
        error: Exception | None = None,
    ) -> GeologicalEventMesh | None:
        if self.config.strict_optional_provider:
            exception = RuntimeError(
                f"Rocky failed for Stage-E {event.kind} event {event.event_id}: {reason}"
            )
            if error is not None:
                raise exception from error
            raise exception
        return None

    def _transform_rocky_vertices(
        self, event: GeologicalEvent, mesh: Any
    ) -> tuple[tuple[float, float, float], ...]:
        bounds_min, _bounds_max = mesh.bounds()
        contact = np.asarray(event.contact_point, dtype=float)
        heading, side, up = self._event_basis(event)
        contact -= up * (
            max(float(self.config.ground_embed_fraction), 0.0) * max(event.radius_z * 2.0, 0.0)
        )
        vertices: list[tuple[float, float, float]] = []
        for vertex in mesh.vertices:
            local_x = float(vertex.x)
            local_y = float(vertex.y - bounds_min.y)
            local_z = float(vertex.z)
            world = contact + heading * local_x + side * local_z + up * local_y
            vertices.append(tuple(float(value) for value in world))
        return tuple(vertices)

    @staticmethod
    def _event_basis(
        event: GeologicalEvent,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        up = np.asarray(event.contact_normal, dtype=float)
        up_length = float(np.linalg.norm(up))
        if up_length <= 1e-12:
            up = np.array((0.0, 0.0, 1.0), dtype=float)
        else:
            up /= up_length
        heading = np.array(
            (math.cos(event.angle), math.sin(event.angle), 0.0),
            dtype=float,
        )
        heading -= up * float(np.dot(heading, up))
        heading_length = float(np.linalg.norm(heading))
        if heading_length <= 1e-12:
            reference = (
                np.array((1.0, 0.0, 0.0), dtype=float)
                if abs(float(up[0])) < 0.9
                else np.array((0.0, 1.0, 0.0), dtype=float)
            )
            heading = np.cross(reference, up)
            heading_length = max(float(np.linalg.norm(heading)), 1e-12)
        heading /= heading_length
        side = np.cross(up, heading)
        side /= max(float(np.linalg.norm(side)), 1e-12)
        return heading, side, up

    def _rocky_subdivisions(self, size: float) -> int:
        if size < 0.25:
            base = 1
        elif size < 0.75:
            base = 2
        elif size < 2.5:
            base = 3
        else:
            base = 4
        scaled = round(base * self.config.rocky_resolution_scale)
        return max(1, min(int(self.config.rocky_max_subdivisions), scaled))

    @staticmethod
    def _rocky_size_class(size: float) -> str:
        if size < 0.25:
            return "cobble"
        if size < 0.75:
            return "floor_cobble"
        if size < 2.5:
            return "step_rock"
        return "rover_obstacle"

    def _rocky_profile(self, event: GeologicalEvent) -> tuple[str, str, str, str]:
        if event.kind == "collapse":
            return "collapsed_ceiling_block", "fractured_block", "fractured_cliff", "box"
        profiles = (
            ("rounded_boulder", "smooth_basalt", "dark_basalt", "icosphere"),
            ("angular_boulder", "rough_basalt", "dark_basalt", "icosphere"),
            ("vesicular_chunk", "vesicular_lava", "porous_lava", "icosphere"),
            ("flat_slab", "flat_lava_slab", "layered_cliff", "box"),
            ("ropy_lava_fragment", "ropy_lava_clast", "porous_lava", "icosphere"),
            ("eroded_irregular", "weird_erosion", "dry_boulder", "icosphere"),
        )
        profile_index = (event.event_id + int(self.config.random_seed or 0)) % len(profiles)
        if event.kind == "boulder":
            profile_index = (profile_index + 1) % len(profiles)
        return profiles[profile_index]

    def _event_seed(self, event: GeologicalEvent) -> int:
        base_seed = self.config.random_seed or 0
        return (
            int(
                (
                    base_seed * 1_000_003
                    + event.event_id * 9_176
                    + event.segment_id * 131
                    + event.sample_index
                )
                % (2**31 - 1)
            )
            or 1
        )

    def _sample_radius(
        self,
        kind: str,
        rng: np.random.Generator,
        *,
        maximum_radius: float | None = None,
    ) -> float:
        ranges = {
            "rock": self.config.rock_radius_range,
            "boulder": self.config.boulder_radius_range,
            "collapse": self.config.collapse_radius_range,
            "choke": self.config.choke_radius_range,
            "infill": self.config.infill_radius_range,
        }
        minimum, maximum = ranges[kind]
        if maximum_radius is not None:
            maximum = min(maximum, max(float(maximum_radius), minimum))
        if kind == "rock":
            fraction = float(rng.random()) ** self.config.rock_size_bias
        elif kind == "boulder":
            fraction = float(rng.random()) ** self.config.boulder_size_bias
        else:
            fraction = float(rng.random())
        return float(minimum + fraction * (maximum - minimum))

    @staticmethod
    def _emit_progress(
        progress: EventProgressCallback | None,
        phase: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if progress is not None:
            progress(phase, current, max(total, 1), message)


__all__ = [
    "EventProgressCallback",
    "GeologicalEvent",
    "GeologicalEventConfig",
    "GeologicalEventField",
    "GeologicalEventMesh",
    "GeologicalEventGenerator",
]


def _load_rocky_api(source_path: str) -> dict[str, Any] | None:
    if source_path:
        source = Path(source_path)
        if not source.is_absolute():
            source = Path(__file__).resolve().parents[2] / source
        if source.exists():
            source_text = str(source)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
    try:
        from rocky import BatchConfig, RockGenerator
        from rocky.config import ParameterRanges
        from rocky.layers import RockParameters
        from rocky.pipeline import _choose_texture_set, _discover_texture_sets
    except ImportError:
        return None

    def choose_texture_set(
        texture_sets: list[dict[str, Path]], rng: np.random.Generator, material_type: str
    ) -> dict[str, Path]:
        import random

        py_rng = random.Random(int(rng.integers(1, 2**31 - 1)))
        return _choose_texture_set(texture_sets, py_rng, material_type)

    return {
        "BatchConfig": BatchConfig,
        "RockGenerator": RockGenerator,
        "ParameterRanges": ParameterRanges,
        "RockParameters": RockParameters,
        "discover_texture_sets": _discover_texture_sets,
        "choose_texture_set": choose_texture_set,
    }
