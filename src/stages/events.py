"""Stage E geological event and coarse debris generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from stages.section_field import SectionField, SectionSample


@dataclass(frozen=True)
class GeologicalEventConfig:
    """Parameters controlling mesh-stage geological event placement."""

    random_seed: int | None = None
    rock_density_per_100m: float = 0.40
    boulder_density_per_100m: float = 0.11
    geological_event_density_per_100m: float = 0.09
    collapse_event_fraction: float = 0.35
    choke_event_fraction: float = 0.30
    infill_event_fraction: float = 0.35
    minimum_event_spacing: float = 22.0
    rock_radius_range: tuple[float, float] = (0.8, 2.4)
    boulder_radius_range: tuple[float, float] = (2.4, 6.2)
    collapse_radius_range: tuple[float, float] = (4.0, 9.0)
    choke_radius_range: tuple[float, float] = (3.0, 7.0)
    infill_radius_range: tuple[float, float] = (5.0, 12.0)
    max_lateral_floor_fraction: float = 0.62
    mesh_latitude_segments: int = 8
    mesh_longitude_segments: int = 14
    use_rocky_meshes: bool = True
    rocky_source_path: str = "../Rocky/src"
    rocky_texture_dir: str = "../Rocky/textures"
    rocky_output_dir: str = "outputs/rocky_stage_e"
    rocky_resolution_scale: float = 0.70
    rocky_max_subdivisions: int = 5


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

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def max_radius(self) -> float:
        return max(self.radius_x, self.radius_y, self.radius_z)


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

    def summary(self) -> dict[str, float]:
        counts = {kind: 0 for kind in ("rock", "boulder", "collapse", "choke", "infill")}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
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
            "mean_severity": float(np.mean([event.severity for event in self.events])) if self.events else 0.0,
        }


class GeologicalEventGenerator:
    """Place Stage-E mesh events from host-aware section samples."""

    def __init__(self, config: GeologicalEventConfig | None = None) -> None:
        self.config = config or GeologicalEventConfig()
        self._rocky_api = _load_rocky_api(self.config.rocky_source_path) if self.config.use_rocky_meshes else None

    def generate(self, section_field: SectionField) -> GeologicalEventField:
        rng = np.random.default_rng(self.config.random_seed)
        samples = [
            sample
            for segment_field in section_field.segment_fields
            for sample in segment_field.samples
        ]
        if not samples:
            return GeologicalEventField(config=self.config, events=())

        events: list[GeologicalEvent] = []
        occupied_positions: list[np.ndarray] = []
        rock_count, boulder_count, collapse_count, choke_count, infill_count = self._counts_from_density(samples)
        recipes = (
            ("rock", rock_count, "floor_debris"),
            ("boulder", boulder_count, "large_breakdown"),
            ("collapse", collapse_count, "roof_breakdown"),
            ("choke", choke_count, "constriction"),
            ("infill", infill_count, "floor_infill"),
        )
        for kind, count, material_hint in recipes:
            candidates = self._rank_candidates(kind, samples)
            for _ in range(max(count, 0)):
                sample = self._choose_candidate(rng, candidates, occupied_positions)
                if sample is None:
                    break
                event = self._build_event(
                    event_id=len(events),
                    kind=kind,
                    material_hint=material_hint,
                    sample=sample,
                    rng=rng,
                )
                events.append(event)
                occupied_positions.append(np.array((event.x, event.y, event.z), dtype=float))

        events.sort(key=lambda event: (event.segment_id, event.sample_index, event.event_id))
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
            )
            for index, event in enumerate(events)
        )
        return GeologicalEventField(
            config=self.config,
            events=normalized_events,
            meshes=tuple(self._build_event_mesh(event) for event in normalized_events),
        )

    def _counts_from_density(self, samples: list[SectionSample]) -> tuple[int, int, int, int, int]:
        total_length = self._total_sampled_length(samples)
        length_units = total_length / 100.0
        rock_count = max(0, int(round(self.config.rock_density_per_100m * length_units)))
        boulder_count = max(0, int(round(self.config.boulder_density_per_100m * length_units)))
        geological_event_count = max(0, int(round(self.config.geological_event_density_per_100m * length_units)))
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
            cover_score = 1.0 / max(sample.cover_thickness, 1.0)
            weak_roof_score = 1.0 / max(sample.roof_thickness, 1.0)
            junction_score = sample.junction_blend_weight
            width_score = sample.tube_width
            narrow_score = 1.0 / max(sample.tube_width, 1.0)
            if kind == "collapse":
                score = 3.0 * weak_roof_score + 1.2 * junction_score + 0.05 * width_score
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

    def _choose_candidate(
        self,
        rng: np.random.Generator,
        candidates: list[tuple[float, SectionSample]],
        occupied_positions: list[np.ndarray],
    ) -> SectionSample | None:
        if not candidates:
            return None
        limit = min(len(candidates), 240)
        weights = np.array([score for score, _sample in candidates[:limit]], dtype=float)
        weights /= max(float(weights.sum()), 1e-9)
        for _attempt in range(80):
            index = int(rng.choice(limit, p=weights))
            sample = candidates[index][1]
            position = np.array((sample.x, sample.y, sample.z), dtype=float)
            if all(
                float(np.linalg.norm(position - occupied)) >= self.config.minimum_event_spacing
                for occupied in occupied_positions
            ):
                return sample
        return candidates[int(rng.integers(0, limit))][1]

    def _build_event(
        self,
        *,
        event_id: int,
        kind: str,
        material_hint: str,
        sample: SectionSample,
        rng: np.random.Generator,
    ) -> GeologicalEvent:
        radius = self._sample_radius(kind, rng)
        severity = float(np.clip(rng.normal(0.62, 0.18), 0.22, 1.0))
        normal = np.array(sample.normal, dtype=float)
        binormal = np.array(sample.binormal, dtype=float)
        tangent = np.array(sample.tangent, dtype=float)
        floor_z = min(point[1] for point in sample.profile_points) if sample.profile_points else -0.5 * sample.tube_height
        lateral_limit = max(0.0, 0.5 * sample.tube_width * self.config.max_lateral_floor_fraction)
        lateral = float(rng.uniform(-lateral_limit, lateral_limit))
        floor = np.array((sample.x, sample.y, sample.z), dtype=float) + normal * lateral + binormal * floor_z

        if kind == "choke":
            center = np.array((sample.x, sample.y, sample.z), dtype=float) + normal * float(rng.uniform(-0.25, 0.25) * sample.tube_width)
            radius_x = min(radius * 1.35, sample.tube_width * 0.42)
            radius_y = min(radius * 0.85, sample.tube_width * 0.32)
            radius_z = min(radius * 0.95, sample.tube_height * 0.46)
            z = center[2]
        elif kind == "collapse":
            radius_x = min(radius * 1.35, sample.tube_width * 0.55)
            radius_y = min(radius, sample.tube_width * 0.45)
            radius_z = min(radius * 0.70, sample.tube_height * 0.42)
            z = floor[2] + radius_z * 0.44
            center = floor
        elif kind == "infill":
            radius_x = min(radius * 1.8, sample.tube_width * 0.70)
            radius_y = min(radius * 1.25, sample.tube_width * 0.55)
            radius_z = min(radius * 0.38, sample.tube_height * 0.30)
            z = floor[2] + radius_z * 0.35
            center = floor
        else:
            radius_x = min(radius * float(rng.uniform(0.75, 1.35)), sample.tube_width * 0.36)
            radius_y = min(radius * float(rng.uniform(0.75, 1.35)), sample.tube_width * 0.36)
            radius_z = min(radius * float(rng.uniform(0.55, 1.05)), sample.tube_height * 0.36)
            z = floor[2] + radius_z * (0.42 if kind == "boulder" else 0.36)
            center = floor

        x, y = center[0], center[1]
        x += tangent[0] * float(rng.uniform(-0.35, 0.35) * max(radius_x, radius_y))
        y += tangent[1] * float(rng.uniform(-0.35, 0.35) * max(radius_x, radius_y))

        return GeologicalEvent(
            event_id=event_id,
            kind=kind,
            segment_id=sample.segment_id,
            sample_index=sample.index,
            x=float(x),
            y=float(y),
            z=float(z),
            surface_z=float(sample.surface_z),
            floor_z=float(floor[2]),
            radius_x=max(float(radius_x), 0.25),
            radius_y=max(float(radius_y), 0.25),
            radius_z=max(float(radius_z), 0.25),
            angle=float(math.atan2(sample.tangent[1], sample.tangent[0]) + rng.uniform(-0.7, 0.7)),
            severity=severity,
            material_hint=material_hint,
        )

    def _build_event_mesh(self, event: GeologicalEvent) -> GeologicalEventMesh:
        if self._rocky_api is not None and event.kind in {"rock", "boulder", "collapse"}:
            rocky_mesh = self._build_rocky_event_mesh(event)
            if rocky_mesh is not None:
                return rocky_mesh

        lat_segments = max(int(self.config.mesh_latitude_segments), 4)
        lon_segments = max(int(self.config.mesh_longitude_segments), 6)
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        cos_angle = math.cos(event.angle)
        sin_angle = math.sin(event.angle)
        center = np.array((event.x, event.y, event.z), dtype=float)

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
                world_x = center[0] + local_x * cos_angle - local_y * sin_angle
                world_y = center[1] + local_x * sin_angle + local_y * cos_angle
                world_z = center[2] + local_z
                vertices.append((float(world_x), float(world_y), float(world_z)))

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
        except KeyError:
            return None

        radius_xy = max(event.radius_x, event.radius_y)
        target_height = max(0.20, event.radius_z * 2.0)
        diameter = max(0.24, radius_xy * 2.0)
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
            placement_role="navigation_obstacle" if event.kind != "rock" else "floor_scatter",
            max_height=max(target_height * 1.05, 0.25),
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
        state = RockGenerator(config).generate_one(params)
        if state.mesh is None:
            return None

        texture_sets = discover_texture_sets(config.texture_dir)
        material_maps = choose_texture_set(texture_sets, np.random.default_rng(seed), material_type)
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
        )

    def _transform_rocky_vertices(self, event: GeologicalEvent, mesh: Any) -> tuple[tuple[float, float, float], ...]:
        bounds_min, _bounds_max = mesh.bounds()
        contact_z = self._event_contact_z(event)
        cos_angle = math.cos(event.angle)
        sin_angle = math.sin(event.angle)
        vertices: list[tuple[float, float, float]] = []
        for vertex in mesh.vertices:
            local_x = float(vertex.x)
            local_y = float(vertex.y - bounds_min.y)
            local_z = float(vertex.z)
            world_x = event.x + local_x * cos_angle - local_z * sin_angle
            world_y = event.y + local_x * sin_angle + local_z * cos_angle
            world_z = contact_z + local_y
            vertices.append((float(world_x), float(world_y), float(world_z)))
        return tuple(vertices)

    @staticmethod
    def _event_contact_z(event: GeologicalEvent) -> float:
        return float(event.floor_z)

    def _rocky_subdivisions(self, size: float) -> int:
        if size < 0.75:
            base = 2
        elif size < 2.5:
            base = 3
        else:
            base = 4
        scaled = round(base * self.config.rocky_resolution_scale)
        return max(1, min(int(self.config.rocky_max_subdivisions), scaled))

    @staticmethod
    def _rocky_size_class(size: float) -> str:
        if size < 0.75:
            return "floor_cobble"
        if size < 2.5:
            return "step_rock"
        return "rover_obstacle"

    def _rocky_profile(self, event: GeologicalEvent) -> tuple[str, str, str, str]:
        profile_index = self._event_seed(event) % 5
        if event.kind == "collapse":
            return "collapsed_ceiling_block", "fractured_block", "fractured_cliff", "box"
        profiles = (
            ("rounded_boulder", "smooth_basalt", "dark_basalt", "icosphere"),
            ("angular_boulder", "rough_basalt", "dark_basalt", "icosphere"),
            ("vesicular_chunk", "vesicular_lava", "porous_lava", "icosphere"),
            ("flat_slab", "flat_lava_slab", "layered_cliff", "box"),
            ("ropy_lava_fragment", "ropy_lava_clast", "porous_lava", "icosphere"),
        )
        if event.kind == "boulder":
            profile_index = (profile_index + 1) % len(profiles)
        return profiles[profile_index]

    def _event_seed(self, event: GeologicalEvent) -> int:
        base_seed = self.config.random_seed or 0
        return int((base_seed * 1_000_003 + event.event_id * 9_176 + event.segment_id * 131 + event.sample_index) % (2**31 - 1)) or 1

    def _sample_radius(self, kind: str, rng: np.random.Generator) -> float:
        ranges = {
            "rock": self.config.rock_radius_range,
            "boulder": self.config.boulder_radius_range,
            "collapse": self.config.collapse_radius_range,
            "choke": self.config.choke_radius_range,
            "infill": self.config.infill_radius_range,
        }
        minimum, maximum = ranges[kind]
        return float(rng.uniform(minimum, maximum))


__all__ = [
    "GeologicalEvent",
    "GeologicalEventConfig",
    "GeologicalEventField",
    "GeologicalEventMesh",
    "GeologicalEventGenerator",
]


def _load_rocky_api(source_path: str) -> dict[str, Any] | None:
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

    def choose_texture_set(texture_sets: list[dict[str, Path]], rng: np.random.Generator, material_type: str) -> dict[str, Path]:
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
