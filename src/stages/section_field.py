"""Stage C: geometry-ready section field generation around the cave network."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from stages.network import CaveJunction, CaveNetwork, CaveSegment


@dataclass(frozen=True)
class SectionFieldConfig:
    """Parameters controlling stage-C section-field generation."""

    random_seed: int | None = None
    base_height_ratio: float = 0.64
    minimum_height_ratio: float = 0.50
    maximum_height_ratio: float = 0.84
    maximum_tube_width: float = 12.0
    chamber_max_tube_width: float = 24.0
    minimum_sample_spacing: float = 8.0
    maximum_sample_spacing: float = 34.0
    curvature_spacing_weight: float = 120.0
    width_gradient_spacing_weight: float = 34.0
    junction_spacing_weight: float = 1.15
    profile_resolution: int = 28
    floor_flatness_base: float = 0.48
    floor_flatness_width_weight: float = 0.12
    roof_arch_base: float = 1.08
    roof_arch_roof_weight: float = 0.18
    lateral_skew_amplitude: float = 0.10
    junction_pre_widen_gain: float = 0.18
    junction_constant_envelope_gain: float = 0.05
    chamber_widen_gain: float = 0.45
    minimum_roof_thickness: float = 6.0
    maximum_centerline_depth: float = 26.0
    preferred_cover_fraction: float = 0.34


@dataclass(frozen=True)
class SectionJunctionInfluence:
    """One junction influence contributing to a section sample."""

    junction_id: int
    kind: str
    weight: float
    split_style: str
    merge_style: str
    capacity_bias: float


@dataclass(frozen=True)
class SectionSample:
    """One geometry-ready section sample along a cave segment."""

    index: int
    segment_id: int
    segment_arc_length: float
    x: float
    y: float
    z: float
    surface_z: float
    cover_thickness: float
    roof_thickness: float
    centerline_depth: float
    tangent: tuple[float, float, float]
    normal: tuple[float, float, float]
    binormal: tuple[float, float, float]
    tube_width: float
    tube_height: float
    floor_flatness: float
    roof_arch: float
    lateral_skew: float
    junction_blend_weight: float
    junction_influences: tuple[SectionJunctionInfluence, ...]
    profile_points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SegmentSectionField:
    """Section-field samples for one cave-network segment."""

    segment_id: int
    connected_junction_ids: tuple[int, ...]
    samples: tuple[SectionSample, ...]


@dataclass(frozen=True)
class SectionField:
    """Stage-C output for section controls and geometry-ready profile samples."""

    config: SectionFieldConfig
    segment_fields: tuple[SegmentSectionField, ...]
    dominant_route_segment_ids: tuple[int, ...]

    def summary(self) -> dict[str, float]:
        sample_count = sum(len(segment_field.samples) for segment_field in self.segment_fields)
        if sample_count == 0:
            return {
                "segment_field_count": 0.0,
                "sample_count": 0.0,
                "dominant_route_segment_count": 0.0,
                "max_junction_blend_weight": 0.0,
                "mean_tube_width": 0.0,
                "min_tube_width": 0.0,
                "max_tube_width": 0.0,
                "mean_tube_height": 0.0,
                "min_tube_height": 0.0,
                "max_tube_height": 0.0,
            }

        all_samples = [
            sample
            for segment_field in self.segment_fields
            for sample in segment_field.samples
        ]
        tube_widths = np.array([sample.tube_width for sample in all_samples], dtype=float)
        tube_heights = np.array([sample.tube_height for sample in all_samples], dtype=float)
        return {
            "segment_field_count": float(len(self.segment_fields)),
            "sample_count": float(sample_count),
            "dominant_route_segment_count": float(len(self.dominant_route_segment_ids)),
            "max_junction_blend_weight": float(
                max(sample.junction_blend_weight for sample in all_samples)
            ),
            "mean_tube_width": float(np.mean(tube_widths)),
            "min_tube_width": float(np.min(tube_widths)),
            "max_tube_width": float(np.max(tube_widths)),
            "mean_tube_height": float(np.mean(tube_heights)),
            "min_tube_height": float(np.min(tube_heights)),
            "max_tube_height": float(np.max(tube_heights)),
        }


class SectionFieldGenerator:
    """Build a smooth lava-tube section field around the cave-network skeleton."""

    def __init__(self, config: SectionFieldConfig | None = None) -> None:
        self.config = config or SectionFieldConfig()

    def generate(self, cave_network: CaveNetwork) -> SectionField:
        rng = np.random.default_rng(self.config.random_seed)
        segment_lookup = {segment.segment_id: segment for segment in cave_network.segments}
        dominant_route_segment_ids = self._build_dominant_route_segment_ids(
            cave_network,
            segment_lookup,
        )
        generation_order = self._build_generation_order(
            cave_network=cave_network,
            segment_lookup=segment_lookup,
            dominant_route_segment_ids=dominant_route_segment_ids,
        )
        node_normal_preferences: dict[int, tuple[float, float, float]] = {}
        segment_fields_by_id: dict[int, SegmentSectionField] = {}
        segment_fields: list[SegmentSectionField] = []
        for segment_id in generation_order:
            segment = segment_lookup[segment_id]
            connected_junctions = tuple(
                junction
                for junction in cave_network.junctions
                if segment.segment_id in junction.segment_ids
            )
            arc_positions = self._build_adaptive_arc_positions(segment, connected_junctions)
            phase = float(rng.uniform(-math.pi, math.pi))
            samples = self._build_segment_samples(
                segment=segment,
                connected_junctions=connected_junctions,
                arc_positions=arc_positions,
                phase=phase,
                initial_normal=(
                    node_normal_preferences.get(segment.start_node_id)
                    or node_normal_preferences.get(segment.end_node_id)
                ),
            )
            if samples:
                node_normal_preferences.setdefault(segment.start_node_id, samples[0].normal)
                node_normal_preferences.setdefault(segment.end_node_id, samples[-1].normal)
            segment_field = SegmentSectionField(
                segment_id=segment.segment_id,
                connected_junction_ids=tuple(
                    junction.junction_id for junction in connected_junctions
                ),
                samples=tuple(samples),
            )
            segment_fields_by_id[segment.segment_id] = segment_field
        for segment in cave_network.segments:
            segment_fields.append(
                SegmentSectionField(
                    segment_id=segment.segment_id,
                    connected_junction_ids=segment_fields_by_id[segment.segment_id].connected_junction_ids,
                    samples=segment_fields_by_id[segment.segment_id].samples,
                )
            )
        return SectionField(
            config=self.config,
            segment_fields=tuple(segment_fields),
            dominant_route_segment_ids=dominant_route_segment_ids,
        )

    def _build_dominant_route_segment_ids(
        self,
        cave_network: CaveNetwork,
        segment_lookup: dict[int, CaveSegment],
    ) -> tuple[int, ...]:
        dominant_pairs = list(
            zip(
                cave_network.dominant_route_node_ids,
                cave_network.dominant_route_node_ids[1:],
            )
        )
        route_segment_ids: list[int] = []
        for start_node_id, end_node_id in dominant_pairs:
            for segment in segment_lookup.values():
                if (
                    segment.start_node_id,
                    segment.end_node_id,
                ) == (start_node_id, end_node_id) or (
                    segment.start_node_id,
                    segment.end_node_id,
                ) == (end_node_id, start_node_id):
                    route_segment_ids.append(segment.segment_id)
                    break
        return tuple(route_segment_ids)

    def _build_generation_order(
        self,
        *,
        cave_network: CaveNetwork,
        segment_lookup: dict[int, CaveSegment],
        dominant_route_segment_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        node_lookup = {node.node_id: node for node in cave_network.nodes}
        ordered_ids = list(dominant_route_segment_ids)
        ordered_set = set(ordered_ids)
        remaining_ids = [
            segment.segment_id
            for segment in cave_network.segments
            if segment.segment_id not in ordered_set
        ]
        remaining_ids.sort(
            key=lambda segment_id: (
                min(
                    node_lookup[segment_lookup[segment_id].start_node_id].along_position,
                    node_lookup[segment_lookup[segment_id].end_node_id].along_position,
                ),
                max(
                    node_lookup[segment_lookup[segment_id].start_node_id].along_position,
                    node_lookup[segment_lookup[segment_id].end_node_id].along_position,
                ),
                segment_lookup[segment_id].segment_id,
            )
        )
        ordered_ids.extend(remaining_ids)
        return tuple(ordered_ids)

    def _build_adaptive_arc_positions(
        self,
        segment: CaveSegment,
        connected_junctions: tuple[CaveJunction, ...],
    ) -> tuple[float, ...]:
        if not segment.points:
            return ()
        total_length = segment.total_length
        if math.isclose(total_length, 0.0):
            return (0.0,)

        positions = [0.0]
        current = 0.0
        while current < total_length:
            spacing = self._adaptive_spacing(
                segment=segment,
                arc_length=current,
                connected_junctions=connected_junctions,
            )
            next_position = min(total_length, current + spacing)
            if math.isclose(next_position, current):
                break
            positions.append(next_position)
            current = next_position
        if not math.isclose(positions[-1], total_length):
            positions.append(total_length)
        return tuple(float(position) for position in positions)

    def _adaptive_spacing(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        connected_junctions: tuple[CaveJunction, ...],
    ) -> float:
        curvature = self._estimate_curvature(segment, arc_length)
        width_gradient = self._estimate_width_gradient(segment, arc_length)
        junction_factor = self._junction_proximity_weight(
            segment=segment,
            arc_length=arc_length,
            connected_junctions=connected_junctions,
        )
        spacing = self.config.maximum_sample_spacing / (
            1.0
            + self.config.curvature_spacing_weight * curvature
            + self.config.width_gradient_spacing_weight * width_gradient
            + self.config.junction_spacing_weight * junction_factor
        )
        return float(
            np.clip(
                spacing,
                self.config.minimum_sample_spacing,
                self.config.maximum_sample_spacing,
            )
        )

    def _build_segment_samples(
        self,
        *,
        segment: CaveSegment,
        connected_junctions: tuple[CaveJunction, ...],
        arc_positions: tuple[float, ...],
        phase: float,
        initial_normal: tuple[float, float, float] | None,
    ) -> list[SectionSample]:
        samples: list[SectionSample] = []
        previous_normal: tuple[float, float, float] | None = initial_normal
        for index, arc_length in enumerate(arc_positions):
            x_coord = self._interpolate_attr(segment, arc_length, "x")
            y_coord = self._interpolate_attr(segment, arc_length, "y")
            surface_z = self._interpolate_attr(segment, arc_length, "elevation")
            cover_thickness = self._interpolate_attr(segment, arc_length, "cover_thickness")
            tangent = self._build_tangent(segment, arc_length)
            normal, binormal = self._build_frame(tangent, previous_normal)
            previous_normal = normal

            width = self._smoothed_width(segment, arc_length)
            height_ratio = self._height_ratio(segment, arc_length)
            tube_height = width * height_ratio
            floor_flatness = self._floor_flatness(segment, arc_length, width)
            roof_arch = self._roof_arch(segment, arc_length)
            lateral_skew = self._lateral_skew(
                segment=segment,
                arc_length=arc_length,
                phase=phase,
            )
            (
                tube_width,
                tube_height,
                floor_flatness,
                roof_arch,
                lateral_skew,
                junction_blend_weight,
                junction_influences,
            ) = self._apply_junction_blending(
                segment=segment,
                arc_length=arc_length,
                connected_junctions=connected_junctions,
                tube_width=width,
                tube_height=tube_height,
                floor_flatness=floor_flatness,
                roof_arch=roof_arch,
                lateral_skew=lateral_skew,
            )
            z_coord, roof_thickness, centerline_depth = self._build_centerline_elevation(
                segment=segment,
                arc_length=arc_length,
                surface_z=surface_z,
                cover_thickness=cover_thickness,
                tube_height=tube_height,
            )
            profile_points = self._build_profile_points(
                tube_width=tube_width,
                tube_height=tube_height,
                floor_flatness=floor_flatness,
                roof_arch=roof_arch,
                lateral_skew=lateral_skew,
            )
            samples.append(
                SectionSample(
                    index=index,
                    segment_id=segment.segment_id,
                    segment_arc_length=arc_length,
                    x=x_coord,
                    y=y_coord,
                    z=z_coord,
                    surface_z=surface_z,
                    cover_thickness=cover_thickness,
                    roof_thickness=roof_thickness,
                    centerline_depth=centerline_depth,
                    tangent=tangent,
                    normal=normal,
                    binormal=binormal,
                    tube_width=tube_width,
                    tube_height=tube_height,
                    floor_flatness=floor_flatness,
                    roof_arch=roof_arch,
                    lateral_skew=lateral_skew,
                    junction_blend_weight=junction_blend_weight,
                    junction_influences=junction_influences,
                    profile_points=profile_points,
                )
            )
        return samples

    def _height_ratio(self, segment: CaveSegment, arc_length: float) -> float:
        roof_competence = self._interpolate_attr(segment, arc_length, "roof_competence")
        width = self._smoothed_width(segment, arc_length)
        raw_ratio = (
            self.config.base_height_ratio
            + 0.08 * (roof_competence - 0.5)
            - 0.06 * np.clip((width - 9.5) / 7.5, 0.0, 1.0)
        )
        return float(
            np.clip(
                raw_ratio,
                self.config.minimum_height_ratio,
                self.config.maximum_height_ratio,
            )
        )

    def _floor_flatness(
        self,
        segment: CaveSegment,
        arc_length: float,
        width: float,
    ) -> float:
        growth_cost = self._interpolate_attr(segment, arc_length, "growth_cost")
        flatness = (
            self.config.floor_flatness_base
            + self.config.floor_flatness_width_weight
            * np.clip((width - 7.0) / 6.0, 0.0, 1.0)
            + 0.06 * growth_cost
        )
        return float(np.clip(flatness, 0.28, 0.92))

    def _roof_arch(self, segment: CaveSegment, arc_length: float) -> float:
        roof_competence = self._interpolate_attr(segment, arc_length, "roof_competence")
        growth_cost = self._interpolate_attr(segment, arc_length, "growth_cost")
        arch = (
            self.config.roof_arch_base
            + self.config.roof_arch_roof_weight * roof_competence
            - 0.08 * growth_cost
        )
        return float(np.clip(arch, 0.92, 1.36))

    def _lateral_skew(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        phase: float,
    ) -> float:
        total_length = max(segment.total_length, 1.0)
        wavelength = max(0.75 * total_length, 70.0)
        return float(
            self.config.lateral_skew_amplitude
            * math.sin((2.0 * math.pi * arc_length / wavelength) + phase)
        )

    def _apply_junction_blending(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        connected_junctions: tuple[CaveJunction, ...],
        tube_width: float,
        tube_height: float,
        floor_flatness: float,
        roof_arch: float,
        lateral_skew: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        tuple[SectionJunctionInfluence, ...],
    ]:
        max_weight = 0.0
        width_scale = 1.0
        height_scale = 1.0
        flatness_delta = 0.0
        arch_delta = 0.0
        skew_scale = 1.0
        width_cap = self.config.maximum_tube_width
        influences: list[SectionJunctionInfluence] = []

        for junction in connected_junctions:
            anchor = self._junction_anchor_arc(segment, junction)
            weight = math.exp(-abs(arc_length - anchor) / max(junction.blend_length, 1.0))
            max_weight = max(max_weight, weight)
            influences.append(
                SectionJunctionInfluence(
                    junction_id=junction.junction_id,
                    kind=junction.kind,
                    weight=float(weight),
                    split_style=junction.split_style,
                    merge_style=junction.merge_style,
                    capacity_bias=junction.capacity_bias,
                )
            )
            if junction.split_style == "pre_widen_then_split" or junction.merge_style == "pre_widen_then_split":
                width_scale += self.config.junction_pre_widen_gain * weight * junction.capacity_bias
                height_scale += 0.10 * weight * junction.capacity_bias
                flatness_delta += 0.10 * weight
                arch_delta += 0.06 * weight
            else:
                width_scale += self.config.junction_constant_envelope_gain * weight * (junction.capacity_bias - 0.85)
                height_scale += 0.03 * weight * junction.capacity_bias
                flatness_delta += 0.04 * weight
                arch_delta += 0.02 * weight
            if junction.kind == "crossing":
                skew_scale *= 0.85
            elif junction.kind == "chamber":
                chamber_weight = weight**1.8
                width_scale += self.config.chamber_widen_gain * chamber_weight
                height_scale += 0.18 * chamber_weight
                flatness_delta += 0.10 * chamber_weight
                arch_delta += 0.08 * chamber_weight
                width_cap = max(
                    width_cap,
                    self.config.maximum_tube_width
                    + chamber_weight * (self.config.chamber_max_tube_width - self.config.maximum_tube_width),
                )

        filtered_influences = tuple(
            sorted(
                (
                    influence
                    for influence in influences
                    if influence.weight >= 0.08
                ),
                key=lambda influence: influence.weight,
                reverse=True,
            )
        )
        return (
            min(tube_width * width_scale, width_cap),
            min(tube_height * height_scale, width_cap * self.config.maximum_height_ratio),
            float(np.clip(floor_flatness + flatness_delta, 0.28, 0.96)),
            float(np.clip(roof_arch + arch_delta, 0.92, 1.45)),
            lateral_skew * skew_scale,
            float(max_weight),
            filtered_influences,
        )

    def _build_centerline_elevation(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        surface_z: float,
        cover_thickness: float,
        tube_height: float,
    ) -> tuple[float, float, float]:
        roof_competence = self._interpolate_attr(segment, arc_length, "roof_competence")
        growth_cost = self._interpolate_attr(segment, arc_length, "growth_cost")
        preferred_roof = (
            self.config.minimum_roof_thickness
            + self.config.preferred_cover_fraction * max(cover_thickness - self.config.minimum_roof_thickness, 0.0)
            + 1.6 * max(roof_competence - 0.5, 0.0)
            - 1.2 * growth_cost
        )
        max_centerline_depth = max(
            0.55 * tube_height,
            min(cover_thickness - 0.9, self.config.maximum_centerline_depth),
        )
        roof_thickness = float(
            np.clip(
                preferred_roof,
                self.config.minimum_roof_thickness,
                max(self.config.minimum_roof_thickness, max_centerline_depth - 0.5 * tube_height),
            )
        )
        centerline_depth = float(
            np.clip(
                roof_thickness + 0.5 * tube_height,
                0.55 * tube_height,
                max_centerline_depth,
            )
        )
        return (
            surface_z - centerline_depth,
            roof_thickness,
            centerline_depth,
        )

    def _junction_anchor_arc(self, segment: CaveSegment, junction: CaveJunction) -> float:
        if segment.start_node_id in junction.node_ids:
            return 0.0
        if segment.end_node_id in junction.node_ids:
            return segment.total_length
        return min(
            (point.arc_length for point in segment.points),
            key=lambda arc_length: math.hypot(
                self._interpolate_attr(segment, arc_length, "x") - junction.center_x,
                self._interpolate_attr(segment, arc_length, "y") - junction.center_y,
            ),
        )

    def _junction_proximity_weight(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        connected_junctions: tuple[CaveJunction, ...],
    ) -> float:
        if not connected_junctions:
            return 0.0
        return max(
            math.exp(
                -abs(arc_length - self._junction_anchor_arc(segment, junction))
                / max(junction.blend_length, 1.0)
            )
            for junction in connected_junctions
        )

    def _build_profile_points(
        self,
        *,
        tube_width: float,
        tube_height: float,
        floor_flatness: float,
        roof_arch: float,
        lateral_skew: float,
    ) -> tuple[tuple[float, float], ...]:
        half_width = 0.5 * tube_width
        half_height = 0.5 * tube_height
        resolution = max(self.config.profile_resolution // 2, 8)
        x_values = np.linspace(-half_width, half_width, resolution, dtype=float)
        normalized = np.clip(np.abs(x_values) / max(half_width, 1.0), 0.0, 1.0)
        top_exp = float(np.clip(1.78 - 0.24 * (roof_arch - 1.0), 1.35, 2.05))
        bottom_exp = float(np.clip(3.0 + 2.4 * floor_flatness, 2.6, 5.5))
        floor_depth_factor = float(np.clip(0.74 - 0.22 * floor_flatness, 0.45, 0.78))
        skew_offset = lateral_skew * half_width

        roof_profile = [
            (
                float(x_coord + skew_offset * (1.0 - normalized_value**2)),
                float(
                    half_height
                    * max(0.0, 1.0 - normalized_value**top_exp) ** (1.0 / top_exp)
                ),
            )
            for x_coord, normalized_value in zip(x_values, normalized, strict=True)
        ]
        floor_profile = [
            (
                float(x_coord + 0.55 * skew_offset * (1.0 - normalized_value**2)),
                float(
                    -half_height
                    * floor_depth_factor
                    * max(0.0, 1.0 - normalized_value**bottom_exp) ** (1.0 / bottom_exp)
                ),
            )
            for x_coord, normalized_value in zip(reversed(x_values), reversed(normalized), strict=True)
        ]
        closed_profile = floor_profile + roof_profile + [floor_profile[0]]
        return tuple(closed_profile)

    def _smoothed_width(self, segment: CaveSegment, arc_length: float) -> float:
        window = max(12.0, 0.08 * max(segment.total_length, 1.0))
        positions = np.array(
            [
                max(0.0, arc_length - window),
                arc_length,
                min(segment.total_length, arc_length + window),
            ],
            dtype=float,
        )
        widths = np.array(
            [self._interpolate_attr(segment, value, "width") for value in positions],
            dtype=float,
        )
        return float(np.mean(widths))

    def _estimate_width_gradient(self, segment: CaveSegment, arc_length: float) -> float:
        delta = max(8.0, 0.05 * max(segment.total_length, 1.0))
        start_width = self._interpolate_attr(segment, max(0.0, arc_length - delta), "width")
        end_width = self._interpolate_attr(segment, min(segment.total_length, arc_length + delta), "width")
        return abs(end_width - start_width) / max(2.0 * delta, 1.0)

    def _estimate_curvature(self, segment: CaveSegment, arc_length: float) -> float:
        delta = max(6.0, 0.05 * max(segment.total_length, 1.0))
        previous = self._interpolate_position(segment, max(0.0, arc_length - delta))
        current = self._interpolate_position(segment, arc_length)
        next_position = self._interpolate_position(segment, min(segment.total_length, arc_length + delta))
        vector_a = np.array(
            [
                current[0] - previous[0],
                current[1] - previous[1],
                current[2] - previous[2],
            ],
            dtype=float,
        )
        vector_b = np.array(
            [
                next_position[0] - current[0],
                next_position[1] - current[1],
                next_position[2] - current[2],
            ],
            dtype=float,
        )
        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        if math.isclose(norm_a, 0.0) or math.isclose(norm_b, 0.0):
            return 0.0
        cos_angle = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
        angle = math.acos(cos_angle)
        return angle / max(0.5 * (norm_a + norm_b), 1.0)

    def _build_tangent(
        self,
        segment: CaveSegment,
        arc_length: float,
    ) -> tuple[float, float, float]:
        delta = max(4.0, 0.03 * max(segment.total_length, 1.0))
        previous = self._interpolate_position(segment, max(0.0, arc_length - delta))
        next_position = self._interpolate_position(segment, min(segment.total_length, arc_length + delta))
        vector = np.array(
            [
                next_position[0] - previous[0],
                next_position[1] - previous[1],
                next_position[2] - previous[2],
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(vector))
        if math.isclose(norm, 0.0):
            return (1.0, 0.0, 0.0)
        tangent = vector / norm
        return (float(tangent[0]), float(tangent[1]), float(tangent[2]))

    def _build_frame(
        self,
        tangent: tuple[float, float, float],
        previous_normal: tuple[float, float, float] | None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        tangent_vector = np.array(tangent, dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.cross(up, tangent_vector)
        normal_norm = float(np.linalg.norm(normal))
        if math.isclose(normal_norm, 0.0):
            normal = (
                np.array(previous_normal, dtype=float)
                if previous_normal is not None
                else np.array([1.0, 0.0, 0.0], dtype=float)
            )
        else:
            normal /= normal_norm
        if previous_normal is not None and float(np.dot(normal, np.array(previous_normal, dtype=float))) < 0.0:
            normal *= -1.0
        binormal = np.cross(tangent_vector, normal)
        binormal_norm = float(np.linalg.norm(binormal))
        if math.isclose(binormal_norm, 0.0):
            binormal = up
        else:
            binormal /= binormal_norm
        return (
            (float(normal[0]), float(normal[1]), float(normal[2])),
            (float(binormal[0]), float(binormal[1]), float(binormal[2])),
        )

    def _interpolate_position(
        self,
        segment: CaveSegment,
        arc_length: float,
    ) -> tuple[float, float, float]:
        return (
            self._interpolate_attr(segment, arc_length, "x"),
            self._interpolate_attr(segment, arc_length, "y"),
            self._interpolate_attr(segment, arc_length, "elevation"),
        )

    @staticmethod
    def _interpolate_attr(
        segment: CaveSegment,
        arc_length: float,
        attr: str,
    ) -> float:
        if not segment.points:
            return 0.0
        if len(segment.points) == 1:
            return float(getattr(segment.points[0], attr))
        arc_values = np.array([point.arc_length for point in segment.points], dtype=float)
        attr_values = np.array([getattr(point, attr) for point in segment.points], dtype=float)
        return float(np.interp(arc_length, arc_values, attr_values))
