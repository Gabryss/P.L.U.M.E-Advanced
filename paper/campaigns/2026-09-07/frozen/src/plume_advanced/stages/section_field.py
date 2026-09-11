"""Stage C: geometry-ready section field generation around the cave network."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from plume_advanced.procedural import procedural_rng
from plume_advanced.stability import RoofStabilityModel
from plume_advanced.stages.network import CaveJunction, CaveNetwork, CaveSegment


@dataclass(frozen=True)
class SectionFieldConfig:
    """Parameters controlling stage-C section-field generation."""

    random_seed: int | None = None
    base_height_ratio: float = 0.66
    minimum_height_ratio: float = 0.34
    maximum_height_ratio: float = 0.96
    width_scale_median: float = 0.84
    width_scale_log_sigma: float = 0.60
    width_longitudinal_variation: float = 0.18
    height_ratio_variation: float = 0.18
    height_ratio_longitudinal_variation: float = 0.07
    minimum_tube_width: float = 1.0
    minimum_tube_height: float = 0.35
    maximum_tube_width: float = 12.0
    chamber_max_tube_width: float = 24.0
    minimum_sample_spacing: float = 8.0
    maximum_sample_spacing: float = 34.0
    sampling_policy: str = "adaptive"
    uniform_sample_spacing: float = 16.0
    reference_sample_spacing: float = 2.0
    curvature_spacing_weight: float = 120.0
    width_gradient_spacing_weight: float = 34.0
    junction_spacing_weight: float = 1.15
    profile_resolution: int = 28
    floor_flatness_base: float = 0.36
    floor_flatness_width_weight: float = 0.12
    floor_relief_base: float = 0.13
    floor_relief_variation: float = 0.035
    roof_arch_base: float = 1.08
    roof_arch_roof_weight: float = 0.18
    wall_roughness_base: float = 0.14
    wall_roughness_variation: float = 0.045
    morphology_gradient_strength: float = 0.32
    morphology_correlation_length: float = 140.0
    morphology_regime_strength: float = 0.72
    morphology_family_spread: float = 0.82
    morphology_flux_width_gain: float = 0.16
    morphology_age_width_gain: float = 0.10
    morphology_floor_relief_max: float = 0.42
    morphology_wall_roughness_max: float = 0.30
    profile_shape_variation: float = 0.92
    lateral_skew_amplitude: float = 0.16
    centerline_wobble_amplitude: float = 2.4
    centerline_wobble_wavelength: float = 95.0
    junction_pre_widen_gain: float = 0.18
    junction_constant_envelope_gain: float = 0.05
    chamber_widen_gain: float = 0.45
    drained_pool_width_scale: float = 1.0
    drained_pool_height_ratio_limit: float = 0.42
    drained_pool_floor_flatness: float = 0.84
    drained_pool_roof_arch: float = 1.04
    drained_pool_transition_power: float = 1.35
    minimum_roof_thickness: float = 6.0
    maximum_centerline_depth: float = 26.0
    preferred_cover_fraction: float = 0.34
    vertical_level_spacing: float = 0.0  # zero: actual passage height plus rock clearance
    minimum_vertical_clearance: float = 2.0
    maximum_uphill_grade: float = 0.015
    level_transition_fraction: float = 0.18
    gravity_m_s2: float = 9.80665
    rock_density_kg_m3: float = 2900.0
    effective_tensile_strength_pa: float = 3_000_000.0
    roof_safety_factor: float = 1.5
    bench_strength: float = 0.6
    floor_incision_ratio: float = 0.10

    @property
    def roof_stability_model(self) -> RoofStabilityModel:
        return RoofStabilityModel(
            self.gravity_m_s2, self.rock_density_kg_m3,
            self.effective_tensile_strength_pa, self.roof_safety_factor,
        )


@dataclass(frozen=True)
class SectionJunctionInfluence:
    """One junction influence contributing to a section sample."""

    junction_id: int
    kind: str
    weight: float
    split_style: str
    merge_style: str
    capacity_bias: float
    blend_length_m: float = 0.0
    chamber_type: str = ""
    room_weight: float = 0.0
    target_width_m: float = 0.0
    target_height_m: float = 0.0


@dataclass(frozen=True)
class _SegmentMorphologyState:
    """Fixed-draw latent morphology shared by every sample on one segment."""

    width_scale: float
    height_ratio_offset: float
    floor_relief: float
    wall_roughness: float
    skew_bias: float
    primary_phase: float
    secondary_phase: float
    floor_phase: float
    shape_bias: float
    roof_bias: float
    floor_bias: float
    asymmetry_bias: float
    parent_segment_id: int | None = None
    regime: str = "balanced"


@dataclass(frozen=True)
class _WorldMorphologyRegime:
    """Continuous world-level morphology controls shared by all segments."""

    width_bias: float
    height_bias: float
    floor_bias: float
    roof_bias: float
    shape_bias: float
    asymmetry_bias: float


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
    lava_flux: float = 0.0
    lava_temperature_k: float = 0.0
    lava_age_s: float = 0.0
    flow_maturity: float = 0.0
    morphology_regime: str = "balanced"
    morphology_family_score: float = 0.0
    parent_morphology_segment_id: int | None = None
    junction_blend_length_m: float = 0.0
    floor_world_z: float = 0.0
    roof_world_z: float = 0.0
    maximum_stable_width_m: float = 0.0
    maximum_stable_height_m: float = 0.0
    roof_demand_ratio: float = 0.0
    collapse_required: bool = False


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
                "mean_lava_flux": 0.0,
                "mean_lava_temperature_k": 0.0,
                "max_lava_age_s": 0.0,
                "width_coefficient_of_variation": 0.0,
                "height_ratio_standard_deviation": 0.0,
                "mean_shape_change_per_100m": 0.0,
                "morphology_regime_count": 0.0,
                "morphology_family_score_standard_deviation": 0.0,
                "drained_pool_sample_count": 0.0,
                "drained_pool_max_width_m": 0.0,
                "drained_pool_mean_aspect_ratio": 0.0,
                "unstable_section_count": 0.0,
                "maximum_roof_demand_ratio": 0.0,
            }

        all_samples = [
            sample for segment_field in self.segment_fields for sample in segment_field.samples
        ]
        tube_widths = np.array([sample.tube_width for sample in all_samples], dtype=float)
        tube_heights = np.array([sample.tube_height for sample in all_samples], dtype=float)
        height_ratios = tube_heights / np.maximum(tube_widths, 1e-9)
        shape_change_rates: list[float] = []
        pool_samples = [
            sample
            for sample in all_samples
            if any(
                influence.chamber_type == "drained_lava_pool" and influence.room_weight >= 0.08
                for influence in sample.junction_influences
            )
        ]
        width_scale = max(float(np.mean(tube_widths)), 1e-9)
        height_scale = max(float(np.mean(tube_heights)), 1e-9)
        for segment_field in self.segment_fields:
            for start, end in zip(segment_field.samples, segment_field.samples[1:]):
                spacing = max(end.segment_arc_length - start.segment_arc_length, 1e-9)
                delta = np.asarray(
                    [
                        (end.tube_width - start.tube_width) / width_scale,
                        (end.tube_height - start.tube_height) / height_scale,
                        end.floor_flatness - start.floor_flatness,
                        end.roof_arch - start.roof_arch,
                        end.lateral_skew - start.lateral_skew,
                    ],
                    dtype=float,
                )
                shape_change_rates.append(100.0 * float(np.linalg.norm(delta)) / spacing)
        return {
            "segment_field_count": float(len(self.segment_fields)),
            "unstable_section_count": float(sum(s.collapse_required for s in all_samples)),
            "maximum_roof_demand_ratio": max(s.roof_demand_ratio for s in all_samples),
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
            "mean_lava_flux": float(np.mean([sample.lava_flux for sample in all_samples])),
            "mean_lava_temperature_k": float(
                np.mean([sample.lava_temperature_k for sample in all_samples])
            ),
            "max_lava_age_s": float(max(sample.lava_age_s for sample in all_samples)),
            "width_coefficient_of_variation": float(
                np.std(tube_widths) / max(float(np.mean(tube_widths)), 1e-9)
            ),
            "height_ratio_standard_deviation": float(np.std(height_ratios)),
            "mean_shape_change_per_100m": float(
                np.mean(shape_change_rates) if shape_change_rates else 0.0
            ),
            "morphology_regime_count": float(
                len({sample.morphology_regime for sample in all_samples})
            ),
            "morphology_family_score_standard_deviation": float(
                np.std([sample.morphology_family_score for sample in all_samples])
            ),
            "drained_pool_sample_count": float(len(pool_samples)),
            "drained_pool_max_width_m": float(
                max((sample.tube_width for sample in pool_samples), default=0.0)
            ),
            "drained_pool_mean_aspect_ratio": float(
                np.mean(
                    [sample.tube_width / max(sample.tube_height, 1e-9) for sample in pool_samples]
                )
                if pool_samples
                else 0.0
            ),
        }


class SectionFieldGenerator:
    """Build a smooth lava-tube section field around the cave-network skeleton."""

    def __init__(self, config: SectionFieldConfig | None = None) -> None:
        self.config = config or SectionFieldConfig()
        self._world_regime = self._sample_world_regime()

    def generate(self, cave_network: CaveNetwork) -> SectionField:
        segment_lookup = {segment.segment_id: segment for segment in cave_network.segments}
        flow_points = [point for segment in cave_network.segments for point in segment.points]
        maximum_lava_age_s = max((point.age_s for point in flow_points), default=0.0)
        maximum_lava_temperature_k = max(
            (point.temperature_k for point in flow_points),
            default=0.0,
        )
        minimum_lava_temperature_k = min(
            (point.temperature_k for point in flow_points),
            default=maximum_lava_temperature_k,
        )
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
        incoming_morphologies: dict[int, list[tuple[int, _SegmentMorphologyState]]] = {}
        segment_fields_by_id: dict[int, SegmentSectionField] = {}
        segment_fields: list[SegmentSectionField] = []
        for segment_id in generation_order:
            segment = segment_lookup[segment_id]
            connected_junctions = tuple(
                junction
                for junction in cave_network.junctions
                if segment.segment_id in junction.segment_ids
            )
            arc_positions = self._build_arc_positions(segment, connected_junctions)
            parent_record = self._directed_parent_morphology(
                incoming_morphologies.get(segment.start_node_id, ())
            )
            morphology = self._sample_segment_morphology(
                procedural_rng(
                    self.config.random_seed,
                    "segment-morphology",
                    segment.segment_id,
                ),
                parent=parent_record[0] if parent_record else None,
                parent_segment_id=parent_record[1] if parent_record else None,
            )
            samples = self._build_segment_samples(
                segment=segment,
                connected_junctions=connected_junctions,
                arc_positions=arc_positions,
                morphology=morphology,
                maximum_lava_age_s=maximum_lava_age_s,
                maximum_lava_temperature_k=maximum_lava_temperature_k,
                minimum_lava_temperature_k=minimum_lava_temperature_k,
                initial_normal=(
                    node_normal_preferences.get(segment.start_node_id)
                    or node_normal_preferences.get(segment.end_node_id)
                ),
            )
            if samples:
                node_normal_preferences.setdefault(segment.start_node_id, samples[0].normal)
                node_normal_preferences.setdefault(segment.end_node_id, samples[-1].normal)
                incoming_morphologies.setdefault(segment.end_node_id, []).append(
                    (segment.segment_id, morphology)
                )
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
                    connected_junction_ids=segment_fields_by_id[
                        segment.segment_id
                    ].connected_junction_ids,
                    samples=segment_fields_by_id[segment.segment_id].samples,
                )
            )
        segment_fields = self._harmonize_connections(cave_network, segment_fields)
        # Assess final profiles after junction blending, grade adjustments and
        # frame transport. Nominal height/cover controls are not the final roof.
        segment_fields = [
            replace(field, samples=tuple(self._assess_roof(sample) for sample in field.samples))
            for field in segment_fields
        ]
        return SectionField(
            config=self.config,
            segment_fields=tuple(segment_fields),
            dominant_route_segment_ids=dominant_route_segment_ids,
        )

    def _assess_roof(self, sample: SectionSample) -> SectionSample:
        profile = np.asarray(sample.profile_points, dtype=float)
        elevations = sample.z + profile[:, 0] * sample.normal[2] + profile[:, 1] * sample.binormal[2]
        floor, roof = float(np.min(elevations)), float(np.max(elevations))
        assessment = self.config.roof_stability_model.assess(
            width_m=float(np.ptp(profile[:, 0])), height_m=roof - floor,
            floor_depth_m=sample.surface_z - floor,
        )
        return replace(
            sample, roof_thickness=assessment.roof_thickness_m,
            floor_world_z=floor, roof_world_z=roof,
            maximum_stable_width_m=assessment.maximum_width_m,
            maximum_stable_height_m=assessment.maximum_height_m,
            roof_demand_ratio=assessment.demand_ratio,
            collapse_required=assessment.failed,
        )

    def _harmonize_connections(
        self, network: CaveNetwork, fields: list[SegmentSectionField]
    ) -> list[SegmentSectionField]:
        """Share floor elevation and ease branch directions at graph nodes.

        Morphology remains independent away from a connection. Only endpoints
        sharing a graph node participate, so an underpass keeps its clearance.
        """
        by_id = {field.segment_id: field for field in fields}
        connections: dict[int, list[tuple[int, int]]] = {}
        for segment in network.segments:
            if not by_id[segment.segment_id].samples:
                continue
            for node, index in ((segment.start_node_id, 0), (segment.end_node_id, -1)):
                connections.setdefault(node, []).append((segment.segment_id, index))
        targets: dict[tuple[int, int], tuple[float, np.ndarray, SectionSample]] = {}
        for incident in connections.values():
            if len(incident) < 2:
                continue
            endpoints = [by_id[sid].samples[index] for sid, index in incident]
            floor = min(self._sample_floor(sample) for sample in endpoints)
            # Use the narrowest incident envelope as the shared node target.
            # This preserves exact continuity even when one incident branch is
            # chamber-expanded but another is constrained by passage caps.
            reference = min(endpoints, key=lambda sample: sample.tube_width)
            direction = np.asarray(reference.tangent, dtype=float)
            for (sid, index), endpoint in zip(incident, endpoints):
                oriented = direction if np.dot(direction, endpoint.tangent) >= 0.0 else -direction
                targets[(sid, index)] = (floor, oriented, reference)

        result = []
        for field in fields:
            samples = field.samples
            if len(samples) < 2 or not any(
                (field.segment_id, index) in targets for index in (0, -1)
            ):
                result.append(field)
                continue
            length = samples[-1].segment_arc_length
            updated = []
            for sample in samples:
                xy = np.asarray((sample.x, sample.y))
                tangent = np.asarray(sample.tangent)
                profile = np.asarray(sample.profile_points)
                width, height = sample.tube_width, sample.tube_height
                floor_delta = 0.0
                if math.isclose(sample.segment_arc_length, 0.0, abs_tol=1e-9):
                    endpoint_indices: tuple[int, ...] = (0,)
                elif math.isclose(sample.segment_arc_length, length, abs_tol=1e-9):
                    endpoint_indices = (-1,)
                else:
                    endpoint_indices = (0, -1)
                for index in endpoint_indices:
                    target = targets.get((field.segment_id, index))
                    if target is None:
                        continue
                    endpoint = samples[index]
                    reach = min(2.0 * endpoint.tube_width, 0.45 * length)
                    along = sample.segment_arc_length - endpoint.segment_arc_length
                    u = max(0.0, 1.0 - abs(along) / max(reach, 1e-9))
                    weight = u * u * (3.0 - 2.0 * u)
                    target_floor, direction, reference = target
                    desired_xy = np.asarray((endpoint.x, endpoint.y)) + along * direction[:2]
                    correction = desired_xy - np.asarray((sample.x, sample.y))
                    correction *= min(
                        1.0, 0.4 * endpoint.tube_width / max(np.linalg.norm(correction), 1e-9)
                    )
                    xy = xy + weight * correction
                    tangent = (1.0 - weight) * tangent + weight * direction
                    floor_delta += weight * (target_floor - self._sample_floor(endpoint))
                    target_profile = np.asarray(reference.profile_points).copy()
                    if target_profile.shape == profile.shape:
                        profile = (1.0 - weight) * profile + weight * target_profile
                    width += weight * (reference.tube_width - sample.tube_width)
                    height += weight * (reference.tube_height - sample.tube_height)
                chamber_caps = []
                for influence in sample.junction_influences:
                    if influence.kind != "chamber":
                        continue
                    if influence.chamber_type == "drained_lava_pool":
                        chamber_caps.append(
                            self.config.maximum_tube_width
                            + influence.room_weight
                            * max(
                                influence.target_width_m - self.config.maximum_tube_width,
                                0.0,
                            )
                        )
                    else:
                        chamber_caps.append(
                            self.config.maximum_tube_width
                            + influence.weight**1.8
                            * (self.config.chamber_max_tube_width - self.config.maximum_tube_width)
                        )
                width_cap = max([self.config.maximum_tube_width, *chamber_caps])
                limited_width = min(width, width_cap)
                limited_height = min(height, width_cap * self.config.maximum_height_ratio)
                profile = profile * np.asarray((limited_width / width, limited_height / height))
                width, height = limited_width, limited_height
                tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
                normal, binormal = self._build_frame(tuple(tangent), sample.normal)
                new_floor_offset = float(
                    np.min(profile[:, 0] * normal[2] + profile[:, 1] * binormal[2])
                )
                z = self._sample_floor(sample) + floor_delta - new_floor_offset
                updated.append(
                    replace(
                        sample,
                        x=float(xy[0]),
                        y=float(xy[1]),
                        z=z,
                        tube_width=width,
                        tube_height=height,
                        profile_points=tuple(
                            (float(point[0]), float(point[1])) for point in profile
                        ),
                        tangent=(float(tangent[0]), float(tangent[1]), float(tangent[2])),
                        normal=normal,
                        binormal=binormal,
                        centerline_depth=sample.surface_z - z,
                        roof_thickness=sample.surface_z - z - 0.5 * height,
                    )
                )
            # Recompute frames from the shaped centerline. Blending old frames
            # alone can point a section backwards on a short, curved branch.
            desired_floors = [self._sample_floor(sample) for sample in updated]
            for _ in range(2):
                centers = np.asarray([(sample.x, sample.y, sample.z) for sample in updated])
                steps = np.diff(centers, axis=0)
                steps /= np.maximum(np.linalg.norm(steps, axis=1, keepdims=True), 1e-9)
                tangents = np.vstack((steps[0], steps[:-1] + steps[1:], steps[-1]))
                tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
                previous_normal = updated[0].normal
                reframed = []
                for sample, tangent, floor in zip(updated, tangents, desired_floors):
                    normal, binormal = self._build_frame(tuple(tangent), previous_normal)
                    previous_normal = normal
                    profile = np.asarray(sample.profile_points)
                    offset = float(np.min(profile[:, 0] * normal[2] + profile[:, 1] * binormal[2]))
                    z = floor - offset
                    reframed.append(
                        replace(
                            sample,
                            z=z,
                            tangent=(float(tangent[0]), float(tangent[1]), float(tangent[2])),
                            normal=normal,
                            binormal=binormal,
                            centerline_depth=sample.surface_z - z,
                            roof_thickness=sample.surface_z - z - 0.5 * sample.tube_height,
                        )
                    )
                updated = reframed
            result.append(replace(field, samples=tuple(updated)))
        return result

    @staticmethod
    def _sample_floor(sample: SectionSample) -> float:
        profile = np.asarray(sample.profile_points)
        return sample.z + float(
            np.min(profile[:, 0] * sample.normal[2] + profile[:, 1] * sample.binormal[2])
        )

    def _build_arc_positions(
        self,
        segment: CaveSegment,
        connected_junctions: tuple[CaveJunction, ...],
    ) -> tuple[float, ...]:
        policy = self.config.sampling_policy
        if policy == "adaptive":
            return self._build_adaptive_arc_positions(segment, connected_junctions)
        spacing = (
            self.config.uniform_sample_spacing
            if policy == "uniform"
            else self.config.reference_sample_spacing
        )
        return self._build_uniform_arc_positions(segment, spacing)

    @staticmethod
    def _build_uniform_arc_positions(
        segment: CaveSegment,
        spacing: float,
    ) -> tuple[float, ...]:
        if not segment.points:
            return ()
        if math.isclose(segment.total_length, 0.0):
            return (0.0,)
        count = max(1, int(math.ceil(segment.total_length / spacing)))
        return tuple(
            float(value) for value in np.linspace(0.0, segment.total_length, count + 1, dtype=float)
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
            candidates = [
                segment
                for segment in segment_lookup.values()
                if (segment.start_node_id, segment.end_node_id) == (start_node_id, end_node_id)
            ]
            if candidates:
                route_segment_ids.append(
                    max(candidates, key=lambda segment: segment.mean_flux).segment_id
                )
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
        morphology: _SegmentMorphologyState,
        maximum_lava_age_s: float,
        maximum_lava_temperature_k: float,
        minimum_lava_temperature_k: float,
        initial_normal: tuple[float, float, float] | None,
    ) -> list[SectionSample]:
        samples: list[SectionSample] = []
        previous_normal: tuple[float, float, float] | None = initial_normal
        for index, arc_length in enumerate(arc_positions):
            history_morphology = self._apply_emplacement_morphology(
                segment,
                morphology,
            )
            local_morphology = self._morphology_at(
                segment=segment,
                arc_length=arc_length,
                base=history_morphology,
            )
            x_coord = self._interpolate_attr(segment, arc_length, "x")
            y_coord = self._interpolate_attr(segment, arc_length, "y")
            surface_z = self._interpolate_attr(segment, arc_length, "elevation")
            cover_thickness = self._interpolate_attr(segment, arc_length, "cover_thickness")
            tangent = self._build_tangent(segment, arc_length)
            normal, binormal = self._build_frame(tangent, previous_normal)
            previous_normal = normal

            lava_flux = self._interpolate_attr(segment, arc_length, "flux")
            lava_temperature_k = self._interpolate_attr(
                segment,
                arc_length,
                "temperature_k",
            )
            lava_age_s = self._interpolate_attr(segment, arc_length, "age_s")
            age_fraction = float(np.clip(lava_age_s / max(maximum_lava_age_s, 1e-9), 0.0, 1.0))
            temperature_span = max(
                maximum_lava_temperature_k - minimum_lava_temperature_k,
                1e-9,
            )
            cooling_fraction = float(
                np.clip(
                    (maximum_lava_temperature_k - lava_temperature_k) / temperature_span,
                    0.0,
                    1.0,
                )
            )
            flow_maturity = 0.60 * age_fraction + 0.40 * cooling_fraction

            width = self._section_width(segment, arc_length, local_morphology)
            height_ratio = self._height_ratio(
                segment,
                arc_length,
                local_morphology,
                flow_maturity,
            )
            raw_tube_height = width * height_ratio
            height_softness = max(0.15 * self.config.minimum_tube_height, 0.05)
            tube_height = float(
                self.config.minimum_tube_height
                + height_softness
                * np.logaddexp(
                    0.0,
                    (raw_tube_height - self.config.minimum_tube_height) / height_softness,
                )
            )
            floor_flatness = self._floor_flatness(
                segment,
                arc_length,
                width,
                flow_maturity,
                local_morphology,
            )
            roof_arch = self._roof_arch(segment, arc_length, local_morphology)
            lateral_skew = self._lateral_skew(
                segment=segment,
                arc_length=arc_length,
                phase=local_morphology.primary_phase,
                bias=local_morphology.skew_bias,
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
            junction_blend_length = max(
                (influence.blend_length_m for influence in junction_influences),
                default=0.0,
            )
            z_coord, roof_thickness, centerline_depth = self._build_centerline_elevation(
                segment=segment,
                arc_length=arc_length,
                surface_z=surface_z,
                cover_thickness=cover_thickness,
                tube_height=tube_height,
            )
            lateral_offset, vertical_offset = self._centerline_wobble_offsets(
                segment=segment,
                arc_length=arc_length,
                phase=local_morphology.primary_phase,
                junction_blend_weight=junction_blend_weight,
            )
            x_coord += lateral_offset * normal[0]
            y_coord += lateral_offset * normal[1]
            z_coord += vertical_offset
            centerline_depth -= vertical_offset
            roof_thickness = centerline_depth - 0.5 * tube_height
            profile_points = self._build_profile_points(
                tube_width=tube_width,
                tube_height=tube_height,
                floor_flatness=floor_flatness,
                roof_arch=roof_arch,
                lateral_skew=lateral_skew,
                wall_roughness=float(
                    np.clip(
                        local_morphology.wall_roughness * (0.85 + 0.30 * flow_maturity),
                        0.0,
                        self.config.morphology_wall_roughness_max,
                    )
                ),
                floor_relief=local_morphology.floor_relief
                * (
                    1.0
                    - 0.78
                    * max(
                        (
                            influence.room_weight
                            for influence in junction_influences
                            if influence.chamber_type == "drained_lava_pool"
                        ),
                        default=0.0,
                    )
                ),
                shape_bias=local_morphology.shape_bias,
                roof_bias=local_morphology.roof_bias,
                floor_bias=local_morphology.floor_bias,
                asymmetry_bias=local_morphology.asymmetry_bias,
                roughness_phase=self._morphology_phase(
                    segment,
                    arc_length,
                    local_morphology.secondary_phase,
                    wavelength_fraction=0.38,
                ),
                floor_phase=self._morphology_phase(
                    segment,
                    arc_length,
                    local_morphology.floor_phase,
                    wavelength_fraction=0.52,
                ),
                bench_strength=self.config.bench_strength
                * float(np.clip(local_morphology.floor_bias + 0.35, 0.0, 1.0))
                * (0.25 + 0.75 * flow_maturity),
                floor_incision_ratio=self.config.floor_incision_ratio
                * (0.5 + 0.5 * math.sin(self._morphology_phase(
                    segment, arc_length, local_morphology.floor_phase,
                    wavelength_fraction=0.65,
                )))
                * (1.0 - junction_blend_weight),
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
                    lava_flux=lava_flux,
                    lava_temperature_k=lava_temperature_k,
                    lava_age_s=lava_age_s,
                    flow_maturity=flow_maturity,
                    morphology_regime=local_morphology.regime,
                    morphology_family_score=float(local_morphology.shape_bias),
                    parent_morphology_segment_id=local_morphology.parent_segment_id,
                    junction_blend_length_m=junction_blend_length,
                )
            )
        return self._apply_vertical_level_profile(segment, samples)

    def _apply_vertical_level_profile(
        self,
        segment: CaveSegment,
        samples: list[SectionSample],
    ) -> list[SectionSample]:
        """Fit a smooth level displacement to available cover and flow grade.

        A level label requests separation; short or shallow reaches may have
        insufficient space to realize it. Endpoints remain exact graph joins.
        """

        if segment.z_level == 0 or len(samples) < 3 or segment.total_length <= 1e-6:
            return samples

        arc = np.asarray([sample.segment_arc_length for sample in samples], dtype=float)
        progress = np.clip(arc / max(segment.total_length, 1e-9), 0.0, 1.0)
        # A full-span smooth rise/fall has no artificial flat deck or short
        # exit ramp. Limit its amplitude as a whole instead of clipping points
        # against the roof, which used to create plateaus and sharp corners.
        envelope = np.sin(math.pi * progress) ** 2
        base_z = np.asarray([sample.z for sample in samples], dtype=float)
        surface_z = np.asarray([sample.surface_z for sample in samples], dtype=float)
        tube_height = np.asarray([sample.tube_height for sample in samples], dtype=float)
        cover = np.asarray([sample.cover_thickness for sample in samples], dtype=float)
        separation = max(
            self.config.vertical_level_spacing,
            float(np.max(tube_height)) + self.config.minimum_vertical_clearance,
        )

        minimum_depth = self.config.minimum_roof_thickness + 0.5 * tube_height
        maximum_depth = np.maximum(
            minimum_depth,
            cover - self.config.minimum_vertical_clearance,
        )
        minimum_z = surface_z - maximum_depth
        maximum_z = surface_z - minimum_depth
        unit_offset = float(segment.z_level) * envelope
        limits = [separation]
        positive, negative = unit_offset > 1e-9, unit_offset < -1e-9
        if np.any(positive):
            limits.append(float(np.min((maximum_z[positive] - base_z[positive]) / unit_offset[positive])))
        if np.any(negative):
            limits.append(float(np.min((minimum_z[negative] - base_z[negative]) / unit_offset[negative])))
        growing = np.diff(unit_offset) > 1e-9
        if np.any(growing):
            # Never introduce a new excessive uphill reach. Existing host
            # gradients are handled separately from the level displacement.
            allowance = np.maximum(
                self.config.maximum_uphill_grade * np.diff(arc) - np.diff(base_z), 0.0
            )
            limits.append(float(np.min(allowance[growing] / np.diff(unit_offset)[growing])))
        amplitude = max(0.0, min(limits))
        profile_z = base_z + amplitude * unit_offset
        profile_z[0], profile_z[-1] = base_z[0], base_z[-1]

        resolved: list[SectionSample] = []
        for sample, z_coord in zip(samples, profile_z, strict=True):
            centerline_depth = sample.surface_z - float(z_coord)
            resolved.append(
                replace(
                    sample,
                    z=float(z_coord),
                    centerline_depth=centerline_depth,
                    roof_thickness=centerline_depth - 0.5 * sample.tube_height,
                )
            )
        return resolved

    def _centerline_wobble_offsets(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        phase: float,
        junction_blend_weight: float,
    ) -> tuple[float, float]:
        total_length = max(segment.total_length, 1.0)
        if total_length < 2.5 * self.config.maximum_sample_spacing:
            return 0.0, 0.0

        endpoint_envelope = math.sin(math.pi * np.clip(arc_length / total_length, 0.0, 1.0))
        endpoint_envelope = max(endpoint_envelope, 0.0) ** 1.35
        junction_envelope = 1.0 - 0.75 * np.clip(junction_blend_weight, 0.0, 1.0)
        envelope = endpoint_envelope * junction_envelope
        if envelope <= 1e-6:
            return 0.0, 0.0

        wavelength = max(self.config.centerline_wobble_wavelength, 1.0)
        segment_phase = phase + 0.73 * segment.segment_id
        wave = math.sin(
            (2.0 * math.pi * arc_length / wavelength) + segment_phase
        ) + 0.42 * math.sin(
            (2.0 * math.pi * arc_length / (0.47 * wavelength)) + 1.7 * segment_phase
        )
        lateral = self.config.centerline_wobble_amplitude * envelope * wave
        vertical = (
            0.22
            * self.config.centerline_wobble_amplitude
            * envelope
            * math.sin((2.0 * math.pi * arc_length / (1.35 * wavelength)) + 0.6 * segment_phase)
        )
        return float(lateral), float(vertical)

    def _sample_world_regime(self) -> _WorldMorphologyRegime:
        rng = procedural_rng(self.config.random_seed, "world-morphology-regime")
        spread = max(self.config.morphology_regime_strength, 0.0)
        return _WorldMorphologyRegime(
            width_bias=float(rng.normal(0.0, 0.34 * spread)),
            height_bias=float(rng.normal(0.0, 0.18 * spread)),
            floor_bias=float(rng.normal(0.0, 0.28 * spread)),
            roof_bias=float(rng.normal(0.0, 0.26 * spread)),
            shape_bias=float(rng.normal(0.0, 0.62 * spread)),
            asymmetry_bias=float(rng.normal(0.0, 0.38 * spread)),
        )

    def _directed_parent_morphology(
        self,
        incoming: tuple[tuple[int, _SegmentMorphologyState], ...]
        | list[tuple[int, _SegmentMorphologyState]],
    ) -> tuple[_SegmentMorphologyState, int] | None:
        """Combine only upstream (segment end-node) states deterministically."""

        if not incoming:
            return None
        ordered = sorted(incoming, key=lambda item: item[0])
        if len(ordered) == 1:
            segment_id, state = ordered[0]
            return state, segment_id
        states = [state for _, state in ordered]

        def mean(name: str) -> float:
            return float(np.mean([getattr(state, name) for state in states]))

        reference = states[0]
        combined = replace(
            reference,
            width_scale=mean("width_scale"),
            height_ratio_offset=mean("height_ratio_offset"),
            floor_relief=mean("floor_relief"),
            wall_roughness=mean("wall_roughness"),
            skew_bias=mean("skew_bias"),
            shape_bias=mean("shape_bias"),
            roof_bias=mean("roof_bias"),
            floor_bias=mean("floor_bias"),
            asymmetry_bias=mean("asymmetry_bias"),
        )
        return combined, ordered[0][0]

    def _sample_segment_morphology(
        self,
        rng: np.random.Generator,
        *,
        parent: _SegmentMorphologyState | None = None,
        parent_segment_id: int | None = None,
    ) -> _SegmentMorphologyState:
        world = self._world_regime
        width_deviate = float(np.clip(rng.normal() + world.width_bias, -2.75, 2.75))
        height_deviate = float(np.clip(rng.normal() + world.height_bias, -2.5, 2.5))
        floor_deviate = float(np.clip(rng.normal() + world.floor_bias, -2.5, 2.5))
        roughness_deviate = float(np.clip(rng.normal(), -2.4, 2.4))
        skew_deviate = float(np.clip(rng.normal() + world.asymmetry_bias, -2.5, 2.5))
        shape_deviate = float(
            np.clip(
                rng.normal() * self.config.morphology_family_spread + world.shape_bias,
                -2.5,
                2.5,
            )
        )
        roof_deviate = float(np.clip(rng.normal() + world.roof_bias, -2.5, 2.5))
        floor_shape_deviate = float(np.clip(rng.normal() + world.floor_bias, -2.5, 2.5))
        width_scale = float(
            np.exp(
                math.log(self.config.width_scale_median)
                + self.config.width_scale_log_sigma * width_deviate
            )
        )
        height_offset = self.config.height_ratio_variation * height_deviate
        shape_bias = self.config.profile_shape_variation * shape_deviate
        roof_bias = 0.42 * roof_deviate
        floor_bias = 0.40 * floor_shape_deviate
        skew_bias = 0.62 * self.config.lateral_skew_amplitude * skew_deviate
        if parent is not None:
            # Daughters inherit the parent family and scale at a junction; only
            # a bounded fraction of the latent state diverges downstream.
            inheritance = 0.68
            width_scale = float(
                np.exp(
                    inheritance * math.log(max(parent.width_scale, 1e-9))
                    + (1.0 - inheritance) * math.log(max(width_scale, 1e-9))
                )
            )
            height_offset = (
                inheritance * parent.height_ratio_offset + (1.0 - inheritance) * height_offset
            )
            shape_bias = inheritance * parent.shape_bias + (1.0 - inheritance) * shape_bias
            roof_bias = inheritance * parent.roof_bias + (1.0 - inheritance) * roof_bias
            floor_bias = inheritance * parent.floor_bias + (1.0 - inheritance) * floor_bias
            skew_bias = inheritance * parent.skew_bias + (1.0 - inheritance) * skew_bias
        floor_relief = float(
            np.clip(
                self.config.floor_relief_base
                + 1.55 * self.config.floor_relief_variation * floor_deviate
                + 0.055 * floor_bias,
                0.0,
                self.config.morphology_floor_relief_max,
            )
        )
        wall_roughness = float(
            np.clip(
                self.config.wall_roughness_base
                + self.config.wall_roughness_variation * roughness_deviate
                + 0.018 * abs(shape_bias),
                0.0,
                self.config.morphology_wall_roughness_max,
            )
        )
        regime = self._classify_morphology_regime(shape_bias, floor_bias, roof_bias)
        return _SegmentMorphologyState(
            width_scale=float(np.clip(width_scale, 0.28, 2.85)),
            height_ratio_offset=float(height_offset),
            floor_relief=floor_relief,
            wall_roughness=wall_roughness,
            skew_bias=float(skew_bias),
            primary_phase=float(rng.uniform(-math.pi, math.pi)),
            secondary_phase=float(rng.uniform(-math.pi, math.pi)),
            floor_phase=float(rng.uniform(-math.pi, math.pi)),
            shape_bias=float(np.clip(shape_bias, -2.0, 2.0)),
            roof_bias=float(np.clip(roof_bias, -1.2, 1.2)),
            floor_bias=float(np.clip(floor_bias, -1.2, 1.2)),
            asymmetry_bias=float(np.clip(world.asymmetry_bias + 0.45 * skew_deviate, -1.5, 1.5)),
            parent_segment_id=parent_segment_id,
            regime=regime,
        )

    @staticmethod
    def _classify_morphology_regime(shape_bias: float, floor_bias: float, roof_bias: float) -> str:
        score = shape_bias + 0.35 * roof_bias
        if score > 0.72:
            return "keyhole"
        if score < -0.72:
            return "triangular"
        if floor_bias > 0.55:
            return "benched"
        if roof_bias < -0.55:
            return "rectangular"
        if shape_bias > 0.28:
            return "arched"
        return "balanced"

    def _morphology_at(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        base: _SegmentMorphologyState,
    ) -> _SegmentMorphologyState:
        """Evaluate a continuous, node-anchored morphology field."""

        progress = float(np.clip(arc_length / max(segment.total_length, 1e-9), 0.0, 1.0))
        blend = progress * progress * (3.0 - 2.0 * progress)
        interior_envelope = max(math.sin(math.pi * progress), 0.0) ** 0.75
        strength = self.config.morphology_gradient_strength
        start_point = segment.points[0]
        end_point = segment.points[-1]

        def spatial_value(label: str, x_coord: float, y_coord: float) -> float:
            field_rng = procedural_rng(
                self.config.random_seed,
                "section-spatial-morphology",
                label,
            )
            angle_a = float(field_rng.uniform(-math.pi, math.pi))
            angle_b = angle_a + float(field_rng.uniform(0.75, 2.35))
            phase_a = float(field_rng.uniform(-math.pi, math.pi))
            phase_b = float(field_rng.uniform(-math.pi, math.pi))
            projection_a = x_coord * math.cos(angle_a) + y_coord * math.sin(angle_a)
            projection_b = x_coord * math.cos(angle_b) + y_coord * math.sin(angle_b)
            wavelength = self.config.morphology_correlation_length
            value = math.sin(2.0 * math.pi * projection_a / wavelength + phase_a)
            value += 0.52 * math.sin(2.0 * math.pi * projection_b / (0.63 * wavelength) + phase_b)
            return value / 1.52

        def gradient(label: str) -> float:
            route_rng = procedural_rng(
                self.config.random_seed,
                "section-route-morphology",
                label,
                segment.segment_id,
            )
            start_value = spatial_value(label, start_point.x, start_point.y)
            end_value = spatial_value(label, end_point.x, end_point.y)
            node_value = (1.0 - blend) * start_value + blend * end_value
            phase = float(route_rng.uniform(-math.pi, math.pi))
            secondary_phase = float(route_rng.uniform(-math.pi, math.pi))
            wave = math.sin(
                2.0 * math.pi * arc_length / self.config.morphology_correlation_length + phase
            )
            wave += 0.42 * math.sin(
                2.0 * math.pi * arc_length / (0.53 * self.config.morphology_correlation_length)
                + secondary_phase
            )
            return float(np.clip(0.72 * node_value + 0.55 * interior_envelope * wave, -2.0, 2.0))

        width_gradient = gradient("width")
        height_gradient = gradient("height")
        floor_gradient = gradient("floor")
        roughness_gradient = gradient("roughness")
        skew_gradient = gradient("skew")
        shape_gradient = gradient("shape")
        roof_shape_gradient = gradient("roof-shape")
        floor_shape_gradient = gradient("floor-shape")
        base_width_log = math.log(
            max(base.width_scale, 1e-9) / max(self.config.width_scale_median, 1e-9)
        )

        def blended_phase(label: str, interior_phase: float) -> float:
            start_phase = math.pi * spatial_value(
                f"{label}-phase",
                start_point.x,
                start_point.y,
            )
            end_phase = math.pi * spatial_value(
                f"{label}-phase",
                end_point.x,
                end_point.y,
            )
            phase_delta = math.atan2(
                math.sin(end_phase - start_phase),
                math.cos(end_phase - start_phase),
            )
            return start_phase + blend * phase_delta + interior_envelope * 0.35 * interior_phase

        return replace(
            base,
            width_scale=float(
                np.clip(
                    self.config.width_scale_median
                    * math.exp(
                        0.28 * interior_envelope * base_width_log + 0.58 * strength * width_gradient
                    ),
                    0.32,
                    2.60,
                )
            ),
            height_ratio_offset=float(
                0.28 * interior_envelope * base.height_ratio_offset
                + 1.50 * strength * height_gradient
            ),
            floor_relief=float(
                np.clip(
                    self.config.floor_relief_base
                    + 0.30 * interior_envelope * (base.floor_relief - self.config.floor_relief_base)
                    + 0.55 * strength * self.config.floor_relief_variation * floor_gradient,
                    0.0,
                    self.config.morphology_floor_relief_max,
                )
            ),
            wall_roughness=float(
                np.clip(
                    self.config.wall_roughness_base
                    + 0.30
                    * interior_envelope
                    * (base.wall_roughness - self.config.wall_roughness_base)
                    + 0.70 * strength * self.config.wall_roughness_variation * roughness_gradient,
                    0.0,
                    self.config.morphology_wall_roughness_max,
                )
            ),
            skew_bias=float(
                0.30 * interior_envelope * base.skew_bias
                + 0.72 * strength * self.config.lateral_skew_amplitude * skew_gradient
            ),
            shape_bias=float(
                np.clip(
                    0.88 * interior_envelope * base.shape_bias
                    + 1.55 * strength * self.config.profile_shape_variation * shape_gradient,
                    -2.0,
                    2.0,
                )
            ),
            roof_bias=float(
                np.clip(
                    0.78 * interior_envelope * base.roof_bias
                    + 0.92 * strength * roof_shape_gradient,
                    -1.2,
                    1.2,
                )
            ),
            floor_bias=float(
                np.clip(
                    0.78 * interior_envelope * base.floor_bias
                    + 0.92 * strength * floor_shape_gradient,
                    -1.2,
                    1.2,
                )
            ),
            primary_phase=blended_phase("primary", base.primary_phase),
            secondary_phase=blended_phase("secondary", base.secondary_phase),
            floor_phase=blended_phase("floor", base.floor_phase),
        )

    def _apply_emplacement_morphology(
        self,
        segment: CaveSegment,
        morphology: _SegmentMorphologyState,
    ) -> _SegmentMorphologyState:
        """Translate preserved emplacement history into section character."""

        phase_value = segment.metadata.get("emplacement_phase_count", 1)
        birth_value = segment.metadata.get("birth_phase", 0)
        phase_count = max(
            int(phase_value) if isinstance(phase_value, (int, float)) else 1,
            1,
        )
        birth_phase = int(birth_value) if isinstance(birth_value, (int, float)) else 0
        relative_age = 1.0 - birth_phase / max(phase_count - 1, 1)
        width_scale = morphology.width_scale * (0.94 + 0.12 * relative_age)
        floor_relief = morphology.floor_relief * (0.82 + 0.36 * relative_age)
        wall_roughness = morphology.wall_roughness * (0.84 + 0.32 * relative_age)

        roof_state = segment.metadata.get("roof_state", "intact_tube")
        if roof_state == "open_channel":
            width_scale *= 1.12
            floor_relief *= 1.12
        elif roof_state == "partial_roof":
            width_scale *= 1.05
            wall_roughness *= 1.12
        elif roof_state == "skylight_prone":
            wall_roughness *= 1.18

        if bool(segment.metadata.get("vertical_capture", False)):
            width_scale *= 0.94
            floor_relief *= 0.88

        formation_state = str(segment.metadata.get("formation_state", ""))
        if formation_state in {"coalesced", "vertically_captured"}:
            morphology = replace(
                morphology,
                shape_bias=morphology.shape_bias + 0.18,
                floor_bias=morphology.floor_bias + 0.16,
            )
        elif formation_state in {"thermally_abandoned", "stranded"}:
            morphology = replace(
                morphology,
                shape_bias=morphology.shape_bias - 0.14,
                floor_bias=morphology.floor_bias + 0.22,
            )

        return replace(
            morphology,
            width_scale=float(np.clip(width_scale, 0.30, 2.85)),
            floor_relief=float(np.clip(floor_relief, 0.0, self.config.morphology_floor_relief_max)),
            wall_roughness=float(
                np.clip(wall_roughness, 0.0, self.config.morphology_wall_roughness_max)
            ),
        )

    def _section_width(
        self,
        segment: CaveSegment,
        arc_length: float,
        morphology: _SegmentMorphologyState,
    ) -> float:
        base = self._smoothed_width(segment, arc_length)
        phase = self._morphology_phase(
            segment,
            arc_length,
            morphology.primary_phase,
            wavelength_fraction=0.46,
        )
        modulation = math.exp(
            self.config.width_longitudinal_variation
            * (0.72 * math.sin(phase) + 0.28 * math.sin(2.17 * phase + 0.6))
        )
        flux = self._interpolate_attr(segment, arc_length, "flux")
        flux_reference = max(segment.mean_flux, 1e-9)
        flux_scale = float(np.clip(flux / flux_reference, 0.55, 1.65))
        phase_value = segment.metadata.get("emplacement_phase_count", 1)
        try:
            phase_count = float(phase_value) if isinstance(phase_value, (str, int, float)) else 1.0
        except (TypeError, ValueError):
            phase_count = 1.0
        age_scale = (
            1.0
            + self.config.morphology_age_width_gain
            * float(np.clip(phase_count, 1.0, 8.0) - 1.0)
            / 7.0
        )
        process_scale = 1.0 + self.config.morphology_flux_width_gain * (flux_scale - 1.0)
        return float(
            np.clip(
                base * morphology.width_scale * modulation * process_scale * age_scale,
                self.config.minimum_tube_width,
                self.config.maximum_tube_width,
            )
        )

    def _height_ratio(
        self,
        segment: CaveSegment,
        arc_length: float,
        morphology: _SegmentMorphologyState,
        flow_maturity: float,
    ) -> float:
        roof_competence = self._interpolate_attr(segment, arc_length, "roof_competence")
        width = self._smoothed_width(segment, arc_length)
        interior_envelope = self._segment_interior_envelope(segment, arc_length)
        raw_ratio = (
            self.config.base_height_ratio
            + 0.08 * (roof_competence - 0.5)
            - 0.06 * np.clip((width - 9.5) / 7.5, 0.0, 1.0)
            - 0.055 * flow_maturity
            + 0.10 * morphology.roof_bias
            + 0.08 * morphology.shape_bias
            + interior_envelope * morphology.height_ratio_offset
            + self.config.height_ratio_longitudinal_variation
            * interior_envelope
            * math.sin(
                self._morphology_phase(
                    segment,
                    arc_length,
                    morphology.secondary_phase,
                    wavelength_fraction=0.55,
                )
            )
        )
        return float(
            np.clip(
                raw_ratio,
                self.config.minimum_height_ratio,
                self.config.maximum_height_ratio,
            )
        )

    @staticmethod
    def _segment_interior_envelope(segment: CaveSegment, arc_length: float) -> float:
        if segment.total_length <= 1e-9:
            return 0.0
        progress = float(np.clip(arc_length / segment.total_length, 0.0, 1.0))
        return float(max(math.sin(math.pi * progress), 0.0) ** 0.65)

    def _floor_flatness(
        self,
        segment: CaveSegment,
        arc_length: float,
        width: float,
        flow_maturity: float,
        morphology: _SegmentMorphologyState,
    ) -> float:
        growth_cost = self._interpolate_attr(segment, arc_length, "growth_cost")
        relief_delta = self.config.floor_relief_base - morphology.floor_relief
        flatness = (
            self.config.floor_flatness_base
            + self.config.floor_flatness_width_weight * np.clip((width - 7.0) / 6.0, 0.0, 1.0)
            + 0.06 * growth_cost
            + 0.045 * flow_maturity
            + 0.85 * relief_delta
            + 0.045 * morphology.floor_bias
        )
        return float(np.clip(flatness, 0.28, 0.92))

    def _roof_arch(
        self,
        segment: CaveSegment,
        arc_length: float,
        morphology: _SegmentMorphologyState,
    ) -> float:
        roof_competence = self._interpolate_attr(segment, arc_length, "roof_competence")
        growth_cost = self._interpolate_attr(segment, arc_length, "growth_cost")
        arch = (
            self.config.roof_arch_base
            + self.config.roof_arch_roof_weight * roof_competence
            - 0.08 * growth_cost
            - 0.55 * (morphology.wall_roughness - self.config.wall_roughness_base)
            + 0.10 * morphology.height_ratio_offset
            + 0.10 * morphology.roof_bias
            - 0.06 * morphology.shape_bias
        )
        return float(np.clip(arch, 0.92, 1.36))

    def _lateral_skew(
        self,
        *,
        segment: CaveSegment,
        arc_length: float,
        phase: float,
        bias: float,
    ) -> float:
        total_length = max(segment.total_length, 1.0)
        wavelength = max(0.75 * total_length, 70.0)
        interior_envelope = self._segment_interior_envelope(segment, arc_length)
        return float(
            bias
            + self.config.lateral_skew_amplitude
            * interior_envelope
            * (1.0 + 0.55 * np.clip(abs(bias), 0.0, 1.0))
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
            distance = abs(arc_length - anchor)
            junction_weight = math.exp(-distance / max(junction.blend_length, 1.0))
            metadata = getattr(junction, "metadata", {}) or {}
            chamber_type = str(metadata.get("chamber_type", ""))
            room_weight = 0.0
            target_width = 0.0
            target_height = 0.0
            influence_length = float(junction.blend_length)
            if chamber_type == "drained_lava_pool" and junction.kind == "chamber":
                pool_length = max(float(metadata.get("pool_length_m", 0.0) or 0.0), 1.0)
                half_length = max(0.5 * pool_length, 0.75 * tube_width, 1.0)
                if distance < half_length:
                    progress = float(np.clip(distance / half_length, 0.0, 1.0))
                    room_weight = math.cos(0.5 * math.pi * progress) ** (
                        2.0 * self.config.drained_pool_transition_power
                    )
                requested_width = float(metadata.get("pool_width_m", 0.0) or 0.0)
                target_width = float(
                    np.clip(
                        requested_width * self.config.drained_pool_width_scale,
                        tube_width,
                        self.config.chamber_max_tube_width,
                    )
                )
                requested_height = float(metadata.get("pool_depth_m", 0.0) or 0.0)
                pool_height_cap = min(
                    1.5 * tube_height,
                    target_width * self.config.drained_pool_height_ratio_limit,
                )
                target_height = float(
                    np.clip(
                        min(requested_height, pool_height_cap),
                        max(self.config.minimum_tube_height, 0.85 * tube_height),
                        max(tube_height, pool_height_cap),
                    )
                )
                influence_length = pool_length
            weight = max(junction_weight, room_weight)
            max_weight = max(max_weight, weight)
            influences.append(
                SectionJunctionInfluence(
                    junction_id=junction.junction_id,
                    kind=junction.kind,
                    weight=float(weight),
                    split_style=junction.split_style,
                    merge_style=junction.merge_style,
                    capacity_bias=junction.capacity_bias,
                    blend_length_m=influence_length,
                    chamber_type=chamber_type,
                    room_weight=float(room_weight),
                    target_width_m=target_width,
                    target_height_m=target_height,
                )
            )
            if (
                junction.split_style == "pre_widen_then_split"
                or junction.merge_style == "pre_widen_then_split"
            ):
                width_scale += (
                    self.config.junction_pre_widen_gain * junction_weight * junction.capacity_bias
                )
                height_scale += 0.10 * junction_weight * junction.capacity_bias
                flatness_delta += 0.10 * junction_weight
                arch_delta += 0.06 * junction_weight
            else:
                width_scale += (
                    self.config.junction_constant_envelope_gain
                    * junction_weight
                    * (junction.capacity_bias - 0.85)
                )
                height_scale += 0.03 * junction_weight * junction.capacity_bias
                flatness_delta += 0.04 * junction_weight
                arch_delta += 0.02 * junction_weight
            if junction.kind == "crossing":
                skew_scale *= 0.85
            elif chamber_type == "drained_lava_pool":
                if target_width > 0.0:
                    target_width_scale = target_width / max(tube_width, 1e-9)
                    width_scale = max(
                        width_scale,
                        (1.0 - room_weight) * width_scale + room_weight * target_width_scale,
                    )
                    width_cap = max(width_cap, target_width)
                if target_height > 0.0:
                    target_height_scale = target_height / max(tube_height, 1e-9)
                    height_scale = (
                        1.0 - room_weight
                    ) * height_scale + room_weight * target_height_scale
                desired_flatness_delta = self.config.drained_pool_floor_flatness - floor_flatness
                flatness_delta = max(
                    flatness_delta,
                    room_weight * desired_flatness_delta,
                )
                arch_delta += room_weight * (self.config.drained_pool_roof_arch - roof_arch)
                skew_scale *= 1.0 - 0.45 * room_weight
            elif junction.kind == "chamber":
                chamber_weight = junction_weight**1.8
                width_scale += self.config.chamber_widen_gain * chamber_weight
                height_scale += 0.18 * chamber_weight
                flatness_delta += 0.10 * chamber_weight
                arch_delta += 0.08 * chamber_weight
                if weight >= 0.08:
                    width_cap = max(
                        width_cap,
                        self.config.maximum_tube_width
                        + chamber_weight
                        * (self.config.chamber_max_tube_width - self.config.maximum_tube_width),
                    )

        filtered_influences = tuple(
            sorted(
                (influence for influence in influences if influence.weight >= 0.08),
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
            + self.config.preferred_cover_fraction
            * max(cover_thickness - self.config.minimum_roof_thickness, 0.0)
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
        starts_here = segment.start_node_id in junction.node_ids
        ends_here = segment.end_node_id in junction.node_ids
        if starts_here and ends_here:
            start_distance = math.hypot(
                segment.points[0].x - junction.center_x,
                segment.points[0].y - junction.center_y,
            )
            end_distance = math.hypot(
                segment.points[-1].x - junction.center_x,
                segment.points[-1].y - junction.center_y,
            )
            return 0.0 if start_distance <= end_distance else segment.total_length
        if starts_here:
            return 0.0
        if ends_here:
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
        wall_roughness: float,
        floor_relief: float,
        shape_bias: float,
        roof_bias: float,
        floor_bias: float,
        asymmetry_bias: float,
        roughness_phase: float,
        floor_phase: float,
        bench_strength: float = 0.0,
        floor_incision_ratio: float = 0.0,
    ) -> tuple[tuple[float, float], ...]:
        half_width = 0.5 * tube_width
        half_height = 0.5 * tube_height
        resolution = max(self.config.profile_resolution // 2, 8)
        # Cluster vertices at the wall turns rather than leaving a long flat
        # bevel between uniformly spaced roof/floor samples.
        x_values = -half_width * np.cos(np.linspace(0.0, math.pi, resolution))
        normalized = np.clip(np.abs(x_values) / max(half_width, 1e-9), 0.0, 1.0)
        top_exp = float(
            np.clip(
                1.78 - 0.24 * (roof_arch - 1.0) - 0.48 * roof_bias - 0.42 * shape_bias,
                0.72,
                4.20,
            )
        )
        bottom_exp = float(
            np.clip(3.0 + 2.4 * floor_flatness - 1.35 * floor_bias - 0.72 * shape_bias, 1.25, 8.5)
        )
        floor_depth_factor = float(
            np.clip(0.74 - 0.22 * floor_flatness + 0.07 * floor_bias, 0.44, 0.86)
        )
        roof_scale = float(np.clip(1.0 + 0.14 * roof_bias + 0.10 * shape_bias, 0.78, 1.28))
        floor_scale = float(np.clip(1.0 + 0.22 * floor_bias - 0.08 * shape_bias, 0.70, 1.36))
        skew_offset = (lateral_skew + 0.12 * asymmetry_bias) * half_width
        roof_skew_offset = skew_offset * (1.0 + 2.0 * np.clip(asymmetry_bias, -0.8, 0.8))
        floor_skew_offset = skew_offset * (1.0 - 0.35 * np.clip(asymmetry_bias, -0.8, 0.8))

        def envelope(value: float) -> float:
            return max(0.0, 1.0 - value * value) ** 0.72

        def wall_relief(value: float) -> float:
            harmonic = math.sin(3.0 * math.pi * value + roughness_phase)
            harmonic += 0.55 * math.sin(7.0 * math.pi * value + 1.37 * roughness_phase)
            return half_height * wall_roughness * envelope(value) * harmonic / 1.55

        def floor_relief_offset(value: float) -> float:
            harmonic = math.sin(2.0 * math.pi * value + floor_phase)
            harmonic += 0.45 * math.sin(5.0 * math.pi * value - 0.73 * floor_phase)
            return half_height * floor_relief * envelope(value) * harmonic / 1.45

        roof_profile = [
            (
                float(x_coord + roof_skew_offset * (1.0 - normalized_value**2)),
                float(
                    roof_scale
                    * half_height
                    * max(0.0, 1.0 - normalized_value**top_exp) ** (1.0 / top_exp)
                    + wall_relief(float(x_coord / max(half_width, 1e-9)))
                    + 0.34
                    * half_height
                    * asymmetry_bias
                    * envelope(float(normalized_value))
                    * float(x_coord / max(half_width, 1e-9))
                ),
            )
            for x_coord, normalized_value in zip(x_values, normalized, strict=True)
        ]
        floor_profile = [
            (
                float(x_coord + 0.55 * floor_skew_offset * (1.0 - normalized_value**2)),
                float(
                    -floor_scale
                    * half_height
                    * floor_depth_factor
                    * max(0.0, 1.0 - normalized_value**bottom_exp) ** (1.0 / bottom_exp)
                    + 0.55 * wall_relief(float(x_coord / max(half_width, 1e-9)))
                    + 1.35 * floor_relief_offset(float(x_coord / max(half_width, 1e-9)))
                ),
            )
            for x_coord, normalized_value in zip(
                reversed(x_values), reversed(normalized), strict=True
            )
        ]
        # Preserve a former lava level as a ledge on each side; a narrow
        # incised channel changes the floor independently of the roof curve.
        # All profiles retain the same vertex count for longitudinal blending.
        bench_level = -half_height * floor_depth_factor * (1.0 - 0.70 * bench_strength)
        shaped_floor = []
        for (x, y), u in zip(floor_profile, reversed(normalized), strict=True):
            ledge_weight = float(np.clip((u - 0.55) / 0.12, 0.0, 1.0))
            y += ledge_weight * max(bench_level - y, 0.0) if bench_strength > 0.0 else 0.0
            y -= tube_height * floor_incision_ratio * math.exp(-(u / 0.22) ** 2)
            shaped_floor.append((x, y))
        closed_profile = shaped_floor + roof_profile + [shaped_floor[0]]
        # Keep the semantic contour envelope consistent with declared section
        # dimensions even when roughness/asymmetry pushes a wall outward.
        contour = np.asarray(closed_profile, dtype=float)
        extent = np.ptp(contour, axis=0)
        limit = np.asarray((tube_width, tube_height), dtype=float)
        scale = np.minimum(1.0, limit / np.maximum(extent, 1e-9))
        center = 0.5 * (np.min(contour, axis=0) + np.max(contour, axis=0))
        contour[:, 0] = center[0] + (contour[:, 0] - center[0]) * scale[0]
        # Limit incision at the requested envelope without moving the ceiling.
        contour[:, 1] = np.maximum(contour[:, 1], np.max(contour[:, 1]) - tube_height)
        contour[-1] = contour[0]
        return tuple((float(point[0]), float(point[1])) for point in contour)

    @staticmethod
    def _morphology_phase(
        segment: CaveSegment,
        arc_length: float,
        phase: float,
        *,
        wavelength_fraction: float,
    ) -> float:
        wavelength = max(70.0, wavelength_fraction * max(segment.total_length, 1.0))
        return float((2.0 * math.pi * arc_length / wavelength) + phase)

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
        end_width = self._interpolate_attr(
            segment, min(segment.total_length, arc_length + delta), "width"
        )
        return abs(end_width - start_width) / max(2.0 * delta, 1.0)

    def _estimate_curvature(self, segment: CaveSegment, arc_length: float) -> float:
        delta = max(6.0, 0.05 * max(segment.total_length, 1.0))
        previous = self._interpolate_position(segment, max(0.0, arc_length - delta))
        current = self._interpolate_position(segment, arc_length)
        next_position = self._interpolate_position(
            segment, min(segment.total_length, arc_length + delta)
        )
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
        next_position = self._interpolate_position(
            segment, min(segment.total_length, arc_length + delta)
        )
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

    @staticmethod
    def _build_frame(
        tangent: tuple[float, float, float],
        previous_normal: tuple[float, float, float] | None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        tangent_vector = np.array(tangent, dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        tangent_norm = float(np.linalg.norm(tangent_vector))
        if math.isclose(tangent_norm, 0.0):
            tangent_vector = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            tangent_vector /= tangent_norm

        # The second profile coordinate represents physical vertical: positive
        # values are the roof and negative values are the floor.  Construct it
        # by projecting world-up into the section plane, rather than choosing a
        # sign from the preceding segment.  Negating the lateral axis for frame
        # continuity also negates ``tangent x normal`` and used to turn whole
        # sections upside down at some junctions.
        binormal = up - float(np.dot(up, tangent_vector)) * tangent_vector
        binormal_norm = float(np.linalg.norm(binormal))
        if not math.isclose(binormal_norm, 0.0):
            binormal /= binormal_norm
            normal = np.cross(binormal, tangent_vector)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
        else:
            # A vertical conduit has no section-plane direction that points
            # upward.  Preserve the transported lateral direction only for
            # this degenerate case, after projecting it off the tangent.
            normal = (
                np.array(previous_normal, dtype=float)
                if previous_normal is not None
                else np.array([1.0, 0.0, 0.0], dtype=float)
            )
            normal -= float(np.dot(normal, tangent_vector)) * tangent_vector
            normal_norm = float(np.linalg.norm(normal))
            if math.isclose(normal_norm, 0.0):
                fallback = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(float(np.dot(fallback, tangent_vector))) > 0.9:
                    fallback = np.array([0.0, 1.0, 0.0], dtype=float)
                normal = fallback - float(np.dot(fallback, tangent_vector)) * tangent_vector
                normal_norm = float(np.linalg.norm(normal))
            normal /= max(normal_norm, 1e-12)
            binormal = np.cross(tangent_vector, normal)
            binormal /= max(float(np.linalg.norm(binormal)), 1e-12)
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
