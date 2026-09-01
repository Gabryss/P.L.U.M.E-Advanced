"""Validated, renderer-independent inputs for the Manim video project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ManimHostData:
    """Frozen Stage-A scalar fields prepared for explanatory animation."""

    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    fields: dict[str, np.ndarray]
    semantic_sha256: str


@dataclass(frozen=True)
class ManimSegment:
    """One plan-view network segment prepared for animation."""

    segment_id: int
    kind: str
    z_level: int
    width_m: float
    age_start_s: float
    age_end_s: float
    points_xyz_m: np.ndarray


@dataclass(frozen=True)
class ManimSection:
    """One cross-section profile associated with a network sample."""

    segment_id: int
    arc_length_m: float
    center_xyz_m: np.ndarray
    normal_xyz: np.ndarray
    width_m: float
    height_m: float
    profile_points_m: np.ndarray


@dataclass(frozen=True)
class ManimPrototypeData:
    """Artifact-backed inputs used by the graph-to-geometry prototype."""

    segments: tuple[ManimSegment, ...]
    sections: tuple[ManimSection, ...]
    hero_segment_id: int
    network_semantic_sha256: str
    section_semantic_sha256: str

    @property
    def hero_segment(self) -> ManimSegment:
        return next(segment for segment in self.segments if segment.segment_id == self.hero_segment_id)

    @property
    def hero_sections(self) -> tuple[ManimSection, ...]:
        return tuple(
            section for section in self.sections if section.segment_id == self.hero_segment_id
        )

    @property
    def longest_section_segment_id(self) -> int:
        """Return the physically longest segment with enough profiles to explain."""

        sample_counts: dict[int, int] = {}
        for section in self.sections:
            sample_counts[section.segment_id] = sample_counts.get(section.segment_id, 0) + 1
        candidates = [
            segment
            for segment in self.segments
            if sample_counts.get(segment.segment_id, 0) >= 5
        ]
        if not candidates:
            candidates = [
                segment
                for segment in self.segments
                if sample_counts.get(segment.segment_id, 0) >= 2
            ]
        if not candidates:
            raise ValueError("No network segment has enough section samples for animation")

        def physical_length(segment: ManimSegment) -> float:
            differences = np.diff(segment.points_xyz_m, axis=0)
            return float(np.sum(np.linalg.norm(differences, axis=1)))

        return max(candidates, key=lambda segment: (physical_length(segment), -segment.segment_id)).segment_id


def load_manim_prototype_data(
    network_path: str | Path,
    section_metadata_path: str | Path,
    *,
    hero_segment_id: int | None = None,
) -> ManimPrototypeData:
    """Load stable Stage-B/C artifacts without invoking generation."""

    network_file = Path(network_path)
    section_metadata_file = Path(section_metadata_path)
    network_payload = _read_json(network_file)
    section_metadata = _read_json(section_metadata_file)
    _require_schema(network_payload, "plume.cave-network.v1", network_file)
    _require_schema(section_metadata, "plume.section-field.v1", section_metadata_file)

    segments = _load_segments(network_payload, network_file)
    npz_name = section_metadata.get("npz_file")
    if not isinstance(npz_name, str) or not npz_name:
        raise ValueError(f"{section_metadata_file} does not declare a non-empty npz_file")
    section_npz_file = section_metadata_file.with_name(npz_name)
    sections = _load_sections(section_npz_file)
    chosen_segment_id = _choose_hero_segment(segments, sections, hero_segment_id)

    return ManimPrototypeData(
        segments=segments,
        sections=sections,
        hero_segment_id=chosen_segment_id,
        network_semantic_sha256=_require_hash(network_payload, network_file),
        section_semantic_sha256=_require_hash(section_metadata, section_metadata_file),
    )


def load_manim_host_data(metadata_path: str | Path) -> ManimHostData:
    """Load the frozen Stage-A field artifact used by the Stage-A scene."""

    metadata_file = Path(metadata_path)
    metadata = _read_json(metadata_file)
    _require_schema(metadata, "plume.manim-host-fields.v1", metadata_file)
    npz_name = metadata.get("npz_file")
    if not isinstance(npz_name, str) or not npz_name:
        raise ValueError(f"{metadata_file} does not declare a non-empty npz_file")
    npz_path = metadata_file.with_name(npz_name)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Stage-A NPZ declared by metadata is missing: {npz_path}")
    field_names = metadata.get("field_names")
    if not isinstance(field_names, list) or not all(isinstance(name, str) for name in field_names):
        raise ValueError(f"{metadata_file} does not declare field_names")
    with np.load(npz_path, allow_pickle=False) as payload:
        required = {"x_coords_m", "y_coords_m", *field_names}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {', '.join(missing)}")
        x_coords = np.asarray(payload["x_coords_m"], dtype=float)
        y_coords = np.asarray(payload["y_coords_m"], dtype=float)
        fields = {name: np.asarray(payload[name], dtype=float) for name in field_names}
    expected_shape = (len(y_coords), len(x_coords))
    for name, values in fields.items():
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid {name} shape or values in {npz_path}")
    return ManimHostData(
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        fields=fields,
        semantic_sha256=_require_hash(metadata, metadata_file),
    )


def evenly_spaced_sections(
    sections: tuple[ManimSection, ...],
    count: int,
) -> tuple[ManimSection, ...]:
    """Select stable, approximately even samples including both endpoints."""

    if count <= 0 or not sections:
        return ()
    ordered = tuple(sorted(sections, key=lambda item: item.arc_length_m))
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return tuple(ordered[int(index)] for index in indices)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required PLUME video artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _require_schema(payload: dict[str, object], expected: str, path: Path) -> None:
    actual = payload.get("schema")
    if actual != expected:
        raise ValueError(f"Unsupported schema in {path}: expected {expected!r}, got {actual!r}")


def _require_hash(payload: dict[str, object], path: Path) -> str:
    value = payload.get("semantic_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path} does not contain a valid semantic_sha256")
    return value


def _load_segments(
    payload: dict[str, object],
    path: Path,
) -> tuple[ManimSegment, ...]:
    records = payload.get("segments")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a segments list")
    segments: list[ManimSegment] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{path} contains a non-object segment record")
        centerline = record.get("centerline")
        if not isinstance(centerline, list) or len(centerline) < 2:
            continue
        points = np.asarray(
            [
                (
                    float(point["x"]),
                    float(point["y"]),
                    float(point["elevation"]) - float(point["cover_thickness"]),
                )
                for point in centerline
                if isinstance(point, dict)
            ],
            dtype=float,
        )
        if points.shape[0] < 2 or not np.all(np.isfinite(points)):
            continue
        segments.append(
            ManimSegment(
                segment_id=int(record["segment_id"]),
                kind=str(record.get("kind", "unknown")),
                z_level=int(record.get("z_level", 0)),
                width_m=float(record.get("mean_width_m", 1.0)),
                age_start_s=float(record.get("age_start_s") or 0.0),
                age_end_s=float(record.get("age_end_s") or 0.0),
                points_xyz_m=points,
            )
        )
    if not segments:
        raise ValueError(f"{path} contains no renderable network segments")
    return tuple(sorted(segments, key=lambda item: item.segment_id))


def _load_sections(npz_path: Path) -> tuple[ManimSection, ...]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"Section NPZ declared by metadata is missing: {npz_path}")
    required = {
        "segment_id",
        "arc_length_m",
        "center_xyz_m",
        "normal",
        "width_m",
        "height_m",
        "profile_offsets",
        "profile_points",
    }
    with np.load(npz_path, allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {', '.join(missing)}")
        segment_ids = np.asarray(payload["segment_id"], dtype=int)
        arc_lengths = np.asarray(payload["arc_length_m"], dtype=float)
        centers = np.asarray(payload["center_xyz_m"], dtype=float)
        normals = np.asarray(payload["normal"], dtype=float)
        widths = np.asarray(payload["width_m"], dtype=float)
        heights = np.asarray(payload["height_m"], dtype=float)
        offsets = np.asarray(payload["profile_offsets"], dtype=int)
        profile_points = np.asarray(payload["profile_points"], dtype=float)
    sample_count = len(segment_ids)
    if not (
        len(arc_lengths) == sample_count
        and centers.shape == (sample_count, 3)
        and normals.shape == (sample_count, 3)
        and len(widths) == sample_count
        and len(heights) == sample_count
        and len(offsets) == sample_count + 1
    ):
        raise ValueError(f"Inconsistent section array shapes in {npz_path}")
    sections = []
    for index in range(sample_count):
        start, end = int(offsets[index]), int(offsets[index + 1])
        profile = profile_points[start:end]
        if profile.shape[0] < 3:
            continue
        sections.append(
            ManimSection(
                segment_id=int(segment_ids[index]),
                arc_length_m=float(arc_lengths[index]),
                center_xyz_m=centers[index],
                normal_xyz=normals[index],
                width_m=float(widths[index]),
                height_m=float(heights[index]),
                profile_points_m=profile,
            )
        )
    if not sections:
        raise ValueError(f"{npz_path} contains no renderable section profiles")
    return tuple(sections)


def _choose_hero_segment(
    segments: tuple[ManimSegment, ...],
    sections: tuple[ManimSection, ...],
    requested: int | None,
) -> int:
    segment_ids = {segment.segment_id for segment in segments}
    counts: dict[int, int] = {}
    for section in sections:
        if section.segment_id in segment_ids:
            counts[section.segment_id] = counts.get(section.segment_id, 0) + 1
    if requested is not None:
        if requested not in counts:
            raise ValueError(f"Requested hero segment {requested} has no renderable section samples")
        return requested
    if not counts:
        raise ValueError("Network and section artifacts have no segment IDs in common")
    topology_kinds = {
        "chamber_braid",
        "inner_bypass",
        "island_bypass",
        "ladder",
        "underpass",
    }
    segment_lookup = {segment.segment_id: segment for segment in segments}
    topology_candidates = [
        segment_id
        for segment_id, count in counts.items()
        if count >= 3 and segment_lookup[segment_id].kind in topology_kinds
    ]
    candidates = topology_candidates or list(counts)
    return max(candidates, key=lambda segment_id: (counts[segment_id], -segment_id))


__all__ = [
    "ManimPrototypeData",
    "ManimHostData",
    "ManimSection",
    "ManimSegment",
    "evenly_spaced_sections",
    "load_manim_host_data",
    "load_manim_prototype_data",
]
