"""Tests for artifact-backed Manim inputs without requiring Manim itself."""

import json
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.media.manim_data import (
    evenly_spaced_sections,
    load_manim_host_data,
    load_manim_prototype_data,
)


def test_load_manim_prototype_data_selects_segment_with_most_samples(tmp_path: Path) -> None:
    network_path, section_path = _write_artifacts(tmp_path)

    data = load_manim_prototype_data(network_path, section_path)

    assert data.hero_segment_id == 7
    assert len(data.segments) == 2
    assert len(data.hero_sections) == 3
    assert data.longest_section_segment_id in {3, 7}
    assert np.allclose(data.hero_sections[0].normal_xyz, [1.0, 0.0, 0.0])
    assert data.network_semantic_sha256 == "a" * 64
    assert data.section_semantic_sha256 == "b" * 64


def test_load_manim_prototype_data_rejects_unknown_schema(tmp_path: Path) -> None:
    network_path, section_path = _write_artifacts(tmp_path)
    payload = json.loads(network_path.read_text(encoding="utf-8"))
    payload["schema"] = "unknown"
    network_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported schema"):
        load_manim_prototype_data(network_path, section_path)


def test_evenly_spaced_sections_includes_endpoints(tmp_path: Path) -> None:
    network_path, section_path = _write_artifacts(tmp_path)
    sections = load_manim_prototype_data(network_path, section_path).hero_sections

    selected = evenly_spaced_sections(sections, 2)

    assert [section.arc_length_m for section in selected] == [0.0, 10.0]


def test_load_manim_host_data_validates_and_loads_fields(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "stage_a_host_fields.npz",
        x_coords_m=np.asarray([0.0, 1.0]),
        y_coords_m=np.asarray([0.0, 1.0, 2.0]),
        elevation=np.arange(6, dtype=float).reshape(3, 2),
        routing_cost=np.ones((3, 2)),
    )
    metadata = tmp_path / "stage_a_host_fields.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "plume.manim-host-fields.v1",
                "field_names": ["elevation", "routing_cost"],
                "npz_file": "stage_a_host_fields.npz",
                "semantic_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )

    data = load_manim_host_data(metadata)

    assert data.fields["elevation"].shape == (3, 2)
    assert data.semantic_sha256 == "c" * 64


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    network_path = tmp_path / "stage_b_network.json"
    network_path.write_text(
        json.dumps(
            {
                "schema": "plume.cave-network.v1",
                "semantic_sha256": "a" * 64,
                "segments": [
                    _segment(3, 0.0),
                    _segment(7, 2.0, kind="island_bypass"),
                ],
            }
        ),
        encoding="utf-8",
    )
    profiles = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
        * 4,
        dtype=float,
    )
    np.savez_compressed(
        tmp_path / "stage_c_sections.npz",
        segment_id=np.asarray([3, 7, 7, 7]),
        arc_length_m=np.asarray([0.0, 0.0, 5.0, 10.0]),
        center_xyz_m=np.asarray(
            [[0.0, 0.0, -2.0], [2.0, 0.0, -2.0], [3.0, 0.0, -2.0], [4.0, 0.0, -2.0]]
        ),
        normal=np.asarray([[1.0, 0.0, 0.0]] * 4),
        width_m=np.asarray([2.0, 2.0, 2.5, 2.0]),
        height_m=np.asarray([1.5, 1.5, 1.8, 1.5]),
        profile_offsets=np.asarray([0, 4, 8, 12, 16]),
        profile_points=profiles,
    )
    section_path = tmp_path / "stage_c_sections.json"
    section_path.write_text(
        json.dumps(
            {
                "schema": "plume.section-field.v1",
                "npz_file": "stage_c_sections.npz",
                "semantic_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    return network_path, section_path


def _segment(
    segment_id: int,
    x_offset: float,
    *,
    kind: str = "backbone",
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "kind": kind,
        "z_level": 0,
        "mean_width_m": 2.0,
        "centerline": [
            {
                "x": x_offset,
                "y": 0.0,
                "elevation": 10.0,
                "cover_thickness": 2.0,
            },
            {
                "x": x_offset + 1.0,
                "y": 1.0,
                "elevation": 9.0,
                "cover_thickness": 2.0,
            },
        ],
    }
