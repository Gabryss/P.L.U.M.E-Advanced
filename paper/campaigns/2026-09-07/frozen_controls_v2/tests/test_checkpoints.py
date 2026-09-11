"""Resumable stage checkpoint integrity and invalidation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from plume_advanced.config import load_project_config
from plume_advanced.pipeline import StageCheckpointStore, pipeline_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_round_trip_reuses_valid_artifact(tmp_path: Path) -> None:
    store = StageCheckpointStore(
        tmp_path / "checkpoints",
        "fingerprint-a",
    )
    builds = 0

    def build() -> dict[str, int]:
        nonlocal builds
        builds += 1
        return {"value": 42}

    first, first_reused = store.load_or_build("network", build, resume=True)
    second, second_reused = store.load_or_build("network", build, resume=True)

    assert first == second == {"value": 42}
    assert not first_reused
    assert second_reused
    assert builds == 1


def test_checkpoint_rejects_changed_fingerprint_and_tampered_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    original = StageCheckpointStore(
        root,
        "fingerprint-a",
    )
    original.save("geometry", (1, 2, 3))

    changed = StageCheckpointStore(
        root,
        "fingerprint-b",
    )
    assert changed.load("geometry") is None

    payload = root / "geometry.pickle"
    payload.write_bytes(payload.read_bytes() + b"tampered")
    assert original.load("geometry") is None


def test_checkpoint_writes_leave_no_temporary_files(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    store = StageCheckpointStore(root, "fingerprint")
    store.save("floor atlas", "ready")

    assert store.load("floor atlas") == "ready"
    assert {path.name for path in root.iterdir()} == {
        "floor_atlas.json",
        "floor_atlas.pickle",
    }


def test_failed_checkpoint_serialization_preserves_previous_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    store = StageCheckpointStore(root, "fingerprint")
    store.save("host", {"complete": True})

    with pytest.raises((AttributeError, TypeError)):
        store.save("host", lambda: None)

    assert store.load("host") == {"complete": True}
    assert not tuple(root.glob(".*.tmp"))


def test_pipeline_fingerprint_tracks_inputs_and_source_layout(
    tmp_path: Path,
) -> None:
    project = load_project_config(ROOT / "src" / "plume_advanced" / "default_project.toml")
    input_path = tmp_path / "input.txt"
    input_path.write_text("first", encoding="utf-8")
    source_root = tmp_path / "source"
    package = source_root / "src" / "plume_advanced"
    package.mkdir(parents=True)
    source = package / "stage.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    initial = pipeline_fingerprint(project, inputs=(input_path,), source_root=source_root)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    source_changed = pipeline_fingerprint(
        project,
        inputs=(input_path,),
        source_root=source_root,
    )
    input_path.write_text("second", encoding="utf-8")
    input_changed = pipeline_fingerprint(
        project,
        inputs=(input_path,),
        source_root=source_root,
    )

    assert initial != source_changed
    assert source_changed != input_changed
