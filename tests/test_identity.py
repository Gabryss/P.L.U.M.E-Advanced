"""Reproducibility must identify executing code and the actual dependency runtime."""

import hashlib
import json
from pathlib import Path

import pytest

from plume_advanced import identity
from plume_advanced.config import load_project_config
from plume_advanced.pipeline import StageCheckpointStore, checkpoints, pipeline_fingerprint


@pytest.mark.parametrize("content", [b"", b"x", b"a" * 1_000_000])
def test_streamed_file_identity(tmp_path, content):
    path = tmp_path / "file"
    path.write_bytes(content)
    assert identity.sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_executing_package_identity_is_independent_of_working_directory(tmp_path, monkeypatch):
    expected = identity.package_source_hash()
    monkeypatch.chdir(tmp_path)
    assert identity.package_source_hash() == expected
    assert expected != hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValueError, match="No Python source"):
        identity.package_source_hash(tmp_path)


def test_package_identity_tracks_file_names_and_content(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("x=1")
    first = identity.package_source_hash(tmp_path)
    source.write_text("x=2")
    second = identity.package_source_hash(tmp_path)
    source.rename(tmp_path / "b.py")
    assert len({first, second, identity.package_source_hash(tmp_path)}) == 3


def test_git_failure_is_unknown_not_clean(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(identity.subprocess, "run", unavailable)
    assert identity.git_identity(tmp_path) == {"revision": None, "dirty": None}


def test_runtime_versions_include_geometry_and_optional_dependencies():
    versions = identity.dependency_versions()
    assert {"numpy", "scipy", "xatlas", "rocky", "trimesh", "scikit-image"} <= versions.keys()
    assert all(isinstance(value, str) and value for value in versions.values())


def test_dependency_change_invalidates_checkpoint_and_fingerprint(tmp_path, monkeypatch):
    project = load_project_config(Path(__file__).parents[1] / "config/earth_short_single.toml")
    store = StageCheckpointStore(tmp_path / "cache", "same")
    store.save("host", [42])
    before = pipeline_fingerprint(project, inputs=(), source_root=tmp_path)
    monkeypatch.setattr(
        checkpoints, "runtime_identity", lambda: {"dependencies": {"numpy": "changed"}}
    )
    assert store.load("host") is None
    assert pipeline_fingerprint(project, inputs=(), source_root=tmp_path) != before


def test_lock_change_invalidates_fingerprint(tmp_path):
    project = load_project_config(Path(__file__).parents[1] / "config/earth_short_single.toml")
    lock = tmp_path / "uv.lock"
    lock.write_text("first")
    before = pipeline_fingerprint(project, inputs=(), source_root=tmp_path)
    lock.write_text("second")
    assert pipeline_fingerprint(project, inputs=(), source_root=tmp_path) != before


@pytest.mark.parametrize("damage", ["truncated", "metadata", "schema", "runtime", "missing"])
def test_corrupt_checkpoints_rebuild_without_deserializing(tmp_path, damage):
    store = StageCheckpointStore(tmp_path, "same")
    store.save("stage", {"value": 42})
    path = tmp_path / "stage.json"
    metadata = json.loads(path.read_text())
    if damage == "truncated":
        (tmp_path / "stage.pickle").write_bytes(b"bad")
    elif damage == "metadata":
        path.write_text("{")
    elif damage == "missing":
        (tmp_path / "stage.pickle").unlink()
    else:
        metadata[damage] = "old"
        path.write_text(json.dumps(metadata))
    result, reused = store.load_or_build("stage", lambda: {"value": 43}, resume=True)
    assert result == {"value": 43} and not reused
