"""Validated, atomic stage artifacts for resumable deterministic generation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from plume_advanced.config import ProjectConfig, project_config_manifest

Artifact = TypeVar("Artifact")
CHECKPOINT_SCHEMA = "plume.stage-checkpoint.v1"


def pipeline_fingerprint(
    project_config: ProjectConfig,
    *,
    inputs: tuple[Path, ...],
    source_root: Path,
) -> str:
    """Hash resolved configuration, inputs, and production source code."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            project_config_manifest(project_config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for path in sorted({value.resolve() for value in inputs}, key=str):
        if path.is_file():
            digest.update(str(path).encode("utf-8"))
            digest.update(_file_hash(path).encode("ascii"))
    package_root = source_root / "src" / "plume_advanced"
    if not package_root.is_dir():
        package_root = Path(__file__).resolve().parents[1]
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(_file_hash(path).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class StageCheckpointStore:
    """Store only artifacts matching the current complete pipeline fingerprint."""

    root: Path
    fingerprint: str

    def load(self, stage: str) -> Any | None:
        payload_path, metadata_path = self._paths(stage)
        if not payload_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema") != CHECKPOINT_SCHEMA
                or metadata.get("stage") != stage
                or metadata.get("fingerprint") != self.fingerprint
                or metadata.get("python") != platform.python_version()
                or metadata.get("payload_sha256") != _file_hash(payload_path)
            ):
                return None
            with payload_path.open("rb") as source:
                return pickle.load(source)
        except (
            OSError,
            ValueError,
            TypeError,
            EOFError,
            AttributeError,
            ImportError,
            json.JSONDecodeError,
            pickle.PickleError,
        ):
            return None

    def save(self, stage: str, artifact: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload_path, metadata_path = self._paths(stage)
        temporary_payload: Path | None = None
        temporary_metadata: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=f".{stage}.",
                suffix=".pickle.tmp",
                delete=False,
            ) as temporary:
                temporary_payload = Path(temporary.name)
                pickle.dump(artifact, temporary, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.flush()
                os.fsync(temporary.fileno())
            metadata = {
                "schema": CHECKPOINT_SCHEMA,
                "stage": stage,
                "fingerprint": self.fingerprint,
                "python": platform.python_version(),
                "artifact_type": f"{type(artifact).__module__}.{type(artifact).__qualname__}",
                "payload_sha256": _file_hash(temporary_payload),
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{stage}.",
                suffix=".json.tmp",
                delete=False,
            ) as temporary:
                temporary_metadata = Path(temporary.name)
                temporary.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_payload.replace(payload_path)
            temporary_payload = None
            temporary_metadata.replace(metadata_path)
            temporary_metadata = None
        finally:
            for temporary_path in (temporary_payload, temporary_metadata):
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def load_or_build(
        self,
        stage: str,
        builder: Callable[[], Artifact],
        *,
        resume: bool,
    ) -> tuple[Artifact, bool]:
        if resume:
            cached = self.load(stage)
            if cached is not None:
                return cached, True
        artifact = builder()
        self.save(stage, artifact)
        return cast(Artifact, artifact), False

    def _paths(self, stage: str) -> tuple[Path, Path]:
        safe_stage = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in stage
        ).strip("_")
        if not safe_stage:
            raise ValueError("Checkpoint stage name cannot be empty")
        return self.root / f"{safe_stage}.pickle", self.root / f"{safe_stage}.json"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["CHECKPOINT_SCHEMA", "StageCheckpointStore", "pipeline_fingerprint"]
