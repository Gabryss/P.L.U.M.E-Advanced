"""Strict loader for the frozen paper experiment declaration."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KNOWN_TOP_LEVEL = {
    "schema_version",
    "general",
    "datasets",
    "morphometry",
    "controllability",
    "host_ablation",
    "sampling_ablation",
    "scalability",
    "export_consistency",
    "determinism",
}


@dataclass(frozen=True)
class EvaluationConfig:
    path: Path
    schema_version: int
    output_root: Path
    project_config: Path
    bootstrap_iterations: int
    confidence_level: float
    bootstrap_seed: int
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a TOML table")
        return dict(value)

    def seed_file(self, section: str) -> Path:
        value = self.section(section).get("seed_file")
        if not value:
            raise ValueError(f"{section}.seed_file is required")
        return _resolve(self.path.parent, value)

    def seeds(self, section: str) -> tuple[int, ...]:
        path = self.seed_file(section)
        seeds: list[int] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            content = line.split("#", 1)[0].strip()
            if content:
                seeds.append(int(content))
        if not seeds:
            raise ValueError(f"seed file is empty: {path}")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"seed file contains duplicates: {path}")
        return tuple(seeds)

    def pdc_root(self, override: str | Path | None = None) -> Path:
        if override is not None:
            return Path(override).expanduser().resolve()
        dataset = self.section("datasets").get("pdc", {})
        environment_name = str(dataset.get("path_env", "PLUME_PDC_ROOT"))
        value = os.environ.get(environment_name)
        if not value:
            raise ValueError(f"PDC path is not set; pass --data-root or define {environment_name}")
        return Path(value).expanduser().resolve()

    def pdc_cave_partition(self, name: str) -> tuple[str, ...]:
        path = self.pdc_partition_path(name)
        cave_ids = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not cave_ids or len(cave_ids) != len(set(cave_ids)):
            raise ValueError(f"PDC {name} cave partition is empty or contains duplicates: {path}")
        return cave_ids

    def pdc_partition_path(self, name: str) -> Path:
        if name not in {"calibration", "evaluation"}:
            raise ValueError("PDC partition must be calibration or evaluation")
        dataset = self.section("datasets").get("pdc", {})
        value = dataset.get(f"{name}_caves")
        if not value:
            raise ValueError(f"datasets.pdc.{name}_caves is required")
        return _resolve(self.path.parent, value)


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    unknown = set(raw) - KNOWN_TOP_LEVEL
    if unknown:
        raise ValueError("Unknown evaluation configuration keys: " + ", ".join(sorted(unknown)))
    schema_version = int(raw.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"Unsupported experiment schema_version: {schema_version}")
    general = raw.get("general", {})
    allowed_general = {
        "output_root",
        "project_config",
        "bootstrap_iterations",
        "confidence_level",
        "bootstrap_seed",
    }
    unknown_general = set(general) - allowed_general
    if unknown_general:
        raise ValueError("Unknown general keys: " + ", ".join(sorted(unknown_general)))
    iterations = int(general.get("bootstrap_iterations", 2000))
    confidence = float(general.get("confidence_level", 0.95))
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_iterations and confidence_level are invalid")
    return EvaluationConfig(
        path=config_path,
        schema_version=schema_version,
        output_root=_resolve(config_path.parent, general.get("output_root", "outputs")),
        project_config=_resolve(
            config_path.parent,
            general.get("project_config", "../config/project.toml"),
        ),
        bootstrap_iterations=iterations,
        confidence_level=confidence,
        bootstrap_seed=int(general.get("bootstrap_seed", 20260101)),
        raw=raw,
    )


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


__all__ = ["EvaluationConfig", "load_evaluation_config"]
