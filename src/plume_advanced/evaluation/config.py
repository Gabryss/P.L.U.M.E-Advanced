"""Strict loader for scientific experiments and their packaged defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plume_advanced.config import ProjectConfig, load_project_config
from plume_advanced.evaluation.config_schema import validate_evaluation_config

DEFAULT_CONFIG = Path(__file__).with_name("resources") / "experiments.toml"

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
    asset_directory: Path | None = None

    def load_project(self, **overrides: Any) -> ProjectConfig:
        return load_project_config(
            self.project_config, asset_directory=self.asset_directory, **overrides
        )

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
        if any(not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError(f"seed file requires unsigned 32-bit integers: {path}")
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


def load_evaluation_config(path: str | Path | None = None) -> EvaluationConfig:
    """Resolve custom paths beside the TOML; write bundled runs in the working directory."""
    config_path = Path(DEFAULT_CONFIG if path is None else path).resolve()
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    validate_evaluation_config(raw)
    schema_version = raw["schema_version"]
    general = raw.get("general", {})
    iterations = int(general.get("bootstrap_iterations", 2000))
    confidence = float(general.get("confidence_level", 0.95))
    bundled = config_path == DEFAULT_CONFIG.resolve()
    project_path = _resolve(config_path.parent, general.get("project_config", "../config/project.toml"))
    return EvaluationConfig(
        path=config_path,
        schema_version=schema_version,
        output_root=_resolve(
            Path.cwd() if bundled else config_path.parent,
            general.get("output_root", "outputs"),
        ),
        project_config=project_path,
        bootstrap_iterations=iterations,
        confidence_level=confidence,
        bootstrap_seed=int(general.get("bootstrap_seed", 20260101)),
        raw=raw,
        asset_directory=(
            _resolve(config_path.parent, general["asset_directory"])
            if "asset_directory" in general else
            (Path.cwd() / "config").resolve() if bundled else project_path.parent
        ),
    )


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


__all__ = ["DEFAULT_CONFIG", "EvaluationConfig", "load_evaluation_config"]
