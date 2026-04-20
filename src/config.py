"""Project configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from stages import (
    BranchMergeConfig,
    GraphConfig,
    GridConfig,
    HostFieldConfig,
    TerrainWave,
)


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    host_field: HostFieldConfig
    graph: GraphConfig
    branching: BranchMergeConfig


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load the project TOML configuration file."""

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    return ProjectConfig(
        host_field=_build_host_field_config(raw_config.get("host_field", {})),
        graph=_build_graph_config(raw_config.get("graph", {})),
        branching=_build_branching_config(raw_config.get("branching", {})),
    )


def _build_host_field_config(raw_config: dict[str, Any]) -> HostFieldConfig:
    config_data = dict(raw_config)

    grid_data = config_data.pop("grid", {})
    wave_data = config_data.pop("waves", None)

    grid = GridConfig(**grid_data)
    waves = None
    if wave_data is not None:
        waves = tuple(TerrainWave(**wave) for wave in wave_data)

    if waves is None:
        return HostFieldConfig(grid=grid, **config_data)

    return HostFieldConfig(grid=grid, waves=waves, **config_data)


def _build_graph_config(raw_config: dict[str, Any]) -> GraphConfig:
    return GraphConfig(**raw_config)


def _build_branching_config(raw_config: dict[str, Any]) -> BranchMergeConfig:
    return BranchMergeConfig(**raw_config)
