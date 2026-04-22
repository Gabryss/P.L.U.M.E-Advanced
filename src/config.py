"""Project configuration loading utilities for the active cave-network pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from stages.host_field import GridConfig, HostFieldConfig, TerrainWave
from stages.network import CaveNetworkConfig
from stages.section_field import SectionFieldConfig


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    procedural_seed: int | None
    host_field: HostFieldConfig
    network: CaveNetworkConfig
    section_field: SectionFieldConfig


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load the project TOML configuration file."""

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    procedural_seed = raw_config.get("procedural_seed")

    return ProjectConfig(
        procedural_seed=procedural_seed,
        host_field=_build_host_field_config(
            raw_config.get("host_field", {}),
            procedural_seed=procedural_seed,
        ),
        network=_build_network_config(
            raw_config.get("network", {}),
            procedural_seed=procedural_seed,
        ),
        section_field=_build_section_field_config(
            raw_config.get("section_field", {}),
            procedural_seed=procedural_seed,
        ),
    )


def _build_host_field_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
) -> HostFieldConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed

    grid_data = config_data.pop("grid", {})
    wave_data = config_data.pop("waves", None)

    grid = GridConfig(**grid_data)
    waves = None
    if wave_data is not None:
        waves = tuple(TerrainWave(**wave) for wave in wave_data)

    if waves is None:
        return HostFieldConfig(
            grid=grid,
            seed_point=tuple(config_data.pop("seed_point", HostFieldConfig.seed_point)),
            **config_data,
        )

    return HostFieldConfig(
        grid=grid,
        waves=waves,
        seed_point=tuple(config_data.pop("seed_point", HostFieldConfig.seed_point)),
        **config_data,
    )


def _build_network_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
) -> CaveNetworkConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    return CaveNetworkConfig(**config_data)


def _build_section_field_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
) -> SectionFieldConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    return SectionFieldConfig(**config_data)
