"""Project configuration loading utilities for the active cave-network pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

import numpy as np

from stages.geometry import GeometryConfig
from stages.host_field import GridConfig, HostFieldConfig, TerrainWave
from stages.network import BraidGrammarConfig, CaveNetworkConfig
from stages.section_field import SectionFieldConfig


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    procedural_seed: int | None
    host_field: HostFieldConfig
    network: CaveNetworkConfig
    section_field: SectionFieldConfig
    geometry: GeometryConfig


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
        geometry=_build_geometry_config(
            raw_config.get("geometry", {}),
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
    range_data = config_data.pop("ranges", {})
    wave_range_data = config_data.pop("wave_ranges", None)

    grid = GridConfig(**grid_data)
    rng = np.random.default_rng(procedural_seed)

    for key, value_range in range_data.items():
        if key in {"seed_point_x", "seed_point_y"}:
            continue
        config_data[key] = _sample_numeric_range(rng, value_range)

    if "seed_point" in config_data:
        seed_point = tuple(config_data.pop("seed_point"))
    elif "seed_point_x" in range_data or "seed_point_y" in range_data:
        default_x, default_y = HostFieldConfig.seed_point
        seed_point = (
            _sample_numeric_range(rng, range_data.get("seed_point_x", default_x)),
            _sample_numeric_range(rng, range_data.get("seed_point_y", default_y)),
        )
    else:
        seed_point = HostFieldConfig.seed_point

    waves: tuple[TerrainWave, ...] | None = None
    if wave_data is not None:
        waves = tuple(TerrainWave(**wave) for wave in wave_data)
    elif wave_range_data is not None:
        waves = _sample_terrain_waves(rng, wave_range_data)

    if waves is None:
        return HostFieldConfig(
            grid=grid,
            seed_point=seed_point,
            **config_data,
        )

    return HostFieldConfig(
        grid=grid,
        waves=waves,
        seed_point=seed_point,
        **config_data,
    )


def _sample_terrain_waves(
    rng: np.random.Generator,
    wave_range_data: dict[str, Any],
) -> tuple[TerrainWave, ...]:
    count = _sample_integer_range(rng, wave_range_data.get("count", 3))
    if count < 0:
        raise ValueError("host_field.wave_ranges.count must be non-negative")

    return tuple(
        TerrainWave(
            amplitude=_sample_numeric_range(rng, wave_range_data.get("amplitude", 8.0)),
            wavelength=_sample_numeric_range(rng, wave_range_data.get("wavelength", 1200.0)),
            angle_degrees=_sample_numeric_range(rng, wave_range_data.get("angle_degrees", 0.0)),
            phase=_sample_numeric_range(rng, wave_range_data.get("phase", 0.0)),
        )
        for _ in range(count)
    )


def _sample_numeric_range(
    rng: np.random.Generator,
    value_or_range: Any,
) -> float:
    if isinstance(value_or_range, list):
        if len(value_or_range) != 2:
            raise ValueError(f"Expected a [min, max] range, got {value_or_range!r}")

        minimum = float(value_or_range[0])
        maximum = float(value_or_range[1])
        if minimum > maximum:
            raise ValueError(f"Invalid range with min > max: {value_or_range!r}")
        if minimum == maximum:
            return minimum
        return float(rng.uniform(minimum, maximum))

    return float(value_or_range)


def _sample_integer_range(
    rng: np.random.Generator,
    value_or_range: Any,
) -> int:
    if isinstance(value_or_range, list):
        if len(value_or_range) != 2:
            raise ValueError(f"Expected a [min, max] range, got {value_or_range!r}")

        minimum = int(value_or_range[0])
        maximum = int(value_or_range[1])
        if minimum > maximum:
            raise ValueError(f"Invalid range with min > max: {value_or_range!r}")
        return int(rng.integers(minimum, maximum + 1))

    return int(value_or_range)


def _build_network_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
) -> CaveNetworkConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    braid_grammar_data = config_data.pop("braid_grammar", {})
    if braid_grammar_data:
        config_data["braid_grammar"] = BraidGrammarConfig(
            **{
                key: _to_range_tuple(value)
                if isinstance(value, list)
                else value
                for key, value in braid_grammar_data.items()
            }
        )
    return CaveNetworkConfig(**config_data)


def _to_range_tuple(value: list[Any]) -> tuple[Any, Any]:
    if len(value) != 2:
        raise ValueError(f"Expected a [min, max] range, got {value!r}")
    return (value[0], value[1])


def _build_section_field_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
) -> SectionFieldConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    return SectionFieldConfig(**config_data)


def _build_geometry_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
) -> GeometryConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    return GeometryConfig(**config_data)
