"""Project configuration loading utilities for the active cave-network pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tomllib
from typing import Any

import numpy as np

from stages.events import GeologicalEventConfig
from stages.geometry import GeometryConfig
from stages.host_field import GridConfig, HostFieldConfig, TerrainWave
from stages.network import BraidGrammarConfig, CaveNetworkConfig
from stages.section_field import SectionFieldConfig
from world import (
    ExportConfig,
    RunConfig,
    SUPPORTED_EVENT_KINDS,
    StageSeeds,
    WorldConfig,
    build_export_config,
    build_run_config,
    derive_stage_seeds,
    resolve_world_config,
)

CURRENT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    schema_version: int
    procedural_seed: int | None
    stage_seeds: StageSeeds
    world: WorldConfig
    run: RunConfig
    export: ExportConfig
    host_field: HostFieldConfig
    network: CaveNetworkConfig
    section_field: SectionFieldConfig
    events: GeologicalEventConfig
    geometry: GeometryConfig


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load the project TOML configuration file."""

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    schema_version = int(raw_config.get("schema_version", 1))
    if schema_version not in {1, CURRENT_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported schema_version {schema_version}; "
            f"supported versions are 1 and {CURRENT_SCHEMA_VERSION}"
        )
    procedural_seed = raw_config.get("procedural_seed")
    if procedural_seed is not None:
        procedural_seed = int(procedural_seed)
    stage_seeds = derive_stage_seeds(procedural_seed)
    world = resolve_world_config(raw_config.get("world"))
    run = build_run_config(raw_config.get("run"))
    export = build_export_config(raw_config.get("export"))

    host_field = _build_host_field_config(
        raw_config.get("host_field", {}),
        procedural_seed=stage_seeds.host,
    )
    network = _build_network_config(
        raw_config.get("network", {}),
        procedural_seed=stage_seeds.network,
        world=world,
    )
    section_field = _build_section_field_config(
        raw_config.get("section_field", {}),
        procedural_seed=stage_seeds.sections,
        world=world,
    )
    events = _build_event_config(
        raw_config.get("events", {}),
        procedural_seed=stage_seeds.events,
        world=world,
    )
    geometry = _build_geometry_config(
        raw_config.get("geometry", {}),
        procedural_seed=stage_seeds.geometry,
        world=world,
    )
    if run.dev_mode:
        host_field, network = _apply_dev_mode(host_field, network, run)
    _validate_pipeline_configs(
        host_field=host_field,
        network=network,
        section_field=section_field,
        events=events,
        geometry=geometry,
    )

    return ProjectConfig(
        schema_version=schema_version,
        procedural_seed=procedural_seed,
        stage_seeds=stage_seeds,
        world=world,
        run=run,
        export=export,
        host_field=host_field,
        network=network,
        section_field=section_field,
        events=events,
        geometry=geometry,
    )


def project_config_manifest(project_config: ProjectConfig) -> dict[str, Any]:
    """Return the completely resolved, JSON-serializable project settings."""

    manifest = asdict(project_config)
    manifest["canonical_coordinates"] = {
        "length_unit": "metre",
        "handedness": "right",
        "up_axis": "Z",
    }
    manifest["world"]["example_roof_demand_ratio"] = project_config.world.roof_demand_ratio(
        project_config.world.body.maximum_passage_width_m,
        max(project_config.section_field.minimum_roof_thickness, 0.1),
    )
    return manifest


def write_project_config_manifest(
    project_config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    """Write the resolved configuration used for one generation."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(project_config_manifest(project_config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


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
    world: WorldConfig,
) -> CaveNetworkConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    maximum_width = world.body.maximum_passage_width_m
    config_data.setdefault("base_passage_radius", 0.38 * maximum_width)
    config_data.setdefault("minimum_passage_radius", 0.22 * maximum_width)
    config_data.setdefault("maximum_passage_radius", 0.50 * maximum_width)
    config_data.setdefault("chamber_radius", 0.50 * world.body.maximum_room_width_m)
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
    world: WorldConfig,
) -> SectionFieldConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    config_data.setdefault(
        "maximum_tube_width",
        world.body.maximum_passage_width_m,
    )
    config_data.setdefault(
        "chamber_max_tube_width",
        world.body.maximum_room_width_m,
    )
    return SectionFieldConfig(**config_data)


def _build_geometry_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
    world: WorldConfig,
) -> GeometryConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    config_data.setdefault("tunnel_radius_scale", 1.0)
    config_data.setdefault("chamber_radius_scale", 1.0)
    config_data.setdefault("junction_radius_scale", 1.0)
    config_data.setdefault(
        "minimum_radius",
        min(3.5, max(0.5, 0.15 * world.body.maximum_passage_width_m)),
    )
    return GeometryConfig(**config_data)


def _build_event_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
    world: WorldConfig,
) -> GeologicalEventConfig:
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    if "enabled_kinds" in config_data:
        config_data["enabled_kinds"] = tuple(
            str(kind).strip().lower()
            for kind in config_data["enabled_kinds"]
        )
    enabled_kinds = set(
        config_data.get("enabled_kinds", GeologicalEventConfig.enabled_kinds)
    )
    unknown_kinds = enabled_kinds - SUPPORTED_EVENT_KINDS
    if unknown_kinds:
        raise ValueError(
            "events.enabled_kinds contains unsupported values: "
            + ", ".join(sorted(unknown_kinds))
        )
    config_data.setdefault("gravity_m_s2", world.body.gravity_m_s2)
    config_data.setdefault("rock_density_kg_m3", world.material.bulk_density_kg_m3)
    config_data.setdefault(
        "effective_tensile_strength_pa",
        world.material.effective_tensile_strength_pa,
    )
    for key in (
        "rock_radius_range",
        "boulder_radius_range",
        "collapse_radius_range",
        "choke_radius_range",
        "infill_radius_range",
    ):
        if key in config_data:
            config_data[key] = _to_range_tuple(config_data[key])
    return GeologicalEventConfig(**config_data)


def _apply_dev_mode(
    host_field: HostFieldConfig,
    network: CaveNetworkConfig,
    run: RunConfig,
) -> tuple[HostFieldConfig, CaveNetworkConfig]:
    """Crop generation extent while retaining body-scale passage dimensions."""

    target_height = min(host_field.grid.height, run.dev_max_route_length_m)
    if target_height >= host_field.grid.height:
        target_grid = host_field.grid
        scale = 1.0
    else:
        scale = target_height / host_field.grid.height
        target_ny = max(
            32,
            int(round(target_height / max(host_field.grid.spacing_y, 1e-6))) + 1,
        )
        target_grid = replace(
            host_field.grid,
            height=target_height,
            ny=target_ny,
        )

    half_width = 0.5 * target_grid.width
    half_height = 0.5 * target_grid.height
    seed_x = float(np.clip(host_field.seed_point[0], -0.45 * half_width, 0.45 * half_width))
    seed_y = float(np.clip(host_field.seed_point[1], -0.44 * target_grid.height, -0.28 * target_grid.height))
    seed_y = float(np.clip(seed_y, -0.90 * half_height, 0.90 * half_height))
    host_field = replace(
        host_field,
        grid=target_grid,
        seed_point=(seed_x, seed_y),
    )

    grammar = network.braid_grammar
    zone_min, zone_max = grammar.zone_count
    zone_cap = run.dev_max_braid_zones
    if zone_cap == 0:
        zone_count = (0, 0)
    else:
        zone_count = (min(zone_min, zone_cap), min(zone_max, zone_cap))
    grammar = replace(grammar, zone_count=zone_count)
    network = replace(
        network,
        braid_grammar=grammar,
        trace_max_steps=max(48, int(round(network.trace_max_steps * scale))),
        spur_count=min(network.spur_count, max(1, run.dev_max_braid_zones)),
        channel_count_samples=max(8, int(round(network.channel_count_samples * max(scale, 0.35)))),
    )
    return host_field, network


def _validate_pipeline_configs(
    *,
    host_field: HostFieldConfig,
    network: CaveNetworkConfig,
    section_field: SectionFieldConfig,
    events: GeologicalEventConfig,
    geometry: GeometryConfig,
) -> None:
    if host_field.grid.width <= 0.0 or host_field.grid.height <= 0.0:
        raise ValueError("host_field.grid width and height must be positive")
    if host_field.grid.nx < 2 or host_field.grid.ny < 2:
        raise ValueError("host_field.grid nx and ny must be at least 2")

    radii = (
        network.minimum_passage_radius,
        network.base_passage_radius,
        network.maximum_passage_radius,
    )
    if any(radius <= 0.0 for radius in radii):
        raise ValueError("network passage radii must be positive")
    if not radii[0] <= radii[1] <= radii[2]:
        raise ValueError(
            "network radii must satisfy minimum_passage_radius <= "
            "base_passage_radius <= maximum_passage_radius"
        )
    if section_field.maximum_tube_width <= 0.0:
        raise ValueError("section_field.maximum_tube_width must be positive")
    if section_field.chamber_max_tube_width < section_field.maximum_tube_width:
        raise ValueError(
            "section_field.chamber_max_tube_width cannot be smaller than "
            "maximum_tube_width"
        )

    density_names = (
        "rock_density_per_100m",
        "boulder_density_per_100m",
        "geological_event_density_per_100m",
    )
    for name in density_names:
        if getattr(events, name) < 0.0:
            raise ValueError(f"events.{name} cannot be negative")
    for name in (
        "minimum_event_spacing",
        "minimum_rock_spacing",
        "minimum_boulder_spacing",
    ):
        if getattr(events, name) < 0.0:
            raise ValueError(f"events.{name} cannot be negative")
    geological_fraction_names = (
        "collapse_event_fraction",
        "choke_event_fraction",
        "infill_event_fraction",
    )
    if any(getattr(events, name) < 0.0 for name in geological_fraction_names):
        raise ValueError("events collapse/choke/infill fractions cannot be negative")
    geological_fraction_sum = sum(
        getattr(events, name)
        for name in geological_fraction_names
    )
    if (
        events.geological_event_density_per_100m > 0.0
        and geological_fraction_sum <= 0.0
    ):
        raise ValueError(
            "events collapse/choke/infill fractions must sum to a positive value"
        )
    if not 0.0 <= events.ground_embed_fraction <= 0.5:
        raise ValueError("events.ground_embed_fraction must be in [0, 0.5]")
    for name in (
        "rock_radius_range",
        "boulder_radius_range",
        "collapse_radius_range",
        "choke_radius_range",
        "infill_radius_range",
    ):
        minimum, maximum = getattr(events, name)
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(
                f"events.{name} must contain positive ordered [min, max] values"
            )

    if geometry.voxel_size <= 0.0:
        raise ValueError("geometry.voxel_size must be positive")
    if geometry.chunk_size < 2:
        raise ValueError("geometry.chunk_size must be at least 2")
    if min(
        geometry.tunnel_radius_scale,
        geometry.chamber_radius_scale,
        geometry.junction_radius_scale,
    ) <= 0.0:
        raise ValueError("geometry radius scales must be positive")
    if geometry.structural_event_blend < 0.0:
        raise ValueError("geometry.structural_event_blend cannot be negative")
