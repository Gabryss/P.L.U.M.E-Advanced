"""Project configuration loading utilities for the active cave-network pipeline."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.procedural import procedural_rng
from plume_advanced.stages.events import GeologicalEventConfig
from plume_advanced.stages.floor_map import FloorMapConfig
from plume_advanced.stages.geometry import GeometryConfig
from plume_advanced.stages.host_field import (
    GridConfig,
    HostFieldConfig,
    RoutingWeights,
    TerrainWave,
)
from plume_advanced.stages.network import (
    BraidGrammarConfig,
    CaveNetworkConfig,
    EmplacementHistoryConfig,
    LobeGrowthConfig,
)
from plume_advanced.stages.section_field import SectionFieldConfig
from plume_advanced.world import (
    SUPPORTED_EVENT_KINDS,
    ExportConfig,
    FlowRegimeConfig,
    RunConfig,
    StageSeeds,
    WorldConfig,
    build_export_config,
    build_flow_regime_config,
    build_run_config,
    derive_stage_seeds,
    resolve_world_config,
)

CURRENT_SCHEMA_VERSION = 3
SUPPORTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "procedural_seed",
        "world",
        "flow_regime",
        "run",
        "export",
        "host_field",
        "network",
        "section_field",
        "floor_map",
        "events",
        "geometry",
    }
)


def _reject_unknown_keys(
    path: str,
    data: dict[str, Any],
    config_type: type[Any],
    *,
    extras: frozenset[str] = frozenset(),
) -> None:
    """Reject misspelled nested settings with their complete TOML path."""

    supported = {field.name for field in fields(config_type)} | extras
    unknown = set(data) - supported
    if unknown:
        qualified = ", ".join(f"{path}.{key}" for key in sorted(unknown))
        raise ValueError(f"Unknown configuration keys: {qualified}")


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    schema_version: int
    procedural_seed: int | None
    stage_seeds: StageSeeds
    world: WorldConfig
    flow_regime: FlowRegimeConfig
    run: RunConfig
    export: ExportConfig
    host_field: HostFieldConfig
    network: CaveNetworkConfig
    section_field: SectionFieldConfig
    floor_map: FloorMapConfig
    events: GeologicalEventConfig
    geometry: GeometryConfig


def load_project_config(
    path: str | Path,
    *,
    world_body: str | None = None,
    dev_mode: bool | None = None,
) -> ProjectConfig:
    """Load the project TOML configuration file.

    ``world_body`` is a CLI-oriented override. When supplied, the body's
    default material replaces any material selected for the original body.
    """

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    source_schema_version = int(raw_config.get("schema_version", 1))
    if source_schema_version not in {1, 2, CURRENT_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported schema_version {source_schema_version}; "
            f"supported versions are 1, 2, and {CURRENT_SCHEMA_VERSION}"
        )
    raw_config = _migrate_project_config(raw_config, source_schema_version)
    unknown_top_level = set(raw_config) - SUPPORTED_TOP_LEVEL_KEYS
    if unknown_top_level:
        raise ValueError(
            "Unknown top-level configuration keys: " + ", ".join(sorted(unknown_top_level))
        )
    if world_body is not None:
        world_data = dict(raw_config.get("world", {}))
        world_data["body"] = world_body
        world_data.pop("material", None)
        raw_config["world"] = world_data
    if dev_mode is not None:
        run_data = dict(raw_config.get("run", {}))
        run_data["dev_mode"] = dev_mode
        raw_config["run"] = run_data

    schema_version = CURRENT_SCHEMA_VERSION
    procedural_seed = raw_config.get("procedural_seed")
    if procedural_seed is not None:
        procedural_seed = int(procedural_seed)
    stage_seeds = derive_stage_seeds(procedural_seed)
    world = resolve_world_config(raw_config.get("world"))
    flow_regime = build_flow_regime_config(raw_config.get("flow_regime"))
    run = build_run_config(raw_config.get("run"))
    export = build_export_config(raw_config.get("export"))

    host_field = _build_host_field_config(
        raw_config.get("host_field", {}),
        procedural_seed=stage_seeds.host,
        world=world,
        flow_regime=flow_regime,
    )
    network = _build_network_config(
        raw_config.get("network", {}),
        procedural_seed=stage_seeds.network,
        world=world,
        flow_regime=flow_regime,
    )
    section_field = _build_section_field_config(
        raw_config.get("section_field", {}),
        procedural_seed=stage_seeds.sections,
        world=world,
        flow_regime=flow_regime,
    )
    floor_map = _build_floor_map_config(
        raw_config.get("floor_map", {}),
        world=world,
        flow_regime=flow_regime,
    )
    events = _build_event_config(
        raw_config.get("events", {}),
        procedural_seed=stage_seeds.events,
        world=world,
    )
    events = _resolve_event_asset_paths(events, config_path.parent)
    geometry = _build_geometry_config(
        raw_config.get("geometry", {}),
        procedural_seed=stage_seeds.geometry,
        world=world,
        run=run,
    )
    geometry = _resolve_geometry_asset_paths(geometry, config_path.parent)
    if run.dev_mode:
        host_field, network = _apply_dev_mode(host_field, network, run)
    _validate_pipeline_configs(
        host_field=host_field,
        network=network,
        section_field=section_field,
        floor_map=floor_map,
        events=events,
        geometry=geometry,
    )

    return ProjectConfig(
        schema_version=schema_version,
        procedural_seed=procedural_seed,
        stage_seeds=stage_seeds,
        world=world,
        flow_regime=flow_regime,
        run=run,
        export=export,
        host_field=host_field,
        network=network,
        section_field=section_field,
        floor_map=floor_map,
        events=events,
        geometry=geometry,
    )


def _migrate_project_config(
    raw_config: dict[str, Any],
    source_schema_version: int,
) -> dict[str, Any]:
    """Normalize supported historical schemas into the active schema."""

    migrated = dict(raw_config)
    if source_schema_version >= CURRENT_SCHEMA_VERSION:
        return migrated

    export = dict(migrated.get("export", {}))
    unsupported_enabled = [
        key for key in ("generate_lods", "generate_wall_shell") if bool(export.get(key, False))
    ]
    if export.get("generate_visual", True) is False:
        unsupported_enabled.append("generate_visual = false")
    if unsupported_enabled:
        raise ValueError(
            "Cannot migrate removed export capabilities: " + ", ".join(unsupported_enabled)
        )
    for key in (
        "quality",
        "generate_visual",
        "generate_lods",
        "generate_wall_shell",
        "wall_thickness_m",
    ):
        export.pop(key, None)
    if export:
        migrated["export"] = export
    elif "export" in migrated:
        migrated["export"] = {}

    world = dict(migrated.get("world", {}))
    for key in (
        "atmosphere",
        "erosion_regime",
        "surface_deposit",
        "cohesion_mpa",
        "friction_angle_degrees",
        "mean_joint_spacing_m",
    ):
        world.pop(key, None)
    if world:
        migrated["world"] = world
    elif "world" in migrated:
        migrated["world"] = {}
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    return migrated


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
    world: WorldConfig,
    flow_regime: FlowRegimeConfig,
) -> HostFieldConfig:
    _reject_unknown_keys(
        "host_field",
        raw_config,
        HostFieldConfig,
        extras=frozenset({"apply_body_scaling", "ranges", "wave_ranges", "routing_weights"}),
    )
    config_data = dict(raw_config)
    apply_body_scaling = bool(config_data.pop("apply_body_scaling", True))
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    config_data.setdefault("gravity_m_s2", world.body.gravity_m_s2)
    config_data.setdefault("rock_density_kg_m3", world.material.bulk_density_kg_m3)
    config_data.setdefault(
        "effective_tensile_strength_pa",
        world.material.effective_tensile_strength_pa,
    )
    config_data.setdefault("material_quality", world.material.rock_mass_quality)
    config_data.setdefault("material_weathering", world.material.weathering)
    config_data.setdefault(
        "characteristic_passage_span_m",
        world.body.maximum_passage_width_m,
    )

    grid_data = config_data.pop("grid", {})
    wave_data = config_data.pop("waves", None)
    range_data = config_data.pop("ranges", {})
    wave_range_data = config_data.pop("wave_ranges", None)
    routing_weights_data = config_data.pop("routing_weights", {})

    _reject_unknown_keys("host_field.grid", grid_data, GridConfig)
    if wave_data is not None:
        for index, wave in enumerate(wave_data):
            _reject_unknown_keys(f"host_field.waves[{index}]", wave, TerrainWave)
    if wave_range_data is not None:
        _reject_unknown_keys(
            "host_field.wave_ranges",
            wave_range_data,
            TerrainWave,
            extras=frozenset({"count"}),
        )
    _reject_unknown_keys(
        "host_field.routing_weights",
        routing_weights_data,
        RoutingWeights,
    )
    config_data["routing_weights"] = RoutingWeights(**routing_weights_data)
    supported_ranges = {field.name for field in fields(HostFieldConfig)} | {
        "seed_point_x",
        "seed_point_y",
    }
    unknown_ranges = set(range_data) - supported_ranges
    if unknown_ranges:
        qualified = ", ".join(f"host_field.ranges.{key}" for key in sorted(unknown_ranges))
        raise ValueError(f"Unknown configuration keys: {qualified}")

    grid = GridConfig(**grid_data)
    rng = procedural_rng(procedural_seed, "host-config-ranges")

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

    (
        grid,
        seed_point,
        waves,
        config_data,
    ) = _apply_body_host_scaling(
        grid=grid,
        seed_point=seed_point,
        waves=waves,
        config_data=config_data,
        world=world,
        flow_regime=flow_regime,
        enabled=apply_body_scaling,
    )

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


def _resolved_horizontal_scale(
    world: WorldConfig,
    flow_regime: FlowRegimeConfig,
) -> float:
    supply_cooling_ratio = flow_regime.supply_rate_scale / flow_regime.cooling_rate_scale
    flow_scale = math.sqrt(float(np.clip(supply_cooling_ratio, 0.25, 4.0)))
    return world.body.host_horizontal_scale * flow_scale


def _resolved_vertical_scale(
    world: WorldConfig,
    flow_regime: FlowRegimeConfig,
) -> float:
    inflation_scale = 0.80 + 0.40 * flow_regime.inflation
    return world.body.host_vertical_scale * inflation_scale


def _resolved_route_length(
    world: WorldConfig,
    flow_regime: FlowRegimeConfig,
) -> float:
    transport_scale = math.sqrt(flow_regime.supply_rate_scale / flow_regime.cooling_rate_scale)
    return (
        world.body.default_route_length_m
        * flow_regime.duration_scale
        * float(np.clip(transport_scale, 0.5, 2.0))
    )


def _apply_body_host_scaling(
    *,
    grid: GridConfig,
    seed_point: tuple[float, float],
    waves: tuple[TerrainWave, ...] | None,
    config_data: dict[str, Any],
    world: WorldConfig,
    flow_regime: FlowRegimeConfig,
    enabled: bool,
) -> tuple[
    GridConfig,
    tuple[float, float],
    tuple[TerrainWave, ...] | None,
    dict[str, Any],
]:
    """Scale the host's physical correlation lengths while preserving resolution."""

    if not enabled:
        config_data.setdefault("body_spatial_scale", 1.0)
        config_data.setdefault("body_vertical_scale", 1.0)
        config_data.setdefault("body_fracture_scale", 1.0)
        config_data.setdefault("target_route_length_m", grid.height)
        return grid, seed_point, waves, config_data

    horizontal_scale = _resolved_horizontal_scale(world, flow_regime)
    vertical_scale = _resolved_vertical_scale(world, flow_regime)
    fracture_scale = world.body.host_fracture_scale * math.sqrt(flow_regime.cooling_rate_scale)
    target_route_length = _resolved_route_length(world, flow_regime)

    base_width = grid.width
    base_height = grid.height
    spacing_x = max(grid.spacing_x, 1e-6)
    spacing_y = max(grid.spacing_y, 1e-6)
    scaled_width = base_width * horizontal_scale
    scaled_height = max(base_height, 1.10 * target_route_length)
    grid = replace(
        grid,
        width=scaled_width,
        height=scaled_height,
        nx=max(32, int(round(scaled_width / spacing_x)) + 1),
        ny=max(32, int(round(scaled_height / spacing_y)) + 1),
    )
    seed_point = (
        float(seed_point[0]) * horizontal_scale,
        float(seed_point[1]) * scaled_height / max(base_height, 1e-6),
    )

    for key in ("corridor_width",):
        config_data[key] = (
            float(config_data.get(key, getattr(HostFieldConfig, key))) * horizontal_scale
        )
    for key in ("fracture_zone_center_offset", "fracture_zone_width"):
        config_data[key] = (
            float(config_data.get(key, getattr(HostFieldConfig, key))) * fracture_scale
        )
    for key in (
        "longitudinal_drop",
        "corridor_depth",
        "volcanic_layer_thickness",
        "minimum_stable_cover",
    ):
        config_data[key] = (
            float(config_data.get(key, getattr(HostFieldConfig, key))) * vertical_scale
        )

    source_waves = waves if waves is not None else HostFieldConfig().waves
    waves = tuple(
        replace(
            wave,
            amplitude=wave.amplitude * vertical_scale,
            wavelength=wave.wavelength * horizontal_scale,
        )
        for wave in source_waves
    )
    config_data.setdefault("body_spatial_scale", horizontal_scale)
    config_data.setdefault("body_vertical_scale", vertical_scale)
    config_data.setdefault("body_fracture_scale", fracture_scale)
    config_data.setdefault("target_route_length_m", target_route_length)
    return grid, seed_point, waves, config_data


def _build_network_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
    world: WorldConfig,
    flow_regime: FlowRegimeConfig,
) -> CaveNetworkConfig:
    _reject_unknown_keys("network", raw_config, CaveNetworkConfig)
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    maximum_width = world.body.maximum_passage_width_m
    spatial_scale = _resolved_horizontal_scale(world, flow_regime)
    target_route_length = _resolved_route_length(world, flow_regime)
    config_data.setdefault("body_spatial_scale", spatial_scale)
    config_data.setdefault("target_route_length_m", target_route_length)
    config_data.setdefault("source_flux", flow_regime.supply_rate_scale)
    config_data.setdefault(
        "cooling_k_per_m",
        CaveNetworkConfig.cooling_k_per_m * flow_regime.cooling_rate_scale,
    )
    config_data.setdefault("base_passage_radius", 0.38 * maximum_width)
    config_data.setdefault("minimum_passage_radius", 0.22 * maximum_width)
    config_data.setdefault("maximum_passage_radius", 0.50 * maximum_width)
    config_data.setdefault("chamber_radius", 0.50 * world.body.maximum_room_width_m)
    config_data.setdefault(
        "chamber_radius_fraction",
        0.45 + 0.45 * flow_regime.inflation,
    )
    config_data.setdefault("minimum_branch_offset_widths", 1.25)
    config_data.setdefault("paint_flux_chambers", False)
    for key in ("source_band_length", "source_band_half_width", "sink_margin"):
        config_data[key] = (
            float(config_data.get(key, getattr(CaveNetworkConfig, key))) * spatial_scale
        )
    config_data["max_uphill_step"] = float(
        config_data.get(
            "max_uphill_step",
            CaveNetworkConfig.max_uphill_step,
        )
    ) * _resolved_vertical_scale(world, flow_regime)
    config_data["spur_max_steps"] = max(
        1,
        int(
            round(
                float(
                    config_data.get(
                        "spur_max_steps",
                        CaveNetworkConfig.spur_max_steps,
                    )
                )
                * spatial_scale
            )
        ),
    )
    distributary_scale = 0.70 + flow_regime.distributary_tendency
    config_data["spur_count"] = max(
        0,
        int(
            round(
                float(config_data.get("spur_count", CaveNetworkConfig.spur_count))
                * distributary_scale
            )
        ),
    )
    braid_grammar_data = dict(config_data.pop("braid_grammar", {}))
    _reject_unknown_keys(
        "network.braid_grammar",
        braid_grammar_data,
        BraidGrammarConfig,
    )
    branch_length_scale = math.sqrt(spatial_scale)
    branch_length_range = _to_range_tuple(
        braid_grammar_data.get(
            "half_length_fraction",
            list(BraidGrammarConfig.half_length_fraction),
        )
    )
    braid_grammar_data["half_length_fraction"] = [
        float(np.clip(value * branch_length_scale, 0.02, 0.24)) for value in branch_length_range
    ]
    branch_abundance_scale = 0.75 + 0.50 * flow_regime.distributary_tendency
    for key, default_range, minimum in (
        ("zone_count", BraidGrammarConfig.zone_count, 0),
        ("branches_per_zone", BraidGrammarConfig.branches_per_zone, 2),
    ):
        value_range = _to_range_tuple(braid_grammar_data.get(key, list(default_range)))
        braid_grammar_data[key] = [
            max(minimum, int(round(value * branch_abundance_scale))) for value in value_range
        ]
    braid_grammar_values: Any = {
        key: _to_range_tuple(value) if isinstance(value, list) else value
        for key, value in braid_grammar_data.items()
    }
    config_data["braid_grammar"] = BraidGrammarConfig(**braid_grammar_values)
    lobe_growth_data = dict(config_data.pop("lobe_growth", {}))
    _reject_unknown_keys(
        "network.lobe_growth",
        lobe_growth_data,
        LobeGrowthConfig,
    )
    lobe_path_range = _to_range_tuple(
        lobe_growth_data.get("path_count", list(LobeGrowthConfig.path_count))
    )
    lobe_growth_data["path_count"] = [
        max(0, int(round(value * branch_abundance_scale))) for value in lobe_path_range
    ]
    lobe_growth_values: Any = {
        key: _to_range_tuple(value) if isinstance(value, list) else value
        for key, value in lobe_growth_data.items()
    }
    config_data["lobe_growth"] = LobeGrowthConfig(**lobe_growth_values)
    emplacement_history_data = dict(config_data.pop("emplacement_history", {}))
    _reject_unknown_keys(
        "network.emplacement_history",
        emplacement_history_data,
        EmplacementHistoryConfig,
    )
    emplacement_history_data.setdefault(
        "drained_pool_max_width_m",
        world.body.maximum_room_width_m,
    )
    emplacement_history_values: Any = {
        key: _to_range_tuple(value) if isinstance(value, list) else value
        for key, value in emplacement_history_data.items()
    }
    config_data["emplacement_history"] = EmplacementHistoryConfig(**emplacement_history_values)
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
    flow_regime: FlowRegimeConfig,
) -> SectionFieldConfig:
    _reject_unknown_keys("section_field", raw_config, SectionFieldConfig)
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    config_data.setdefault(
        "maximum_tube_width",
        world.body.maximum_passage_width_m,
    )
    config_data.setdefault(
        "minimum_tube_width",
        0.5,
    )
    config_data.setdefault(
        "minimum_tube_height",
        0.35,
    )
    config_data.setdefault("gravity_m_s2", world.body.gravity_m_s2)
    config_data.setdefault("rock_density_kg_m3", world.material.bulk_density_kg_m3)
    config_data.setdefault("effective_tensile_strength_pa", world.material.effective_tensile_strength_pa)
    config_data.setdefault("roof_safety_factor", world.body.roof_safety_factor)
    config_data.setdefault(
        "chamber_max_tube_width",
        world.body.maximum_room_width_m,
    )
    spatial_scale = _resolved_horizontal_scale(world, flow_regime)
    vertical_scale = _resolved_vertical_scale(world, flow_regime)
    for key in (
        "minimum_sample_spacing",
        "maximum_sample_spacing",
        "uniform_sample_spacing",
        "reference_sample_spacing",
        "centerline_wobble_amplitude",
        "centerline_wobble_wavelength",
        "morphology_correlation_length",
    ):
        config_data[key] = (
            float(config_data.get(key, getattr(SectionFieldConfig, key))) * spatial_scale
        )
    for key in (
        "minimum_roof_thickness",
        "maximum_centerline_depth",
        "vertical_level_spacing",
        "minimum_vertical_clearance",
    ):
        config_data[key] = (
            float(config_data.get(key, getattr(SectionFieldConfig, key))) * vertical_scale
        )
    config_data["chamber_widen_gain"] = float(
        config_data.get(
            "chamber_widen_gain",
            SectionFieldConfig.chamber_widen_gain,
        )
    ) * (0.70 + 0.60 * flow_regime.inflation)
    return SectionFieldConfig(**config_data)


def _build_floor_map_config(
    raw_config: dict[str, Any],
    *,
    world: WorldConfig,
    flow_regime: FlowRegimeConfig,
) -> FloorMapConfig:
    _reject_unknown_keys("floor_map", raw_config, FloorMapConfig)
    config_data = dict(raw_config)
    spatial_scale = _resolved_horizontal_scale(world, flow_regime)
    for key in ("lateral_spacing_m", "plan_resolution_m"):
        config_data[key] = float(config_data.get(key, getattr(FloorMapConfig, key))) * spatial_scale
    config_data["minimum_clearance_m"] = float(
        config_data.get(
            "minimum_clearance_m",
            FloorMapConfig.minimum_clearance_m,
        )
    ) * _resolved_vertical_scale(world, flow_regime)
    return FloorMapConfig(**config_data)


def _build_geometry_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
    world: WorldConfig,
    run: RunConfig,
) -> GeometryConfig:
    _reject_unknown_keys("geometry", raw_config, GeometryConfig)
    config_data = dict(raw_config)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    resolution_policy = str(config_data.pop("resolution_policy", "fixed")).strip().lower()
    if resolution_policy not in {"fixed", "body"}:
        raise ValueError("geometry.resolution_policy must be one of: body, fixed")
    if resolution_policy == "body":
        if "voxel_size" in config_data:
            raise ValueError(
                "geometry.voxel_size cannot be combined with "
                'geometry.resolution_policy = "body"; use the fixed policy '
                "for an explicit voxel size"
            )
        target_samples = {
            "preview": 10.0,
            "standard": 16.0,
            "production": 20.0,
        }[run.quality]
        production_scale = {
            "preview": 2.0,
            "standard": 1.2,
            "production": 1.0,
        }[run.quality]
        sampling_voxel_size = world.body.maximum_passage_width_m / target_samples
        quality_voxel_size = world.body.production_voxel_size_m * production_scale
        config_data["voxel_size"] = min(
            sampling_voxel_size,
            quality_voxel_size,
        )
    else:
        target_samples = world.body.maximum_passage_width_m / float(
            config_data.get("voxel_size", GeometryConfig.voxel_size)
        )
    config_data["resolution_policy"] = resolution_policy
    config_data["resolution_quality"] = run.quality
    config_data["characteristic_passage_width_m"] = world.body.maximum_passage_width_m
    config_data["target_samples_across_passage"] = target_samples
    config_data.setdefault("tunnel_radius_scale", 1.0)
    config_data.setdefault("chamber_radius_scale", 1.0)
    config_data.setdefault("junction_radius_scale", 1.0)
    config_data.setdefault(
        "minimum_radius",
        min(3.5, max(0.5, 0.15 * world.body.maximum_passage_width_m)),
    )
    return GeometryConfig(**config_data)


def _resolve_geometry_asset_paths(
    geometry: GeometryConfig,
    config_directory: Path,
) -> GeometryConfig:
    def resolve(value: str) -> str:
        if not value:
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (config_directory / path).resolve()
        return str(path)

    return replace(
        geometry,
        cave_diffuse_texture=resolve(geometry.cave_diffuse_texture),
        cave_normal_texture=resolve(geometry.cave_normal_texture),
        cave_roughness_texture=resolve(geometry.cave_roughness_texture),
        cave_displacement_texture=resolve(geometry.cave_displacement_texture),
    )


def _build_event_config(
    raw_config: dict[str, Any],
    *,
    procedural_seed: int | None,
    world: WorldConfig,
) -> GeologicalEventConfig:
    _reject_unknown_keys("events", raw_config, GeologicalEventConfig)
    config_data = dict(raw_config)
    config_data.setdefault("roof_safety_factor", world.body.roof_safety_factor)
    if "random_seed" not in config_data:
        config_data["random_seed"] = procedural_seed
    if "enabled_kinds" in config_data:
        config_data["enabled_kinds"] = tuple(
            str(kind).strip().lower() for kind in config_data["enabled_kinds"]
        )
    enabled_kinds = set(config_data.get("enabled_kinds", GeologicalEventConfig.enabled_kinds))
    unknown_kinds = enabled_kinds - SUPPORTED_EVENT_KINDS
    if unknown_kinds:
        raise ValueError(
            "events.enabled_kinds contains unsupported values: " + ", ".join(sorted(unknown_kinds))
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
        "boulder_satellite_count_range",
        "boulder_halo_radius_range_m",
        "collapse_fragment_count_range",
        "collapse_talus_radius_range_m",
        "minor_cluster_count_range",
        "minor_cluster_radius_range_m",
    ):
        if key in config_data:
            config_data[key] = _to_range_tuple(config_data[key])
    return GeologicalEventConfig(**config_data)


def _resolve_event_asset_paths(
    events: GeologicalEventConfig,
    config_directory: Path,
) -> GeologicalEventConfig:
    """Resolve Rocky development, texture, and scratch paths like cave assets."""

    def resolve(value: str) -> str:
        if not value:
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (config_directory / path).resolve()
        return str(path)

    return replace(
        events,
        rocky_source_path=resolve(events.rocky_source_path),
        rocky_texture_dir=resolve(events.rocky_texture_dir),
        rocky_output_dir=resolve(events.rocky_output_dir),
    )


def _apply_dev_mode(
    host_field: HostFieldConfig,
    network: CaveNetworkConfig,
    run: RunConfig,
) -> tuple[HostFieldConfig, CaveNetworkConfig]:
    """Crop generation extent while retaining body-scale passage dimensions."""

    representative_extent_scale = math.sqrt(max(host_field.target_route_length_m, 1.0) / 5_000.0)
    target_height = min(
        host_field.grid.height,
        run.dev_max_route_length_m * representative_extent_scale,
    )
    if target_height >= host_field.grid.height:
        target_grid = host_field.grid
    else:
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
    seed_y = float(
        np.clip(host_field.seed_point[1], -0.44 * target_grid.height, -0.28 * target_grid.height)
    )
    seed_y = float(np.clip(seed_y, -0.90 * half_height, 0.90 * half_height))
    host_field = replace(
        host_field,
        grid=target_grid,
        seed_point=(seed_x, seed_y),
    )

    grammar = network.braid_grammar
    zone_min, zone_max = grammar.zone_count
    zone_cap = int(round(run.dev_max_braid_zones * representative_extent_scale))
    if zone_cap == 0:
        zone_count = (0, 0)
    else:
        zone_count = (min(zone_min, zone_cap), min(zone_max, zone_cap))
    grammar = replace(grammar, zone_count=zone_count)
    lobe_growth = network.lobe_growth
    path_min, path_max = lobe_growth.path_count
    path_cap = max(1, 3 * max(zone_cap, 1))
    lobe_growth = replace(
        lobe_growth,
        path_count=(min(path_min, path_cap), min(path_max, path_cap)),
    )
    network = replace(
        network,
        braid_grammar=grammar,
        lobe_growth=lobe_growth,
        target_route_length_m=min(network.target_route_length_m, target_height),
        trace_max_steps=max(48, target_grid.ny),
        spur_count=min(network.spur_count, max(1, zone_cap)),
        channel_count_samples=max(
            8,
            int(round(network.channel_count_samples * representative_extent_scale)),
        ),
    )
    return host_field, network


def _validate_pipeline_configs(
    *,
    host_field: HostFieldConfig,
    network: CaveNetworkConfig,
    section_field: SectionFieldConfig,
    floor_map: FloorMapConfig,
    events: GeologicalEventConfig,
    geometry: GeometryConfig,
) -> None:
    if host_field.grid.width <= 0.0 or host_field.grid.height <= 0.0:
        raise ValueError("host_field.grid width and height must be positive")
    if host_field.grid.nx < 2 or host_field.grid.ny < 2:
        raise ValueError("host_field.grid nx and ny must be at least 2")
    if (
        min(
            host_field.body_spatial_scale,
            host_field.body_vertical_scale,
            host_field.body_fracture_scale,
            host_field.target_route_length_m,
        )
        <= 0.0
    ):
        raise ValueError("host_field body scales and target route must be positive")
    host_field.routing_weights.resolved()

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
    if network.minimum_branch_offset_widths <= 0.0:
        raise ValueError("network.minimum_branch_offset_widths must be positive")
    if network.target_route_length_m <= 0.0:
        raise ValueError("network.target_route_length_m must be positive")
    if network.source_count <= 0:
        raise ValueError("network.source_count must be positive")
    if network.source_band_length <= 0.0 or network.source_band_half_width <= 0.0:
        raise ValueError("network source-band dimensions must be positive")
    if network.sink_margin < 0.0:
        raise ValueError("network.sink_margin cannot be negative")
    if network.trace_max_steps <= 0 or network.spur_max_steps <= 0:
        raise ValueError("network trace step limits must be positive")
    if network.spur_count < 0 or network.occupancy_smoothing_passes < 0:
        raise ValueError("network spur and smoothing counts cannot be negative")
    if network.channel_count_samples < 2:
        raise ValueError("network.channel_count_samples must be at least 2")
    if network.max_uphill_step < 0.0:
        raise ValueError("network.max_uphill_step cannot be negative")
    if network.growth_cost_weight < 0.0 or network.corridor_weight < 0.0:
        raise ValueError("network routing support weights cannot be negative")
    if network.growth_cost_weight + network.corridor_weight <= 0.0:
        raise ValueError("at least one network routing support weight must be positive")
    if not 0.0 <= network.chamber_flux_quantile <= 1.0:
        raise ValueError("network.chamber_flux_quantile must be in [0, 1]")
    if network.chamber_radius <= 0.0:
        raise ValueError("network.chamber_radius must be positive")
    if (
        min(
            network.source_flux,
            network.source_temperature_k,
            network.nominal_flow_speed_m_s,
        )
        <= 0.0
    ):
        raise ValueError("network source flow values must be positive")
    if network.cooling_k_per_m < 0.0:
        raise ValueError("network.cooling_k_per_m cannot be negative")
    if not 0.0 < network.chamber_radius_fraction <= 1.0:
        raise ValueError("network.chamber_radius_fraction must be in (0, 1]")
    if network.growth_model not in {"hybrid_lobe", "legacy_braid"}:
        raise ValueError("network.growth_model must be hybrid_lobe or legacy_braid")
    if network.emplacement_backend not in {
        "internal",
        "downflow_reference",
        "flowy",
    }:
        raise ValueError(
            "network.emplacement_backend must be internal, downflow_reference, or flowy"
        )
    if network.flowy_timeout_s <= 0.0:
        raise ValueError("network.flowy_timeout_s must be positive")
    if network.downflow_ensemble_size <= 0:
        raise ValueError("network.downflow_ensemble_size must be positive")
    if network.emplacement_backend == "flowy" and not network.flowy_executable:
        raise ValueError("network.flowy_executable is required when emplacement_backend='flowy'")
    if not 0.0 <= network.network_density <= 3.0:
        raise ValueError("network.network_density must be in [0, 3]")
    if network.lobe_launch_rate < 0.0:
        raise ValueError("network.lobe_launch_rate cannot be negative")
    for name, value in (
        ("loop_probability", network.loop_probability),
        ("capture_probability", network.capture_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"network.{name} must be in [0, 1]")
    if network.chamber_gain < 0.0:
        raise ValueError("network.chamber_gain cannot be negative")
    lobe = network.lobe_growth
    for name, value_range in (
        ("path_count", lobe.path_count),
        ("maximum_steps", lobe.maximum_steps),
        ("branch_flux_fraction", lobe.branch_flux_fraction),
    ):
        if value_range[0] > value_range[1]:
            raise ValueError(f"network.lobe_growth.{name} must have min <= max")
    if lobe.path_count[0] < 0 or lobe.maximum_steps[0] <= 0:
        raise ValueError("network.lobe_growth path counts and step limits are invalid")
    if lobe.minimum_persistence_steps <= 0:
        raise ValueError("network.lobe_growth.minimum_persistence_steps must be positive")
    if lobe.minimum_persistence_steps > lobe.maximum_steps[1]:
        raise ValueError(
            "network.lobe_growth.minimum_persistence_steps cannot exceed maximum_steps"
        )
    positive_lobe_values = {
        "minimum_anchor_spacing_fraction": lobe.minimum_anchor_spacing_fraction,
        "perturbation_correlation_cells": lobe.perturbation_correlation_cells,
        "candidate_temperature": lobe.candidate_temperature,
        "exposed_cooling_multiplier": lobe.exposed_cooling_multiplier,
        "retirement_temperature_k": lobe.retirement_temperature_k,
        "deposition_spread_cells": lobe.deposition_spread_cells,
    }
    if any(value <= 0.0 for value in positive_lobe_values.values()):
        raise ValueError("network.lobe_growth positive controls must be greater than zero")
    if lobe.terrain_perturbation_m < 0.0:
        raise ValueError("network.lobe_growth.terrain_perturbation_m cannot be negative")
    if lobe.deposition_feedback_m < 0.0:
        raise ValueError("network.lobe_growth.deposition_feedback_m cannot be negative")
    if not 0.0 <= lobe.backbone_curvature_fraction <= 1.0:
        raise ValueError("network.lobe_growth.backbone_curvature_fraction must be in [0, 1]")
    if lobe.backbone_curvature_wavelength_fraction <= 0.0:
        raise ValueError(
            "network.lobe_growth.backbone_curvature_wavelength_fraction must be positive"
        )
    if not 0.0 <= lobe.backbone_curvature_secondary_fraction <= 1.0:
        raise ValueError(
            "network.lobe_growth.backbone_curvature_secondary_fraction must be in [0, 1]"
        )
    if (
        min(
            lobe.inertia_weight,
            lobe.perturbed_slope_weight,
            lobe.downstream_potential_weight,
            lobe.initial_divergence_weight,
            lobe.channel_avoidance_weight,
            lobe.channel_reuse_weight,
            lobe.branch_flux_fraction[0],
            lobe.breakout_capacity_weight,
            lobe.breakout_confinement_weight,
            lobe.breakout_curvature_weight,
            lobe.breakout_blockage_weight,
        )
        < 0.0
    ):
        raise ValueError("network.lobe_growth weights and flux fractions cannot be negative")
    if not 0.0 <= lobe.retired_path_fraction <= 1.0:
        raise ValueError("network.lobe_growth.retired_path_fraction must be in [0, 1]")
    if not 0.0 < lobe.minimum_viable_flux_fraction <= 1.0:
        raise ValueError("network.lobe_growth.minimum_viable_flux_fraction must be in (0, 1]")
    if not 0.0 <= lobe.coalescence_flux_return_fraction <= 1.0:
        raise ValueError("network.lobe_growth.coalescence_flux_return_fraction must be in [0, 1]")
    history = network.emplacement_history
    for name, value_range in (
        ("phase_count", history.phase_count),
        ("active_phase_span", history.active_phase_span),
    ):
        if value_range[0] > value_range[1]:
            raise ValueError(f"network.emplacement_history.{name} must have min <= max")
        if value_range[0] <= 0:
            raise ValueError(f"network.emplacement_history.{name} values must be positive")
    if history.maximum_absolute_level < 0:
        raise ValueError("network.emplacement_history.maximum_absolute_level cannot be negative")
    for name, value in (
        ("stacked_lobe_fraction", history.stacked_lobe_fraction),
        ("chamber_formation_probability", history.chamber_formation_probability),
        (
            "vertical_capture_chamber_probability",
            history.vertical_capture_chamber_probability,
        ),
        ("roof_failure_probability", history.roof_failure_probability),
        ("reoccupation_probability", history.reoccupation_probability),
        ("breakout_probability", history.breakout_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"network.emplacement_history.{name} must be in [0, 1]")
    if history.phase_flux_budget_fraction < 0.0:
        raise ValueError(
            "network.emplacement_history.phase_flux_budget_fraction cannot be negative"
        )
    if not 0.0 <= history.retirement_flux_threshold <= 1.0:
        raise ValueError("network.emplacement_history.retirement_flux_threshold must be in [0, 1]")
    if (
        history.drained_pool_count[0] > history.drained_pool_count[1]
        or history.drained_pool_count[0] < 0
    ):
        raise ValueError(
            "network.emplacement_history.drained_pool_count must be a nonnegative range"
        )
    if history.drained_pool_min_spacing_m < 0.0:
        raise ValueError(
            "network.emplacement_history.drained_pool_min_spacing_m cannot be negative"
        )
    for name, value in (
        ("drained_pool_probability", history.drained_pool_probability),
        ("drained_pool_flux_quantile", history.drained_pool_flux_quantile),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"network.emplacement_history.{name} must be in [0, 1]")
    for name, value_range in (
        ("drained_pool_length_m", history.drained_pool_length_m),
        ("drained_pool_width_ratio", history.drained_pool_width_ratio),
        ("drained_pool_depth_m", history.drained_pool_depth_m),
    ):
        if value_range[0] <= 0.0 or value_range[0] > value_range[1]:
            raise ValueError(f"network.emplacement_history.{name} must be positive min <= max")
    if history.drained_pool_min_spacing_m == 0.0:
        raise ValueError("network.emplacement_history.drained_pool_min_spacing_m must be positive")
    if history.drained_pool_max_width_m <= 0.0:
        raise ValueError("network.emplacement_history.drained_pool_max_width_m must be positive")
    grammar = network.braid_grammar
    for name, value_range in (
        ("zone_count", grammar.zone_count),
        ("half_length_fraction", grammar.half_length_fraction),
        ("branches_per_zone", grammar.branches_per_zone),
        ("lateral_offset_scale", grammar.lateral_offset_scale),
        ("start_shift_fraction", grammar.start_shift_fraction),
        ("end_shift_fraction", grammar.end_shift_fraction),
        ("skew", grammar.skew),
        ("wobble", grammar.wobble),
        ("ladder_rung_count", grammar.ladder_rung_count),
        ("chamber_radius_scale", grammar.chamber_radius_scale),
    ):
        if value_range[0] > value_range[1]:
            raise ValueError(f"network.braid_grammar.{name} must have min <= max")
    if grammar.zone_count[0] < 0 or grammar.branches_per_zone[0] < 2:
        raise ValueError("network braid counts must be non-negative with at least two branches")
    if grammar.ladder_rung_count[0] < 0:
        raise ValueError("network.braid_grammar.ladder_rung_count cannot be negative")
    if not 0.0 <= grammar.min_center_spacing <= 1.0:
        raise ValueError("network.braid_grammar.min_center_spacing must be in [0, 1]")
    for name, probability in (
        ("underpass_probability", grammar.underpass_probability),
        ("ladder_probability", grammar.ladder_probability),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"network.braid_grammar.{name} must be in [0, 1]")
    if not 0.0 < section_field.minimum_tube_width <= section_field.maximum_tube_width:
        raise ValueError(
            "section_field widths must satisfy 0 < minimum_tube_width <= maximum_tube_width"
        )
    if section_field.minimum_tube_height <= 0.0:
        raise ValueError("section_field.minimum_tube_height must be positive")
    section_field.roof_stability_model  # validates finite physical inputs
    for name in ("bench_strength", "floor_incision_ratio"):
        value = getattr(section_field, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"section_field.{name} must be finite and in [0, 1]")
    if not (
        0.0
        < section_field.minimum_height_ratio
        <= section_field.base_height_ratio
        <= section_field.maximum_height_ratio
    ):
        raise ValueError(
            "section_field height ratios must satisfy "
            "0 < minimum_height_ratio <= base_height_ratio <= maximum_height_ratio"
        )
    if section_field.width_scale_median <= 0.0:
        raise ValueError("section_field.width_scale_median must be positive")
    nonnegative_morphology_values = {
        "width_scale_log_sigma": section_field.width_scale_log_sigma,
        "width_longitudinal_variation": section_field.width_longitudinal_variation,
        "height_ratio_variation": section_field.height_ratio_variation,
        "height_ratio_longitudinal_variation": (section_field.height_ratio_longitudinal_variation),
        "floor_relief_base": section_field.floor_relief_base,
        "floor_relief_variation": section_field.floor_relief_variation,
        "wall_roughness_base": section_field.wall_roughness_base,
        "wall_roughness_variation": section_field.wall_roughness_variation,
        "morphology_gradient_strength": section_field.morphology_gradient_strength,
        "morphology_regime_strength": section_field.morphology_regime_strength,
        "morphology_family_spread": section_field.morphology_family_spread,
        "morphology_flux_width_gain": section_field.morphology_flux_width_gain,
        "morphology_age_width_gain": section_field.morphology_age_width_gain,
        "morphology_floor_relief_max": section_field.morphology_floor_relief_max,
        "morphology_wall_roughness_max": section_field.morphology_wall_roughness_max,
        "profile_shape_variation": section_field.profile_shape_variation,
    }
    if any(value < 0.0 for value in nonnegative_morphology_values.values()):
        names = ", ".join(
            name for name, value in nonnegative_morphology_values.items() if value < 0.0
        )
        raise ValueError(f"section_field morphology amplitudes cannot be negative: {names}")
    if section_field.morphology_correlation_length <= 0.0:
        raise ValueError("section_field.morphology_correlation_length must be positive")
    if section_field.morphology_floor_relief_max < section_field.floor_relief_base:
        raise ValueError("section_field.morphology_floor_relief_max must be >= floor_relief_base")
    if section_field.morphology_wall_roughness_max < section_field.wall_roughness_base:
        raise ValueError(
            "section_field.morphology_wall_roughness_max must be >= wall_roughness_base"
        )
    if section_field.maximum_uphill_grade < 0.0:
        raise ValueError("section_field.maximum_uphill_grade cannot be negative")
    if not 0.0 < section_field.level_transition_fraction < 0.5:
        raise ValueError("section_field.level_transition_fraction must be in (0, 0.5)")
    if section_field.chamber_max_tube_width < section_field.maximum_tube_width:
        raise ValueError(
            "section_field.chamber_max_tube_width cannot be smaller than maximum_tube_width"
        )
    if section_field.drained_pool_width_scale <= 0.0:
        raise ValueError("section_field.drained_pool_width_scale must be positive")
    if not 0.0 < section_field.drained_pool_height_ratio_limit <= 1.0:
        raise ValueError("section_field.drained_pool_height_ratio_limit must be in (0, 1]")
    if not 0.0 <= section_field.drained_pool_floor_flatness <= 1.0:
        raise ValueError("section_field.drained_pool_floor_flatness must be in [0, 1]")
    if section_field.drained_pool_roof_arch <= 0.0:
        raise ValueError("section_field.drained_pool_roof_arch must be positive")
    if section_field.drained_pool_transition_power <= 0.0:
        raise ValueError("section_field.drained_pool_transition_power must be positive")
    if section_field.sampling_policy not in {"adaptive", "uniform", "reference"}:
        raise ValueError("section_field.sampling_policy must be adaptive, uniform, or reference")
    if (
        min(
            section_field.minimum_sample_spacing,
            section_field.maximum_sample_spacing,
            section_field.uniform_sample_spacing,
            section_field.reference_sample_spacing,
        )
        <= 0.0
    ):
        raise ValueError("section_field sample spacings must be positive")
    if section_field.vertical_level_spacing < 0.0 or section_field.minimum_vertical_clearance <= 0.0:
        raise ValueError("section_field level spacing must be nonnegative and clearance positive")
    if floor_map.lateral_spacing_m <= 0.0:
        raise ValueError("floor_map.lateral_spacing_m must be positive")
    if floor_map.plan_resolution_m <= 0.0:
        raise ValueError("floor_map.plan_resolution_m must be positive")
    if not 0.0 < floor_map.maximum_lateral_fraction < 1.0:
        raise ValueError("floor_map.maximum_lateral_fraction must be in (0, 1)")
    if floor_map.minimum_clearance_m < 0.0:
        raise ValueError("floor_map.minimum_clearance_m cannot be negative")

    density_names = (
        "rock_density_per_100m",
        "boulder_density_per_100m",
        "rock_density_per_100m2",
        "boulder_density_per_100m2",
        "geological_event_density_per_100m",
        "minor_cluster_density_per_1000m2",
    )
    for name in density_names:
        if getattr(events, name) < 0.0:
            raise ValueError(f"events.{name} cannot be negative")
    if events.debris_density_basis not in {"floor_area", "length"}:
        raise ValueError("events.debris_density_basis must be floor_area or length")
    if events.rock_population_multiplier <= 0.0:
        raise ValueError("events.rock_population_multiplier must be positive")
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
    geological_fraction_sum = sum(getattr(events, name) for name in geological_fraction_names)
    if events.geological_event_density_per_100m > 0.0 and geological_fraction_sum <= 0.0:
        raise ValueError("events collapse/choke/infill fractions must sum to a positive value")
    if not 0.0 <= events.ground_embed_fraction <= 0.5:
        raise ValueError("events.ground_embed_fraction must be in [0, 0.5]")
    if not 0.0 <= events.clustered_debris_fraction <= 1.0:
        raise ValueError("events.clustered_debris_fraction must be in [0, 1]")
    if events.collapse_cluster_radius_scale <= 0.0:
        raise ValueError("events.collapse_cluster_radius_scale must be positive")
    if not 0.0 < events.collapse_cluster_spacing_scale <= 1.0:
        raise ValueError("events.collapse_cluster_spacing_scale must be in (0, 1]")
    for name in (
        "gallery_width_size_fraction",
        "gallery_clearance_size_fraction",
        "roof_block_size_fraction",
        "background_contact_spacing",
        "rover_width_m",
        "rover_max_lateral_slope",
    ):
        if getattr(events, name) <= 0.0:
            raise ValueError(f"events.{name} must be positive")
    if not 0.0 < events.boulder_max_height_fraction <= 1.0:
        raise ValueError("events.boulder_max_height_fraction must be in (0, 1]")
    if events.edge_accumulation_strength < 0.0:
        raise ValueError("events.edge_accumulation_strength cannot be negative")
    if events.placement_jitter_m < 0.0:
        raise ValueError("events.placement_jitter_m cannot be negative")
    if events.rover_side_margin_m < 0.0:
        raise ValueError("events.rover_side_margin_m cannot be negative")
    if not 0.0 <= events.clean_floor_fraction < 1.0:
        raise ValueError("events.clean_floor_fraction must be in [0, 1)")
    if events.debris_patch_length_m <= 0.0:
        raise ValueError("events.debris_patch_length_m must be positive")
    if (
        events.wall_scree_fraction < 0.0
        or events.transported_lag_fraction < 0.0
        or (events.wall_scree_fraction + events.transported_lag_fraction > 1.0)
    ):
        raise ValueError(
            "events wall_scree_fraction and transported_lag_fraction must "
            "be non-negative and sum to at most 1"
        )
    for name in (
        "boulder_satellite_count_range",
        "collapse_fragment_count_range",
        "minor_cluster_count_range",
    ):
        minimum, maximum = getattr(events, name)
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"events.{name} must contain non-negative ordered values")
    for name in (
        "boulder_halo_radius_range_m",
        "collapse_talus_radius_range_m",
        "minor_cluster_radius_range_m",
    ):
        minimum, maximum = getattr(events, name)
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(f"events.{name} must contain positive ordered values")
    for name in (
        "rock_radius_range",
        "boulder_radius_range",
        "collapse_radius_range",
        "choke_radius_range",
        "infill_radius_range",
    ):
        minimum, maximum = getattr(events, name)
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(f"events.{name} must contain positive ordered [min, max] values")
    for name in ("rock_size_bias", "boulder_size_bias"):
        if getattr(events, name) <= 0.0:
            raise ValueError(f"events.{name} must be positive")
    if events.rocky_resolution_scale <= 0.0:
        raise ValueError("events.rocky_resolution_scale must be positive")
    if not 1 <= events.rocky_max_subdivisions <= 6:
        raise ValueError("events.rocky_max_subdivisions must be in [1, 6]")

    if geometry.voxel_size <= 0.0:
        raise ValueError("geometry.voxel_size must be positive")
    for name in ("floor_roughness_scale", "roof_roughness_scale", "surface_wall_relief_m",
                 "surface_roof_relief_m", "surface_floor_relief_m", "surface_crust_relief_m"):
        value = getattr(geometry, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"geometry.{name} must be finite and non-negative")
    for name in ("surface_feature_scale_m", "surface_normal_filter_voxels"):
        value = getattr(geometry, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"geometry.{name} must be finite and positive")
    if geometry.storage_mode not in {"auto", "dense", "tiled"}:
        raise ValueError("geometry.storage_mode must be auto, dense, or tiled")
    if geometry.max_dense_voxels <= 0:
        raise ValueError("geometry.max_dense_voxels must be positive")
    if geometry.embedded_texture_max_size <= 0:
        raise ValueError("geometry.embedded_texture_max_size must be positive")
    if not 0.0 <= geometry.cave_normal_scale <= 10.0:
        raise ValueError("geometry.cave_normal_scale must be in [0, 10]")
    if not 0 <= geometry.cave_smoothing_iterations <= 50:
        raise ValueError("geometry.cave_smoothing_iterations must be in [0, 50]")
    if geometry.cave_displacement_scale_m < 0.0:
        raise ValueError("geometry.cave_displacement_scale_m must be non-negative")
    if not 0.0 < geometry.cave_displacement_midlevel < 1.0:
        raise ValueError("geometry.cave_displacement_midlevel must be in (0, 1)")
    if geometry.resolution_policy not in {"body", "fixed"}:
        raise ValueError("geometry.resolution_policy must be one of: body, fixed")
    if geometry.resolution_quality not in {"preview", "standard", "production"}:
        raise ValueError("geometry.resolution_quality must be preview, standard, or production")
    if (
        min(
            geometry.characteristic_passage_width_m,
            geometry.target_samples_across_passage,
            geometry.characteristic_samples_across_passage,
        )
        <= 0.0
    ):
        raise ValueError("geometry passage-resolution values must be positive")
    if geometry.chunk_size < 2:
        raise ValueError("geometry.chunk_size must be at least 2")
    if (
        min(
            geometry.tunnel_radius_scale,
            geometry.chamber_radius_scale,
            geometry.junction_radius_scale,
        )
        <= 0.0
    ):
        raise ValueError("geometry radius scales must be positive")
    if geometry.structural_event_blend < 0.0:
        raise ValueError("geometry.structural_event_blend cannot be negative")
