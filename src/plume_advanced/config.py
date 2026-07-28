"""Project configuration loading utilities for the active cave-network pipeline."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.stages.events import GeologicalEventConfig
from plume_advanced.stages.floor_map import FloorMapConfig
from plume_advanced.stages.geometry import GeometryConfig
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, TerrainWave
from plume_advanced.stages.network import BraidGrammarConfig, CaveNetworkConfig
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

CURRENT_SCHEMA_VERSION = 2
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
) -> ProjectConfig:
    """Load the project TOML configuration file.

    ``world_body`` is a CLI-oriented override. When supplied, the body's
    default material replaces any material selected for the original body.
    """

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    unknown_top_level = set(raw_config) - SUPPORTED_TOP_LEVEL_KEYS
    if unknown_top_level:
        raise ValueError(
            "Unknown top-level configuration keys: "
            + ", ".join(sorted(unknown_top_level))
        )
    if world_body is not None:
        world_data = dict(raw_config.get("world", {}))
        world_data["body"] = world_body
        world_data.pop("material", None)
        raw_config["world"] = world_data

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
        extras=frozenset(
            {"apply_body_scaling", "ranges", "wave_ranges"}
        ),
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
    supported_ranges = {field.name for field in fields(HostFieldConfig)} | {
        "seed_point_x",
        "seed_point_y",
    }
    unknown_ranges = set(range_data) - supported_ranges
    if unknown_ranges:
        qualified = ", ".join(
            f"host_field.ranges.{key}" for key in sorted(unknown_ranges)
        )
        raise ValueError(f"Unknown configuration keys: {qualified}")

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
    supply_cooling_ratio = (
        flow_regime.supply_rate_scale / flow_regime.cooling_rate_scale
    )
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
    transport_scale = math.sqrt(
        flow_regime.supply_rate_scale / flow_regime.cooling_rate_scale
    )
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
    fracture_scale = (
        world.body.host_fracture_scale
        * math.sqrt(flow_regime.cooling_rate_scale)
    )
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
            float(config_data.get(key, getattr(HostFieldConfig, key)))
            * horizontal_scale
        )
    for key in ("fracture_zone_center_offset", "fracture_zone_width"):
        config_data[key] = (
            float(config_data.get(key, getattr(HostFieldConfig, key)))
            * fracture_scale
        )
    for key in (
        "longitudinal_drop",
        "corridor_depth",
        "volcanic_layer_thickness",
        "minimum_stable_cover",
    ):
        config_data[key] = (
            float(config_data.get(key, getattr(HostFieldConfig, key)))
            * vertical_scale
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
            float(config_data.get(key, getattr(CaveNetworkConfig, key)))
            * spatial_scale
        )
    config_data["max_uphill_step"] = (
        float(
            config_data.get(
                "max_uphill_step",
                CaveNetworkConfig.max_uphill_step,
            )
        )
        * _resolved_vertical_scale(world, flow_regime)
    )
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
        float(np.clip(value * branch_length_scale, 0.02, 0.24))
        for value in branch_length_range
    ]
    branch_abundance_scale = 0.75 + 0.50 * flow_regime.distributary_tendency
    for key, default_range, minimum in (
        ("zone_count", BraidGrammarConfig.zone_count, 0),
        ("branches_per_zone", BraidGrammarConfig.branches_per_zone, 2),
    ):
        value_range = _to_range_tuple(
            braid_grammar_data.get(key, list(default_range))
        )
        braid_grammar_data[key] = [
            max(minimum, int(round(value * branch_abundance_scale)))
            for value in value_range
        ]
    braid_grammar_values: Any = {
        key: _to_range_tuple(value) if isinstance(value, list) else value
        for key, value in braid_grammar_data.items()
    }
    config_data["braid_grammar"] = BraidGrammarConfig(**braid_grammar_values)
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
        "chamber_max_tube_width",
        world.body.maximum_room_width_m,
    )
    spatial_scale = _resolved_horizontal_scale(world, flow_regime)
    vertical_scale = _resolved_vertical_scale(world, flow_regime)
    for key in (
        "minimum_sample_spacing",
        "maximum_sample_spacing",
        "centerline_wobble_amplitude",
        "centerline_wobble_wavelength",
    ):
        config_data[key] = (
            float(config_data.get(key, getattr(SectionFieldConfig, key)))
            * spatial_scale
        )
    for key in (
        "minimum_roof_thickness",
        "maximum_centerline_depth",
        "vertical_level_spacing",
        "minimum_vertical_clearance",
    ):
        config_data[key] = (
            float(config_data.get(key, getattr(SectionFieldConfig, key)))
            * vertical_scale
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
        config_data[key] = (
            float(config_data.get(key, getattr(FloorMapConfig, key)))
            * spatial_scale
        )
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
    resolution_policy = str(
        config_data.pop("resolution_policy", "fixed")
    ).strip().lower()
    if resolution_policy not in {"fixed", "body"}:
        raise ValueError(
            "geometry.resolution_policy must be one of: body, fixed"
        )
    if resolution_policy == "body":
        if "voxel_size" in config_data:
            raise ValueError(
                "geometry.voxel_size cannot be combined with "
                "geometry.resolution_policy = \"body\"; use the fixed policy "
                "for an explicit voxel size"
            )
        target_samples = {
            "preview": 10.0,
            "standard": 14.0,
            "production": 20.0,
        }[run.quality]
        production_scale = {
            "preview": 2.0,
            "standard": 1.4,
            "production": 1.0,
        }[run.quality]
        sampling_voxel_size = (
            world.body.maximum_passage_width_m / target_samples
        )
        quality_voxel_size = (
            world.body.production_voxel_size_m * production_scale
        )
        config_data["voxel_size"] = min(
            sampling_voxel_size,
            quality_voxel_size,
        )
    else:
        target_samples = (
            world.body.maximum_passage_width_m
            / float(config_data.get("voxel_size", GeometryConfig.voxel_size))
        )
    config_data["resolution_policy"] = resolution_policy
    config_data["resolution_quality"] = run.quality
    config_data["characteristic_passage_width_m"] = (
        world.body.maximum_passage_width_m
    )
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

    representative_extent_scale = math.sqrt(
        max(host_field.target_route_length_m, 1.0) / 5_000.0
    )
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
    seed_y = float(np.clip(host_field.seed_point[1], -0.44 * target_grid.height, -0.28 * target_grid.height))
    seed_y = float(np.clip(seed_y, -0.90 * half_height, 0.90 * half_height))
    host_field = replace(
        host_field,
        grid=target_grid,
        seed_point=(seed_x, seed_y),
    )

    grammar = network.braid_grammar
    zone_min, zone_max = grammar.zone_count
    zone_cap = int(
        round(run.dev_max_braid_zones * representative_extent_scale)
    )
    if zone_cap == 0:
        zone_count = (0, 0)
    else:
        zone_count = (min(zone_min, zone_cap), min(zone_max, zone_cap))
    grammar = replace(grammar, zone_count=zone_count)
    network = replace(
        network,
        braid_grammar=grammar,
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
    if min(
        host_field.body_spatial_scale,
        host_field.body_vertical_scale,
        host_field.body_fracture_scale,
        host_field.target_route_length_m,
    ) <= 0.0:
        raise ValueError("host_field body scales and target route must be positive")

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
    if network.source_count <= 0:
        raise ValueError("network.source_count must be positive")
    if min(
        network.source_flux,
        network.source_temperature_k,
        network.nominal_flow_speed_m_s,
    ) <= 0.0:
        raise ValueError("network source flow values must be positive")
    if network.cooling_k_per_m < 0.0:
        raise ValueError("network.cooling_k_per_m cannot be negative")
    if not 0.0 < network.chamber_radius_fraction <= 1.0:
        raise ValueError("network.chamber_radius_fraction must be in (0, 1]")
    if section_field.maximum_tube_width <= 0.0:
        raise ValueError("section_field.maximum_tube_width must be positive")
    if section_field.chamber_max_tube_width < section_field.maximum_tube_width:
        raise ValueError(
            "section_field.chamber_max_tube_width cannot be smaller than "
            "maximum_tube_width"
        )
    if min(
        section_field.vertical_level_spacing,
        section_field.minimum_vertical_clearance,
    ) <= 0.0:
        raise ValueError("section_field vertical separation values must be positive")
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
    if not 0.0 <= events.clustered_debris_fraction <= 1.0:
        raise ValueError("events.clustered_debris_fraction must be in [0, 1]")
    if events.collapse_cluster_radius_scale <= 0.0:
        raise ValueError("events.collapse_cluster_radius_scale must be positive")
    if not 0.0 < events.collapse_cluster_spacing_scale <= 1.0:
        raise ValueError("events.collapse_cluster_spacing_scale must be in (0, 1]")
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
    if geometry.storage_mode not in {"auto", "dense", "tiled"}:
        raise ValueError("geometry.storage_mode must be auto, dense, or tiled")
    if geometry.max_dense_voxels <= 0:
        raise ValueError("geometry.max_dense_voxels must be positive")
    if geometry.embedded_texture_max_size <= 0:
        raise ValueError("geometry.embedded_texture_max_size must be positive")
    if geometry.resolution_policy not in {"body", "fixed"}:
        raise ValueError("geometry.resolution_policy must be one of: body, fixed")
    if geometry.resolution_quality not in {"preview", "standard", "production"}:
        raise ValueError(
            "geometry.resolution_quality must be preview, standard, or production"
        )
    if min(
        geometry.characteristic_passage_width_m,
        geometry.target_samples_across_passage,
        geometry.characteristic_samples_across_passage,
    ) <= 0.0:
        raise ValueError("geometry passage-resolution values must be positive")
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
