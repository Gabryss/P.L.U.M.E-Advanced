"""Resolved celestial, geological, run-mode, and export configuration.

The generator works in metres in a canonical right-handed, Z-up coordinate
system.  Celestial-body profiles affect generation; exporters alone are
responsible for converting units and axes for target applications.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from plume_advanced.procedural import derive_subseed
from plume_advanced.stability import RoofStabilityModel

SUPPORTED_EVENT_KINDS = frozenset({"rock", "boulder", "collapse", "choke", "infill"})
SUPPORTED_EXPORT_TARGETS = frozenset(
    {"all", "neutral", "blender", "ue5", "unity", "gazebo", "omniverse"}
)
SUPPORTED_EXPORT_FORMATS = frozenset({"auto", "glb", "obj", "usd"})
SUPPORTED_QUALITY_LEVELS = frozenset({"preview", "standard", "production"})
EXPORT_FORMATS_BY_TARGET = {
    "all": frozenset({"auto"}),
    "neutral": frozenset({"glb", "obj"}),
    "blender": frozenset({"glb", "obj"}),
    "ue5": frozenset({"glb", "obj"}),
    "unity": frozenset({"glb", "obj"}),
    "gazebo": frozenset({"obj"}),
    "omniverse": frozenset({"usd"}),
}

APPLICATION_EXPORT_FORMATS = {
    "blender": "glb",
    "ue5": "glb",
    "unity": "glb",
    "gazebo": "obj",
    "omniverse": "usd",
}


@dataclass(frozen=True)
class CelestialBodyProfile:
    """Procedural defaults for one celestial environment.

    Passage and route limits are conservative project defaults rather than
    universal geological maxima.  Users can override every value in TOML.
    """

    name: str
    gravity_m_s2: float
    default_material: str
    maximum_passage_width_m: float
    maximum_room_width_m: float
    default_route_length_m: float
    production_voxel_size_m: float
    host_horizontal_scale: float
    host_vertical_scale: float
    host_fracture_scale: float
    roof_safety_factor: float = 1.5


@dataclass(frozen=True)
class GeologicalMaterialProfile:
    """Rock-mass parameters used by procedural stability surrogates."""

    name: str
    bulk_density_kg_m3: float
    intact_tensile_strength_mpa: float
    rock_mass_quality: float
    weathering: float

    @property
    def effective_tensile_strength_pa(self) -> float:
        """Return a deliberately conservative fractured-rock strength."""

        quality = float(np.clip(self.rock_mass_quality, 0.05, 1.0))
        weathering_factor = 1.0 - 0.65 * float(np.clip(self.weathering, 0.0, 1.0))
        return self.intact_tensile_strength_mpa * 1_000_000.0 * quality * weathering_factor


@dataclass(frozen=True)
class WorldConfig:
    """Fully resolved physical context consumed by procedural stages."""

    body: CelestialBodyProfile
    material: GeologicalMaterialProfile

    @property
    def roof_stability_model(self) -> RoofStabilityModel:
        return RoofStabilityModel(
            gravity_m_s2=self.body.gravity_m_s2,
            rock_density_kg_m3=self.material.bulk_density_kg_m3,
            effective_tensile_strength_pa=self.material.effective_tensile_strength_pa,
            safety_factor=self.body.roof_safety_factor,
        )

    def roof_demand_ratio(self, span_m: float, roof_thickness_m: float) -> float:
        """Estimate gravity-driven roof demand relative to rock-mass capacity.

        This is a fast procedural surrogate, not a finite-element analysis.
        """

        span = max(float(span_m), 0.0)
        thickness = max(float(roof_thickness_m), 0.1)
        return self.roof_stability_model.assess(
            width_m=span, height_m=0.0, floor_depth_m=thickness
        ).demand_ratio


@dataclass(frozen=True)
class RunConfig:
    """Controls generation extent without changing the selected world physics."""

    dev_mode: bool = False
    quality: str = "standard"
    dev_max_route_length_m: float = 1500.0
    dev_max_lobe_paths: int = 6
    render_diagnostics: bool = True
    overwrite_outputs: bool = False


@dataclass(frozen=True)
class FlowRegimeConfig:
    """Procedural eruption controls independent of the celestial body.

    These are dimensionless surrogates, not a thermofluid simulation. They
    separate lava-supply history from gravity and rock stability so that two
    environments on the same body can still have different morphologies.
    """

    supply_rate_scale: float = 1.0
    duration_scale: float = 1.0
    inflation: float = 0.50
    distributary_tendency: float = 0.55
    cooling_rate_scale: float = 1.0


@dataclass(frozen=True)
class ExportConfig:
    """Target selection and target-independent export requirements."""

    target: str = "blender"
    file_format: str = "glb"
    generate_collision: bool = True
    max_visual_triangles: int = 0
    max_asset_bytes: int = 0


@dataclass(frozen=True)
class StageSeeds:
    """Stable named seeds so optional stages cannot perturb each other."""

    host: int | None
    network: int | None
    sections: int | None
    events: int | None
    geometry: int | None


BODY_PRESETS: dict[str, CelestialBodyProfile] = {
    "earth": CelestialBodyProfile(
        name="earth",
        gravity_m_s2=9.80665,
        default_material="terrestrial_basalt",
        maximum_passage_width_m=10.0,
        # PDC calibration sections reach 26.3 m and the mapped Valentine
        # compound pool exceeds 22.9 m. Keep ordinary passages at 10 m while
        # allowing sparse, explicitly labelled rooms to reach that endmember.
        maximum_room_width_m=28.0,
        default_route_length_m=5_000.0,
        production_voxel_size_m=0.5,
        host_horizontal_scale=1.0,
        host_vertical_scale=1.0,
        host_fracture_scale=1.0,
    ),
    "mars": CelestialBodyProfile(
        name="mars",
        gravity_m_s2=3.71,
        default_material="martian_basalt",
        maximum_passage_width_m=50.0,
        maximum_room_width_m=100.0,
        default_route_length_m=15_000.0,
        production_voxel_size_m=1.0,
        host_horizontal_scale=2.25,
        host_vertical_scale=1.70,
        host_fracture_scale=1.80,
    ),
    "moon": CelestialBodyProfile(
        name="moon",
        gravity_m_s2=1.62,
        default_material="mare_basalt",
        maximum_passage_width_m=100.0,
        maximum_room_width_m=200.0,
        default_route_length_m=30_000.0,
        production_voxel_size_m=2.0,
        host_horizontal_scale=3.25,
        host_vertical_scale=2.30,
        host_fracture_scale=2.60,
    ),
}


MATERIAL_PRESETS: dict[str, GeologicalMaterialProfile] = {
    "terrestrial_basalt": GeologicalMaterialProfile(
        name="terrestrial_basalt",
        bulk_density_kg_m3=2_900.0,
        intact_tensile_strength_mpa=8.0,
        rock_mass_quality=0.55,
        weathering=0.30,
    ),
    "martian_basalt": GeologicalMaterialProfile(
        name="martian_basalt",
        bulk_density_kg_m3=2_950.0,
        intact_tensile_strength_mpa=9.0,
        rock_mass_quality=0.62,
        weathering=0.14,
    ),
    "mare_basalt": GeologicalMaterialProfile(
        name="mare_basalt",
        bulk_density_kg_m3=3_050.0,
        intact_tensile_strength_mpa=10.0,
        rock_mass_quality=0.68,
        weathering=0.04,
    ),
}


def resolve_world_config(raw_config: dict[str, Any] | None) -> WorldConfig:
    """Resolve a world selection and optional physical overrides."""

    data = dict(raw_config or {})
    body_name = str(data.pop("body", "earth")).strip().lower()
    if body_name not in BODY_PRESETS:
        choices = ", ".join(sorted(BODY_PRESETS))
        raise ValueError(f"world.body must be one of: {choices}; got {body_name!r}")

    base_body = BODY_PRESETS[body_name]
    material_name = str(data.pop("material", base_body.default_material)).strip().lower()
    if material_name not in MATERIAL_PRESETS:
        choices = ", ".join(sorted(MATERIAL_PRESETS))
        raise ValueError(f"world.material must be one of: {choices}; got {material_name!r}")

    body_keys = {
        "gravity_m_s2",
        "maximum_passage_width_m",
        "maximum_room_width_m",
        "default_route_length_m",
        "production_voxel_size_m",
        "host_horizontal_scale",
        "host_vertical_scale",
        "host_fracture_scale",
        "roof_safety_factor",
    }
    material_keys = {
        "bulk_density_kg_m3",
        "intact_tensile_strength_mpa",
        "rock_mass_quality",
        "weathering",
    }
    unknown = set(data) - body_keys - material_keys
    if unknown:
        raise ValueError(f"Unknown world configuration keys: {', '.join(sorted(unknown))}")

    body_overrides = {key: data[key] for key in body_keys if key in data}
    material_overrides = {key: data[key] for key in material_keys if key in data}
    body = replace(base_body, **body_overrides)
    material = replace(MATERIAL_PRESETS[material_name], **material_overrides)
    _validate_world(body, material)
    return WorldConfig(body=body, material=material)


def build_run_config(raw_config: dict[str, Any] | None) -> RunConfig:
    data = dict(raw_config or {})
    _reject_unknown_keys("run", data, RunConfig)
    if "quality" in data:
        data["quality"] = str(data["quality"]).strip().lower()
    config = RunConfig(**data)
    if config.quality not in SUPPORTED_QUALITY_LEVELS:
        raise ValueError(
            f"run.quality must be one of: {', '.join(sorted(SUPPORTED_QUALITY_LEVELS))}"
        )
    if config.dev_max_route_length_m <= 0.0:
        raise ValueError("run.dev_max_route_length_m must be positive")
    if config.dev_max_lobe_paths < 1:
        raise ValueError("run.dev_max_lobe_paths must be positive")
    return config


def build_flow_regime_config(
    raw_config: dict[str, Any] | None,
) -> FlowRegimeConfig:
    """Build and validate independent lava-supply morphology controls."""

    data = dict(raw_config or {})
    _reject_unknown_keys("flow_regime", data, FlowRegimeConfig)
    config = FlowRegimeConfig(**data)
    positive = {
        "supply_rate_scale": config.supply_rate_scale,
        "duration_scale": config.duration_scale,
        "cooling_rate_scale": config.cooling_rate_scale,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"flow_regime.{name} must be positive")
    for name, value in {
        "inflation": config.inflation,
        "distributary_tendency": config.distributary_tendency,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"flow_regime.{name} must be in [0, 1]")
    return config


def build_export_config(raw_config: dict[str, Any] | None) -> ExportConfig:
    data = dict(raw_config or {})
    _reject_unknown_keys(
        "export",
        data,
        ExportConfig,
        aliases=frozenset({"format"}),
    )
    if "format" in data:
        if "file_format" in data:
            raise ValueError("Use only one of export.format or export.file_format")
        data["file_format"] = data.pop("format")
    for key in ("target", "file_format"):
        if key in data:
            data[key] = str(data[key]).strip().lower()
    config = ExportConfig(**data)
    for name in ("max_visual_triangles", "max_asset_bytes"):
        value = getattr(config, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"export.{name} must be a nonnegative integer (0 disables the limit)")
    if config.target not in SUPPORTED_EXPORT_TARGETS:
        raise ValueError(
            f"export.target must be one of: {', '.join(sorted(SUPPORTED_EXPORT_TARGETS))}"
        )
    if config.file_format not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(
            f"export.format must be one of: {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}"
        )
    supported_formats = EXPORT_FORMATS_BY_TARGET[config.target]
    if config.file_format not in supported_formats:
        raise ValueError(
            f"export target {config.target!r} supports format(s): "
            f"{', '.join(sorted(supported_formats))}; got {config.file_format!r}"
        )
    return config


def _reject_unknown_keys(
    path: str,
    data: dict[str, Any],
    config_type: type[Any],
    *,
    aliases: frozenset[str] = frozenset(),
) -> None:
    supported = {field.name for field in fields(config_type)} | aliases
    unknown = set(data) - supported
    if unknown:
        qualified = ", ".join(f"{path}.{key}" for key in sorted(unknown))
        raise ValueError(f"Unknown configuration keys: {qualified}")


def derive_stage_seeds(procedural_seed: int | None) -> StageSeeds:
    """Derive reproducible seeds by label rather than by call order."""

    labels = ("host", "network", "sections", "events", "geometry")
    if procedural_seed is None:
        return StageSeeds(**{label: None for label in labels})

    values: dict[str, int] = {}
    for label in labels:
        values[label] = derive_subseed(procedural_seed, "stage", label)
    return StageSeeds(**values)


def _validate_world(
    body: CelestialBodyProfile,
    material: GeologicalMaterialProfile,
) -> None:
    WorldConfig(body, material).roof_stability_model
    positive_body_values = {
        "gravity_m_s2": body.gravity_m_s2,
        "maximum_passage_width_m": body.maximum_passage_width_m,
        "maximum_room_width_m": body.maximum_room_width_m,
        "default_route_length_m": body.default_route_length_m,
        "production_voxel_size_m": body.production_voxel_size_m,
        "host_horizontal_scale": body.host_horizontal_scale,
        "host_vertical_scale": body.host_vertical_scale,
        "host_fracture_scale": body.host_fracture_scale,
    }
    for name, value in positive_body_values.items():
        if float(value) <= 0.0:
            raise ValueError(f"world.{name} must be positive")
    if body.maximum_room_width_m < body.maximum_passage_width_m:
        raise ValueError("world.maximum_room_width_m cannot be smaller than passage width")

    positive_material_values = {
        "bulk_density_kg_m3": material.bulk_density_kg_m3,
        "intact_tensile_strength_mpa": material.intact_tensile_strength_mpa,
    }
    for name, value in positive_material_values.items():
        if float(value) <= 0.0:
            raise ValueError(f"world.{name} must be positive")
    if not 0.0 < material.rock_mass_quality <= 1.0:
        raise ValueError("world.rock_mass_quality must be in (0, 1]")
    if not 0.0 <= material.weathering <= 1.0:
        raise ValueError("world.weathering must be in [0, 1]")


__all__ = [
    "BODY_PRESETS",
    "MATERIAL_PRESETS",
    "CelestialBodyProfile",
    "ExportConfig",
    "EXPORT_FORMATS_BY_TARGET",
    "FlowRegimeConfig",
    "GeologicalMaterialProfile",
    "RunConfig",
    "SUPPORTED_EVENT_KINDS",
    "SUPPORTED_EXPORT_FORMATS",
    "SUPPORTED_EXPORT_TARGETS",
    "StageSeeds",
    "WorldConfig",
    "build_export_config",
    "build_flow_regime_config",
    "build_run_config",
    "derive_stage_seeds",
    "resolve_world_config",
]
