"""Resolved celestial, geological, run-mode, and export configuration.

The generator works in metres in a canonical right-handed, Z-up coordinate
system.  Celestial-body profiles affect generation; exporters alone are
responsible for converting units and axes for target applications.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import zlib
from typing import Any

import numpy as np


SUPPORTED_EVENT_KINDS = frozenset({"rock", "boulder", "collapse", "choke", "infill"})
SUPPORTED_EXPORT_TARGETS = frozenset(
    {"neutral", "blender", "ue5", "unity", "gazebo", "omniverse"}
)
SUPPORTED_EXPORT_FORMATS = frozenset({"glb", "obj", "fbx", "usd", "usdc", "dae"})
SUPPORTED_QUALITY_LEVELS = frozenset({"preview", "standard", "production"})


@dataclass(frozen=True)
class CelestialBodyProfile:
    """Procedural defaults for one celestial environment.

    Passage and route limits are conservative project defaults rather than
    universal geological maxima.  Users can override every value in TOML.
    """

    name: str
    gravity_m_s2: float
    atmosphere: str
    erosion_regime: str
    surface_deposit: str
    default_material: str
    maximum_passage_width_m: float
    maximum_room_width_m: float
    default_route_length_m: float
    production_voxel_size_m: float


@dataclass(frozen=True)
class GeologicalMaterialProfile:
    """Rock-mass parameters used by procedural stability surrogates."""

    name: str
    bulk_density_kg_m3: float
    intact_tensile_strength_mpa: float
    cohesion_mpa: float
    friction_angle_degrees: float
    rock_mass_quality: float
    mean_joint_spacing_m: float
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

    def roof_demand_ratio(self, span_m: float, roof_thickness_m: float) -> float:
        """Estimate gravity-driven roof demand relative to rock-mass capacity.

        This is a fast procedural surrogate, not a finite-element analysis.
        """

        span = max(float(span_m), 0.0)
        thickness = max(float(roof_thickness_m), 0.1)
        demand = (
            self.material.bulk_density_kg_m3
            * self.body.gravity_m_s2
            * span
            * span
            / thickness
        )
        return float(demand / max(self.material.effective_tensile_strength_pa, 1.0))


@dataclass(frozen=True)
class RunConfig:
    """Controls generation extent without changing the selected world physics."""

    dev_mode: bool = False
    quality: str = "standard"
    dev_max_route_length_m: float = 1500.0
    dev_max_braid_zones: int = 2
    render_diagnostics: bool = True


@dataclass(frozen=True)
class ExportConfig:
    """Target selection and target-independent export requirements."""

    target: str = "blender"
    file_format: str = "glb"
    quality: str = "standard"
    generate_visual: bool = True
    generate_collision: bool = True
    generate_lods: bool = True
    generate_wall_shell: bool = False
    wall_thickness_m: float = 0.5


@dataclass(frozen=True)
class StageSeeds:
    """Stable named seeds so optional stages cannot perturb each other."""

    host: int | None
    network: int | None
    sections: int | None
    events: int | None
    geometry: int | None
    surface: int | None
    export: int | None


BODY_PRESETS: dict[str, CelestialBodyProfile] = {
    "earth": CelestialBodyProfile(
        name="earth",
        gravity_m_s2=9.80665,
        atmosphere="dense",
        erosion_regime="weathering_and_water",
        surface_deposit="weathered_basalt",
        default_material="terrestrial_basalt",
        maximum_passage_width_m=10.0,
        maximum_room_width_m=20.0,
        default_route_length_m=5_000.0,
        production_voxel_size_m=0.5,
    ),
    "mars": CelestialBodyProfile(
        name="mars",
        gravity_m_s2=3.71,
        atmosphere="thin",
        erosion_regime="aeolian_and_thermal",
        surface_deposit="martian_dust",
        default_material="martian_basalt",
        maximum_passage_width_m=50.0,
        maximum_room_width_m=100.0,
        default_route_length_m=15_000.0,
        production_voxel_size_m=1.0,
    ),
    "moon": CelestialBodyProfile(
        name="moon",
        gravity_m_s2=1.62,
        atmosphere="vacuum",
        erosion_regime="impact_and_thermal",
        surface_deposit="lunar_regolith",
        default_material="mare_basalt",
        maximum_passage_width_m=100.0,
        maximum_room_width_m=200.0,
        default_route_length_m=30_000.0,
        production_voxel_size_m=2.0,
    ),
}


MATERIAL_PRESETS: dict[str, GeologicalMaterialProfile] = {
    "terrestrial_basalt": GeologicalMaterialProfile(
        name="terrestrial_basalt",
        bulk_density_kg_m3=2_900.0,
        intact_tensile_strength_mpa=8.0,
        cohesion_mpa=12.0,
        friction_angle_degrees=38.0,
        rock_mass_quality=0.55,
        mean_joint_spacing_m=1.8,
        weathering=0.30,
    ),
    "martian_basalt": GeologicalMaterialProfile(
        name="martian_basalt",
        bulk_density_kg_m3=2_950.0,
        intact_tensile_strength_mpa=9.0,
        cohesion_mpa=14.0,
        friction_angle_degrees=40.0,
        rock_mass_quality=0.62,
        mean_joint_spacing_m=2.5,
        weathering=0.14,
    ),
    "mare_basalt": GeologicalMaterialProfile(
        name="mare_basalt",
        bulk_density_kg_m3=3_050.0,
        intact_tensile_strength_mpa=10.0,
        cohesion_mpa=16.0,
        friction_angle_degrees=42.0,
        rock_mass_quality=0.68,
        mean_joint_spacing_m=3.2,
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
        "atmosphere",
        "erosion_regime",
        "surface_deposit",
        "maximum_passage_width_m",
        "maximum_room_width_m",
        "default_route_length_m",
        "production_voxel_size_m",
    }
    material_keys = {
        "bulk_density_kg_m3",
        "intact_tensile_strength_mpa",
        "cohesion_mpa",
        "friction_angle_degrees",
        "rock_mass_quality",
        "mean_joint_spacing_m",
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
    if "quality" in data:
        data["quality"] = str(data["quality"]).strip().lower()
    config = RunConfig(**data)
    if config.quality not in SUPPORTED_QUALITY_LEVELS:
        raise ValueError(
            f"run.quality must be one of: {', '.join(sorted(SUPPORTED_QUALITY_LEVELS))}"
        )
    if config.dev_max_route_length_m <= 0.0:
        raise ValueError("run.dev_max_route_length_m must be positive")
    if config.dev_max_braid_zones < 0:
        raise ValueError("run.dev_max_braid_zones must be non-negative")
    return config


def build_export_config(raw_config: dict[str, Any] | None) -> ExportConfig:
    data = dict(raw_config or {})
    if "format" in data:
        if "file_format" in data:
            raise ValueError("Use only one of export.format or export.file_format")
        data["file_format"] = data.pop("format")
    for key in ("target", "file_format", "quality"):
        if key in data:
            data[key] = str(data[key]).strip().lower()
    config = ExportConfig(**data)
    if config.target not in SUPPORTED_EXPORT_TARGETS:
        raise ValueError(
            f"export.target must be one of: {', '.join(sorted(SUPPORTED_EXPORT_TARGETS))}"
        )
    if config.file_format not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(
            f"export.format must be one of: {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}"
        )
    if config.quality not in SUPPORTED_QUALITY_LEVELS:
        raise ValueError(
            f"export.quality must be one of: {', '.join(sorted(SUPPORTED_QUALITY_LEVELS))}"
        )
    if config.wall_thickness_m <= 0.0:
        raise ValueError("export.wall_thickness_m must be positive")
    return config


def derive_stage_seeds(procedural_seed: int | None) -> StageSeeds:
    """Derive reproducible seeds by label rather than by call order."""

    labels = ("host", "network", "sections", "events", "geometry", "surface", "export")
    if procedural_seed is None:
        return StageSeeds(**{label: None for label in labels})

    values: dict[str, int] = {}
    for label in labels:
        label_code = zlib.crc32(label.encode("utf-8"))
        sequence = np.random.SeedSequence([int(procedural_seed), label_code])
        values[label] = int(sequence.generate_state(1, dtype=np.uint32)[0])
    return StageSeeds(**values)


def _validate_world(
    body: CelestialBodyProfile,
    material: GeologicalMaterialProfile,
) -> None:
    positive_body_values = {
        "gravity_m_s2": body.gravity_m_s2,
        "maximum_passage_width_m": body.maximum_passage_width_m,
        "maximum_room_width_m": body.maximum_room_width_m,
        "default_route_length_m": body.default_route_length_m,
        "production_voxel_size_m": body.production_voxel_size_m,
    }
    for name, value in positive_body_values.items():
        if float(value) <= 0.0:
            raise ValueError(f"world.{name} must be positive")
    if body.maximum_room_width_m < body.maximum_passage_width_m:
        raise ValueError("world.maximum_room_width_m cannot be smaller than passage width")

    positive_material_values = {
        "bulk_density_kg_m3": material.bulk_density_kg_m3,
        "intact_tensile_strength_mpa": material.intact_tensile_strength_mpa,
        "cohesion_mpa": material.cohesion_mpa,
        "mean_joint_spacing_m": material.mean_joint_spacing_m,
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
    "GeologicalMaterialProfile",
    "RunConfig",
    "SUPPORTED_EVENT_KINDS",
    "SUPPORTED_EXPORT_FORMATS",
    "SUPPORTED_EXPORT_TARGETS",
    "StageSeeds",
    "WorldConfig",
    "build_export_config",
    "build_run_config",
    "derive_stage_seeds",
    "resolve_world_config",
]
