from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DomainConfig:
    grid_shape: tuple[int, int, int]
    voxel_size: float


@dataclass(slots=True)
class MaterialLayerConfig:
    name: str
    thickness: float
    hardness: float
    solubility: float
    permeability: float
    base_porosity: float


@dataclass(slots=True)
class GeologyConfig:
    interface_undulation: float
    fracture_strength: float
    fracture_wavelengths: tuple[float, float, float]
    layers: list[MaterialLayerConfig]


@dataclass(slots=True)
class HydroConfig:
    source_face: str
    sink_face: str
    recharge_head: float
    drainage_head: float
    solver_iterations: int


@dataclass(slots=True)
class DissolutionConfig:
    iterations: int
    dt: float
    porosity_gain: float
    solid_loss: float
    permeability_fracture_gain: float
    permeability_porosity_gain: float
    fracture_amplification: float
    exposure_gain: float
    undersaturation: float
    void_threshold: float


@dataclass(slots=True)
class MonitoringConfig:
    progress_bar_enabled: bool
    log_interval: int
    plot_interval: int
    slice_interval: int
    mesh_preview_interval: int
    histogram_interval: int


@dataclass(slots=True)
class DatasetConfig:
    output_dir: str
    sample_count: int


@dataclass(slots=True)
class RunConfig:
    name: str
    seed: int
    domain: DomainConfig
    geology: GeologyConfig
    hydro: HydroConfig
    dissolution: DissolutionConfig
    monitoring: MonitoringConfig
    dataset: DatasetConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_keys(data: dict[str, Any], section: str, keys: list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing keys in '{section}': {', '.join(missing)}")


def _parse_layers(raw_layers: list[dict[str, Any]]) -> list[MaterialLayerConfig]:
    if not raw_layers:
        raise ValueError("At least one geology layer must be configured.")

    layers = [
        MaterialLayerConfig(
            name=str(layer["name"]),
            thickness=float(layer["thickness"]),
            hardness=float(layer["hardness"]),
            solubility=float(layer["solubility"]),
            permeability=float(layer["permeability"]),
            base_porosity=float(layer["base_porosity"]),
        )
        for layer in raw_layers
    ]
    total_thickness = sum(layer.thickness for layer in layers)
    if total_thickness <= 0:
        raise ValueError("Layer thickness values must sum to a positive number.")
    return layers


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    _require_keys(
        raw,
        "root",
        ["name", "seed", "domain", "geology", "hydro", "dissolution", "monitoring", "dataset"],
    )

    domain = raw["domain"]
    geology = raw["geology"]
    hydro = raw["hydro"]
    dissolution = raw["dissolution"]
    monitoring = raw["monitoring"]
    dataset = raw["dataset"]

    _require_keys(domain, "domain", ["grid_shape", "voxel_size"])
    _require_keys(geology, "geology", ["interface_undulation", "fracture_strength", "fracture_wavelengths", "layers"])
    _require_keys(hydro, "hydro", ["source_face", "sink_face", "recharge_head", "drainage_head", "solver_iterations"])
    _require_keys(
        dissolution,
        "dissolution",
        [
            "iterations",
            "dt",
            "porosity_gain",
            "solid_loss",
            "permeability_fracture_gain",
            "permeability_porosity_gain",
            "fracture_amplification",
            "exposure_gain",
            "undersaturation",
            "void_threshold",
        ],
    )
    _require_keys(
        monitoring,
        "monitoring",
        ["progress_bar_enabled", "log_interval", "plot_interval", "slice_interval", "mesh_preview_interval", "histogram_interval"],
    )
    _require_keys(dataset, "dataset", ["output_dir", "sample_count"])

    return RunConfig(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        domain=DomainConfig(
            grid_shape=tuple(int(value) for value in domain["grid_shape"]),
            voxel_size=float(domain["voxel_size"]),
        ),
        geology=GeologyConfig(
            interface_undulation=float(geology["interface_undulation"]),
            fracture_strength=float(geology["fracture_strength"]),
            fracture_wavelengths=tuple(float(value) for value in geology["fracture_wavelengths"]),
            layers=_parse_layers(geology["layers"]),
        ),
        hydro=HydroConfig(
            source_face=str(hydro["source_face"]),
            sink_face=str(hydro["sink_face"]),
            recharge_head=float(hydro["recharge_head"]),
            drainage_head=float(hydro["drainage_head"]),
            solver_iterations=int(hydro["solver_iterations"]),
        ),
        dissolution=DissolutionConfig(
            iterations=int(dissolution["iterations"]),
            dt=float(dissolution["dt"]),
            porosity_gain=float(dissolution["porosity_gain"]),
            solid_loss=float(dissolution["solid_loss"]),
            permeability_fracture_gain=float(dissolution["permeability_fracture_gain"]),
            permeability_porosity_gain=float(dissolution["permeability_porosity_gain"]),
            fracture_amplification=float(dissolution["fracture_amplification"]),
            exposure_gain=float(dissolution["exposure_gain"]),
            undersaturation=float(dissolution["undersaturation"]),
            void_threshold=float(dissolution["void_threshold"]),
        ),
        monitoring=MonitoringConfig(
            progress_bar_enabled=bool(monitoring["progress_bar_enabled"]),
            log_interval=int(monitoring["log_interval"]),
            plot_interval=int(monitoring["plot_interval"]),
            slice_interval=int(monitoring["slice_interval"]),
            mesh_preview_interval=int(monitoring["mesh_preview_interval"]),
            histogram_interval=int(monitoring["histogram_interval"]),
        ),
        dataset=DatasetConfig(
            output_dir=str(dataset["output_dir"]),
            sample_count=int(dataset["sample_count"]),
        ),
    )


def write_config_snapshot(config: RunConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

