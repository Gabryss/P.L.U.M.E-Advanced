from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import DomainConfig


Array = np.ndarray


@dataclass(slots=True)
class GeologyState:
    material_id: Array
    rock_hardness: Array
    solubility: Array
    permeability: Array
    fracture_density: Array
    layer_depth: Array
    base_porosity: Array


@dataclass(slots=True)
class HydroState:
    hydraulic_head: Array
    water_flux: Array
    flow_vector: Array


@dataclass(slots=True)
class SpeleogenesisState:
    dissolution_damage: Array
    porosity: Array
    solid_fraction: Array
    void_mask: Array
    effective_permeability: Array


@dataclass(slots=True)
class SimulationState:
    domain: DomainConfig
    geology: GeologyState
    hydro: HydroState
    speleogenesis: SpeleogenesisState

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.domain.grid_shape

    def phi(self, void_threshold: float) -> Array:
        return self.speleogenesis.solid_fraction - void_threshold

