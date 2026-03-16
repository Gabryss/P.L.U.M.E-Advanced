from __future__ import annotations

import numpy as np

from src.config import DissolutionConfig
from src.state import GeologyState, HydroState, SpeleogenesisState


def initialize_speleogenesis(geology: GeologyState, config: DissolutionConfig) -> SpeleogenesisState:
    porosity = np.clip(geology.base_porosity.copy(), 0.0, 1.0)
    solid_fraction = np.clip(1.0 - 0.5 * porosity, 0.0, 1.0)
    effective_permeability = compute_effective_permeability(geology, porosity, config)
    void_mask = solid_fraction < config.void_threshold
    return SpeleogenesisState(
        dissolution_damage=np.zeros_like(geology.permeability),
        porosity=porosity,
        solid_fraction=solid_fraction,
        void_mask=void_mask,
        effective_permeability=effective_permeability,
    )


def compute_effective_permeability(
    geology: GeologyState,
    porosity: np.ndarray,
    config: DissolutionConfig,
) -> np.ndarray:
    baseline = geology.permeability
    fracture_boost = 1.0 + config.permeability_fracture_gain * geology.fracture_density
    porosity_boost = 1.0 + config.permeability_porosity_gain * porosity
    return np.clip(baseline * fracture_boost * porosity_boost, 1e-6, None)


def _neighbor_void_fraction(void_mask: np.ndarray) -> np.ndarray:
    exposure = np.zeros_like(void_mask, dtype=float)

    exposure[1:, :, :] += void_mask[:-1, :, :]
    exposure[:-1, :, :] += void_mask[1:, :, :]
    exposure[:, 1:, :] += void_mask[:, :-1, :]
    exposure[:, :-1, :] += void_mask[:, 1:, :]
    exposure[:, :, 1:] += void_mask[:, :, :-1]
    exposure[:, :, :-1] += void_mask[:, :, 1:]

    return exposure / 6.0


def update_speleogenesis(
    geology: GeologyState,
    hydro: HydroState,
    speleo: SpeleogenesisState,
    config: DissolutionConfig,
) -> SpeleogenesisState:
    fracture_factor = 1.0 + config.fracture_amplification * geology.fracture_density
    exposure_factor = 1.0 + config.exposure_gain * _neighbor_void_fraction(speleo.void_mask)
    hardness_factor = 1.0 / np.clip(geology.rock_hardness, 1e-3, None)

    dissolution_rate = (
        hydro.water_flux
        * geology.solubility
        * fracture_factor
        * exposure_factor
        * hardness_factor
        * config.undersaturation
    )
    dissolution_damage = speleo.dissolution_damage + config.dt * dissolution_rate
    porosity = np.clip(speleo.porosity + config.porosity_gain * config.dt * dissolution_rate, 0.0, 1.0)
    solid_fraction = np.clip(speleo.solid_fraction - config.solid_loss * config.dt * dissolution_rate, 0.0, 1.0)
    effective_permeability = compute_effective_permeability(geology, porosity, config)
    void_mask = solid_fraction < config.void_threshold

    return SpeleogenesisState(
        dissolution_damage=dissolution_damage,
        porosity=porosity,
        solid_fraction=solid_fraction,
        void_mask=void_mask,
        effective_permeability=effective_permeability,
    )

