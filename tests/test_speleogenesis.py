from __future__ import annotations

import numpy as np

from src.config import load_config
from src.geology.generator import generate_geology
from src.hydro.solver import solve_hydraulic_head
from src.speleogenesis.evolution import initialize_speleogenesis, update_speleogenesis
from src.utils.random import make_rng


def test_speleogenesis_update_keeps_fields_bounded() -> None:
    config = load_config("config/default.yaml")
    geology = generate_geology(config.domain, config.geology, make_rng(config.seed))
    speleo = initialize_speleogenesis(geology, config.dissolution)
    hydro = solve_hydraulic_head(config.domain, config.hydro, speleo.effective_permeability)
    updated = update_speleogenesis(geology, hydro, speleo, config.dissolution)

    assert np.all(updated.porosity >= 0.0)
    assert np.all(updated.porosity <= 1.0)
    assert np.all(updated.solid_fraction >= 0.0)
    assert np.all(updated.solid_fraction <= 1.0)
    assert np.all(updated.effective_permeability >= 0.0)

