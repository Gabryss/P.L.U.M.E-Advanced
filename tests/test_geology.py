from __future__ import annotations

import numpy as np

from src.config import load_config
from src.geology.generator import generate_geology
from src.utils.random import make_rng


def test_geology_generation_is_reproducible() -> None:
    config = load_config("config/default.yaml")
    geology_a = generate_geology(config.domain, config.geology, make_rng(config.seed))
    geology_b = generate_geology(config.domain, config.geology, make_rng(config.seed))

    assert np.array_equal(geology_a.material_id, geology_b.material_id)
    assert np.allclose(geology_a.fracture_density, geology_b.fracture_density)
    assert geology_a.material_id.shape == config.domain.grid_shape

