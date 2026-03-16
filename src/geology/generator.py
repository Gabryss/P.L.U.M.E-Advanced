from __future__ import annotations

import numpy as np

from src.config import DomainConfig, GeologyConfig
from src.state import GeologyState


def _layer_boundaries(domain: DomainConfig, config: GeologyConfig, rng: np.random.Generator) -> np.ndarray:
    nz, ny, nx = domain.grid_shape
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    cumulative = np.cumsum([layer.thickness for layer in config.layers[:-1]], dtype=float)
    cumulative /= sum(layer.thickness for layer in config.layers)

    boundaries = []
    for idx, base_fraction in enumerate(cumulative, start=1):
        phase_x = rng.uniform(0.0, 2.0 * np.pi)
        phase_y = rng.uniform(0.0, 2.0 * np.pi)
        wave = (
            np.sin(xx * (0.7 + 0.15 * idx) + phase_x)
            + 0.6 * np.cos(yy * (0.9 + 0.12 * idx) + phase_y)
            + 0.35 * np.sin((xx + yy) * (0.45 + 0.1 * idx))
        )
        normalized_wave = wave / np.max(np.abs(wave))
        boundary = (base_fraction * nz) + config.interface_undulation * normalized_wave
        boundaries.append(boundary)

    return np.stack(boundaries, axis=0) if boundaries else np.empty((0, ny, nx), dtype=float)


def _fracture_field(domain: DomainConfig, config: GeologyConfig, rng: np.random.Generator) -> np.ndarray:
    nz, ny, nx = domain.grid_shape
    z = np.linspace(0.0, 1.0, nz, endpoint=True)[:, None, None]
    y = np.linspace(0.0, 1.0, ny, endpoint=False)[None, :, None]
    x = np.linspace(0.0, 1.0, nx, endpoint=False)[None, None, :]

    field = np.zeros((nz, ny, nx), dtype=float)
    for idx, wavelength in enumerate(config.fracture_wavelengths, start=1):
        angle = rng.uniform(0.0, np.pi)
        direction = np.cos(angle) * x + np.sin(angle) * y
        phase = rng.uniform(0.0, 2.0 * np.pi)
        band = np.sin((direction / max(wavelength, 1e-3)) * 2.0 * np.pi + phase)
        field += (1.0 / idx) * band

    field += 0.25 * np.sin((z * 3.0 + y * 2.0 + x) * 2.0 * np.pi)
    field += 0.1 * rng.standard_normal((nz, ny, nx))
    field -= field.min()
    field /= max(field.max(), 1e-6)
    return np.clip(field * config.fracture_strength, 0.0, 1.0)


def generate_geology(domain: DomainConfig, config: GeologyConfig, rng: np.random.Generator) -> GeologyState:
    nz, ny, nx = domain.grid_shape
    boundaries = _layer_boundaries(domain, config, rng)
    depth = np.arange(nz, dtype=float)[:, None, None]

    material_id = np.zeros((nz, ny, nx), dtype=int)
    for idx, boundary in enumerate(boundaries):
        material_id += depth > boundary

    hardness = np.empty((nz, ny, nx), dtype=float)
    solubility = np.empty((nz, ny, nx), dtype=float)
    permeability = np.empty((nz, ny, nx), dtype=float)
    base_porosity = np.empty((nz, ny, nx), dtype=float)

    for idx, layer in enumerate(config.layers):
        mask = material_id == idx
        hardness[mask] = layer.hardness
        solubility[mask] = layer.solubility
        permeability[mask] = layer.permeability
        base_porosity[mask] = layer.base_porosity

    fracture_density = _fracture_field(domain, config, rng)
    layer_depth = depth / max(nz - 1, 1)

    return GeologyState(
        material_id=material_id,
        rock_hardness=hardness,
        solubility=solubility,
        permeability=permeability,
        fracture_density=fracture_density,
        layer_depth=np.broadcast_to(layer_depth, (nz, ny, nx)),
        base_porosity=base_porosity,
    )
