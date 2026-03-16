from __future__ import annotations

from collections import deque

import numpy as np

from src.state import SimulationState


def connected_components(mask: np.ndarray) -> tuple[int, int]:
    visited = np.zeros_like(mask, dtype=bool)
    component_count = 0
    largest_size = 0
    shape = mask.shape

    for start in np.argwhere(mask):
        z, y, x = (int(value) for value in start)
        if visited[z, y, x]:
            continue

        component_count += 1
        queue = deque([(z, y, x)])
        visited[z, y, x] = True
        size = 0

        while queue:
            cz, cy, cx = queue.popleft()
            size += 1
            for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if not (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                    continue
                if visited[nz, ny, nx] or not mask[nz, ny, nx]:
                    continue
                visited[nz, ny, nx] = True
                queue.append((nz, ny, nx))

        largest_size = max(largest_size, size)

    return component_count, largest_size


def compute_metrics(state: SimulationState, iteration: int, elapsed_seconds: float) -> dict[str, float | int]:
    void_mask = state.speleogenesis.void_mask
    components, largest_component = connected_components(void_mask)
    return {
        "iteration": iteration,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "cave_volume_fraction": float(void_mask.mean()),
        "void_voxel_count": int(void_mask.sum()),
        "porous_voxel_count": int((state.speleogenesis.porosity > 0.2).sum()),
        "mean_porosity": float(state.speleogenesis.porosity.mean()),
        "mean_solid_fraction": float(state.speleogenesis.solid_fraction.mean()),
        "mean_effective_permeability": float(state.speleogenesis.effective_permeability.mean()),
        "max_effective_permeability": float(state.speleogenesis.effective_permeability.max()),
        "mean_water_flux": float(state.hydro.water_flux.mean()),
        "max_water_flux": float(state.hydro.water_flux.max()),
        "connected_components": components,
        "largest_component_size": largest_component,
    }


def numerical_warnings(state: SimulationState) -> list[str]:
    warnings: list[str] = []
    speleo = state.speleogenesis

    if np.any(speleo.solid_fraction < 0.0) or np.any(speleo.solid_fraction > 1.0):
        warnings.append("solid_fraction out of bounds")
    if np.any(speleo.porosity < 0.0) or np.any(speleo.porosity > 1.0):
        warnings.append("porosity out of bounds")
    if np.any(speleo.effective_permeability < 0.0):
        warnings.append("negative permeability encountered")
    if float(state.hydro.water_flux.max()) > 10.0:
        warnings.append("water_flux unusually large")
    if float(state.hydro.water_flux.max()) == 0.0:
        warnings.append("water_flux collapsed to zero")

    return warnings

