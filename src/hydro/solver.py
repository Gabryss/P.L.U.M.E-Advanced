from __future__ import annotations

import numpy as np

from src.config import DomainConfig, HydroConfig
from src.state import HydroState


def _face_mask(shape: tuple[int, int, int], face: str) -> np.ndarray:
    nz, ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    if face == "top":
        mask[0, :, :] = True
    elif face == "bottom":
        mask[-1, :, :] = True
    elif face == "front":
        mask[:, 0, :] = True
    elif face == "back":
        mask[:, -1, :] = True
    elif face == "left":
        mask[:, :, 0] = True
    elif face == "right":
        mask[:, :, -1] = True
    else:
        raise ValueError(f"Unsupported boundary face: {face}")
    return mask


def _initial_head(shape: tuple[int, int, int], source_face: str, sink_face: str, recharge_head: float, drainage_head: float) -> np.ndarray:
    nz, ny, nx = shape
    axes = {
        "top": np.linspace(recharge_head, drainage_head, nz)[:, None, None],
        "bottom": np.linspace(drainage_head, recharge_head, nz)[:, None, None],
        "front": np.linspace(recharge_head, drainage_head, ny)[None, :, None],
        "back": np.linspace(drainage_head, recharge_head, ny)[None, :, None],
        "left": np.linspace(recharge_head, drainage_head, nx)[None, None, :],
        "right": np.linspace(drainage_head, recharge_head, nx)[None, None, :],
    }
    source_bias = np.broadcast_to(axes[source_face], shape)
    sink_bias = np.broadcast_to(axes[sink_face], shape)
    return 0.5 * (source_bias + sink_bias)


def solve_hydraulic_head(
    domain: DomainConfig,
    config: HydroConfig,
    effective_permeability: np.ndarray,
) -> HydroState:
    shape = domain.grid_shape
    source_mask = _face_mask(shape, config.source_face)
    sink_mask = _face_mask(shape, config.sink_face)

    head = _initial_head(shape, config.source_face, config.sink_face, config.recharge_head, config.drainage_head)
    head[source_mask] = config.recharge_head
    head[sink_mask] = config.drainage_head

    permeability = np.clip(effective_permeability, 1e-6, None)
    for _ in range(config.solver_iterations):
        accum = np.zeros_like(head)
        weights = np.zeros_like(head)

        w = 0.5 * (permeability[1:, :, :] + permeability[:-1, :, :])
        accum[1:, :, :] += w * head[:-1, :, :]
        accum[:-1, :, :] += w * head[1:, :, :]
        weights[1:, :, :] += w
        weights[:-1, :, :] += w

        w = 0.5 * (permeability[:, 1:, :] + permeability[:, :-1, :])
        accum[:, 1:, :] += w * head[:, :-1, :]
        accum[:, :-1, :] += w * head[:, 1:, :]
        weights[:, 1:, :] += w
        weights[:, :-1, :] += w

        w = 0.5 * (permeability[:, :, 1:] + permeability[:, :, :-1])
        accum[:, :, 1:] += w * head[:, :, :-1]
        accum[:, :, :-1] += w * head[:, :, 1:]
        weights[:, :, 1:] += w
        weights[:, :, :-1] += w

        updated = np.divide(accum, weights, out=head.copy(), where=weights > 0.0)
        updated[source_mask] = config.recharge_head
        updated[sink_mask] = config.drainage_head
        head = updated

    gradients = np.gradient(head, domain.voxel_size)
    flow_components = np.stack([-permeability * gradient for gradient in gradients], axis=0)
    water_flux = np.linalg.norm(flow_components, axis=0)
    return HydroState(hydraulic_head=head, water_flux=water_flux, flow_vector=flow_components)

