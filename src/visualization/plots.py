from __future__ import annotations

from pathlib import Path

from src.visualization import _matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np


def save_time_series(metrics_history: list[dict[str, float | int]], path: str | Path) -> None:
    if not metrics_history:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    iterations = [entry["iteration"] for entry in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    series = [
        ("cave_volume_fraction", "Cave Volume Fraction"),
        ("mean_porosity", "Mean Porosity"),
        ("mean_effective_permeability", "Mean Effective Permeability"),
        ("max_water_flux", "Max Water Flux"),
    ]
    for axis, (key, label) in zip(axes.flatten(), series, strict=True):
        axis.plot(iterations, [entry[key] for entry in metrics_history], linewidth=2.0)
        axis.set_title(label)
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def save_histograms(fields: dict[str, np.ndarray], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (name, values) in zip(axes.flatten(), fields.items(), strict=False):
        axis.hist(values.ravel(), bins=30, color="#24577a", alpha=0.85)
        axis.set_title(name)
        axis.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)

