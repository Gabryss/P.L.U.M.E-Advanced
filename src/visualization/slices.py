from __future__ import annotations

from pathlib import Path

from src.visualization import _matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np


def save_central_slices(fields: dict[str, np.ndarray], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(fields), 3, figsize=(12, 4 * len(fields)))
    if len(fields) == 1:
        axes = np.asarray([axes])

    for row, (name, field) in enumerate(fields.items()):
        z, y, x = (size // 2 for size in field.shape)
        slices = [field[z, :, :], field[:, y, :], field[:, :, x]]
        titles = [f"{name} XY", f"{name} XZ", f"{name} YZ"]
        for axis, image, title in zip(axes[row], slices, titles, strict=True):
            axis.imshow(image, cmap="viridis", origin="lower")
            axis.set_title(title)
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)

