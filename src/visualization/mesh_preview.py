from __future__ import annotations

from pathlib import Path

from src.visualization import _matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def save_mesh_preview(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")

    if len(vertices) and len(faces):
        tris = vertices[faces]
        mesh = Poly3DCollection(tris, alpha=0.75, facecolor="#2d7f5e", edgecolor="#173f2f", linewidth=0.1)
        axis.add_collection3d(mesh)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        axis.set_xlim(mins[0], maxs[0])
        axis.set_ylim(mins[1], maxs[1])
        axis.set_zlim(mins[2], maxs[2])

    axis.set_title("Cave Mesh Preview")
    axis.set_xlabel("Z")
    axis.set_ylabel("Y")
    axis.set_zlabel("X")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)

