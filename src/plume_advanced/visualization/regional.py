"""Stage-B network figures with equal metric scales and shared host context."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import map_coordinates


def render_network_comparison(host, comparisons, output, *, window_m=None):
    """Show width envelopes as Stage-B estimates, never claim a meshed surface."""
    angle = np.radians(host.config.flow_angle_degrees)
    basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    origin = np.array(host.config.seed_point)
    polygons, colours, bounds = [], [], []
    individual_sources = set()
    has_shared = False
    palette = [
        "#178091",
        "#e3822c",
        "#498a57",
        "#7353a5",
        "#bd4d6b",
        "#3853a0",
        "#847137",
        "#457272",
    ]
    for _, network in comparisons:
        patches, colors = [], []
        for s in network.segments:
            xy = np.array([(p.x, p.y) for p in s.points])
            tangent = np.gradient(xy, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            half = np.array([p.width for p in s.points])[:, None] / 2
            poly = (np.r_[xy + normal * half, (xy - normal * half)[::-1]] - origin) @ basis.T
            patches.append(poly)
            ids = s.metadata.get("contributing_system_ids", [])
            if len(ids) == 1:
                individual_sources.add(ids[0])
            else:
                has_shared = True
            colors.append(palette[ids[0] % len(palette)] if len(ids) == 1 else "#4b4b4b")
            bounds.append(poly)
        polygons.append(patches)
        colours.append(colors)
    points = np.concatenate(bounds)
    low, high = points.min(axis=0) - 15, points.max(axis=0) + 15
    if window_m is not None and (not np.isfinite(window_m) or window_m <= 0):
        raise ValueError("window_m must be finite and positive")
    windows = (
        np.linspace(low[0], high[0], int(np.ceil((high[0] - low[0]) / window_m)) + 1)
        if window_m is not None
        else np.array([low[0], high[0]])
    )
    views = (
        [
            (index, start, end)
            for index in range(len(comparisons))
            for start, end in zip(windows, windows[1:])
        ]
        if window_m is not None
        else [(i, low[0], high[0]) for i in range(len(comparisons))]
    )
    fig, axes = plt.subplots(
        len(views),
        1,
        figsize=(14, max(3.5, len(views) * 3.5)),
        squeeze=False,
        constrained_layout=True,
    )
    aa, cc = np.meshgrid(np.linspace(low[0], high[0], 900), np.linspace(low[1], high[1], 220))
    xy = np.stack((aa, cc), axis=-1) @ basis + origin
    ix = (xy[..., 0] - host.x_coords[0]) / (host.x_coords[1] - host.x_coords[0])
    iy = (xy[..., 1] - host.y_coords[0]) / (host.y_coords[1] - host.y_coords[0])
    cost = map_coordinates(host.growth_cost, [iy, ix], order=1, mode="nearest")
    for ax, (index, start, end) in zip(axes[:, 0], views):
        label, network = comparisons[index]
        ax.imshow(
            cost,
            extent=[low[0], high[0], low[1], high[1]],
            origin="lower",
            cmap="Greys",
            alpha=0.18,
            vmin=0,
            vmax=1,
        )
        ax.add_collection(
            PolyCollection(polygons[index], facecolors=colours[index], edgecolors="none")
        )
        for kind, marker in (("entry", "o"), ("exit", "^"), ("terminal", "D")):
            positions = np.array([[n.x, n.y] for n in network.nodes if n.kind == kind])
            if len(positions):
                positions = (positions - origin) @ basis.T
                ax.scatter(*positions.T, marker=marker, s=34, facecolor="#222222",
                           edgecolor="white", linewidth=.65, zorder=5)
        ax.set(
            xlim=(start, end),
            ylim=(low[1], high[1]),
            xlabel="Distance along original flow axis (m)",
            ylabel="Lateral distance (m)",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_title(
            f"{label} · {network.config.systems.count} sources · network seed {network.config.random_seed}"
        )
    legend: list[Patch | Line2D] = [
        Patch(color=palette[i % len(palette)], label=f"Source {i + 1} only")
        for i in sorted(individual_sources)
    ]
    if has_shared:
        legend.append(Patch(color="#4b4b4b", label="Multiple sources"))
    for kind, marker, label in (("entry", "o", "Inlet"), ("exit", "^", "Downstream terminus"),
                                ("terminal", "D", "Blind end")):
        if any(n.kind == kind for _, network in comparisons for n in network.nodes):
            legend.append(Line2D([], [], color="#222222", marker=marker, ls="", label=label))
    fig.legend(handles=legend, loc="outside lower center", ncol=min(5, len(legend)),
               frameon=False, fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output
