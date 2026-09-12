"""Metric host and section-envelope previews for interconnected systems."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates

from plume_advanced.stages.network_interconnected import spatial_metrics


def render_interconnected(host, network, sections, output_path):
    """Show actual Stage-C envelopes; never stretch the lateral axis."""
    metrics = spatial_metrics(network, sections)
    axis = np.asarray(network.backend_provenance["flow_direction"])
    cross = np.array([-axis[1], axis[0]])
    origin = np.asarray(host.config.seed_point)
    rows = max(1, int(math.ceil((metrics["downstream_extent_m"] - 1e-8) / 500)))
    fig, axes = plt.subplots(
        rows + 1,
        1,
        figsize=(15, rows * 3.5 + 2.6),
        gridspec_kw={"height_ratios": [3.5] * rows + [1.4]},
        constrained_layout=True,
    )
    lookup = {s.segment_id: s for s in network.segments}
    polygons = []
    palette = (
        "#147d92",
        "#d88836",
        "#51865b",
        "#af587b",
        "#8c754d",
        "#3865a5",
        "#aa663f",
        "#6b798a",
    )
    for field in sections.segment_fields:
        left, right = [], []
        for p in field.samples:
            q = np.asarray(p.profile_points)
            xy = (
                np.array([p.x, p.y])
                - origin
                + q[:, :1] * np.array(p.normal[:2])
                + q[:, 1:] * np.array(p.binormal[:2])
            )
            ac = np.column_stack((xy @ axis, xy @ cross))
            left.append(ac[int(np.argmin(ac[:, 1]))])
            right.append(ac[int(np.argmax(ac[:, 1]))])
        ids = lookup[field.segment_id].metadata.get("system_ids", [])
        colour = palette[ids[0]] if len(ids) == 1 else "#67558c" if ids else "#8c8680"
        polygons.append((np.array(left + right[::-1]), colour))
    vertices = np.concatenate([p for p, _ in polygons])
    ymin, ymax = float(vertices[:, 1].min() - 12), float(vertices[:, 1].max() + 12)
    extent = metrics["downstream_extent_m"]
    for i, ax in enumerate(axes[:-1]):
        start, end = i * 500.0, min((i + 1) * 500.0, extent)
        a, c = np.meshgrid(np.linspace(start, end, 650), np.linspace(ymin, ymax, 180))
        xy = origin[:, None, None] + axis[:, None, None] * a + cross[:, None, None] * c
        ix = (xy[0] - host.x_coords[0]) / (host.x_coords[1] - host.x_coords[0])
        iy = (xy[1] - host.y_coords[0]) / (host.y_coords[1] - host.y_coords[0])
        costs = map_coordinates(host.growth_cost, [iy, ix], order=1, mode="nearest")
        ax.imshow(
            costs,
            extent=[start, end, ymin, ymax],
            origin="lower",
            cmap="Greys",
            vmin=0,
            vmax=max(1, float(host.growth_cost.max())),
            alpha=0.30,
            aspect="equal",
        )
        for poly, colour in polygons:
            if poly[:, 0].max() >= start and poly[:, 0].min() <= end:
                ax.fill(
                    poly[:, 0],
                    poly[:, 1],
                    facecolor=colour,
                    edgecolor=colour,
                    linewidth=0.45,
                    zorder=2,
                )
        nodes = {n.node_id: n for n in network.nodes}
        for event in network.backend_provenance["interaction_events"]:
            n = nodes[event["node_id"]]
            xy = np.array([n.x, n.y]) - origin
            ac = np.array([xy @ axis, xy @ cross])
            if start <= ac[0] <= end:
                ax.scatter(
                    *ac,
                    marker="o" if event["kind"] == "merge" else "D",
                    s=25,
                    color="#202535",
                    edgecolors="white",
                    linewidth=0.65,
                    zorder=3,
                )
        ax.set(xlim=(start - 2, end + 2), ylim=(ymin, ymax), ylabel="Lateral distance (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    legend = [
        Line2D([], [], color=palette[i], lw=5, label=f"Route {i + 1}")
        for i in range(network.config.systems.count)
    ]
    legend += [
        Line2D([], [], color="#67558c", lw=5, label="Shared passage"),
        Line2D([], [], color="#202535", marker="o", ls="", label="Merge"),
        Line2D([], [], color="#202535", marker="D", ls="", label="Split"),
    ]
    axes[0].legend(
        handles=legend, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=6, frameon=False
    )
    ax = axes[-1]
    ax.step(
        metrics["station_m"], metrics["channel_count"], where="mid", color="#147d92", linewidth=1.6
    )
    ax.fill_between(
        metrics["station_m"], metrics["channel_count"], step="mid", color="#147d92", alpha=0.12
    )
    for event in network.backend_provenance["interaction_events"]:
        ax.axvline(event["station_m"], color="#555566", alpha=0.25, linewidth=0.6)
    ax.set(
        xlim=(0, extent),
        ylim=(0.5, network.config.systems.count + 0.5),
        yticks=range(1, network.config.systems.count + 1),
        ylabel="Distinct passages",
        xlabel="Downstream distance (m)",
    )
    ax.grid(alpha=0.15)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    return output
