"""Inspect source systems, shared passages and explicit connections at scale."""

from collections import Counter
from pathlib import Path

import numpy as np


def render_system_network(host, network, output_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    angle = np.radians(host.config.flow_angle_degrees)
    flow, cross = (
        np.array([np.cos(angle), np.sin(angle)]),
        np.array([-np.sin(angle), np.cos(angle)]),
    )
    origin = np.array(host.config.seed_point)
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#887744",
        "#5555AA",
    ]
    incoming = Counter(s.end_node_id for s in network.segments)
    outgoing = Counter(s.start_node_id for s in network.segments)
    merges = [n for n in network.nodes if incoming[n.node_id] > 1]
    splits = [n for n in network.nodes if outgoing[n.node_id] > 1]
    nodes = {n.node_id: n for n in network.nodes}
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.1, 0.7])
    full = fig.add_subplot(grid[0, :])
    zooms = [fig.add_subplot(grid[1, i]) for i in range(2)]
    balance = fig.add_subplot(grid[2, :])

    def draw(ax):
        for s in network.segments:
            xy = np.array([[p.x, p.y] for p in s.points]) - origin
            group = s.metadata["system_ids"]
            ax.plot(
                xy @ flow,
                xy @ cross,
                color=palette[group[0]] if len(group) == 1 else "#242B35",
                lw=1.5 if len(group) == 1 else 2.8,
                solid_capstyle="round",
            )
        for group, marker, color in [(merges, "o", "#242B35"), (splits, "D", "#9A4DB2")]:
            xy = np.array([[n.x, n.y] for n in group]).reshape(-1, 2) - origin
            ax.scatter(
                xy @ flow,
                xy @ cross,
                marker=marker,
                color=color,
                s=35,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Distance along host flow (m)")
        ax.set_ylabel("Lateral distance (m)")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    draw(full)
    handles = [
        Line2D([], [], color=palette[i], lw=2, label=f"System {i + 1}")
        for i in range(network.config.systems.count)
    ]
    handles += [
        Line2D([], [], color="#242B35", lw=3, label="Shared passage"),
        Line2D([], [], marker="o", ls="", color="#242B35", label="Merge"),
        Line2D([], [], marker="D", ls="", color="#9A4DB2", label="Split"),
    ]
    full.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=min(len(handles), 8),
        frameon=False,
    )
    width = 2 * network.config.base_passage_radius
    for ax, group, label in zip(zooms, [merges, splits], ["Merge", "Split"]):
        draw(ax)
        if not group:
            ax.text(
                0.5,
                0.5,
                f"No {label.lower()} in this candidate",
                transform=ax.transAxes,
                ha="center",
            )
            continue
        node = group[0]
        pos = np.array([node.x, node.y]) - origin
        a, c = pos @ flow, pos @ cross
        ax.set_xlim(a - 18 * width, a + 18 * width)
        ax.set_ylim(c - 9 * width, c + 9 * width)
        ax.set_aspect("equal", adjustable="box")
        for s in network.segments:
            if node.node_id in (s.start_node_id, s.end_node_id):
                target = min(8 * width, 0.4 * s.total_length)
                if s.end_node_id == node.node_id:
                    target = s.total_length - target
                p = min(s.points, key=lambda point: abs(point.arc_length - target))
                position = np.array([p.x, p.y]) - origin
                if abs(position @ flow - a) < 18 * width:
                    ax.annotate(
                        f"q = {s.mean_flux:.2f}",
                        (position @ flow, position @ cross),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=9,
                    )
    stations = np.linspace(
        min(n.along_position for n in network.nodes) + 0.01,
        max(n.along_position for n in network.nodes) - 0.01,
        500,
    )
    discharge = [
        sum(
            s.mean_flux
            for s in network.segments
            if nodes[s.start_node_id].along_position <= a < nodes[s.end_node_id].along_position
        )
        for a in stations
    ]
    expected = network.config.source_flux * network.config.systems.count
    balance.axhline(expected, color="#aaaaaa", ls="--", lw=3, label="Total source supply")
    balance.plot(stations, discharge, color="#0072B2", lw=1.6, label="Sum of channel discharge")
    balance.set_ylim(0, expected * 1.3)
    balance.set(xlabel="Distance along host flow (m)", ylabel="Relative discharge")
    balance.legend(loc="lower center", ncol=2, frameon=False)
    balance.grid(alpha=0.18)
    balance.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output
