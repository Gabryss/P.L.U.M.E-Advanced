"""Plan and elevation views of the same optional layered Stage-B network."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plume_advanced.stages.network_layers import segment_xyz


def render_layered_network(host, network, output):
    angle = np.radians(host.config.flow_angle_degrees)
    basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    origin = np.array(host.config.seed_point)
    palette = ["#178091", "#e3822c", "#7353a5", "#498a57"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    positions = {}
    for s in network.segments:
        xyz = segment_xyz(s, network.config.layers)
        plan = (xyz[:, :2] - origin) @ basis.T
        a, b = (int(s.metadata[f"regional_{key}_layer"]) for key in ("start", "end"))
        color = palette[a] if a == b else "#333333"
        # Show the estimated plan width at its metric scale. Constant line
        # weights hid both constrictions and wider sustained passages.
        tangent = np.gradient(plan, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        offset = normal * np.array([p.width for p in s.points])[:, None] / 2
        outline = np.vstack((plan + offset, (plan - offset)[::-1]))
        axes[0].fill(outline[:, 0], outline[:, 1], color=color, alpha=.7, linewidth=0)
        for index, node in ((0, s.start_node_id), (-1, s.end_node_id)):
            positions[node] = (plan[index, 0], plan[index, 1], xyz[index, 2])
        for ax, ordinate in zip(axes, (plan[:, 1], xyz[:, 2])):
            ax.plot(
                plan[:, 0],
                ordinate,
                color=color,
                lw=(.6 if ax is axes[0] else 2.2) if a == b else 2,
                ls="-" if a == b else "--",
            )
        if a != b:
            for ax, ordinate in zip(axes, (plan[:, 1], xyz[:, 2])):
                ax.scatter(plan[[0, -1], 0], ordinate[[0, -1]], c=color, s=20, zorder=5)
    for ax, title, label in zip(
        axes, ("Plan view", "Elevation view"), ("Lateral distance (m)", "Centreline elevation (m)")
    ):
        ax.set(xlabel="Distance along original flow axis (m)", ylabel=label, title=title)
        ax.grid(alpha=0.18)
    for kind, marker in (("entry", "o"), ("exit", "^"), ("terminal", "D")):
        points = np.array([positions[n.node_id] for n in network.nodes
                           if n.kind == kind and n.node_id in positions])
        if len(points):
            for ax, ordinate in zip(axes, (1, 2)):
                ax.scatter(points[:, 0], points[:, ordinate], marker=marker, s=28,
                           color="#222222", edgecolor="white", linewidth=.5, zorder=5)
    # Keep a shared, tight downstream range. Aspect='equal' with adjustable
    # data limits previously expanded the plan axis far beyond the host.
    along = np.array(list(positions.values()))[:, 0]
    for ax in axes:
        ax.set_xlim(float(along.min()) - 15, float(along.max()) + 15)
    legend = [
        Line2D([], [], color=palette[i], lw=3, label=f"Layer {i + 1}")
        for i in range(network.config.layers.count)
    ]
    legend.append(Line2D([], [], color="#333333", ls="--", lw=3, label="Descending connection"))
    legend.extend(Line2D([], [], color="#222222", marker=marker, ls="", label=label)
                  for marker, label in (("o", "Inlet"), ("^", "Outlet"), ("D", "Blind end")))
    axes[0].legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1.12), ncols=4)
    axes[1].set_title("Elevation view · vertical scale differs from plan view")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output
