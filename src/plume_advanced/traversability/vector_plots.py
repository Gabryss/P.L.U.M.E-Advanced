"""Registered geometry previews with explicit legends and equal-distance axes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PALETTE = ["#e9e7e1", "#4b9fa3", "#bd653e"]
LABELS = ["Unmeasured / ambiguous", "Sampled cavity", "Conservative prop mask"]


def render_chart(output, chart, fields, vectors, network):
    name = chart["id"] + "_geometry.png"
    origin, r = fields["origin_xy_m"], float(fields["resolution_m"])
    h, w = fields["physical_state"].shape
    extent = [origin[0], origin[0] + w*r, origin[1], origin[1] + h*r]
    # Long Y-oriented caves should not become thin vertical slivers. This swaps
    # display axes only; numeric arrays and all vector coordinates stay XYZ.
    horizontal = int(h > w)
    order = [horizontal, 1-horizontal]
    display = fields["physical_state"]
    if horizontal:
        extent = extent[2:] + extent[:2]
        display = display.T
    height = 5.5 * min(h, w) / max(h, w) + 2.0
    fig, axes = plt.subplots(1, 2, figsize=(12, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.99, top=1-.55/height, bottom=1.25/height, wspace=.12)
    axes[0].imshow(display, origin="lower", extent=extent,
                   interpolation="nearest", cmap=ListedColormap(PALETTE), vmin=0, vmax=2)
    axes[0].set_title("Physical raster")
    fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(PALETTE, LABELS)],
               loc="lower center", bbox_to_anchor=(.27, .02), fontsize=8)
    for key, colour, style in (("domain_rings", "#aaa7a0", ":"),
                                ("cavity_rings", "#227e85", "-"),
                                ("obstacle_rings", "#bd653e", "-")):
        for ring in vectors[key]:
            xy = np.asarray(ring["xy_m"])
            axes[1].plot(*xy[:, order].T, color=colour, linestyle=style, linewidth=.8)
    for edge in network["edges"]:
        if edge["chart_id"] != chart["id"]:
            continue
        xy = np.array(edge["xyz_m"])[:, :2]
        verified = edge["cavity_witness"]["status"] == "verified"
        axes[1].plot(*xy[:, order].T, color="#293747" if verified else "#bc3679",
                     linestyle="-" if verified else "--", linewidth=1.3)
    axes[1].set_title("Vector outlines and centreline witnesses")
    fig.legend(handles=[
        Line2D([], [], color="#227e85", label="Sampled cavity outline"),
        Line2D([], [], color="#bd653e", label="Prop mask outline"),
        Line2D([], [], color="#aaa7a0", linestyle=":", label="Sampling domain"),
        Line2D([], [], color="#293747", label="Verified witness"),
        Line2D([], [], color="#bc3679", linestyle="--", label="Unresolved witness"),
    ], loc="lower center", bbox_to_anchor=(.77, .02), ncol=2, fontsize=8)
    for ax in axes:
        ax.set(xlim=extent[:2], ylim=extent[2:],
               xlabel=f"{'XY'[horizontal]} (m)", ylabel=f"{'XY'[1-horizontal]} (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
    fig.suptitle(f"{chart['id']} · {r:g} m cells · no robot limits")
    fig.savefig(output/name, dpi=150)
    plt.close(fig)
    return name


def render_network(output, network):
    name = "network_vectors.png"
    layers = sorted({e[k] for e in network["edges"] for k in ("from_layer", "to_layer")})
    positions = np.concatenate([e["xyz_m"] for e in network["edges"]])
    horizontal = int(np.ptp(positions[:, 1]) > np.ptp(positions[:, 0]))
    span = np.ptp(positions, axis=0)
    plan_height = max(1.4, 10.5 * span[1-horizontal] / max(span[horizontal], 1e-6))
    side_height = np.clip(10.5 * span[2] / max(span[horizontal], 1e-6), .65, 3.0)
    fig, axes = plt.subplots(2, 1, figsize=(12, plan_height + side_height + 2.4),
                             height_ratios=[plan_height, side_height], sharex=True,
                             layout="constrained")
    palette = plt.get_cmap("tab10")
    for edge in network["edges"]:
        p = np.asarray(edge["xyz_m"])
        colour = "#bc6532" if edge["kind"] == "ramp" else palette(edge["from_layer"] % 10)
        style = "-" if edge["cavity_witness"]["status"] == "verified" else "--"
        for ax, second in zip(axes, (1-horizontal, 2)):
            ax.plot(p[:, horizontal], p[:, second], color=colour, linestyle=style, linewidth=1.4)
    for node in network["nodes"]:
        p = node["xyz_m"]
        for ax, second in zip(axes, (1-horizontal, 2)):
            if node["degree"] != 2:
                ax.scatter(p[horizontal], p[second], s=12, color="#293747",
                           marker="o" if node["degree"] > 2 else "s", zorder=3)
    handles = [Line2D([], [], color=palette(i % 10), label=f"Layer {i}") for i in layers]
    handles += [Line2D([], [], color="#bc6532", label="Ramp"),
                Line2D([], [], color="#293747", label="Verified witness"),
                Line2D([], [], color="#293747", linestyle="--", label="Unresolved witness"),
                Line2D([], [], color="#293747", marker="s", linestyle="none", label="Graph terminal"),
                Line2D([], [], color="#293747", marker="o", linestyle="none", label="Junction")]
    for ax, axis_name in zip(axes, ("XY"[1-horizontal], "Z")):
        ax.set(xlabel=f"{'XY'[horizontal]} (m)", ylabel=f"{axis_name} (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    axes[0].set_title("Plan view — XY overlap does not join layers")
    axes[1].set_title("Elevation view — explicit ramp connections")
    fig.legend(handles=handles, loc="outside lower center", ncol=min(5, len(handles)), fontsize=9)
    fig.suptitle("Passage graph · cavity witnesses only · props and robot feasibility separate")
    fig.savefig(output/name, dpi=150)
    plt.close(fig)
    return name
