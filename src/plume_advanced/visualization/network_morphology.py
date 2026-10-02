"""Inspectable network morphology with explicit plan-view connection semantics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from plume_advanced.stages.network_morphology import morphology_metrics


def render_network_morphology(host, network, output):
    metrics = morphology_metrics(network)
    angle = np.radians(host.config.flow_angle_degrees)
    basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    origin = np.array(host.config.seed_point)
    palette = ["#178091", "#e3822c", "#7353a5", "#498a57"]
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    plan = fig.add_subplot(layout[0, :])
    widths = fig.add_subplot(layout[1, 0])
    density = fig.add_subplot(layout[1, 1])
    patches, colors = [], []
    for s in network.segments:
        xy = (np.array([(p.x, p.y) for p in s.points]) - origin) @ basis.T
        w = np.array([p.width for p in s.points])
        tangent = np.gradient(xy, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        role = s.metadata.get("regional_route_type")
        color = palette[int(s.metadata.get("regional_start_layer", 0))]
        if role == "descending_ramp":
            plan.plot(*xy.T, color="#333333", ls="--", lw=1.5, zorder=4)
        else:
            patches.append(
                np.r_[xy + normal * w[:, None] / 2, (xy - normal * w[:, None] / 2)[::-1]]
            )
            colors.append(color)
        widths.plot(xy[:, 0], w, color=color, lw=1, alpha=0.8)
    plan.add_collection(PolyCollection(patches, facecolors=colors, edgecolors="none"))
    for n in network.nodes:
        xy = (np.array([n.x, n.y]) - origin) @ basis.T
        if n.kind == "junction":
            plan.scatter(xy[0], xy[1], s=14, facecolor="white", edgecolor="#333333", zorder=5)
        elif n.kind == "terminal":
            plan.scatter(xy[0], xy[1], s=30, marker="x", color="#333333", zorder=5)
        else:
            plan.scatter(
                xy[0], xy[1], s=25, marker=">" if n.kind == "entry" else "s", color="#333333", zorder=5
            )
    plan.autoscale_view()
    plan.set_aspect("equal", adjustable="datalim")
    plan.set(
        ylabel="Lateral distance (m)", title="Width envelopes · circles mark actual graph junctions"
    )
    handles = [
        Line2D([], [], color=palette[i], lw=3, label=f"Layer {i + 1}")
        for i in range(network.config.layers.count if network.config.layers.enabled else 1)
    ]
    handles.extend(
        [
            Line2D([], [], color="#333333", ls="--", label="Ramp"),
            Line2D([], [], color="#333333", marker="x", ls="none", label="Blind end"),
        ]
    )
    plan.legend(handles=handles, ncols=len(handles), loc="upper left", fontsize=9)
    widths.set(ylabel="Estimated width (m)", title="Passage hierarchy and spatial variation")
    bins = np.array(metrics["junction_bin_edges_m"])
    density.bar(
        bins[:-1],
        metrics["junction_counts"],
        width=np.diff(bins),
        align="edge",
        color="#178091",
        edgecolor="white",
    )
    density.set(ylabel="Split / merge nodes", title="Measured local connectivity (100 m bins)")
    for ax in (plan, widths, density):
        ax.set_xlabel("Distance along original flow axis (m)")
        ax.grid(alpha=0.15)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def render_campaign_gallery(root):
    """Plot saved campaign cases, including failures, in bounded six-case pages."""
    root = Path(root)
    cases = json.loads((root / "campaign.json").read_text())["cases"]
    palette = ["#178091", "#e3822c", "#7353a5", "#498a57"]
    outputs = []
    for offset in range(0, len(cases), 6):
        page = cases[offset : offset + 6]
        fig, axes = plt.subplots(
            (len(page) + 1) // 2,
            2,
            figsize=(16, 4 * ((len(page) + 1) // 2)),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, case in zip(axes.ravel(), page):
            directory = root / case["case"]
            cfg = json.loads((directory / "resolved_config.json").read_text())
            layers = cfg["network"]["layers"]
            count = layers["count"] if layers["enabled"] else 1
            title = (
                f"{count} layer(s) · {case['sources']} sources · root seed {case['requested_seed']}"
            )
            ax.set_title(title)
            if not case["accepted"]:
                ax.text(
                    0.5, 0.5, "FAILED — see quality report", ha="center", transform=ax.transAxes
                )
                continue
            artifact = json.loads((directory / "network.json").read_text())
            host = cfg["host_field"]
            angle = np.radians(host["flow_angle_degrees"])
            basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = np.array(host["seed_point"])
            patches, colors = [], []
            for s in artifact["segments"]:
                xy = (np.array([[p["x"], p["y"]] for p in s["centerline"]]) - origin) @ basis.T
                w = np.array([p["width"] for p in s["centerline"]])
                meta = s["metadata"]
                if meta.get("regional_route_type") == "descending_ramp":
                    ax.plot(*xy.T, color="#333333", ls="--", lw=1)
                else:
                    t = np.gradient(xy, axis=0)
                    t /= np.maximum(np.linalg.norm(t, axis=1)[:, None], 1e-9)
                    normal = np.column_stack((-t[:, 1], t[:, 0]))
                    patches.append(
                        np.r_[xy + normal * w[:, None] / 2, (xy - normal * w[:, None] / 2)[::-1]]
                    )
                    colors.append(palette[int(meta.get("regional_start_layer", 0))])
            ax.add_collection(PolyCollection(patches, facecolors=colors, edgecolors="none"))
            for n in artifact["nodes"]:
                xy = (np.array([n["x"], n["y"]]) - origin) @ basis.T
                if n["kind"] == "junction":
                    ax.scatter(xy[0], xy[1], s=6, color="#333333", zorder=5)
                elif n["kind"] == "terminal":
                    ax.scatter(xy[0], xy[1], s=18, marker="x", color="#333333", zorder=5)
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="box")
            ax.set(xlabel="Along-flow distance (m)", ylabel="Lateral distance (m)")
            ax.grid(alpha=0.15)
        for ax in axes.ravel()[len(page) :]:
            ax.set_visible(False)
        output = root / f"comparison_{offset // 6 + 1:02d}.png"
        fig.savefig(output, dpi=170)
        plt.close(fig)
        outputs.append(output)
    return outputs
