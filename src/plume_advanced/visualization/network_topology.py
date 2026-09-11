"""Plan-view diagnostics that expose actual passage envelopes and rock islands."""

from pathlib import Path

import numpy as np

from plume_advanced.stages.network_topology import section_footprint


def render_topology_footprint(network, sections, output_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask, x, y, spacing = section_footprint(network, sections)
    nodes = {n.node_id: n for n in network.nodes}
    first, last = [
        nodes[i] for i in (network.dominant_route_node_ids[0], network.dominant_route_node_ids[-1])
    ]
    origin = np.array([first.x, first.y])
    axis = np.array([last.x, last.y]) - origin
    axis /= np.linalg.norm(axis)
    cross = np.array([-axis[1], axis[0]])
    colors = {
        "trunk": "#0072B2",
        "island_arm": "#D55E00",
        "side_branch": "#009E73",
        "feeder": "#7570B3",
    }
    labels = {
        "trunk": "Main passage",
        "island_arm": "Routes around an island",
        "side_branch": "Blind side branch",
        "feeder": "Source feeder",
    }
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), layout="constrained", sharex=True, sharey=True)
    for segment in network.segments:
        xy = np.array([[p.x, p.y] for p in segment.points]) - origin
        role = segment.metadata.get("topology_role", "trunk")
        axes[0].plot(xy @ axis, xy @ cross, color=colors[role], lw=1.8)
    used = {s.metadata.get("topology_role", "trunk") for s in network.segments}
    axes[0].legend(
        handles=[
            Line2D([], [], color=colors[r], label=label, lw=2)
            for r, label in labels.items()
            if r in used
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        frameon=False,
    )
    extent = [x[0] - spacing / 2, x[-1] + spacing / 2, y[0] - spacing / 2, y[-1] + spacing / 2]
    axes[1].imshow(
        mask,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["white", "#849FA7"]),
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axes[1].contour(x, y, mask.astype(float), levels=[0.5], colors=["#344E58"], linewidths=0.65)
    axes[1].legend(
        handles=[
            Patch(
                facecolor="#849FA7", edgecolor="#344E58", label="Union of Stage-C section envelopes"
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        frameon=False,
    )
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x[0] - 5, x[-1] + 5)
        ax.set_ylim(y[0] - 5, y[-1] + 5)
        ax.set_ylabel("Lateral distance (m)")
        ax.grid(alpha=0.13)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Distance along the dominant passage (m)")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def render_gallery_history(network, output_path, *, dpi=180):
    """Show actual per-phase discharge, including dormant gaps before reuse."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    segments = sorted(network.segments, key=lambda s: s.segment_id)
    flows = np.array([s.metadata["phase_fluxes"] for s in segments])
    phases = flows.shape[1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(segments))), layout="constrained")
    colors = plt.get_cmap("Blues")(np.linspace(0.25, 1, 255))
    cmap = ListedColormap(np.vstack([[0.94, 0.94, 0.94, 1], colors]))
    im = ax.imshow(flows, aspect="auto", vmin=0, vmax=max(float(flows.max()), 1e-9), cmap=cmap)
    ax.set_xticks(range(phases), [f"Phase {p + 1}" for p in range(phases)])
    roles = {
        "trunk": "main passage",
        "feeder": "source inlet",
        "island_arm": "island arm",
        "side_branch": "blind branch",
    }
    ax.set_yticks(
        range(len(segments)),
        [f"{s.segment_id}: {roles[s.metadata['topology_role']]}" for s in segments],
    )
    for i in range(len(segments)):
        for p in range(phases):
            value = flows[i, p]
            ax.text(
                p,
                i,
                "inactive" if value == 0 else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.65 * flows.max() else "#22313b",
                fontsize=9,
            )
    ax.set_xticks(np.arange(phases) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(segments)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    fig.colorbar(im, ax=ax, label="Procedural discharge (reference source units)", shrink=0.75)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
