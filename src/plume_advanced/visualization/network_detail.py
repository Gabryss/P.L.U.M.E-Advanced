"""Paired physical-scale inspection of local detail, including added junctions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plume_advanced.stages.network_layers import segment_xyz


def render_network_detail(host, coarse, detailed, output):
    def route(network, original):
        pieces = {s.start_node_id: s for s in network.segments
                  if s.metadata.get("detail_parent_segment_id", s.segment_id) == original.segment_id}
        coordinates: list[np.ndarray] = []
        widths: list[float] = []
        node = original.start_node_id
        while node in pieces:
            segment = pieces.pop(node)
            skip = 1 if coordinates else 0
            coordinates.extend(segment_xyz(segment, network.config.layers)[skip:])
            widths.extend(p.width for p in segment.points[skip:])
            node = segment.end_node_id
        xyz = np.array(coordinates)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
        return xyz, np.array(widths), arc

    candidates = []
    for original in coarse.segments:
        a, aw, u = route(coarse, original)
        b, bw, t = route(detailed, original)
        interpolated = np.column_stack([np.interp(t/t[-1], u/u[-1], col) for col in np.c_[a, aw].T])
        candidates.append((float(np.max(abs(np.c_[b, bw]-interpolated))), original.segment_id))
    _, sid = max(candidates)
    selected = next(s for s in coarse.segments if s.segment_id == sid)
    angle = np.radians(host.config.flow_angle_degrees)
    basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    origin = np.array(host.config.seed_point)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for network, label, color, style in ((coarse, "Coarse", "#9f6771", "--"),
                                         (detailed, "Detailed", "#107f89", "-")):
        for i, segment in enumerate(network.segments):
            xyz = segment_xyz(segment, network.config.layers)
            xy = (xyz[:, :2]-origin) @ basis.T
            axes[0, 0].plot(*xy.T, color=color, ls=style, lw=1.4, label=label if i == 0 else None)
        xyz, widths, arc = route(network, selected)
        xy = (xyz[:, :2]-origin) @ basis.T
        axes[0, 1].plot(*xy.T, color=color, ls=style, lw=2, label=label)
        axes[1, 0].plot(arc, widths, color=color, ls=style, lw=2, label=label)
        axes[1, 1].plot(arc, xyz[:, 2], color=color, ls=style, lw=2, label=label)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend()
    for ax in axes[0]:
        ax.set(xlabel="Along-flow distance (m)", ylabel="Lateral distance (m)")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_title("Complete network · original routes retained")
    axes[0, 1].set_title(f"Passage {sid} · local plan")
    axes[1, 0].set(xlabel="Distance from passage start (m)", ylabel="Width (m)", title="Width envelope")
    axes[1, 1].set(xlabel="Distance from passage start (m)", ylabel="Elevation (m)", title="Actual centreline elevation")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
