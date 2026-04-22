"""Visualization helpers for the stage-B cave network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stages.host_field import HostField
from stages.network import CaveNetwork


@dataclass(frozen=True)
class CaveNetworkPlotConfig:
    """Figure settings for the cave-network visualization."""

    figure_size: tuple[float, float] = (15.0, 11.0)
    dpi: int = 180


class CaveNetworkPlotter:
    """Render the occupancy-first cave network into a reviewable artifact."""

    def __init__(self, config: CaveNetworkPlotConfig | None = None) -> None:
        self.config = config or CaveNetworkPlotConfig()

    def render(
        self,
        host_field: HostField,
        cave_network: CaveNetwork,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=self.config.figure_size,
            constrained_layout=True,
        )
        fig.suptitle("Stage B - Cave Network", fontsize=16)

        self._draw_map_panel(
            ax=axes[0, 0],
            host_field=host_field,
            values=host_field.elevation,
            cave_network=cave_network,
            title="Terrain With Occupancy And Network",
            cmap="terrain",
            colorbar_label="Elevation",
            overlay_mode="occupancy",
        )
        self._draw_map_panel(
            ax=axes[0, 1],
            host_field=host_field,
            values=host_field.growth_cost,
            cave_network=cave_network,
            title="Growth Cost With Network",
            cmap="magma_r",
            colorbar_label="Cost",
            overlay_mode="segments",
        )
        self._draw_map_panel(
            ax=axes[1, 0],
            host_field=host_field,
            values=np.where(cave_network.occupancy, cave_network.width_field, np.nan),
            cave_network=cave_network,
            title="Occupied Width Field",
            cmap="viridis",
            colorbar_label="Width",
            overlay_mode="width",
        )
        self._draw_profile_panel(ax=axes[1, 1], cave_network=cave_network)

        summary = cave_network.summary()
        summary_line = (
            f"Nodes: {int(summary['node_count'])} | "
            f"Segments: {int(summary['segment_count'])} | "
            f"Loops: {int(summary['loop_count'])} | "
            f"Max channels: {int(summary['max_parallel_channels'])} | "
            f"Dominant route: {summary['dominant_route_length']:.1f}"
        )
        fig.text(0.5, 0.01, summary_line, ha="center", fontsize=10)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_map_panel(
        self,
        *,
        ax,
        host_field: HostField,
        values,
        cave_network: CaveNetwork,
        title: str,
        cmap: str,
        colorbar_label: str,
        overlay_mode: str,
    ) -> None:
        import matplotlib.pyplot as plt

        image = ax.imshow(
            values,
            extent=host_field.extent,
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        if overlay_mode in {"occupancy", "width"}:
            occupancy_alpha = np.where(cave_network.occupancy, 0.72, 0.0)
            ax.imshow(
                np.where(cave_network.occupancy, 1.0, np.nan),
                extent=host_field.extent,
                origin="lower",
                cmap="gray",
                alpha=occupancy_alpha,
                aspect="equal",
            )

        dominant_pairs = set(zip(cave_network.dominant_route_node_ids, cave_network.dominant_route_node_ids[1:]))
        dense_graph = len(cave_network.nodes) >= 250
        for segment in cave_network.segments:
            x_coords = [point.x for point in segment.points]
            y_coords = [point.y for point in segment.points]
            is_dominant = (segment.start_node_id, segment.end_node_id) in dominant_pairs
            linestyle = "-"
            if is_dominant:
                color = "#22d3ee"
                linewidth = 1.4 if dense_graph else 1.8
            elif segment.kind in {"backbone", "braid"}:
                color = "#f8fafc"
                linewidth = 0.55 if dense_graph else 0.8
            elif segment.kind == "island_bypass":
                color = "#e5e7eb"
                linewidth = 0.7 if dense_graph else 0.95
            elif segment.kind == "chamber_braid":
                color = "#fb7185"
                linewidth = 0.8 if dense_graph else 1.0
            elif segment.kind == "ladder":
                color = "#f59e0b"
                linewidth = 0.65 if dense_graph else 0.9
            elif segment.kind == "underpass":
                color = "#c084fc"
                linewidth = 0.7 if dense_graph else 0.95
                linestyle = (0, (3, 2))
            else:
                color = "#facc15"
                linewidth = 0.6 if dense_graph else 0.85
            ax.plot(
                x_coords,
                y_coords,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.82 if dense_graph else 0.9,
            )

        degrees = {node.node_id: 0 for node in cave_network.nodes}
        for segment in cave_network.segments:
            degrees[segment.start_node_id] += 1
            degrees[segment.end_node_id] += 1
        for node in cave_network.nodes:
            color = "#34d399" if degrees[node.node_id] >= 3 else "#e2e8f0"
            if node.kind == "entry":
                color = "#06b6d4"
            elif node.kind == "exit":
                color = "#ef4444"
            elif node.kind == "spur_terminal":
                color = "#fb923c"
            elif node.kind == "chamber":
                color = "#fb7185"
            node_size = 4 + 1.4 * degrees[node.node_id] if dense_graph else 8 + 3.0 * degrees[node.node_id]
            ax.scatter(
                [node.x],
                [node.y],
                c=color,
                s=node_size,
                edgecolors="black",
                linewidths=0.1 if dense_graph else 0.2,
                alpha=0.7 if dense_graph else 0.88,
            )

        colorbar = plt.colorbar(image, ax=ax, shrink=0.9)
        colorbar.set_label(colorbar_label)

    def _draw_profile_panel(self, *, ax, cave_network: CaveNetwork) -> None:
        ax.set_title("Longitudinal Network Profile")
        ax.set_xlabel("Along-flow distance")
        ax.set_ylabel("Parallel channels")
        ax.step(
            cave_network.slice_along_positions,
            cave_network.slice_channel_counts,
            where="mid",
            color="#1d4ed8",
            linewidth=2.2,
            label="Slice channels",
        )
        ax.set_ylim(0.8, max(cave_network.slice_channel_counts) + 0.6)

        secondary_axis = ax.twinx()
        secondary_axis.set_ylabel("Segment width")
        braid_widths = [
            segment.mean_width
            for segment in cave_network.segments
            if segment.kind != "spur"
        ]
        if braid_widths:
            percentiles = np.percentile(braid_widths, [25, 50, 75])
            for percentile, color, label in zip(percentiles, ["#10b981", "#f59e0b", "#ef4444"], ["P25 width", "Median width", "P75 width"], strict=True):
                secondary_axis.axhline(percentile, color=color, linewidth=1.5, linestyle="--", label=label)
        secondary_axis.set_ylim(0.0, max((segment.mean_width for segment in cave_network.segments), default=1.0) * 1.15)

        lines = ax.get_lines() + secondary_axis.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
