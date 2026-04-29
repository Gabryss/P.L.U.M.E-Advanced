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
            constrained_layout=False,
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
        fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.955))

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
        ax.set_title("Longitudinal Network Diagnostics")
        ax.set_xlabel("Along-flow distance")
        ax.set_ylabel("Parallel channels")

        if not cave_network.slice_along_positions:
            ax.text(0.5, 0.5, "No longitudinal samples", ha="center", va="center")
            ax.set_axis_off()
            return

        along_positions = np.array(cave_network.slice_along_positions, dtype=float)
        channel_counts = np.array(cave_network.slice_channel_counts, dtype=float)
        ax.step(
            along_positions,
            channel_counts,
            where="mid",
            color="#2563eb",
            linewidth=1.8,
            label="parallel channels",
        )
        ax.fill_between(
            along_positions,
            0.0,
            channel_counts,
            step="mid",
            color="#93c5fd",
            alpha=0.35,
        )
        ax.set_ylim(0.0, max(channel_counts.max() + 0.8, 2.0))

        secondary_axis = ax.twinx()
        secondary_axis.set_ylabel("Tube width")
        width_profile = self._build_width_profile(cave_network, along_positions)
        if width_profile is not None:
            mean_width, min_width, max_width = width_profile
            secondary_axis.plot(
                along_positions,
                mean_width,
                color="#059669",
                linewidth=2.0,
                label="mean width",
            )
            secondary_axis.fill_between(
                along_positions,
                min_width,
                max_width,
                color="#10b981",
                alpha=0.18,
                label="width range",
            )
            secondary_axis.set_ylim(0.0, max(float(np.nanmax(max_width)) * 1.18, 1.0))

        seen_junction_labels: set[str] = set()
        for junction in cave_network.junctions:
            color = "#dc2626" if junction.kind == "chamber" else "#7c3aed"
            label = "chamber junction" if junction.kind == "chamber" else "split/merge junction"
            if label in seen_junction_labels:
                label = "_nolegend_"
            else:
                seen_junction_labels.add(label)
            ax.axvline(
                junction.along_position,
                color=color,
                linewidth=0.9,
                alpha=0.35,
                label=label,
            )

        lines = ax.get_lines() + secondary_axis.get_lines()
        legend_items = [
            (line, line.get_label())
            for line in lines
            if not line.get_label().startswith("_")
        ]
        ax.legend(
            [line for line, _label in legend_items],
            [label for _line, label in legend_items],
            loc="best",
            fontsize=8,
        )
        ax.grid(True, axis="x", alpha=0.18)

    @staticmethod
    def _build_width_profile(
        cave_network: CaveNetwork,
        along_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if along_positions.size == 0 or not cave_network.segments:
            return None

        nodes_by_id = {node.node_id: node for node in cave_network.nodes}
        bin_width = float(np.median(np.diff(along_positions))) if along_positions.size > 1 else 1.0
        bin_width = max(bin_width, 1.0)
        widths_by_bin: list[list[float]] = [[] for _ in along_positions]

        for segment in cave_network.segments:
            if segment.kind == "spur" or not segment.points:
                continue
            start_node = nodes_by_id.get(segment.start_node_id)
            end_node = nodes_by_id.get(segment.end_node_id)
            if start_node is None or end_node is None:
                continue

            start_along = start_node.along_position
            end_along = end_node.along_position
            segment_length = max(segment.total_length, 1.0)
            for point in segment.points:
                t = np.clip(point.arc_length / segment_length, 0.0, 1.0)
                point_along = (1.0 - t) * start_along + t * end_along
                bin_index = int(np.argmin(np.abs(along_positions - point_along)))
                if abs(float(along_positions[bin_index]) - point_along) <= 1.5 * bin_width:
                    widths_by_bin[bin_index].append(point.width)

        mean_width = np.full(along_positions.shape, np.nan, dtype=float)
        min_width = np.full(along_positions.shape, np.nan, dtype=float)
        max_width = np.full(along_positions.shape, np.nan, dtype=float)
        for index, widths in enumerate(widths_by_bin):
            if not widths:
                continue
            width_values = np.array(widths, dtype=float)
            mean_width[index] = float(np.mean(width_values))
            min_width[index] = float(np.min(width_values))
            max_width[index] = float(np.max(width_values))

        valid = np.isfinite(mean_width)
        if not np.any(valid):
            return None

        valid_positions = along_positions[valid]
        mean_width = np.interp(along_positions, valid_positions, mean_width[valid])
        min_width = np.interp(along_positions, valid_positions, min_width[valid])
        max_width = np.interp(along_positions, valid_positions, max_width[valid])
        return mean_width, min_width, max_width
