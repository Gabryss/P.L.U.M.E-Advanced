"""Visualization helpers for the stage-B cave network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.stages.host_field import HostField
from plume_advanced.stages.network import CaveNetwork


@dataclass(frozen=True)
class CaveNetworkPlotConfig:
    """Figure settings for the cave-network visualization."""

    figure_size: tuple[float, float] = (16.0, 20.0)
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
            4,
            2,
            figsize=self.config.figure_size,
            constrained_layout=self.config.figure_size[1] >= 12.0,
        )

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
        self._draw_emplacement_timeline(ax=axes[2, 0], cave_network=cave_network)
        self._draw_topology_scatter(ax=axes[2, 1], cave_network=cave_network)
        self._draw_breakout_flux(ax=axes[3, 0], cave_network=cave_network)
        self._draw_breakout_process(ax=axes[3, 1], cave_network=cave_network)

        summary = cave_network.summary()
        summary_line = (
            f"Density: {summary['network_density']:.2f} | "
            f"Phases/levels: {int(summary['emplacement_phase_count'])}/"
            f"{int(summary['vertical_level_count'])} | "
            f"Nodes: {int(summary['node_count'])} | "
            f"Segments: {int(summary['segment_count'])} | "
            f"Loops: {int(summary['loop_count'])} | "
            f"Captures/chambers: {int(summary['vertical_capture_count'])}/"
            f"{int(summary['process_chamber_count'])} | "
            f"Drained pools: {int(summary['drained_lava_pool_count'])} | "
            f"Skeleton/visible channels: "
            f"{int(summary['max_parallel_channels'])}/"
            f"{int(summary['max_visible_parallel_channels'])} | "
            f"Dominant route: {summary['dominant_route_length']:.1f}"
        )
        fig.suptitle(f"Stage B - Cave Network\n{summary_line}", fontsize=14)

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

        dominant_pairs = set(
            zip(cave_network.dominant_route_node_ids, cave_network.dominant_route_node_ids[1:])
        )
        dense_graph = len(cave_network.nodes) >= 250
        for segment in cave_network.segments:
            x_coords = [point.x for point in segment.points]
            y_coords = [point.y for point in segment.points]
            is_dominant = (segment.start_node_id, segment.end_node_id) in dominant_pairs
            linestyle: Any = "-"
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
            elif segment.metadata.get("chamber_type") == "drained_lava_pool":
                color = "#f97316"
                linewidth = 1.8 if dense_graph else 2.4
            elif bool(segment.metadata.get("chamber_forming", False)):
                color = "#fb7185"
                linewidth = 1.1 if dense_graph else 1.45
            elif bool(segment.metadata.get("vertical_capture", False)):
                color = "#c084fc"
                linewidth = 0.85 if dense_graph else 1.15
                linestyle = (0, (4, 2))
            elif segment.kind == "anastomosis":
                color = "#38bdff"
                linewidth = 0.75 if dense_graph else 1.0
            elif segment.kind == "distributary":
                color = "#a7f3d0"
                linewidth = 0.7 if dense_graph else 0.95
            elif segment.kind == "ladder":
                color = "#f59e0b"
                linewidth = 0.65 if dense_graph else 0.9
            elif segment.kind == "underpass":
                color = "#c084fc"
                linewidth = 0.7 if dense_graph else 0.95
                linestyle = (0, (3, 2))
            elif segment.kind in {"abandoned_lobe", "stalled_lobe"}:
                color = "#fb923c"
                linewidth = 0.6 if dense_graph else 0.85
                linestyle = (0, (2, 2))
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
            node_size = (
                4 + 1.4 * degrees[node.node_id] if dense_graph else 8 + 3.0 * degrees[node.node_id]
            )
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
        visible_counts = np.array(
            cave_network.slice_visible_channel_counts,
            dtype=float,
        )
        if visible_counts.size == channel_counts.size:
            ax.step(
                along_positions,
                visible_counts,
                where="mid",
                color="#dc2626",
                linewidth=1.5,
                linestyle="--",
                label="visible occupied channels",
            )
        ax.set_ylim(
            0.0,
            max(
                channel_counts.max() + 0.8,
                visible_counts.max() + 0.8 if visible_counts.size else 0.0,
                2.0,
            ),
        )

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
            (line, line.get_label()) for line in lines if not line.get_label().startswith("_")
        ]
        ax.legend(
            [line for line, _label in legend_items],
            [label for _line, label in legend_items],
            loc="best",
            fontsize=8,
        )
        ax.grid(True, axis="x", alpha=0.18)

    @staticmethod
    def _draw_emplacement_timeline(*, ax, cave_network: CaveNetwork) -> None:
        paths: dict[str, dict[str, object]] = {}
        for segment in cave_network.segments:
            path_id = segment.metadata.get("lobe_path_id")
            if path_id is None:
                continue
            key = str(path_id)
            record = paths.setdefault(
                key,
                {
                    "birth": segment.metadata.get("birth_phase", 0),
                    "death": segment.metadata.get("death_phase", 0),
                    "state": segment.metadata.get("formation_state", "unknown"),
                    "level": segment.z_level,
                    "chamber": False,
                },
            )
            record["chamber"] = bool(record["chamber"]) or bool(
                segment.metadata.get("chamber_forming", False)
            )
        if not paths:
            ax.set_title("Emplacement History")
            ax.axis("off")
            return

        def path_number(item: tuple[str, dict[str, object]]) -> int:
            suffix = item[0].rsplit("_", maxsplit=1)[-1]
            return int(suffix) if suffix.isdigit() else 0

        ordered = sorted(paths.items(), key=path_number)
        colors = {
            "coalesced": "#38bdf8",
            "vertically_captured": "#a855f7",
            "thermally_abandoned": "#f97316",
            "stranded": "#f59e0b",
        }
        for row, (path_id, record) in enumerate(ordered):
            birth_value = record["birth"]
            death_value = record["death"]
            level_value = record["level"]
            birth = int(birth_value) if isinstance(birth_value, (int, float)) else 0
            death = int(death_value) if isinstance(death_value, (int, float)) else birth
            state = str(record["state"])
            ax.barh(
                row,
                death - birth + 1,
                left=birth - 0.45,
                height=0.62,
                color=colors.get(state, "#64748b"),
                alpha=0.82,
            )
            level = int(level_value) if isinstance(level_value, (int, float)) else 0
            ax.text(death + 0.58, row, f"L{level:+d}", va="center", fontsize=7)
            if bool(record["chamber"]):
                ax.scatter([death], [row], marker="*", s=44, color="#e11d48", zorder=3)
        ax.set_yticks(range(len(ordered)), [path_id.replace("lobe_", "") for path_id, _ in ordered])
        ax.set_xlabel("Emplacement phase")
        ax.set_ylabel("Lobe path")
        ax.set_title("Seeded Route Lifetimes (★ chamber-forming)")
        ax.grid(True, axis="x", alpha=0.2)
        ax.invert_yaxis()

    @staticmethod
    def _draw_topology_scatter(*, ax, cave_network: CaveNetwork) -> None:
        palette = {
            "backbone": "#0891b2",
            "source_feeder": "#22c55e",
            "anastomosis": "#2563eb",
            "abandoned_lobe": "#f97316",
            "stalled_lobe": "#f59e0b",
        }
        seen: set[str] = set()
        for segment in cave_network.segments:
            if len(segment.points) < 2 or segment.total_length <= 0.0:
                continue
            start = segment.points[0]
            end = segment.points[-1]
            direct = max(float(np.hypot(end.x - start.x, end.y - start.y)), 1e-9)
            sinuosity = segment.total_length / direct
            label = segment.kind if segment.kind not in seen else "_nolegend_"
            seen.add(segment.kind)
            marker = "D" if segment.z_level != 0 else "o"
            ax.scatter(
                [segment.total_length],
                [sinuosity],
                color=palette.get(segment.kind, "#64748b"),
                marker=marker,
                s=22 + 4 * abs(segment.z_level),
                alpha=0.78,
                label=label,
            )
        ax.axhline(1.0, color="#475569", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Segment length (m)")
        ax.set_ylabel("Sinuosity")
        ax.set_title("Persistence–Sinuosity Morphospace (◆ stacked)")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", fontsize=7)

    @staticmethod
    def _breakout_records(cave_network: CaveNetwork) -> list[dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        nodes = {node.node_id: node for node in cave_network.nodes}
        for segment in cave_network.segments:
            path_id = segment.metadata.get("lobe_path_id")
            if path_id is None or segment.metadata.get("branching_process") is None:
                continue
            key = str(path_id)
            along = min(
                nodes[segment.start_node_id].along_position,
                nodes[segment.end_node_id].along_position,
            )
            if key in records:
                records[key]["along"] = min(float(records[key]["along"]), along)
                continue
            records[key] = {
                "path_id": key,
                "along": along,
                "trigger": str(segment.metadata.get("breakout_trigger", "unknown")),
                "outcome": str(segment.metadata.get("formation_state", "unknown")),
                "score": float(segment.metadata.get("breakout_score", 0.0)),
                "parent_before": float(segment.metadata.get("parent_flux_before_split", 0.0)),
                "parent_after": float(segment.metadata.get("parent_flux_after_split", 0.0)),
                "branch_flux": float(segment.metadata.get("initial_flux", 0.0)),
                "split_fraction": float(segment.metadata.get("branch_flux_fraction", 0.0)),
                "returned_flux": float(segment.metadata.get("coalescence_returned_flux", 0.0)),
            }
        return sorted(records.values(), key=lambda item: float(item["along"]))

    @classmethod
    def _draw_breakout_flux(cls, *, ax, cave_network: CaveNetwork) -> None:
        records = cls._breakout_records(cave_network)
        ax.set_title("Finite-Flux Breakout Allocation")
        ax.set_xlabel("Breakout event (downstream order)")
        ax.set_ylabel("Normalized flux")
        if not records:
            ax.text(0.5, 0.5, "No breakout events", ha="center", va="center")
            ax.set_axis_off()
            return
        positions = np.arange(len(records), dtype=float)
        source_flux = max(float(cave_network.config.source_flux), 1e-9)
        before = np.asarray([float(item["parent_before"]) for item in records]) / source_flux
        after = np.asarray([float(item["parent_after"]) for item in records]) / source_flux
        branch = np.asarray([float(item["branch_flux"]) for item in records]) / source_flux
        returned = np.asarray([float(item["returned_flux"]) for item in records]) / source_flux
        ax.plot(positions, before, color="#0f766e", marker="o", ms=3, label="parent before")
        ax.plot(positions, after, color="#64748b", marker=".", label="parent after")
        ax.bar(positions, branch, color="#f97316", alpha=0.48, label="branch allocation")
        ax.bar(positions, returned, color="#38bdf8", alpha=0.72, label="returned at merge")
        ax.axhline(
            cave_network.config.lobe_growth.minimum_viable_flux_fraction,
            color="#dc2626",
            linestyle="--",
            linewidth=1.0,
            label="survival threshold",
        )
        ax.set_xticks(
            positions,
            [str(item["path_id"]).replace("lobe_", "") for item in records],
        )
        ax.legend(fontsize=7, ncols=2)
        ax.grid(True, axis="y", alpha=0.2)

    @classmethod
    def _draw_breakout_process(cls, *, ax, cave_network: CaveNetwork) -> None:
        records = cls._breakout_records(cave_network)
        ax.set_title("Breakout Cause And Survival")
        ax.set_xlabel("Process opportunity score")
        ax.set_ylabel("Allocated parent-flux fraction")
        if not records:
            ax.text(0.5, 0.5, "No breakout events", ha="center", va="center")
            ax.set_axis_off()
            return
        trigger_colors = {
            "capacity_overflow": "#dc2626",
            "margin_avulsion": "#8b5cf6",
            "bend_overflow": "#f59e0b",
            "seeded_blockage": "#475569",
        }
        outcome_markers = {
            "coalesced": "o",
            "vertically_captured": "D",
            "thermally_abandoned": "X",
            "stranded": "s",
        }
        seen: set[str] = set()
        for item in records:
            trigger = str(item["trigger"])
            outcome = str(item["outcome"])
            label = trigger.replace("_", " ") if trigger not in seen else "_nolegend_"
            seen.add(trigger)
            ax.scatter(
                [float(item["score"])],
                [float(item["split_fraction"])],
                color=trigger_colors.get(trigger, "#64748b"),
                marker=outcome_markers.get(outcome, "o"),
                s=42,
                alpha=0.82,
                label=label,
            )
        ax.legend(fontsize=7, title="dominant trigger")
        ax.grid(True, alpha=0.2)

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
            if segment.kind in {"spur", "abandoned_lobe", "stalled_lobe"} or not segment.points:
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
