"""User-facing geological plan and intrinsic cave-floor atlas rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plume_advanced.stages.floor_map import FloorAtlas, FloorCell

_GEOLOGY_COLORS = {
    "bare_basalt": "#4b5563",
    "sediment": "#d6b56d",
    "breakdown": "#7f1d1d",
    "debris": "#b45309",
    "constriction": "#7c3aed",
}


class FloorMapPlotter:
    """Render final cave morphology, geology, clearance, and intrinsic bands."""

    def render(
        self,
        floor_atlas: FloorAtlas,
        output_path: str | Path,
        event_field: Any | None = None,
        cave_network: Any | None = None,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cells = floor_atlas.cells
        figure, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
        if not cells:
            for axis in axes.flat:
                axis.text(0.5, 0.5, "No mapped floor cells", ha="center", va="center")
                axis.set_axis_off()
            figure.savefig(output, dpi=180)
            plt.close(figure)
            return output

        self._draw_world_scalar(
            figure,
            axes[0, 0],
            cells,
            values=[cell.z for cell in cells],
            title="Final cave plan · floor elevation",
            label="Floor elevation (m)",
            cmap="terrain",
            cave_network=cave_network,
        )
        self._draw_world_scalar(
            figure,
            axes[0, 1],
            cells,
            values=[cell.clearance_m for cell in cells],
            title="Final cave plan · measured clearance",
            label="Floor-to-roof clearance (m)",
            cmap="viridis",
            cave_network=cave_network,
        )
        self._draw_geology(
            axes[1, 0],
            cells,
            event_field=event_field,
            cave_network=cave_network,
        )
        self._draw_intrinsic(figure, axes[1, 1], cells, event_field)

        summary = floor_atlas.summary()
        figure.suptitle(
            "PLUME final geological cave map · "
            f"{int(summary['cell_count'])} valid floor cells · "
            f"{int(summary['invalidated_cell_count'])} post-event invalidated · "
            f"{int(summary['segment_count'])} segment bands",
            fontsize=15,
        )
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return output

    def _draw_world_scalar(
        self,
        figure: Any,
        axis: Any,
        cells: tuple[Any, ...],
        *,
        values: list[float],
        title: str,
        label: str,
        cmap: str,
        cave_network: Any | None,
    ) -> None:
        plot = axis.scatter(
            [cell.x for cell in cells],
            [cell.y for cell in cells],
            c=values,
            s=9,
            cmap=cmap,
            linewidths=0,
        )
        figure.colorbar(plot, ax=axis, label=label)
        self._draw_network_context(axis, cave_network)
        axis.set_title(title)
        axis.set_xlabel("World X (m)")
        axis.set_ylabel("World Y (m)")
        axis.set_aspect("equal", adjustable="box")

    def _draw_geology(
        self,
        axis: Any,
        cells: tuple[Any, ...],
        *,
        event_field: Any | None,
        cave_network: Any | None,
    ) -> None:
        for geology_class, color in _GEOLOGY_COLORS.items():
            selected = [cell for cell in cells if cell.geology_class == geology_class]
            if not selected:
                continue
            axis.scatter(
                [cell.x for cell in selected],
                [cell.y for cell in selected],
                c=color,
                s=9 + 13 * np.asarray([cell.event_influence for cell in selected]),
                linewidths=0,
                label=geology_class.replace("_", " "),
            )
        self._draw_network_context(axis, cave_network)
        self._draw_event_markers(axis, event_field)
        axis.set_title("Post-event geological floor classification")
        axis.set_xlabel("World X (m)")
        axis.set_ylabel("World Y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best", fontsize=8)

    def _draw_intrinsic(
        self,
        figure: Any,
        axis: Any,
        cells: tuple[Any, ...],
        event_field: Any | None,
    ) -> None:
        plot = axis.scatter(
            [cell.atlas_x_m for cell in cells],
            [cell.atlas_y_m for cell in cells],
            c=[cell.clearance_m for cell in cells],
            s=8,
            cmap="viridis",
            linewidths=0,
        )
        figure.colorbar(plot, ax=axis, label="Measured clearance (m)")
        cell_lookup = {cell.cell_id: cell for cell in cells}
        props = [
            event
            for event in getattr(event_field, "events", ())
            if event.kind in {"rock", "boulder"}
        ]
        mapped: list[tuple[Any, FloorCell]] = []
        for event in props:
            cell = cell_lookup.get(event.floor_cell_id)
            if cell is not None:
                mapped.append((event, cell))
        if mapped:
            axis.scatter(
                [cell.atlas_x_m for _event, cell in mapped],
                [cell.atlas_y_m for _event, cell in mapped],
                marker="x",
                s=20,
                c=[
                    "#ef4444"
                    if event.cluster_parent_event_id >= 0
                    else "#f59e0b"
                    for event, _cell in mapped
                ],
                linewidths=0.9,
            )
        axis.set_title("Topology-safe intrinsic atlas · debris overlay")
        axis.set_xlabel("Distance along segment (m)")
        axis.set_ylabel("Segment band + lateral offset (m)")

    @staticmethod
    def _draw_network_context(axis: Any, cave_network: Any | None) -> None:
        if cave_network is None:
            return
        for segment in cave_network.segments:
            if len(segment.points) < 2:
                continue
            axis.plot(
                [point.x for point in segment.points],
                [point.y for point in segment.points],
                color="#111827",
                linewidth=0.45,
                alpha=0.30,
                zorder=0,
            )
        degree: dict[int, int] = {}
        for segment in cave_network.segments:
            degree[segment.start_node_id] = degree.get(segment.start_node_id, 0) + 1
            degree[segment.end_node_id] = degree.get(segment.end_node_id, 0) + 1
        termini = [node for node in cave_network.nodes if degree.get(node.node_id, 0) == 1]
        chambers = [node for node in cave_network.nodes if node.kind == "chamber"]
        if termini:
            axis.scatter(
                [node.x for node in termini],
                [node.y for node in termini],
                marker="s",
                facecolors="none",
                edgecolors="#0f172a",
                s=38,
                linewidths=1.0,
                label="generated terminus",
            )
        if chambers:
            axis.scatter(
                [node.x for node in chambers],
                [node.y for node in chambers],
                marker="*",
                c="#facc15",
                edgecolors="#713f12",
                s=75,
                linewidths=0.7,
                label="chamber",
            )

    @staticmethod
    def _draw_event_markers(axis: Any, event_field: Any | None) -> None:
        structural = [
            event
            for event in getattr(event_field, "events", ())
            if event.kind in {"collapse", "choke", "infill"}
        ]
        if structural:
            marker_lookup = {"collapse": "v", "choke": "D", "infill": "^"}
            color_lookup = {
                "collapse": "#dc2626",
                "choke": "#7c3aed",
                "infill": "#ca8a04",
            }
            for kind in ("collapse", "choke", "infill"):
                selected = [event for event in structural if event.kind == kind]
                if selected:
                    axis.scatter(
                        [event.x for event in selected],
                        [event.y for event in selected],
                        marker=marker_lookup[kind],
                        c=color_lookup[kind],
                        s=44,
                        edgecolors="white",
                        linewidths=0.5,
                        label=kind,
                    )
        clustered = [
            event
            for event in getattr(event_field, "events", ())
            if event.cluster_parent_event_id >= 0
        ]
        if clustered:
            axis.scatter(
                [event.x for event in clustered],
                [event.y for event in clustered],
                marker="x",
                c="#ef4444",
                s=24,
                linewidths=0.9,
                label="collapse-clustered debris",
            )
