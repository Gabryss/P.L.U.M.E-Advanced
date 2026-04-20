"""Visualization helpers for the stage-B trunk graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stages.graph import TrunkGraph
from stages.host_field import HostField


@dataclass(frozen=True)
class TrunkGraphPlotConfig:
    """Figure settings for the stage-B trunk graph visualization."""

    figure_size: tuple[float, float] = (15.0, 11.0)
    dpi: int = 180


class TrunkGraphPlotter:
    """Render the stage-B trunk graph into a reviewable artifact."""

    def __init__(self, config: TrunkGraphPlotConfig | None = None) -> None:
        self.config = config or TrunkGraphPlotConfig()

    def render(
        self,
        host_field: HostField,
        trunk_graph: TrunkGraph,
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
        fig.suptitle("Stage B - Trunk Graph", fontsize=16)

        self._draw_map_panel(
            ax=axes[0, 0],
            host_field=host_field,
            values=host_field.elevation,
            trunk_graph=trunk_graph,
            title="Terrain And Trunk Path",
            cmap="terrain",
            colorbar_label="Elevation",
            draw_contours=True,
        )
        self._draw_map_panel(
            ax=axes[0, 1],
            host_field=host_field,
            values=host_field.growth_cost,
            trunk_graph=trunk_graph,
            title="Growth Cost And Trunk Path",
            cmap="magma_r",
            colorbar_label="Cost",
        )
        self._draw_map_panel(
            ax=axes[1, 0],
            host_field=host_field,
            values=host_field.roof_competence,
            trunk_graph=trunk_graph,
            title="Roof Competence And Trunk Path",
            cmap="cividis",
            colorbar_label="Competence",
        )
        self._draw_profile_panel(ax=axes[1, 1], trunk_graph=trunk_graph)

        summary = trunk_graph.summary()
        summary_line = (
            f"Points: {int(summary['point_count'])} | "
            f"Length: {summary['total_length']:.1f} | "
            f"Elevation drop: {summary['elevation_drop']:.1f} | "
            f"Mean growth cost: {summary['mean_growth_cost']:.2f}"
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
        trunk_graph: TrunkGraph,
        title: str,
        cmap: str,
        colorbar_label: str,
        draw_contours: bool = False,
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

        x_coords = [point.x for point in trunk_graph.points]
        y_coords = [point.y for point in trunk_graph.points]
        ax.plot(x_coords, y_coords, color="white", linewidth=2.2)
        ax.scatter([x_coords[0]], [y_coords[0]], c="cyan", s=28, marker="o")
        ax.scatter([x_coords[-1]], [y_coords[-1]], c="red", s=28, marker="o")

        if draw_contours:
            ax.contour(
                host_field.x_coords,
                host_field.y_coords,
                values,
                levels=8,
                colors="black",
                linewidths=0.45,
                alpha=0.45,
            )

        colorbar = plt.colorbar(image, ax=ax, shrink=0.9)
        colorbar.set_label(colorbar_label)

    def _draw_profile_panel(self, *, ax, trunk_graph: TrunkGraph) -> None:
        profile = trunk_graph.profile()

        ax.set_title("Longitudinal Profile")
        ax.set_xlabel("Arc Length")
        ax.set_ylabel("Elevation")
        elevation_line = ax.plot(
            profile["arc_length"],
            profile["elevation"],
            color="#2b4c7e",
            linewidth=2.2,
            label="Elevation",
        )[0]

        secondary_axis = ax.twinx()
        secondary_axis.set_ylabel("Normalized Proxy")
        competence_line = secondary_axis.plot(
            profile["arc_length"],
            profile["roof_competence"],
            color="#4a8c62",
            linewidth=1.7,
            label="Roof competence",
        )[0]
        cost_line = secondary_axis.plot(
            profile["arc_length"],
            profile["growth_cost"],
            color="#bb5a3c",
            linewidth=1.7,
            label="Growth cost",
        )[0]
        secondary_axis.set_ylim(0.0, 1.05)

        lines = [elevation_line, competence_line, cost_line]
        ax.legend(lines, [line.get_label() for line in lines], loc="best")
