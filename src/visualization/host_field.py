"""Visualization helpers for the stage-A host field."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from stages.host_field import HostField


@dataclass(frozen=True)
class HostFieldPlotConfig:
    """Figure settings shared by host-field visualizations."""

    figure_size: tuple[float, float] = (16.0, 11.0)
    dpi: int = 180


class HostFieldPlotter:
    """Render the first-stage host field into a reviewable artifact."""

    def __init__(self, config: HostFieldPlotConfig | None = None) -> None:
        self.config = config or HostFieldPlotConfig()

    def render(self, host_field: HostField, output_path: str | Path) -> Path:
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        panel_specs = [
            {
                "values": host_field.elevation,
                "title": "Terrain Elevation",
                "cmap": "terrain",
                "colorbar_label": "Elevation",
                "draw_contours": True,
            },
            {
                "values": host_field.slope_degrees,
                "title": "Slope Proxy",
                "cmap": "viridis",
                "colorbar_label": "Slope (deg)",
            },
            {
                "values": host_field.cover_thickness,
                "title": "Cover Thickness",
                "cmap": "plasma",
                "colorbar_label": "Cover",
            },
            {
                "values": host_field.roof_competence,
                "title": "Roof Competence",
                "cmap": "cividis",
                "colorbar_label": "Competence",
            },
            {
                "values": host_field.growth_cost,
                "title": "Growth Cost",
                "cmap": "magma_r",
                "colorbar_label": "Cost",
            },
        ]

        columns = 3
        rows = math.ceil(len(panel_specs) / columns)
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=self.config.figure_size,
            constrained_layout=True,
        )
        fig.suptitle("Stage A - Host Field", fontsize=16)

        axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for ax, panel in zip(axes_flat, panel_specs, strict=False):
            self._draw_panel(
                ax=ax,
                host_field=host_field,
                values=panel["values"],
                title=panel["title"],
                cmap=panel["cmap"],
                colorbar_label=panel["colorbar_label"],
                draw_contours=panel.get("draw_contours", False),
            )

        remaining_axes = axes_flat[len(panel_specs) :]
        for ax in remaining_axes:
            ax.axis("off")

        summary = host_field.summary()
        summary_line = (
            f"Mean slope: {summary['slope_mean_deg']:.1f} deg | "
            f"Mean cover: {summary['cover_thickness_mean']:.1f} | "
            f"Mean roof competence: {summary['roof_competence_mean']:.2f} | "
            f"Mean growth cost: {summary['growth_cost_mean']:.2f}"
        )
        fig.text(0.5, 0.01, summary_line, ha="center", fontsize=10)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_panel(
        self,
        *,
        ax,
        host_field: HostField,
        values,
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
        ax.scatter(
            [host_field.config.seed_point[0]],
            [host_field.config.seed_point[1]],
            c="white",
            s=18,
            marker="x",
            linewidths=1.2,
        )

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

        if title == "Growth Cost":
            cost_contours = ax.contour(
                host_field.x_coords,
                host_field.y_coords,
                values,
                levels=6,
                colors="white",
                linewidths=0.7,
                alpha=0.8,
            )
            ax.clabel(cost_contours, fmt="%.2f", fontsize=7)

        colorbar = plt.colorbar(image, ax=ax, shrink=0.9)
        colorbar.set_label(colorbar_label)
