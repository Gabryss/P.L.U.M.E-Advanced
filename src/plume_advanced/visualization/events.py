"""Visualization helpers for Stage-E geological events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plume_advanced.stages.events import GeologicalEventField
from plume_advanced.stages.network import CaveNetwork
from plume_advanced.stages.section_field import SectionField


@dataclass(frozen=True)
class GeologicalEventPlotConfig:
    """Figure settings for Stage-E event diagnostics."""

    figure_size: tuple[float, float] = (15.0, 10.0)
    dpi: int = 180


class GeologicalEventPlotter:
    """Render event placement diagnostics for the mesh-stage event layer."""

    def __init__(self, config: GeologicalEventPlotConfig | None = None) -> None:
        self.config = config or GeologicalEventPlotConfig()

    def render(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        event_field: GeologicalEventField,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size, constrained_layout=True)
        fig.suptitle("Stage E - Grounded Props and Structural Events", fontsize=16)

        self._draw_plan(axes[0, 0], cave_network, event_field)
        self._draw_profile(axes[0, 1], section_field, event_field)
        self._draw_counts(axes[1, 0], event_field)
        self._draw_radius_severity(axes[1, 1], event_field)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_plan(self, ax, cave_network: CaveNetwork, event_field: GeologicalEventField) -> None:
        ax.set_title("Plan Placement")
        for segment in cave_network.segments:
            if not segment.points:
                continue
            ax.plot(
                [point.x for point in segment.points],
                [point.y for point in segment.points],
                color="#64748b",
                linewidth=0.7,
                alpha=0.45,
            )
        colors = self._event_colors()
        for event in event_field.events:
            ax.scatter(
                [event.x],
                [event.y],
                s=10.0 + 5.5 * event.max_radius,
                color=colors.get(event.kind, "#334155"),
                alpha=0.76,
                linewidths=0.3,
                edgecolors="#0f172a",
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    def _draw_profile(
        self,
        ax,
        section_field: SectionField,
        event_field: GeologicalEventField,
    ) -> None:
        ax.set_title("Longitudinal Position / Height")
        for segment_field in section_field.segment_fields:
            if not segment_field.samples:
                continue
            ax.plot(
                [sample.y for sample in segment_field.samples],
                [sample.z for sample in segment_field.samples],
                color="#94a3b8",
                linewidth=0.6,
                alpha=0.45,
            )
        colors = self._event_colors()
        for event in event_field.events:
            ax.scatter(
                [event.y],
                [event.z],
                s=14.0 + 6.5 * event.radius_z,
                color=colors.get(event.kind, "#334155"),
                alpha=0.72,
            )
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.55)

    def _draw_counts(self, ax, event_field: GeologicalEventField) -> None:
        ax.set_title("Event Counts")
        summary = event_field.summary()
        kinds = ["rock", "boulder", "collapse", "choke", "infill"]
        values = [summary[f"{kind}_count"] for kind in kinds]
        colors = [self._event_colors()[kind] for kind in kinds]
        ax.bar(kinds, values, color=colors)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Count")

    def _draw_radius_severity(self, ax, event_field: GeologicalEventField) -> None:
        ax.set_title("Radius / Severity")
        if not event_field.events:
            ax.axis("off")
            return
        colors = self._event_colors()
        for kind in sorted({event.kind for event in event_field.events}):
            events = [event for event in event_field.events if event.kind == kind]
            ax.scatter(
                [event.max_radius for event in events],
                [event.severity for event in events],
                color=colors.get(kind, "#334155"),
                label=kind,
                alpha=0.72,
            )
        ax.set_xlabel("Max radius")
        ax.set_ylabel("Severity")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.55)

    @staticmethod
    def _event_colors() -> dict[str, str]:
        return {
            "rock": "#78716c",
            "boulder": "#44403c",
            "collapse": "#b45309",
            "choke": "#dc2626",
            "infill": "#ca8a04",
        }


__all__ = ["GeologicalEventPlotConfig", "GeologicalEventPlotter"]
