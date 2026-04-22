"""Visualization helpers for the stage-C section field."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stages.network import CaveNetwork
from stages.section_field import SectionField, SectionSample


@dataclass(frozen=True)
class SectionFieldPlotConfig:
    """Figure settings for section-field visualization."""

    figure_size: tuple[float, float] = (15.0, 11.0)
    dpi: int = 180


class SectionFieldPlotter:
    """Render a reviewable artifact for geometry-ready section samples."""

    def __init__(self, config: SectionFieldPlotConfig | None = None) -> None:
        self.config = config or SectionFieldPlotConfig()

    def render(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size, constrained_layout=True)
        fig.suptitle("Stage C - Section Field", fontsize=16)

        self._draw_plan_panel(ax=axes[0, 0], section_field=section_field)
        self._draw_route_profile_panel(
            ax=axes[0, 1],
            cave_network=cave_network,
            section_field=section_field,
        )
        self._draw_cross_section_panel(ax=axes[1, 0], section_field=section_field)
        self._draw_control_panel(
            ax=axes[1, 1],
            cave_network=cave_network,
            section_field=section_field,
        )

        summary = section_field.summary()
        fig.text(
            0.5,
            0.01,
            (
                f"Segments: {int(summary['segment_field_count'])} | "
                f"Samples: {int(summary['sample_count'])} | "
                f"Dominant-route segments: {int(summary['dominant_route_segment_count'])} | "
                f"Max junction blend: {summary['max_junction_blend_weight']:.2f}"
            ),
            ha="center",
            fontsize=10,
        )

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_plan_panel(self, *, ax, section_field: SectionField) -> None:
        import matplotlib.pyplot as plt

        segment_fields = [field for field in section_field.segment_fields if field.samples]
        if not segment_fields:
            ax.set_title("Plan View")
            ax.axis("off")
            return

        widths = [
            sample.tube_width
            for segment_field in segment_fields
            for sample in segment_field.samples
        ]
        width_min = min(widths)
        width_span = max(max(widths) - width_min, 1e-6)
        for segment_field in segment_fields:
            x_values = [sample.x for sample in segment_field.samples]
            y_values = [sample.y for sample in segment_field.samples]
            mean_width = float(np.mean([sample.tube_width for sample in segment_field.samples]))
            width_ratio = (mean_width - width_min) / width_span
            ax.plot(
                x_values,
                y_values,
                color=plt.cm.viridis(width_ratio),
                linewidth=1.0 + 1.4 * width_ratio,
                alpha=0.92,
            )
        for segment_field in segment_fields:
            for sample in segment_field.samples:
                if sample.junction_blend_weight < 0.42:
                    continue
                ax.scatter(
                    [sample.x],
                    [sample.y],
                    c="#fb7185",
                    s=6 + 20 * sample.junction_blend_weight,
                    alpha=0.35,
                    linewidths=0.0,
                )
        ax.set_title("Section Width Plan View")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    def _draw_route_profile_panel(self, *, ax, cave_network: CaveNetwork, section_field: SectionField) -> None:
        route_samples = self._collect_route_samples(cave_network, section_field)
        if not route_samples:
            ax.set_title("Dominant Route Profile")
            ax.axis("off")
            return

        accumulated = 0.0
        distances: list[float] = []
        widths: list[float] = []
        heights: list[float] = []
        elevations: list[float] = []
        previous_sample: SectionSample | None = None
        for sample in route_samples:
            if previous_sample is not None:
                accumulated += float(
                    np.linalg.norm(
                        np.array([sample.x - previous_sample.x, sample.y - previous_sample.y, sample.z - previous_sample.z])
                    )
                )
            distances.append(accumulated)
            widths.append(sample.tube_width)
            heights.append(sample.tube_height)
            elevations.append(sample.z)
            previous_sample = sample

        ax.set_title("Dominant Route Width / Height")
        ax.plot(distances, widths, color="#0f766e", linewidth=2.0, label="Width")
        ax.plot(distances, heights, color="#f59e0b", linewidth=2.0, label="Height")
        ax.set_xlabel("Route distance")
        ax.set_ylabel("Section size")
        secondary_axis = ax.twinx()
        secondary_axis.plot(distances, elevations, color="#334155", linewidth=1.6, linestyle="--", label="Elevation")
        secondary_axis.set_ylabel("Elevation")
        lines = ax.get_lines() + secondary_axis.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

    def _draw_cross_section_panel(self, *, ax, section_field: SectionField) -> None:
        representative_samples = self._select_representative_samples(section_field)
        if not representative_samples:
            ax.set_title("Representative Cross Sections")
            ax.axis("off")
            return

        offsets = np.linspace(-2.8, 2.8, len(representative_samples))
        palette = ["#22c55e", "#06b6d4", "#f59e0b", "#ef4444"]
        for offset, sample, color in zip(offsets, representative_samples, palette, strict=False):
            x_coords = [point[0] + offset * sample.tube_width * 0.28 for point in sample.profile_points]
            y_coords = [point[1] for point in sample.profile_points]
            ax.fill(x_coords, y_coords, color=color, alpha=0.18)
            ax.plot(x_coords, y_coords, color=color, linewidth=1.5)
        ax.set_title("Representative Lava-Tube Sections")
        ax.set_xlabel("Local width")
        ax.set_ylabel("Local height")
        ax.set_aspect("equal")

    def _draw_control_panel(self, *, ax, cave_network: CaveNetwork, section_field: SectionField) -> None:
        route_samples = self._collect_route_samples(cave_network, section_field)
        if not route_samples:
            ax.set_title("Section Controls")
            ax.axis("off")
            return

        distances = [index for index, _sample in enumerate(route_samples)]
        ax.set_title("Section Controls Along Dominant Route")
        ax.plot(
            distances,
            [sample.floor_flatness for sample in route_samples],
            color="#8b5cf6",
            linewidth=2.0,
            label="Floor flatness",
        )
        ax.plot(
            distances,
            [sample.roof_arch for sample in route_samples],
            color="#ec4899",
            linewidth=2.0,
            label="Roof arch",
        )
        ax.plot(
            distances,
            [sample.junction_blend_weight for sample in route_samples],
            color="#ef4444",
            linewidth=1.8,
            linestyle="--",
            label="Junction blend",
        )
        ax.set_xlabel("Adaptive sample index")
        ax.set_ylabel("Control value")
        ax.legend(loc="best", fontsize=8)

    def _collect_route_samples(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
    ) -> list[SectionSample]:
        field_lookup = {
            segment_field.segment_id: segment_field
            for segment_field in section_field.segment_fields
        }
        samples: list[SectionSample] = []
        for segment_id in section_field.dominant_route_segment_ids:
            segment_field = field_lookup.get(segment_id)
            if segment_field is None:
                continue
            samples.extend(segment_field.samples)
        return samples

    def _select_representative_samples(
        self,
        section_field: SectionField,
    ) -> list[SectionSample]:
        all_samples = [
            sample
            for segment_field in section_field.segment_fields
            for sample in segment_field.samples
        ]
        if not all_samples:
            return []
        junction_heavy = max(all_samples, key=lambda sample: sample.junction_blend_weight)
        widest = max(all_samples, key=lambda sample: sample.tube_width)
        tallest = max(all_samples, key=lambda sample: sample.tube_height)
        median = all_samples[len(all_samples) // 2]
        selected: list[SectionSample] = []
        for sample in (median, junction_heavy, widest, tallest):
            if sample not in selected:
                selected.append(sample)
        return selected[:4]
