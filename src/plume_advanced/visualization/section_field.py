"""Visualization helpers for the stage-C section field."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plume_advanced.stages.network import CaveNetwork
from plume_advanced.stages.section_field import SectionField, SectionSample


@dataclass(frozen=True)
class SectionFieldPlotConfig:
    """Figure settings for section-field visualization."""

    figure_size: tuple[float, float] = (16.0, 14.0)
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

        fig, axes = plt.subplots(3, 2, figsize=self.config.figure_size, constrained_layout=True)
        fig.suptitle("Stage C - Section Field", fontsize=16)

        self._draw_plan_panel(ax=axes[0, 0], section_field=section_field)
        self._draw_vertical_profile_panel(
            ax=axes[0, 1],
            cave_network=cave_network,
            section_field=section_field,
        )
        self._draw_route_profile_panel(
            ax=axes[1, 0],
            cave_network=cave_network,
            section_field=section_field,
        )
        self._draw_cross_section_panel(
            ax=axes[1, 1],
            cave_network=cave_network,
            section_field=section_field,
        )
        self._draw_control_panel(
            ax=axes[2, 0],
            cave_network=cave_network,
            section_field=section_field,
        )
        self._draw_morphospace_panel(ax=axes[2, 1], section_field=section_field)

        summary = section_field.summary()
        fig.supxlabel(
            f"Segments: {int(summary['segment_field_count'])} | "
            f"Samples: {int(summary['sample_count'])} | "
            f"Dominant-route segments: {int(summary['dominant_route_segment_count'])} | "
            f"Width CV: {summary['width_coefficient_of_variation']:.2f} | "
            f"Shape change/100m: {summary['mean_shape_change_per_100m']:.2f}",
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
                color=plt.get_cmap("viridis")(width_ratio),
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
        route_samples = self._collect_route_samples(section_field)
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

    def _draw_vertical_profile_panel(
        self,
        *,
        ax,
        cave_network: CaveNetwork,
        section_field: SectionField,
    ) -> None:
        segment_lookup = {segment.segment_id: segment for segment in cave_network.segments}
        node_lookup = {node.node_id: node for node in cave_network.nodes}
        level_colors = {-2: "#6b21a8", -1: "#a855f7", 0: "#0891b2", 1: "#f59e0b", 2: "#dc2626"}
        surface_label_used = False
        level_labels: set[int] = set()
        for segment_field in section_field.segment_fields:
            segment = segment_lookup.get(segment_field.segment_id)
            if segment is None or not segment_field.samples:
                continue
            start_node = node_lookup[segment.start_node_id]
            end_node = node_lookup[segment.end_node_id]
            progress = np.asarray(
                [
                    sample.segment_arc_length / max(segment.total_length, 1e-9)
                    for sample in segment_field.samples
                ],
                dtype=float,
            )
            along = (1.0 - progress) * start_node.along_position + progress * end_node.along_position
            center_z = [sample.z for sample in segment_field.samples]
            surface_z = [sample.surface_z for sample in segment_field.samples]
            if not surface_label_used:
                ax.plot(along, surface_z, color="#64748b", linewidth=0.7, alpha=0.45, label="surface")
                surface_label_used = True
            else:
                ax.plot(along, surface_z, color="#94a3b8", linewidth=0.35, alpha=0.12)
            label = f"level {segment.z_level:+d}" if segment.z_level not in level_labels else "_nolegend_"
            level_labels.add(segment.z_level)
            ax.plot(
                along,
                center_z,
                color=level_colors.get(segment.z_level, "#334155"),
                linewidth=1.7 if bool(segment.metadata.get("vertical_capture", False)) else 0.85,
                alpha=0.86,
                label=label,
            )
        ax.set_title("Longitudinal Vertical Network Profile")
        ax.set_xlabel("Along-flow distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", fontsize=7)

    def _draw_cross_section_panel(
        self,
        *,
        ax,
        cave_network: CaveNetwork,
        section_field: SectionField,
    ) -> None:
        import matplotlib.pyplot as plt

        representative_samples = self._select_route_gradient_samples(
            cave_network,
            section_field,
            count=10,
        )
        if not representative_samples:
            ax.set_title("Cross-section Gradient")
            ax.axis("off")
            return

        columns = 5
        maximum_width = max(sample.tube_width for sample in representative_samples)
        maximum_height = max(sample.tube_height for sample in representative_samples)
        x_spacing = 1.35 * maximum_width
        y_spacing = 1.65 * maximum_height
        palette = plt.get_cmap("plasma")
        for index, sample in enumerate(representative_samples):
            column = index % columns
            row = 1 - index // columns
            center_x = column * x_spacing
            center_y = row * y_spacing
            color = palette(index / max(len(representative_samples) - 1, 1))
            x_coords = [point[0] + center_x for point in sample.profile_points]
            y_coords = [point[1] + center_y for point in sample.profile_points]
            ax.fill(x_coords, y_coords, color=color, alpha=0.14)
            ax.plot(x_coords, y_coords, color=color, linewidth=1.25)
            ax.text(
                center_x,
                center_y - 0.72 * maximum_height,
                f"{index / max(len(representative_samples) - 1, 1):.0%}",
                ha="center",
                va="top",
                fontsize=7,
            )
        ax.set_title("Natural Cross-section Gradient Along Dominant Route")
        ax.set_xlabel("Upstream → downstream progression")
        ax.set_ylabel("Actual section size")
        ax.set_aspect("equal")
        ax.set_yticks([])
        ax.set_xticks([])

    @staticmethod
    def _draw_morphospace_panel(*, ax, section_field: SectionField) -> None:
        import matplotlib.pyplot as plt

        samples = [
            sample
            for segment_field in section_field.segment_fields
            for sample in segment_field.samples
        ]
        if not samples:
            ax.set_title("Section Morphospace")
            ax.axis("off")
            return
        widths = np.asarray([sample.tube_width for sample in samples], dtype=float)
        ratios = np.asarray(
            [sample.tube_height / max(sample.tube_width, 1e-9) for sample in samples],
            dtype=float,
        )
        family_score = np.asarray(
            [sample.morphology_family_score for sample in samples], dtype=float
        )
        floor_relief = np.asarray(
            [1.0 - sample.floor_flatness for sample in samples], dtype=float
        )
        sizes = 10.0 + 22.0 * np.asarray(
            [abs(sample.lateral_skew) for sample in samples], dtype=float
        )
        scatter = ax.scatter(
            widths,
            ratios,
            c=family_score,
            s=sizes + 26.0 * floor_relief,
            cmap="coolwarm",
            alpha=0.58,
            linewidths=0.0,
        )
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.82)
        colorbar.set_label("Morphology family score")
        ax.set_title("Continuous Section Morphospace (family / floor relief)")
        ax.set_xlabel("Tube width (m)")
        ax.set_ylabel("Height / width")
        ax.grid(True, alpha=0.16)

    def _draw_control_panel(self, *, ax, cave_network: CaveNetwork, section_field: SectionField) -> None:
        route_samples = self._collect_route_samples(section_field)
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

    def _select_route_gradient_samples(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
        *,
        count: int,
    ) -> list[SectionSample]:
        route_samples = self._collect_route_samples(section_field)
        if not route_samples or count <= 0:
            return []
        selected_indices = np.linspace(
            0,
            len(route_samples) - 1,
            min(count, len(route_samples)),
            dtype=int,
        )
        return [route_samples[int(index)] for index in selected_indices]
