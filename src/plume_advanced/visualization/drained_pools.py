"""Focused diagnostics for sparse drained-lava pool rooms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plume_advanced.stages.network import CaveJunction, CaveNetwork
from plume_advanced.stages.section_field import SectionField, SectionSample


@dataclass(frozen=True)
class DrainedPoolPlotConfig:
    """Figure settings for drained-room diagnostics."""

    figure_size: tuple[float, float] = (15.0, 11.0)
    dpi: int = 180


class DrainedPoolPlotter:
    """Render planform, transition, section, and dimension evidence."""

    def __init__(self, config: DrainedPoolPlotConfig | None = None) -> None:
        self.config = config or DrainedPoolPlotConfig()

    def render(
        self,
        network: CaveNetwork,
        sections: SectionField,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pools = tuple(
            junction
            for junction in network.junctions
            if junction.metadata.get("chamber_type") == "drained_lava_pool"
        )
        samples_by_pool = {pool.junction_id: self._pool_samples(pool, sections) for pool in pools}
        figure, axes = plt.subplots(
            2,
            2,
            figsize=self.config.figure_size,
            constrained_layout=True,
        )
        self._draw_plan(axes[0, 0], network, pools)
        self._draw_transitions(axes[0, 1], pools, samples_by_pool)
        self._draw_sections(axes[1, 0], pools, samples_by_pool)
        self._draw_dimensions(axes[1, 1], pools)
        figure.suptitle(
            "Drained-lava pool diagnostics — sparse, flow-aligned room endmembers",
            fontsize=14,
        )
        figure.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(figure)
        return output

    @staticmethod
    def _pool_samples(
        pool: CaveJunction,
        sections: SectionField,
    ) -> tuple[tuple[SectionSample, float], ...]:
        records: list[tuple[SectionSample, float]] = []
        for field in sections.segment_fields:
            for sample in field.samples:
                room_weight = max(
                    (
                        influence.room_weight
                        for influence in sample.junction_influences
                        if influence.junction_id == pool.junction_id
                        and influence.chamber_type == "drained_lava_pool"
                    ),
                    default=0.0,
                )
                if room_weight > 0.0:
                    distance = math.hypot(sample.x - pool.center_x, sample.y - pool.center_y)
                    records.append((sample, math.copysign(distance, sample.segment_arc_length)))
        return tuple(records)

    @staticmethod
    def _pool_angle(network: CaveNetwork, pool: CaveJunction) -> float:
        candidates = [
            segment
            for segment in network.segments
            if segment.segment_id in pool.segment_ids and len(segment.points) >= 2
        ]
        if not candidates:
            return 0.0
        segment = min(
            candidates,
            key=lambda item: min(
                math.hypot(point.x - pool.center_x, point.y - pool.center_y)
                for point in item.points
            ),
        )
        if math.hypot(
            segment.points[0].x - pool.center_x,
            segment.points[0].y - pool.center_y,
        ) <= math.hypot(
            segment.points[-1].x - pool.center_x,
            segment.points[-1].y - pool.center_y,
        ):
            first, second = segment.points[0], segment.points[1]
        else:
            first, second = segment.points[-2], segment.points[-1]
        return math.degrees(math.atan2(second.y - first.y, second.x - first.x))

    def _draw_plan(
        self,
        axis,
        network: CaveNetwork,
        pools: tuple[CaveJunction, ...],
    ) -> None:
        from matplotlib.patches import Ellipse

        for segment in network.segments:
            is_pool = segment.metadata.get("chamber_type") == "drained_lava_pool"
            axis.plot(
                [point.x for point in segment.points],
                [point.y for point in segment.points],
                color="#f97316" if is_pool else "#64748b",
                linewidth=2.2 if is_pool else 0.7,
                alpha=0.9 if is_pool else 0.45,
            )
        for pool in pools:
            metadata = pool.metadata
            patch = Ellipse(
                (pool.center_x, pool.center_y),
                width=float(metadata.get("pool_length_m", 0.0)),
                height=float(metadata.get("pool_width_m", 0.0)),
                angle=self._pool_angle(network, pool),
                facecolor="#fb923c33",
                edgecolor="#ea580c",
                linewidth=1.5,
            )
            axis.add_patch(patch)
            axis.text(
                pool.center_x, pool.center_y, str(metadata.get("pool_id", "pool")), fontsize=7
            )
        axis.set_title("Planform with requested room footprints")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_aspect("equal")

    @staticmethod
    def _draw_transitions(axis, pools, samples_by_pool) -> None:
        for index, pool in enumerate(pools):
            records = samples_by_pool[pool.junction_id]
            if not records:
                continue
            segment_ids = sorted({record[0].segment_id for record in records})
            for branch_index, segment_id in enumerate(segment_ids):
                ordered = sorted(
                    (record for record in records if record[0].segment_id == segment_id),
                    key=lambda item: item[1],
                )
                distance = [item[1] for item in ordered]
                label_prefix = (
                    str(pool.metadata.get("pool_id", index))
                    if branch_index == 0
                    else "_nolegend_"
                )
                axis.plot(
                    distance,
                    [item[0].tube_width for item in ordered],
                    color=f"C{index % 10}",
                    alpha=0.82,
                    label=f"{label_prefix} width",
                )
                axis.plot(
                    distance,
                    [item[0].tube_height for item in ordered],
                    color=f"C{index % 10}",
                    linestyle="--",
                    alpha=0.82,
                    label=f"{label_prefix} height",
                )
        axis.set_title("Branch-wise graded section transitions")
        axis.set_xlabel("Plan distance from room centre (m)")
        axis.set_ylabel("Section size (m)")
        if pools:
            axis.legend(fontsize=7)

    @staticmethod
    def _draw_sections(axis, pools, samples_by_pool) -> None:
        chosen = []
        for pool in pools:
            records = samples_by_pool[pool.junction_id]
            if records:
                chosen.append(max(records, key=lambda item: item[0].tube_width)[0])
        for index, sample in enumerate(chosen):
            contour = np.asarray(sample.profile_points, dtype=float)
            axis.plot(
                contour[:, 0],
                contour[:, 1],
                color=f"C{index % 10}",
                linewidth=1.6,
                label=f"room {index + 1}: {sample.tube_width:.1f} × {sample.tube_height:.1f} m",
            )
        axis.axhline(0.0, color="#94a3b8", linewidth=0.5, alpha=0.4)
        axis.set_title("Generated room cross-section endmembers")
        axis.set_xlabel("Lateral offset (m)")
        axis.set_ylabel("Vertical offset (m)")
        axis.set_aspect("equal", adjustable="datalim")
        if chosen:
            axis.legend(fontsize=7)

    @staticmethod
    def _draw_dimensions(axis, pools: tuple[CaveJunction, ...]) -> None:
        if not pools:
            axis.text(0.5, 0.5, "No drained pools selected for this seed", ha="center")
            axis.set_axis_off()
            return
        labels = [str(pool.metadata.get("pool_id", index)) for index, pool in enumerate(pools)]
        positions = np.arange(len(pools), dtype=float)
        width = 0.24
        for offset, key, label, color in (
            (-width, "pool_length_m", "length", "#0f766e"),
            (0.0, "pool_width_m", "width", "#f97316"),
            (width, "pool_depth_m", "depth control", "#7c3aed"),
        ):
            axis.bar(
                positions + offset,
                [float(pool.metadata.get(key, 0.0)) for pool in pools],
                width,
                label=label,
                color=color,
            )
        for index, pool in enumerate(pools):
            axis.text(
                index,
                0.2,
                f"{pool.metadata.get('process_cause', 'unknown')}\n"
                f"outlet ratio {float(pool.metadata.get('pool_outlet_ratio', 0.0)):.1f}×",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=7,
                color="#111827",
            )
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_ylabel("Metres")
        axis.set_title("Room dimensions and process provenance")
        axis.legend(fontsize=8)


__all__ = ["DrainedPoolPlotConfig", "DrainedPoolPlotter"]
