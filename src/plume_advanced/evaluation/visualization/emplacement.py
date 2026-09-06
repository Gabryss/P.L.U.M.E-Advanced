"""Deterministic phase-activity diagrams for Stage-B emplacement history."""

from __future__ import annotations

from pathlib import Path

from plume_advanced.evaluation.metrics.emplacement import emplacement_metrics
from plume_advanced.stages.network import CaveNetwork


def render_emplacement_phase_activity(
    network: CaveNetwork,
    output_path: str | Path,
    *,
    dpi: int = 180,
) -> Path:
    """Render phase activity and lobe lifetimes without requiring a host field.

    The figure is intentionally based only on canonical network records, so it
    can be regenerated from saved Stage-B artifacts.  Missing phase metadata is
    rendered as an explicit unavailable panel rather than guessed history.
    """

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = emplacement_metrics(network)
    phases = diagnostics["phase_activity"]
    phase_available = bool(diagnostics["phase_metadata_available"])
    records = []
    for segment_index, segment in enumerate(network.segments):
        metadata = dict(segment.metadata or {})
        path_id = metadata.get("lobe_path_id", metadata.get("path_id"))
        if path_id is None:
            origin = str(metadata.get("formation_origin", segment.kind))
            path_id = (
                f"{origin}_system"
                if origin in {"backbone", "source_feeder", "spur"}
                else f"segment_{segment.segment_id}_{segment_index}"
            )
        if any(record["path_id"] == str(path_id) for record in records):
            continue
        birth = metadata.get("birth_phase", 0)
        death = metadata.get("death_phase", diagnostics["phase_count"] - 1)
        try:
            birth = int(birth)
            death = max(birth, int(death))
        except (TypeError, ValueError):
            birth, death = 0, diagnostics["phase_count"] - 1
        state = str(metadata.get("formation_state", metadata.get("termination", "unknown")))
        records.append({"path_id": str(path_id), "birth": birth, "death": death, "state": state})
    records.sort(key=lambda record: (record["birth"], record["path_id"]))

    figure_height = max(7.0, 3.8 + 0.32 * len(records))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, figure_height),
        constrained_layout=True,
    )
    axis = axes[0]
    if not phase_available:
        axis.text(0.5, 0.5, "Phase metadata unavailable", ha="center", va="center")
        axis.set_axis_off()
    else:
        phase_numbers = [row["phase"] for row in phases]
        active_paths = [row["active_path_count"] for row in phases]
        active_segments = [row["active_segment_count"] for row in phases]
        allocated = [row["allocated_flux"] for row in phases]
        returned = [row["returned_flux"] for row in phases]
        axis.step(phase_numbers, active_paths, where="mid", linewidth=2.2, label="active paths")
        axis.step(
            phase_numbers, active_segments, where="mid", linewidth=1.5, label="active segments"
        )
        axis.set_ylabel("Active count")
        axis.set_xlabel("Emplacement phase")
        secondary = axis.twinx()
        secondary.plot(
            phase_numbers, allocated, color="#f97316", marker="o", label="allocated flux"
        )
        secondary.plot(phase_numbers, returned, color="#22c55e", marker="s", label="returned flux")
        secondary.set_ylabel("Flux budget")
        handles, labels = axis.get_legend_handles_labels()
        handles2, labels2 = secondary.get_legend_handles_labels()
        axis.legend(handles + handles2, labels + labels2, loc="upper right", fontsize=8)
        axis.set_title("Phase-by-phase emplacement activity and flux budget")
        axis.grid(True, alpha=0.2)

    axis = axes[1]
    if not records:
        axis.text(0.5, 0.5, "No path records", ha="center", va="center")
        axis.set_axis_off()
    else:
        colors = {
            "coalesced": "#38bdf8",
            "vertically_captured": "#a855f7",
            "thermally_abandoned": "#f97316",
            "stranded": "#f59e0b",
            "stalled": "#f59e0b",
        }
        for row, record in enumerate(records):
            width = record["death"] - record["birth"] + 1
            axis.barh(
                row,
                width,
                left=record["birth"] - 0.5,
                height=0.65,
                color=colors.get(record["state"], "#64748b"),
                alpha=0.85,
            )
        axis.set_yticks(
            range(len(records)),
            [record["path_id"] for record in records],
            fontsize=8,
        )
        axis.set_xlabel("Emplacement phase")
        axis.set_ylabel("Path")
        axis.set_title("Path survival / retirement timeline")
        axis.grid(True, axis="x", alpha=0.2)
        axis.invert_yaxis()

    figure.suptitle("Stage-B staged emplacement diagnostics")
    figure.savefig(output, dpi=dpi, metadata={"Software": "PLUME evaluation"})
    plt.close(figure)
    return output


__all__ = ["render_emplacement_phase_activity"]
