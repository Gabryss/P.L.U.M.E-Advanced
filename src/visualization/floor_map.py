"""User-facing plan and intrinsic-atlas rendering for the cave floor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stages.floor_map import FloorAtlas


class FloorMapPlotter:
    """Render a conventional plan view beside the topology-safe floor atlas."""

    def render(
        self,
        floor_atlas: FloorAtlas,
        output_path: str | Path,
        event_field: Any | None = None,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cells = floor_atlas.cells
        figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        if not cells:
            for axis in axes:
                axis.text(0.5, 0.5, "No mapped floor cells", ha="center", va="center")
                axis.set_axis_off()
            figure.savefig(output, dpi=180)
            plt.close(figure)
            return output

        elevation = np.asarray([cell.z for cell in cells], dtype=float)
        plan = axes[0].scatter(
            [cell.x for cell in cells],
            [cell.y for cell in cells],
            c=elevation,
            s=8,
            cmap="terrain",
            linewidths=0,
        )
        figure.colorbar(plan, ax=axes[0], label="Floor elevation (m)")
        axes[0].set_title("World plan view (may contain vertical overlaps)")
        axes[0].set_xlabel("World X (m)")
        axes[0].set_ylabel("World Y (m)")
        axes[0].set_aspect("equal", adjustable="box")

        atlas = axes[1].scatter(
            [cell.atlas_x_m for cell in cells],
            [cell.atlas_y_m for cell in cells],
            c=[cell.clearance_m for cell in cells],
            s=8,
            cmap="viridis",
            linewidths=0,
        )
        figure.colorbar(atlas, ax=axes[1], label="Measured clearance (m)")
        axes[1].set_title("Intrinsic floor atlas (one band per segment)")
        axes[1].set_xlabel("Distance along segment (m)")
        axes[1].set_ylabel("Segment band + lateral offset (m)")

        if event_field is not None:
            props = [
                event
                for event in getattr(event_field, "events", ())
                if event.kind in {"rock", "boulder"}
            ]
            if props:
                axes[0].scatter(
                    [event.x for event in props],
                    [event.y for event in props],
                    marker="x",
                    s=22,
                    c="#d64045",
                    label="rocks / boulders",
                )
                axes[0].legend(loc="best")
                cell_lookup = {cell.cell_id: cell for cell in cells}
                mapped = [
                    (event, cell_lookup.get(event.floor_cell_id))
                    for event in props
                ]
                mapped = [(event, cell) for event, cell in mapped if cell is not None]
                if mapped:
                    axes[1].scatter(
                        [cell.atlas_x_m for _event, cell in mapped],
                        [cell.atlas_y_m for _event, cell in mapped],
                        marker="x",
                        s=22,
                        c="#d64045",
                    )

        summary = floor_atlas.summary()
        figure.suptitle(
            "PLUME cave-floor map · "
            f"{int(summary['cell_count'])} cells · "
            f"{int(summary['segment_count'])} segment bands"
        )
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return output
