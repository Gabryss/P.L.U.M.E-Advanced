"""Visualization helpers for the stage-B.1 mixed branching sub-stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stages.branching_models import (
    DOWNSTREAM_RECONNECT_LOOP,
    LOCAL_BYPASS_LOOP,
    SPUR,
    BranchMergeNetwork,
)
from stages.host_field import HostField


@dataclass(frozen=True)
class BranchMergePlotConfig:
    """Figure settings for the branch / merge visualization."""

    figure_size: tuple[float, float] = (15.0, 11.0)
    dpi: int = 180


class BranchMergePlotter:
    """Render the branch / merge sub-stage into a reviewable artifact."""

    BRANCH_COLORS = {
        LOCAL_BYPASS_LOOP: '#f97316',
        DOWNSTREAM_RECONNECT_LOOP: '#ec4899',
        SPUR: '#14b8a6',
    }

    def __init__(self, config: BranchMergePlotConfig | None = None) -> None:
        self.config = config or BranchMergePlotConfig()

    def render(
        self,
        host_field: HostField,
        branch_network: BranchMergeNetwork,
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
        fig.suptitle('Stage B.1 - Mixed Branch / Merge Review', fontsize=16)

        self._draw_map_panel(
            ax=axes[0, 0],
            host_field=host_field,
            values=host_field.elevation,
            branch_network=branch_network,
            title='Terrain With Mixed Branch Network',
            cmap='terrain',
            colorbar_label='Elevation',
            draw_contours=True,
        )
        self._draw_map_panel(
            ax=axes[0, 1],
            host_field=host_field,
            values=host_field.growth_cost,
            branch_network=branch_network,
            title='Growth Cost With Branch Network',
            cmap='magma_r',
            colorbar_label='Cost',
        )
        self._draw_map_panel(
            ax=axes[1, 0],
            host_field=host_field,
            values=host_field.roof_competence,
            branch_network=branch_network,
            title='Roof Competence With Branch Network',
            cmap='cividis',
            colorbar_label='Competence',
        )
        self._draw_profile_panel(ax=axes[1, 1], branch_network=branch_network)

        summary = branch_network.summary()
        summary_line = (
            f"Candidates: {int(summary['candidate_count'])} | "
            f"Branches: {int(summary['branch_count'])} | "
            f"Local loops: {int(summary['local_bypass_count'])} | "
            f"Downstream loops: {int(summary['downstream_reconnect_count'])} | "
            f"Spurs: {int(summary['spur_count'])}"
        )
        fig.text(0.5, 0.01, summary_line, ha='center', fontsize=10)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        return output

    def _draw_map_panel(
        self,
        *,
        ax,
        host_field: HostField,
        values,
        branch_network: BranchMergeNetwork,
        title: str,
        cmap: str,
        colorbar_label: str,
        draw_contours: bool = False,
    ) -> None:
        import matplotlib.pyplot as plt

        image = ax.imshow(
            values,
            extent=host_field.extent,
            origin='lower',
            cmap=cmap,
            aspect='equal',
        )
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        trunk_x = [point.x for point in branch_network.trunk_graph.points]
        trunk_y = [point.y for point in branch_network.trunk_graph.points]
        ax.plot(trunk_x, trunk_y, color='white', linewidth=2.2, label='Trunk')
        ax.scatter([trunk_x[0]], [trunk_y[0]], c='cyan', s=26, marker='o')
        ax.scatter([trunk_x[-1]], [trunk_y[-1]], c='red', s=26, marker='o')

        for candidate in branch_network.candidates:
            source = branch_network.trunk_graph.points[candidate.trunk_index]
            ax.scatter([source.x], [source.y], c='#38bdf8', s=22, marker='s', alpha=0.7)

        for branch in branch_network.branches:
            branch_x = [point.x for point in branch.points]
            branch_y = [point.y for point in branch.points]
            color = self.BRANCH_COLORS.get(branch.branch_kind, '#f97316')
            ax.plot(branch_x, branch_y, color=color, linewidth=1.9)
            ax.scatter([branch_x[-1]], [branch_y[-1]], c=color, s=24, marker='o')
            source = branch_network.trunk_graph.points[branch.source_trunk_index]
            ax.scatter([source.x], [source.y], c=color, s=32, marker='s', edgecolors='black', linewidths=0.4)
            if branch.target_trunk_index is not None:
                target = branch_network.trunk_graph.points[branch.target_trunk_index]
                ax.scatter([target.x], [target.y], c=color, s=34, marker='D', edgecolors='white', linewidths=0.5)
            if branch.merge_event is not None:
                merge_target = branch_network.trunk_graph.points[branch.merge_event.target_trunk_index]
                ax.scatter([merge_target.x], [merge_target.y], c=color, s=46, marker='*', edgecolors='black', linewidths=0.4)

        if draw_contours:
            ax.contour(
                host_field.x_coords,
                host_field.y_coords,
                values,
                levels=8,
                colors='black',
                linewidths=0.45,
                alpha=0.45,
            )

        colorbar = plt.colorbar(image, ax=ax, shrink=0.9)
        colorbar.set_label(colorbar_label)

    def _draw_profile_panel(self, *, ax, branch_network: BranchMergeNetwork) -> None:
        trunk_profile = branch_network.trunk_graph.profile()

        ax.set_title('Branch Elevation Profiles')
        ax.set_xlabel('Arc Length')
        ax.set_ylabel('Elevation')
        ax.plot(
            trunk_profile['arc_length'],
            trunk_profile['elevation'],
            color='#1d4ed8',
            linewidth=2.1,
            label='Trunk',
        )

        if branch_network.branches:
            for branch in branch_network.branches:
                branch_arc = [point.arc_length for point in branch.points]
                branch_elevation = [point.elevation for point in branch.points]
                label = f"Branch {branch.branch_id} ({branch.branch_kind})"
                ax.plot(
                    branch_arc,
                    branch_elevation,
                    linewidth=1.5,
                    color=self.BRANCH_COLORS.get(branch.branch_kind, '#f97316'),
                    label=label,
                )
        else:
            ax.text(0.5, 0.5, 'No branches generated', ha='center', va='center', transform=ax.transAxes)

        ax.legend(loc='best', fontsize=8)
