#!/usr/bin/env python3
"""Single entrypoint for generating the current cave-network output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(tempfile.gettempdir()) / "plume-advanced-cache"
MPL_CACHE = CACHE_ROOT / "matplotlib"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.geometry_export import export_geometry_obj
from stages.geometry import GeometryGenerator
from stages.host_field import HostFieldGenerator
from stages.network import CaveNetworkGenerator
from stages.section_field import SectionFieldGenerator
from visualization.geometry import GeometryPlotter
from visualization.host_field import HostFieldPlotter
from visualization.network import CaveNetworkPlotter
from visualization.section_field import SectionFieldPlotter


class TerminalProgress:
    """Rich-backed progress reporter for long pipeline runs."""

    def __init__(self, *, width: int = 32) -> None:
        self.console = Console()
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description:<28}"),
            BarColumn(bar_width=width),
            TaskProgressColumn(),
            TextColumn("({task.completed:.0f}/{task.total:.0f})"),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[detail]}"),
            console=self.console,
        )
        self._progress.start()
        self._active_task_id: int | None = None
        self._last_total = 1

    def log(self, message: str) -> None:
        self.console.print(message)

    def start(self, label: str, detail: str = "") -> None:
        self._last_total = 1
        self._active_task_id = self._progress.add_task(
            label,
            total=1,
            completed=0,
            detail=detail or "starting",
        )

    def update(self, current: int, total: int, detail: str = "") -> None:
        if self._active_task_id is None:
            return
        total = max(total, 1)
        current = min(max(current, 0), total)
        self._last_total = total
        self._progress.update(
            self._active_task_id,
            total=total,
            completed=current,
            detail=detail,
        )

    def finish(self, detail: str = "done") -> None:
        if self._active_task_id is None:
            return
        self.update(self._last_total, self._last_total, detail)
        self._progress.stop_task(self._active_task_id)
        self._active_task_id = None

    def close(self) -> None:
        self._progress.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "project.toml",
        help="Path to the project TOML configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "stage_b_cave_network.png",
        help="Path for the generated cave-network visualization.",
    )
    parser.add_argument(
        "--host-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated host-field visualization. "
            "Defaults to a sibling file named stage_a_host_field.png."
        ),
    )
    parser.add_argument(
        "--section-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated section-field visualization. "
            "Defaults to a sibling file named stage_c_section_field.png."
        ),
    )
    parser.add_argument(
        "--geometry-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated geometry-stage technical visualization. "
            "Defaults to a sibling file named stage_d_geometry.png."
        ),
    )
    parser.add_argument(
        "--geometry-presentation-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated geometry-stage presentation visualization. "
            "Defaults to a sibling file named stage_d_geometry_presentation.png."
        ),
    )
    parser.add_argument(
        "--geometry-chunk-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated geometry-stage chunk diagnostics. "
            "Defaults to a sibling file named stage_d_geometry_chunks.png."
        ),
    )
    parser.add_argument(
        "--geometry-mesh-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated geometry-stage OBJ export. "
            "Defaults to a sibling file named stage_d_geometry.obj."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    progress = TerminalProgress()

    progress.start("Configuration", "loading TOML")
    project_config = load_project_config(args.config)
    host_output = args.host_output or args.output.with_name("stage_a_host_field.png")
    section_output = args.section_output or args.output.with_name("stage_c_section_field.png")
    geometry_output = args.geometry_output or args.output.with_name("stage_d_geometry.png")
    geometry_presentation_output = (
        args.geometry_presentation_output
        or geometry_output.with_name("stage_d_geometry_presentation.png")
    )
    geometry_chunk_output = (
        args.geometry_chunk_output
        or geometry_output.with_name("stage_d_geometry_chunks.png")
    )
    geometry_mesh_output = (
        args.geometry_mesh_output or args.output.with_name("stage_d_geometry.obj")
    )
    progress.finish(f"loaded {args.config}")

    progress.start("Stage A - Host Field", "generating scalar fields")
    host_field = HostFieldGenerator(project_config.host_field).generate()
    progress.update(1, 2, "rendering host-field plot")
    host_output_path = HostFieldPlotter().render(host_field, host_output)
    progress.finish(f"wrote {host_output_path.name}")

    progress.start("Stage B - Cave Network", "tracing cave skeleton")
    cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
    network_summary = cave_network.summary()
    progress.update(
        1,
        2,
        (
            f"{int(network_summary['segment_count'])} segments, "
            f"{int(network_summary['junction_count'])} junctions; rendering"
        ),
    )
    network_output_path = CaveNetworkPlotter().render(host_field, cave_network, args.output)
    progress.finish(f"wrote {network_output_path.name}")

    progress.start("Stage C - Section Field", "sampling tunnel profiles")
    section_field = SectionFieldGenerator(project_config.section_field).generate(cave_network)
    section_summary = section_field.summary()
    progress.update(
        1,
        2,
        f"{int(section_summary['sample_count'])} samples; rendering",
    )
    section_output_path = SectionFieldPlotter().render(
        cave_network,
        section_field,
        section_output,
    )
    progress.finish(f"wrote {section_output_path.name}")

    progress.start("Stage D - Geometry", "starting voxel geometry")

    def geometry_progress(phase: str, current: int, total: int, message: str) -> None:
        progress.update(current, total, f"{phase}: {message}")

    cave_geometry = GeometryGenerator(project_config.geometry).generate(
        cave_network,
        section_field,
        progress=geometry_progress,
    )
    progress.finish(
        (
            f"{len(cave_geometry.chunk_meshes)} chunks, "
            f"{len(cave_geometry.assembled_faces)} faces"
        )
    )

    geometry_plotter = GeometryPlotter()
    progress.start("Stage D - Visuals", "rendering diagnostic sheet")
    geometry_output_path = geometry_plotter.render_debug(
        cave_network,
        cave_geometry,
        geometry_output,
    )
    progress.update(1, 4, f"wrote {geometry_output_path.name}; rendering presentation")
    geometry_presentation_output_path = geometry_plotter.render_presentation(
        cave_network,
        cave_geometry,
        geometry_presentation_output,
    )
    progress.update(
        2,
        4,
        f"wrote {geometry_presentation_output_path.name}; rendering chunk diagnostics",
    )
    geometry_chunk_output_path = geometry_plotter.render_chunks(
        cave_network,
        cave_geometry,
        geometry_chunk_output,
    )
    progress.update(3, 4, f"wrote {geometry_chunk_output_path.name}; exporting OBJ")
    geometry_mesh_output_path = export_geometry_obj(cave_geometry, geometry_mesh_output)
    progress.finish(f"wrote {geometry_mesh_output_path.name}")

    progress.close()
    progress.log("Generated cave pipeline artifacts.")
    progress.log(f"Configuration: {args.config}")
    progress.log(f"Stage A visualization: {host_output_path}")
    progress.log(f"Stage B visualization: {network_output_path}")
    progress.log(f"Stage C visualization: {section_output_path}")
    progress.log(f"Stage D diagnostic visualization: {geometry_output_path}")
    progress.log(f"Stage D presentation visualization: {geometry_presentation_output_path}")
    progress.log(f"Stage D chunk diagnostics: {geometry_chunk_output_path}")
    progress.log(f"Stage D mesh export: {geometry_mesh_output_path}")
    for key, value in host_field.summary().items():
        progress.log(f"host_{key}: {value:.3f}")
    for key, value in network_summary.items():
        progress.log(f"network_{key}: {value:.3f}")
    for key, value in section_summary.items():
        progress.log(f"section_{key}: {value:.3f}")
    for key, value in cave_geometry.summary().items():
        progress.log(f"geometry_{key}: {value:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
