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

from config import load_project_config, write_project_config_manifest
from exporters import export_target_asset
from stages.events import GeologicalEventGenerator
from stages.floor_map import FloorMapGenerator, export_floor_atlas
from stages.geometry import GeometryGenerator
from stages.host_field import HostFieldGenerator, export_host_influence_report
from stages.network import CaveNetworkGenerator
from stages.section_field import SectionFieldGenerator
from visualization.geometry import GeometryPlotter
from visualization.events import GeologicalEventPlotter
from visualization.floor_map import FloorMapPlotter
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
        "--event-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated geological event visualization. "
            "Defaults to a sibling file named stage_e_geological_events.png."
        ),
    )
    parser.add_argument(
        "--floor-map-output",
        type=Path,
        default=None,
        help=(
            "Path for the cave-floor map PNG. Matching NPZ and JSON files are "
            "written beside it. Defaults to stage_c_floor_map.png."
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
    parser.add_argument(
        "--geometry-glb-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated geometry-stage GLB scene export. "
            "Defaults to a sibling file named stage_d_geometry.glb."
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
    event_output = args.event_output or args.output.with_name("stage_e_geological_events.png")
    floor_map_output = (
        args.floor_map_output or args.output.with_name("stage_c_floor_map.png")
    )
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
    geometry_glb_output = (
        args.geometry_glb_output or args.output.with_name("stage_d_geometry.glb")
    )
    resolved_config_output = args.output.with_name("resolved_project_config.json")
    resolved_config_path = write_project_config_manifest(
        project_config,
        resolved_config_output,
    )
    progress.finish(
        (
            f"{project_config.world.body.name}/"
            f"{project_config.world.material.name}; "
            f"wrote {resolved_config_path.name}"
        )
    )

    progress.start("Stage A - Host Field", "generating scalar fields")
    host_field = HostFieldGenerator(project_config.host_field).generate()
    host_influence_path = export_host_influence_report(
        host_field,
        host_output.with_name("stage_a_host_influence.json"),
    )
    if project_config.run.render_diagnostics:
        progress.update(1, 2, "rendering host-field plot")
        host_output_path = HostFieldPlotter().render(host_field, host_output)
        progress.finish(f"wrote {host_output_path.name}")
    else:
        host_output_path = None
        progress.finish("diagnostic render disabled")

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
    if project_config.run.render_diagnostics:
        network_output_path = CaveNetworkPlotter().render(
            host_field,
            cave_network,
            args.output,
        )
        progress.finish(f"wrote {network_output_path.name}")
    else:
        network_output_path = None
        progress.finish("diagnostic render disabled")

    progress.start("Stage C - Section Field", "sampling tunnel profiles")
    section_field = SectionFieldGenerator(project_config.section_field).generate(cave_network)
    section_summary = section_field.summary()
    progress.update(
        1,
        2,
        f"{int(section_summary['sample_count'])} samples; rendering",
    )
    if project_config.run.render_diagnostics:
        section_output_path = SectionFieldPlotter().render(
            cave_network,
            section_field,
            section_output,
        )
        progress.finish(f"wrote {section_output_path.name}")
    else:
        section_output_path = None
        progress.finish("diagnostic render disabled")

    def geometry_progress(phase: str, current: int, total: int, message: str) -> None:
        progress.update(current, total, f"{phase}: {message}")

    geometry_generator = GeometryGenerator(project_config.geometry)
    progress.start("Stage D1 - Base Volume", "stamping cave density")
    base_geometry = geometry_generator.build_base_volume(
        cave_network,
        section_field,
        progress=geometry_progress,
    )
    progress.finish(
        f"built {int(base_geometry.summary()['carved_voxel_count'])} cave voxels"
    )

    progress.start("Stage C2 - Floor Atlas", "raycasting traversable floor cells")
    floor_atlas = FloorMapGenerator(project_config.floor_map).generate(
        cave_network,
        section_field,
        base_geometry,
    )
    floor_npz_path, floor_json_path = export_floor_atlas(
        floor_atlas,
        floor_map_output.with_suffix(""),
    )
    floor_summary = floor_atlas.summary()
    progress.finish(
        (
            f"{int(floor_summary['cell_count'])} cells; "
            f"wrote {floor_npz_path.name} and {floor_json_path.name}"
        )
    )

    progress.start("Stage E - Geological Events", "grounding props and modifiers")
    event_field = GeologicalEventGenerator(project_config.events).generate(
        section_field,
        base_geometry,
        floor_atlas,
    )
    event_summary = event_field.summary()
    progress.update(
        1,
        2,
        f"{int(event_summary['event_count'])} events; rendering",
    )
    if project_config.run.render_diagnostics:
        event_output_path = GeologicalEventPlotter().render(
            cave_network,
            section_field,
            event_field,
            event_output,
        )
        floor_map_output_path = FloorMapPlotter().render(
            floor_atlas,
            floor_map_output,
            event_field,
        )
        progress.finish(
            f"wrote {event_output_path.name} and {floor_map_output_path.name}"
        )
    else:
        event_output_path = None
        floor_map_output_path = None
        progress.finish("diagnostic render disabled")

    progress.start("Stage D2 - Final Geometry", "applying structural events")
    cave_geometry = geometry_generator.finalize(
        base_geometry,
        event_field,
        progress=geometry_progress,
    )
    progress.finish(
        (
            f"{len(cave_geometry.chunk_meshes)} chunks, "
            f"{int(cave_geometry.summary()['export_face_count'])} exported faces"
        )
    )

    progress.start("Stage D - Export", "preparing target package")
    geometry_output_path = None
    geometry_presentation_output_path = None
    geometry_chunk_output_path = None
    if project_config.run.render_diagnostics:
        geometry_plotter = GeometryPlotter()
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
    progress.update(
        3,
        4,
        (
            (
                f"wrote {geometry_chunk_output_path.name}; "
                if geometry_chunk_output_path is not None
                else ""
            )
            + "exporting "
            f"{project_config.export.target}"
        ),
    )
    selected_output = (
        geometry_mesh_output
        if project_config.export.file_format == "obj"
        else geometry_glb_output
    )
    export_result = export_target_asset(
        cave_geometry,
        project_config.export,
        selected_output.parent / f"export_{project_config.export.target}",
        asset_name=selected_output.stem,
    )
    progress.finish(f"wrote {export_result.primary_asset.name}")

    progress.close()
    progress.log("Generated cave pipeline artifacts.")
    progress.log(f"Configuration: {args.config}")
    progress.log(f"Resolved configuration: {resolved_config_path}")
    progress.log(f"Host routing influence: {host_influence_path}")
    progress.log(
        "World: "
        f"{project_config.world.body.name} "
        f"({project_config.world.body.gravity_m_s2:.5g} m/s²), "
        f"material={project_config.world.material.name}"
    )
    progress.log(
        "Run mode: "
        f"{'development' if project_config.run.dev_mode else 'production'}, "
        f"quality={project_config.run.quality}"
    )
    progress.log(
        "Export intent: "
        f"target={project_config.export.target}, "
        f"format={project_config.export.file_format}"
    )
    if project_config.run.render_diagnostics:
        progress.log(f"Stage A visualization: {host_output_path}")
        progress.log(f"Stage B visualization: {network_output_path}")
        progress.log(f"Stage C visualization: {section_output_path}")
        progress.log(f"Stage C floor-map visualization: {floor_map_output_path}")
        progress.log(f"Stage E visualization: {event_output_path}")
        progress.log(f"Stage D diagnostic visualization: {geometry_output_path}")
        progress.log(
            f"Stage D presentation visualization: {geometry_presentation_output_path}"
        )
        progress.log(f"Stage D chunk diagnostics: {geometry_chunk_output_path}")
    progress.log(f"Floor atlas arrays: {floor_npz_path}")
    progress.log(f"Floor atlas metadata: {floor_json_path}")
    progress.log(f"Target export: {export_result.primary_asset}")
    for exported_file in export_result.files:
        progress.log(f"  export_file: {exported_file}")
    for warning in export_result.warnings:
        progress.log(f"[yellow]  warning: {warning}[/yellow]")
    progress.log("")
    progress.log("[bold]Key cave metrics[/bold]")
    progress.log(f"total_lava_tube_length_m: {network_summary['total_length']:.3f}")
    progress.log(f"dominant_route_length_m: {network_summary['dominant_route_length']:.3f}")
    progress.log(
        "network_segment_width_m: "
        f"avg={network_summary['mean_segment_width']:.3f}, "
        f"min={network_summary['min_segment_width']:.3f}, "
        f"max={network_summary['max_segment_width']:.3f}"
    )
    progress.log(
        "section_tube_width_m: "
        f"avg={section_summary['mean_tube_width']:.3f}, "
        f"min={section_summary['min_tube_width']:.3f}, "
        f"max={section_summary['max_tube_width']:.3f}"
    )
    progress.log(
        "section_tube_height_m: "
        f"avg={section_summary['mean_tube_height']:.3f}, "
        f"min={section_summary['min_tube_height']:.3f}, "
        f"max={section_summary['max_tube_height']:.3f}"
    )
    progress.log("")
    for key, value in host_field.summary().items():
        progress.log(f"host_{key}: {value:.3f}")
    for key, value in network_summary.items():
        progress.log(f"network_{key}: {value:.3f}")
    for key, value in section_summary.items():
        progress.log(f"section_{key}: {value:.3f}")
    for key, value in event_summary.items():
        progress.log(f"event_{key}: {value:.3f}")
    for key, value in cave_geometry.summary().items():
        progress.log(f"geometry_{key}: {value:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
