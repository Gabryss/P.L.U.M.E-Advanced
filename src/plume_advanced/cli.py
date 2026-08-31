#!/usr/bin/env python3
"""Single entrypoint for generating the current cave-network output."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

SOURCE_ROOT = Path(__file__).resolve().parents[2]
WORKING_ROOT = Path.cwd()
SOURCE_CONFIG = SOURCE_ROOT / "config" / "project.toml"
WORKING_CONFIG = WORKING_ROOT / "config" / "project.toml"
PACKAGED_CONFIG = Path(__file__).with_name("default_project.toml")
if WORKING_CONFIG.is_file():
    ROOT = WORKING_ROOT
    DEFAULT_CONFIG = WORKING_CONFIG
elif SOURCE_CONFIG.is_file():
    ROOT = SOURCE_ROOT
    DEFAULT_CONFIG = SOURCE_CONFIG
else:
    ROOT = WORKING_ROOT
    DEFAULT_CONFIG = PACKAGED_CONFIG
CACHE_ROOT = Path(tempfile.gettempdir()) / "plume-advanced-cache"
MPL_CACHE = CACHE_ROOT / "matplotlib"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

from plume_advanced.config import (
    ProjectConfig,
    load_project_config,
    write_project_config_manifest,
)
from plume_advanced.evaluation.artifacts import (
    export_event_report,
    export_geometry_report,
    export_network_artifact,
    export_section_artifact,
)
from plume_advanced.exporters import export_target_asset
from plume_advanced.output_guard import (
    OutputOverwriteRefused,
    require_output_overwrite_confirmation,
)
from plume_advanced.pipeline import StageCheckpointStore, pipeline_fingerprint
from plume_advanced.run_manifest import write_run_manifest
from plume_advanced.stages.events import GeologicalEventGenerator
from plume_advanced.stages.floor_map import FloorMapGenerator, export_floor_atlas
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator, export_host_influence_report
from plume_advanced.stages.network import CaveNetworkGenerator, export_network_report
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.visualization.events import GeologicalEventPlotter
from plume_advanced.visualization.floor_map import FloorMapPlotter
from plume_advanced.visualization.geometry import GeometryPlotter
from plume_advanced.visualization.host_field import HostFieldPlotter
from plume_advanced.visualization.network import CaveNetworkPlotter
from plume_advanced.visualization.section_field import SectionFieldPlotter


class TerminalProgress:
    """Rich-backed progress reporter for long pipeline runs."""

    _current: ClassVar["TerminalProgress | None"] = None

    def __init__(self, *, width: int = 32) -> None:
        self.console = Console()
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description:<28}"),
            BarColumn(bar_width=width),
            TaskProgressColumn(),
            TextColumn("({task.completed:.0f}/{task.total:.0f})"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[detail]}"),
            console=self.console,
        )
        self._progress.start()
        self._active_task_id: TaskID | None = None
        self._last_total = 1
        type(self)._current = self

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
        if type(self)._current is self:
            type(self)._current = None

    @classmethod
    def close_active(cls) -> None:
        if cls._current is not None:
            cls._current.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the project TOML configuration.",
    )
    parser.add_argument(
        "--body",
        choices=("earth", "mars", "moon"),
        default=None,
        help=("Override world.body for this run and use that body's default geological material."),
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
            "Path for the complete portable OBJ scene export. "
            "Defaults to a sibling file named plume_cave_scene.obj."
        ),
    )
    parser.add_argument(
        "--geometry-glb-output",
        type=Path,
        default=None,
        help=(
            "Path for the complete portable GLB scene export. "
            "Defaults to a sibling file named plume_cave_scene.glb."
        ),
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help=(
            "Bypass the confirmation for non-empty output directories. "
            "Intended for deliberate unattended or debug generation."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse validated stage checkpoints whose configuration, inputs, "
            "Python version, and production source fingerprint still match."
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=None,
        help="Checkpoint directory; defaults to .plume-checkpoints beside the outputs.",
    )
    return parser.parse_args(argv)


def _run_pipeline(argv: list[str] | None = None) -> int:
    run_started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    args = parse_args(argv)
    project_config = load_project_config(args.config, world_body=args.body)
    host_output = args.host_output or args.output.with_name("stage_a_host_field.png")
    section_output = args.section_output or args.output.with_name("stage_c_section_field.png")
    geometry_output = args.geometry_output or args.output.with_name("stage_d_geometry.png")
    event_output = args.event_output or args.output.with_name("stage_e_geological_events.png")
    floor_map_output = args.floor_map_output or args.output.with_name("stage_c_floor_map.png")
    geometry_presentation_output = args.geometry_presentation_output or geometry_output.with_name(
        "stage_d_geometry_presentation.png"
    )
    geometry_chunk_output = args.geometry_chunk_output or geometry_output.with_name(
        "stage_d_geometry_chunks.png"
    )
    geometry_mesh_output = args.geometry_mesh_output or args.output.with_name(
        "plume_cave_scene.obj"
    )
    geometry_glb_output = args.geometry_glb_output or args.output.with_name("plume_cave_scene.glb")
    resolved_config_output = args.output.with_name("resolved_project_config.json")
    run_manifest_output = args.output.with_name("run_manifest.json")
    output_directories = {
        path.parent
        for path in (
            args.output,
            host_output,
            section_output,
            geometry_output,
            event_output,
            floor_map_output,
            geometry_presentation_output,
            geometry_chunk_output,
            geometry_mesh_output,
            geometry_glb_output,
            resolved_config_output,
            run_manifest_output,
        )
    }
    try:
        require_output_overwrite_confirmation(
            output_directories,
            allow_overwrite=(
                args.resume or args.force_overwrite or project_config.run.overwrite_outputs
            ),
        )
    except OutputOverwriteRefused as error:
        print(error, file=sys.stderr)
        return 2

    progress = TerminalProgress()
    progress.start("Configuration", "loaded TOML and checked output")
    resolved_config_path = write_project_config_manifest(
        project_config,
        resolved_config_output,
    )
    stage_timings["configuration_s"] = time.perf_counter() - run_started
    progress.finish(
        (
            f"{project_config.world.body.name}/"
            f"{project_config.world.material.name}; "
            f"wrote {resolved_config_path.name}"
        )
    )
    manifest_inputs = _run_inputs(args.config, project_config)
    completed_outputs: list[Path] = [resolved_config_path]
    checkpoint_root = (
        args.checkpoint_directory
        if args.checkpoint_directory is not None
        else args.output.parent / ".plume-checkpoints"
    )
    checkpoint_store: StageCheckpointStore = StageCheckpointStore(
        checkpoint_root,
        pipeline_fingerprint(
            project_config,
            inputs=manifest_inputs,
            source_root=SOURCE_ROOT,
        ),
    )

    def resumable(stage: str, builder):
        artifact, reused = checkpoint_store.load_or_build(
            stage,
            builder,
            resume=args.resume,
        )
        if reused:
            progress.log(f"[cyan]Resumed {stage} from a validated checkpoint.[/cyan]")
        return artifact

    def checkpoint(stage: str) -> None:
        write_run_manifest(
            project_config,
            run_manifest_output,
            outputs=completed_outputs,
            elapsed_seconds=time.perf_counter() - run_started,
            source_root=ROOT,
            status="running",
            current_stage=stage,
            inputs=manifest_inputs,
            timings=stage_timings,
        )

    checkpoint("host_field")
    progress.start("Stage A - Host Field", "generating scalar fields")
    stage_started = time.perf_counter()
    host_field = resumable(
        "host_field",
        lambda: HostFieldGenerator(project_config.host_field).generate(),
    )
    host_influence_path = export_host_influence_report(
        host_field,
        host_output.with_name("stage_a_host_influence.json"),
        generation_context={
            "body_id": project_config.world.body.name,
            "material_id": project_config.world.material.name,
            "flow_regime": asdict(project_config.flow_regime),
        },
    )
    if project_config.run.render_diagnostics:
        progress.update(1, 2, "rendering host-field plot")
        host_output_path = HostFieldPlotter().render(host_field, host_output)
        progress.finish(f"wrote {host_output_path.name}")
    else:
        host_output_path = None
        progress.finish("diagnostic render disabled")
    completed_outputs.extend(
        path for path in (host_influence_path, host_output_path) if path is not None
    )
    stage_timings["host_s"] = time.perf_counter() - stage_started

    checkpoint("network")
    progress.start("Stage B - Cave Network", "tracing cave skeleton")
    stage_started = time.perf_counter()
    cave_network = resumable(
        "network",
        lambda: CaveNetworkGenerator(project_config.network).generate(host_field),
    )
    network_summary = cave_network.summary()
    network_report_path = export_network_report(
        cave_network,
        args.output.with_name("stage_b_network_report.json"),
    )
    network_artifact_path = export_network_artifact(
        cave_network,
        args.output.with_name("stage_b_network.json"),
    )
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
    completed_outputs.extend(
        path
        for path in (network_report_path, network_artifact_path, network_output_path)
        if path is not None
    )
    stage_timings["network_s"] = time.perf_counter() - stage_started

    checkpoint("section_field")
    progress.start("Stage C - Section Field", "sampling tunnel profiles")
    stage_started = time.perf_counter()
    section_field = resumable(
        "section_field",
        lambda: SectionFieldGenerator(project_config.section_field).generate(cave_network),
    )
    section_summary = section_field.summary()
    section_npz_path, section_json_path = export_section_artifact(
        section_field,
        section_output.with_name("stage_c_sections"),
    )
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
    if section_output_path is not None:
        completed_outputs.append(section_output_path)
    completed_outputs.extend((section_npz_path, section_json_path))
    stage_timings["sections_s"] = time.perf_counter() - stage_started

    def geometry_progress(phase: str, current: int, total: int, message: str) -> None:
        progress.update(current, total, f"{phase}: {message}")

    geometry_generator = GeometryGenerator(project_config.geometry)
    checkpoint("base_geometry")
    progress.start(
        "Stage D1 - Base Volume",
        (
            f"stamping at {project_config.geometry.voxel_size:.3g} m "
            f"({project_config.geometry.resolution_quality})"
        ),
    )
    stage_started = time.perf_counter()
    base_geometry = resumable(
        "base_geometry",
        lambda: geometry_generator.build_base_volume(
            cave_network,
            section_field,
            progress=geometry_progress,
        ),
    )
    progress.finish(f"built {int(base_geometry.summary()['carved_voxel_count'])} cave voxels")
    stage_timings["base_geometry_s"] = time.perf_counter() - stage_started

    floor_map_generator = FloorMapGenerator(project_config.floor_map)
    checkpoint("base_floor_atlas")
    progress.start("Stage C2 - Base Floor Atlas", "raycasting event-placement cells")
    stage_started = time.perf_counter()
    base_floor_atlas = resumable(
        "base_floor_atlas",
        lambda: floor_map_generator.generate(
            cave_network,
            section_field,
            base_geometry,
        ),
    )
    base_floor_summary = base_floor_atlas.summary()
    progress.finish(f"{int(base_floor_summary['cell_count'])} placement cells")
    stage_timings["floor_base_s"] = time.perf_counter() - stage_started

    checkpoint("geological_events")
    progress.start("Stage E - Geological Events", "grounding props and modifiers")
    stage_started = time.perf_counter()

    def event_progress(phase: str, current: int, total: int, message: str) -> None:
        progress.update(current, total, f"{phase}: {message}")

    event_field = resumable(
        "geological_events",
        lambda: GeologicalEventGenerator(project_config.events).generate(
            section_field,
            base_geometry,
            base_floor_atlas,
            progress=event_progress,
        ),
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
        progress.finish(f"wrote {event_output_path.name}")
    else:
        event_output_path = None
        progress.finish("diagnostic render disabled")
    if event_output_path is not None:
        completed_outputs.append(event_output_path)
    stage_timings["events_s"] = time.perf_counter() - stage_started

    checkpoint("final_geometry")
    progress.start("Stage D2 - Final Geometry", "applying structural events")
    stage_started = time.perf_counter()
    cave_geometry = resumable(
        "final_geometry",
        lambda: geometry_generator.finalize(
            base_geometry,
            event_field,
            progress=geometry_progress,
        ),
    )
    progress.finish(
        (
            f"{len(cave_geometry.chunk_meshes)} chunks, "
            f"{int(cave_geometry.summary()['export_face_count'])} exported faces"
        )
    )
    stage_timings["geometry_s"] = (
        stage_timings.get("base_geometry_s", 0.0) + time.perf_counter() - stage_started
    )
    geometry_report_path = export_geometry_report(
        cave_geometry,
        geometry_output.with_name("stage_d_geometry_report.json"),
    )
    completed_outputs.append(geometry_report_path)

    checkpoint("final_floor_atlas")
    progress.start("Stage C3 - Final Floor Map", "relifting post-event geology")
    stage_started = time.perf_counter()
    floor_atlas = resumable(
        "final_floor_atlas",
        lambda: floor_map_generator.revalidate(
            cave_network,
            section_field,
            cave_geometry,
            base_floor_atlas,
            event_field,
        ),
    )
    floor_npz_path, floor_json_path = export_floor_atlas(
        floor_atlas,
        floor_map_output.with_suffix(""),
    )
    floor_summary = floor_atlas.summary()
    if project_config.run.render_diagnostics:
        floor_map_output_path = FloorMapPlotter().render(
            floor_atlas,
            floor_map_output,
            event_field,
            cave_network,
        )
        floor_render_detail = f"; wrote {floor_map_output_path.name}"
    else:
        floor_map_output_path = None
        floor_render_detail = ""
    progress.finish(
        (
            f"{int(floor_summary['cell_count'])} valid, "
            f"{int(floor_summary['invalidated_cell_count'])} invalidated"
            f"{floor_render_detail}"
        )
    )
    completed_outputs.extend((floor_npz_path, floor_json_path))
    if floor_map_output_path is not None:
        completed_outputs.append(floor_map_output_path)
    stage_timings["floor_final_s"] = time.perf_counter() - stage_started
    event_report_path = export_event_report(
        event_field,
        event_output.with_name("stage_e_event_report.json"),
        invalidated_floor_cells=int(floor_summary["invalidated_cell_count"]),
    )
    completed_outputs.append(event_report_path)

    checkpoint("export")
    progress.start("Stage D - Export", "preparing target package")
    stage_started = time.perf_counter()
    geometry_output_path = None
    geometry_presentation_output_path = None
    geometry_chunk_output_path = None
    if project_config.run.render_diagnostics and hasattr(cave_geometry.voxel_grid, "density"):
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
    elif project_config.run.render_diagnostics:
        progress.log(
            "[yellow]Skipping dense Stage-D diagnostic rasters for sparse tiled storage.[/yellow]"
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
        geometry_mesh_output if project_config.export.file_format == "obj" else geometry_glb_output
    )
    export_result = export_target_asset(
        cave_geometry,
        project_config.export,
        selected_output.parent / f"export_{project_config.export.target}",
        asset_name=selected_output.stem,
    )
    progress.finish(f"wrote {export_result.primary_asset.name}")
    stage_timings["export_s"] = time.perf_counter() - stage_started

    progress.close()
    progress.log("Generated cave pipeline artifacts.")
    progress.log(f"Configuration: {args.config}")
    progress.log(f"Resolved configuration: {resolved_config_path}")
    progress.log(f"Host routing influence: {host_influence_path}")
    progress.log(f"Network diagnostics: {network_report_path}")
    progress.log(f"Network semantic artifact: {network_artifact_path}")
    progress.log(f"Section arrays: {section_npz_path}")
    progress.log(f"Section metadata: {section_json_path}")
    progress.log(f"Geometry report: {geometry_report_path}")
    progress.log(f"Event report: {event_report_path}")
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
        progress.log(f"Stage D presentation visualization: {geometry_presentation_output_path}")
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
        "geometry_resolution: "
        f"policy={project_config.geometry.resolution_policy}, "
        f"quality={project_config.geometry.resolution_quality}, "
        f"voxel_size_m={project_config.geometry.voxel_size:.3f}, "
        f"nominal_samples={project_config.geometry.characteristic_samples_across_passage:.1f}"
    )
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

    manifest_outputs = [
        resolved_config_path,
        host_influence_path,
        network_report_path,
        network_artifact_path,
        section_npz_path,
        section_json_path,
        geometry_report_path,
        event_report_path,
        floor_npz_path,
        floor_json_path,
        export_result.primary_asset,
        *export_result.files,
    ]
    manifest_outputs.extend(
        path
        for path in (
            host_output_path,
            network_output_path,
            section_output_path,
            floor_map_output_path,
            event_output_path,
            geometry_output_path,
            geometry_presentation_output_path,
            geometry_chunk_output_path,
        )
        if path is not None
    )
    run_manifest_path = write_run_manifest(
        project_config,
        run_manifest_output,
        outputs=manifest_outputs,
        elapsed_seconds=time.perf_counter() - run_started,
        source_root=ROOT,
        status="complete",
        current_stage="complete",
        inputs=manifest_inputs,
        timings=stage_timings,
    )
    progress.log(f"Run manifest: {run_manifest_path}")

    return 0


def _run_inputs(config_path: Path, project_config: ProjectConfig) -> tuple[Path, ...]:
    candidates = [
        config_path.resolve(),
        SOURCE_ROOT / "pyproject.toml",
        SOURCE_ROOT / "uv.lock",
    ]
    candidates.extend(
        Path(path)
        for path in (
            project_config.geometry.cave_diffuse_texture,
            project_config.geometry.cave_normal_texture,
            project_config.geometry.cave_roughness_texture,
            project_config.geometry.cave_displacement_texture,
        )
        if path
    )
    return tuple(path for path in candidates if path.is_file())


def main(argv: list[str] | None = None) -> int:
    run_started = time.perf_counter()
    try:
        return _run_pipeline(argv)
    except (Exception, KeyboardInterrupt) as error:
        TerminalProgress.close_active()
        try:
            args = parse_args(argv)
            project_config = load_project_config(args.config, world_body=args.body)
            manifest_path = args.output.with_name("run_manifest.json")
            current_stage = "configuration"
            completed_outputs: list[Path] = []
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                current_stage = str(previous.get("current_stage", current_stage))
                completed_outputs.extend(
                    (manifest_path.parent / record["path"]).resolve()
                    for record in previous.get("outputs", ())
                )
            write_run_manifest(
                project_config,
                manifest_path,
                outputs=completed_outputs,
                elapsed_seconds=time.perf_counter() - run_started,
                source_root=ROOT,
                status="failed",
                current_stage=current_stage,
                failed_stage=current_stage,
                error=f"{type(error).__name__}: {error}",
                inputs=_run_inputs(args.config, project_config),
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
