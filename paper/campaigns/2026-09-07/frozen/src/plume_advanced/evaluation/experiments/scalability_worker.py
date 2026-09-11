"""One isolated geometry benchmark case, invoked by the parent monitor."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.experiments.common import config_hash, for_seed
from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-config", type=Path, required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--route-length-m", type=float, required=True)
    parser.add_argument("--storage-mode", choices=("dense", "tiled"), required=True)
    parser.add_argument("--quality", choices=("preview", "standard", "production"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    project = load_project_config(args.project_config, world_body=args.body, dev_mode=False)
    if args.quality is not None and args.quality != project.run.quality:
        raise ValueError("Benchmark quality must match the frozen project's run.quality")
    project = _benchmark_project(project, args.seed, args.route_length_m, args.storage_mode)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    stage = time.perf_counter()
    host = HostFieldGenerator(project.host_field).generate()
    timings["host_s"] = time.perf_counter() - stage
    stage = time.perf_counter()
    network = CaveNetworkGenerator(project.network).generate(host)
    timings["network_s"] = time.perf_counter() - stage
    stage = time.perf_counter()
    sections = SectionFieldGenerator(project.section_field).generate(network)
    timings["sections_s"] = time.perf_counter() - stage
    stage = time.perf_counter()
    generator = GeometryGenerator(project.geometry)
    base = generator.build_base_volume(network, sections)
    timings["volume_s"] = time.perf_counter() - stage
    stage = time.perf_counter()
    geometry = generator.finalize(base)
    timings["meshing_s"] = time.perf_counter() - stage
    timings["geometry_s"] = timings["volume_s"] + timings["meshing_s"]
    if not geometry.assembled_faces:
        raise ValueError("Benchmark produced no mesh")
    timings["total_s"] = time.perf_counter() - started
    payload = {
        "timings": timings,
        "network": network.summary(),
        "sections": sections.summary(),
        "geometry": geometry.summary(),
        "storage_mode_actual": "tiled" if hasattr(geometry.voxel_grid, "tiles") else "dense",
        "bbox_m": geometry.voxel_grid.bounds,
        "voxel_size_m": project.geometry.voxel_size,
        "quality": project.run.quality,
        "project_config_sha256": config_hash(project),
        "timing_scope": "host, network, sections, base density with roof screening, meshing and welding; excludes optional events, appearance, exports and application import",
        "section_resolution": section_resolution_report(sections, project.geometry.voxel_size),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _benchmark_project(project, seed: int, route_length_m: float, storage_mode: str):
    project = for_seed(project, seed)
    ratio = route_length_m / max(project.host_field.target_route_length_m, 1e-9)
    grid = project.host_field.grid
    height = max(1.10 * route_length_m, 32.0 * grid.spacing_y)
    grid = replace(grid, height=height, ny=max(32, int(round(height / grid.spacing_y)) + 1))
    seed_x, seed_y = project.host_field.seed_point
    host = replace(
        project.host_field,
        grid=grid,
        seed_point=(seed_x, seed_y * ratio),
        target_route_length_m=route_length_m,
    )
    network = replace(
        project.network,
        target_route_length_m=route_length_m,
        trace_max_steps=max(48, grid.ny),
    )
    geometry = replace(project.geometry, storage_mode=storage_mode)
    return replace(project, host_field=host, network=network, geometry=geometry)


if __name__ == "__main__":
    raise SystemExit(main())
