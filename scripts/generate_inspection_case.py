#!/usr/bin/env python3
"""Generate a fixed-seed short inspection checkpoint (no rocks).

Pair with complete_inspection_case.py and create_blender_inspection.py.
Checkpoints use pickle and must remain trusted local files.
"""

import argparse
import json
import pickle
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import export_network_artifact, export_section_artifact
from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.identity import package_source_hash
from plume_advanced.pipeline.recovery import build_accepted_base
from plume_advanced.progress import progress_scope
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("single", "multi"))
    parser.add_argument("seed", type=int)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--length", type=float, default=250.0)
    parser.add_argument("--voxel", type=float, default=0.10)
    args = parser.parse_args()
    mode, seed, length, voxel, root = args.mode, args.seed, args.length, args.voxel, args.directory
    if root.exists() and any(root.iterdir()):
        parser.error("Use an empty output directory; existing cases are preserved")
    if not (np.isfinite(length) and length > 0 and np.isfinite(voxel) and voxel > 0):
        parser.error("length and voxel must be finite and positive")
    if seed < 0:
        parser.error("seed must be nonnegative")
    root = root.resolve()
    config_root = Path(__file__).resolve().parents[1] / "config"
    project = load_project_config(
        config_root / "earth_short_interconnected_full.toml"
        if mode == "multi"
        else config_root / "earth_short_single.toml",
        seed_override=seed,
    )
    project = replace(
        project,
        network=replace(project.network, target_route_length_m=float(length)),
        geometry=replace(project.geometry, voxel_size=float(voxel)),
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "recipe.json").write_text(
        json.dumps(
            dict(
                mode=mode, seed=seed, length_m=length, voxel_m=voxel, source=package_source_hash()
            ),
            indent=2,
        )
    )
    last = 0
    trace = (root / "progress.jsonl").open("w")

    def progress(step, current=0, total=1, detail=""):
        nonlocal last
        trace.write(json.dumps(dict(step=step, current=current, total=total, detail=detail)) + "\n")
        trace.flush()
        if time.monotonic() - last > 10 or current == total:
            print(step, current, total, detail, flush=True)
            last = time.monotonic()

    start = time.monotonic()
    source_hash = package_source_hash()
    try:
        with progress_scope(progress):
            host = HostFieldGenerator(project.host_field).generate()
            network = CaveNetworkGenerator(project.network).generate(
                host,
                section_config=project.section_field,
                quality_report_path=root / "network_quality_report.json",
                quality_progress=lambda d: progress("network", detail=d),
            )
            sections = SectionFieldGenerator(project.section_field).generate(network)
            with (root / "initial_inputs.pickle").open("wb") as f:
                pickle.dump((project, host, network, sections), f)
            accepted = build_accepted_base(
                project, host, network, sections,
                report_path=root / "pipeline_recovery.json", progress=progress,
            )
            network, sections, base = accepted.network, accepted.sections, accepted.geometry
            export_network_artifact(network, root / "stage_b_network.json")
            export_section_artifact(sections, root / "stage_c_sections.npz")
            (root / "network_quality_report.json").write_text(json.dumps(network.quality_report, indent=2))
            (root / "resolution.json").write_text(
                json.dumps(section_resolution_report(sections, float(voxel)), indent=2)
            )
            with (root / "inputs.pickle").open("wb") as f:
                pickle.dump((project, host, network, sections), f)
            gen = GeometryGenerator(project.geometry)
            with (root / "base.pickle").open("wb") as f:
                pickle.dump(base, f)
            geometry = gen.finalize(base, progress=progress)
            np.savez_compressed(
                root / "mesh.npz",
                vertices=geometry.assembled_vertices,
                faces=geometry.assembled_faces,
            )
            mesh = trimesh.Trimesh(
                geometry.assembled_vertices, geometry.assembled_faces, process=False
            )
            stats = dict(
                seed=seed,
                mode=mode,
                source=source_hash,
                recovery_outcome=accepted.report["outcome"],
                source_unchanged=source_hash == package_source_hash(),
                elapsed=time.monotonic() - start,
                route_m=network.dominant_route_length,
                total_m=sum(s.total_length for s in network.segments),
                graph_cycles=len(network.segments) - len(network.nodes) + 1,
                components=geometry.component_count,
                watertight=bool(mesh.is_watertight),
                euler=int(mesh.euler_number),
                genus=int((2 * geometry.component_count - mesh.euler_number) // 2),
                faces=len(mesh.faces),
            )
            (root / "geometry_check.json").write_text(json.dumps(stats, indent=2))
            print(stats, flush=True)
    except Exception as error:
        (root / "failure.json").write_text(
            json.dumps(
                dict(
                    error_type=type(error).__name__,
                    message=str(error),
                    source=source_hash,
                    elapsed_seconds=time.monotonic() - start,
                ),
                indent=2,
            )
        )
        traceback.print_exc()
        sys.exit(1)
    finally:
        trace.close()


if __name__ == "__main__":
    main()
