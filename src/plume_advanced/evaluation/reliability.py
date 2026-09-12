"""Isolated seed campaigns exercising the production acceptance and mesh paths.

Every case has a wall-time limit. Failures stay in the report and do not prevent
later cases from running. Replays use a different PYTHONHASHSEED. Screening is
not a geological certificate and a finite campaign cannot prove every seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from plume_advanced.identity import package_source_hash, runtime_identity, sha256_file


@dataclass(frozen=True)
class ReliabilityCase:
    config: str
    seed: int
    body: str | None = None
    scope: str = "sections"
    voxel_size: float | None = None


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def execute_case(case: ReliabilityCase, output: Path) -> dict:
    """Run a real case; caller owns process isolation and failure recording."""
    from plume_advanced.config import load_project_config, project_config_manifest
    from plume_advanced.evaluation.artifacts import (
        host_semantic_hash,
        network_semantic_hash,
        section_semantic_hash,
    )
    from plume_advanced.evaluation.experiments.common import for_seed
    from plume_advanced.progress import progress_scope, report_progress
    from plume_advanced.stages.host_field import HostFieldGenerator
    from plume_advanced.stages.network import CaveNetworkGenerator
    from plume_advanced.stages.section_field import SectionFieldGenerator

    project = load_project_config(case.config, world_body=case.body)
    if not project.network.quality.enabled:
        raise ValueError("Reliability campaigns require enabled production quality screening")
    project = for_seed(project, case.seed)
    if case.voxel_size is not None:
        if case.voxel_size <= 0 or not math.isfinite(case.voxel_size):
            raise ValueError("voxel size must be finite and positive")
        project = replace(project, geometry=replace(project.geometry, voxel_size=case.voxel_size))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", project_config_manifest(project))
    timings = {}

    @contextmanager
    def stage(name):
        start = time.perf_counter()
        report_progress(name, detail="starting")
        try:
            yield
        finally:
            timings[name] = time.perf_counter() - start
            write_json(output / "timings.json", timings)

    with (output / "progress.jsonl").open("w") as trace:

        def progress(step, current, total, detail):
            trace.write(
                json.dumps(dict(step=step, completed=current, total=total, detail=detail)) + "\n"
            )
            trace.flush()

        with progress_scope(progress):
            with stage("host"):
                host = HostFieldGenerator(project.host_field).generate()
                before = host_semantic_hash(host)
            with stage("network"):
                network = CaveNetworkGenerator(project.network).generate(
                    host,
                    section_config=project.section_field,
                    quality_report_path=output / "network_quality.json",
                    quality_progress=lambda detail: report_progress(
                        "Network acceptance", detail=detail
                    ),
                )
            with stage("sections"):
                sections = SectionFieldGenerator(project.section_field).generate(network)
            identity = dict(
                host=before,
                network=network_semantic_hash(network),
                sections=section_semantic_hash(sections),
            )
            if host_semantic_hash(host) != before:
                raise AssertionError("Generation mutated its input host field")
            metrics = {
                "segment_count": len(network.segments),
                "combined_length_m": sum(segment.total_length for segment in network.segments),
                "selected_attempt": network.quality_report["selected_attempt"],
                "repair_pass": network.quality_report["selected_repair_pass"],
                "checks": len(network.quality_report["attempts"][-1]["checks"]),
            }
            if case.scope == "full":
                import numpy as np
                import trimesh

                from plume_advanced.exporters import export_target_asset
                from plume_advanced.run_manifest import write_run_manifest
                from plume_advanced.stages.events import GeologicalEventGenerator
                from plume_advanced.stages.floor_map import FloorMapGenerator
                from plume_advanced.stages.geometry import GeometryGenerator
                from plume_advanced.validation import PortableAssetValidator

                generator = GeometryGenerator(project.geometry)
                with stage("base_geometry"):
                    base = generator.build_base_volume(network, sections, progress=progress)
                floors = FloorMapGenerator(project.floor_map)
                with stage("base_floor"):
                    atlas = floors.generate(network, sections, base)
                with stage("events"):
                    events = GeologicalEventGenerator(project.events).generate(
                        sections, base, atlas, progress=progress
                    )
                with stage("final_geometry"):
                    geometry = generator.finalize(base, events, progress=progress)
                    mesh = trimesh.Trimesh(
                        vertices=geometry.assembled_vertices,
                        faces=geometry.assembled_faces,
                        process=False,
                    )
                    if not (
                        len(mesh.faces)
                        and np.isfinite(mesh.vertices).all()
                        and mesh.is_watertight
                        and mesh.is_winding_consistent
                        and np.all(mesh.area_faces > 0)
                    ):
                        raise AssertionError(
                            "Final cave mesh failed finite/closed/oriented/nondegenerate checks"
                        )
                with stage("final_floor"):
                    final_atlas = floors.revalidate(network, sections, geometry, atlas, events)
                    metrics.update(
                        floor_cells=len(final_atlas.cells),
                        event_count=len(events.events),
                        triangles=len(mesh.faces),
                        density_mib=geometry.summary()["density_memory_mib"],
                    )
                with stage("export"):
                    exported = export_target_asset(
                        geometry,
                        replace(project.export, target="blender", file_format="glb"),
                        output / "export",
                    )
                    metrics["asset_bytes"] = exported.primary_asset.stat().st_size
                    identity["glb"] = sha256_file(exported.primary_asset)
                    manifest = write_run_manifest(
                        project,
                        output / "run_manifest.json",
                        outputs=exported.files,
                        elapsed_seconds=sum(timings.values()),
                        source_root=Path(case.config).parent.parent,
                    )
                with stage("portable_validation"):
                    profile = "textured" if project.geometry.cave_diffuse_texture else "neutral"
                    checks = PortableAssetValidator(
                        exported.primary_asset, run_manifest_path=manifest, material_profile=profile
                    ).validate()
                    write_json(output / "asset_checks.json", [asdict(check) for check in checks])
                    failed = [check.name for check in checks if not check.passed]
                    if failed:
                        raise AssertionError(f"Portable asset checks failed: {failed}")
            return dict(status="passed", identity=identity, metrics=metrics, timings_s=timings)


def run_isolated(
    case: ReliabilityCase, output: Path, *, timeout_s: float, hash_seed: int = 11,
    memory_limit_mib: int = 8192,
) -> dict:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be finite and positive")
    if type(memory_limit_mib) is not int or memory_limit_mib < 0:
        raise ValueError("memory limit must be a nonnegative integer")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "case.json", asdict(case))
    started = time.perf_counter()
    env = dict(
        os.environ,
        PYTHONHASHSEED=str(hash_seed),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with (output / "worker.log").open("w") as log:
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "plume_advanced.evaluation.reliability",
                    "--worker",
                    str(output),
                    "--memory-limit-mib", str(memory_limit_mib),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            result_path = output / "result.json"
            result = (
                json.loads(result_path.read_text())
                if result_path.is_file()
                else {
                    "status": "failed",
                    "reason": f"Worker exited {process.returncode} without a result",
                }
            )
            if process.returncode != 0:
                result["status"] = "failed"
            result["exit_code"] = process.returncode
        except subprocess.TimeoutExpired:
            result = dict(status="timeout", reason=f"Exceeded {timeout_s}s wall-time limit")
    result.update(
        case=asdict(case), elapsed_s=time.perf_counter() - started, python_hash_seed=hash_seed
    )
    write_json(output / "result.json", result)
    return result


def run_campaign(
    cases: list[ReliabilityCase], output: Path, *, timeout_s: float = 180.0, replay: bool = True,
    memory_limit_mib: int = 8192,
) -> dict:
    if not cases or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("At least one case and a positive timeout are required")
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = dict(
        schema="plume.seed-reliability.v1",
        complete=False,
        passed=False,
        source_sha256=package_source_hash(),
        runtime=runtime_identity(),
        memory_limit_mib=memory_limit_mib,
        planned_cases=len(cases),
        replay=replay,
        cases=[],
    )
    for index, case in enumerate(cases):
        result = run_isolated(case, output / f"case_{index:04d}", timeout_s=timeout_s,
                              memory_limit_mib=memory_limit_mib)
        if replay and result["status"] == "passed":
            repeated = run_isolated(
                case, output / f"replay_{index:04d}", timeout_s=timeout_s, hash_seed=37,
                memory_limit_mib=memory_limit_mib,
            )
            result["replay_passed"] = repeated["status"] == "passed" and result[
                "identity"
            ] == repeated.get("identity")
            if not result["replay_passed"]:
                result.update(
                    status="failed", reason="Fresh-process replay mismatch; inspect replay result"
                )
        summary["cases"].append(result)
        write_json(output / "summary.json", summary)
        print(
            f"{index + 1}/{len(cases)} {Path(case.config).stem} {case.body or 'preset'} seed={case.seed}: {result['status']}",
            flush=True,
        )
    summary.update(
        complete=True, passed=all(case["status"] == "passed" for case in summary["cases"])
    )
    summary["source_unchanged"] = summary["source_sha256"] == package_source_hash()
    summary["passed"] &= summary["source_unchanged"]
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 17, 42, 20260910, 4294967295]
    )
    parser.add_argument("--bodies", choices=["earth", "mars", "moon"], nargs="+")
    parser.add_argument("--scope", choices=["sections", "full"], default="sections")
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--memory-limit-mib", type=int, default=8192,
                        help="Linux worker address-space ceiling; 0 disables it. Other platforms record it as unavailable.")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker_source = package_source_hash()
        memory_limited = False
        try:
            if args.memory_limit_mib > 0 and sys.platform == "linux":
                import resource
                limit = args.memory_limit_mib * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                memory_limited = True
            result = execute_case(
                ReliabilityCase(**json.loads((args.worker / "case.json").read_text())), args.worker
            )
        except Exception as error:
            traceback.print_exc()
            result = dict(status="failed", reason=f"{type(error).__name__}: {error}")
        result.update(source_sha256=worker_source, runtime=runtime_identity(),
                      memory_limit_applied=memory_limited)
        if sys.platform == "linux":
            import resource
            result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        write_json(args.worker / "result.json", result)
        return 0 if result["status"] == "passed" else 1
    if not args.configs or not args.output:
        parser.error("--configs and --output are required")
    cases = [
        ReliabilityCase(str(config.resolve()), seed, body, args.scope, args.voxel_size)
        for config in args.configs
        for body in (args.bodies or [None])
        for seed in sorted(set(args.seeds))
    ]
    return (
        0
        if run_campaign(cases, args.output, timeout_s=args.timeout, replay=not args.no_replay,
                        memory_limit_mib=args.memory_limit_mib)[
            "passed"
        ]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
