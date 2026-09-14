"""Isolated seed campaigns exercising the production acceptance and mesh paths.

Every case has a wall-time limit. Failures stay in the report and do not prevent
later cases from running. Replays use a different PYTHONHASHSEED. Screening is
not a geological certificate and a finite campaign cannot prove every seed.
"""

from __future__ import annotations

import argparse
import hashlib
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
from threading import Event, Thread
from typing import Any

from plume_advanced.evaluation.reliability_reports import (
    diagnose,
    process_diagnostic,
    read_result,
    seal_result,
    verify_result,
    write_json,
    write_report,
)
from plume_advanced.evaluation.reliability_state import (
    campaign_lock,
    plan_identity,
    preflight,
    project_inputs,
    read_plan,
)
from plume_advanced.identity import package_source_hash, runtime_identity, sha256_file


@dataclass(frozen=True)
class ReliabilityCase:
    config: str
    seed: int
    body: str | None = None
    scope: str = "sections"
    voxel_size: float | None = None


def execute_case(
    case: ReliabilityCase, output: Path, *, checkpoint_directory: Path | None = None
) -> dict:
    """Run a real case; caller owns process isolation and failure recording."""
    from plume_advanced.config import load_project_config, project_config_manifest
    from plume_advanced.evaluation.artifacts import (
        export_network_artifact,
        export_section_artifact,
        host_semantic_hash,
        network_semantic_hash,
        section_semantic_hash,
    )
    from plume_advanced.progress import progress_scope, report_progress
    from plume_advanced.stages.host_field import HostFieldGenerator
    from plume_advanced.stages.network import CaveNetworkGenerator
    from plume_advanced.stages.section_field import SectionFieldGenerator

    project = load_project_config(case.config, world_body=case.body, seed_override=case.seed)
    if not project.network.quality.enabled:
        raise ValueError("Reliability campaigns require enabled production quality screening")
    if case.voxel_size is not None:
        if case.voxel_size <= 0 or not math.isfinite(case.voxel_size):
            raise ValueError("voxel size must be finite and positive")
        project = replace(project, geometry=replace(project.geometry, voxel_size=case.voxel_size))
    from plume_advanced.pipeline import StageCheckpointStore, pipeline_fingerprint

    output.mkdir(parents=True, exist_ok=True)
    for path in project_inputs(project, Path(case.config)):
        if not path.is_file():
            raise FileNotFoundError(path)
    store = (
        None
        if checkpoint_directory is None
        else StageCheckpointStore(
            checkpoint_directory,
            pipeline_fingerprint(
                project,
                inputs=project_inputs(project, Path(case.config)),
                source_root=Path(__file__).resolve().parents[3],
            ),
        )
    )

    accepted_context = ""

    def cached(name, builder):
        if store is None:
            return builder()
        value, reused = store.load_or_build(name + accepted_context, builder, resume=True)
        report_progress("Checkpoint", detail=f"{name}: {'reused' if reused else 'saved'}")
        return value

    write_json(output / "resolved_config.json", project_config_manifest(project))
    timings = {}
    state = dict(stage="configuration", step="starting", completed=0, total=None, detail="")
    warnings = []
    last_status = 0.0

    @contextmanager
    def stage(name):
        start = time.perf_counter()
        state["stage"] = name
        report_progress(name, detail="starting")
        write_json(output / "status.json", state)
        try:
            yield
        finally:
            timings[name] = time.perf_counter() - start
            write_json(output / "timings.json", timings)

    with (output / "progress.jsonl").open("w") as trace:

        def progress(step, current, total, detail):
            nonlocal last_status
            state.update(step=step, completed=current, total=total, detail=detail)
            if time.monotonic() - last_status >= 1:
                write_json(output / "status.json", state)
                last_status = time.monotonic()
            trace.write(
                json.dumps(
                    dict(
                        stage=state["stage"],
                        step=step,
                        completed=current,
                        total=total,
                        detail=detail,
                    )
                )
                + "\n"
            )
            trace.flush()

        with progress_scope(progress):
            with stage("host"):
                host = cached("host", lambda: HostFieldGenerator(project.host_field).generate())
                before = host_semantic_hash(host)
            with stage("network"):
                network = cached(
                    "network",
                    lambda: CaveNetworkGenerator(project.network).generate(
                        host,
                        section_config=project.section_field,
                        quality_report_path=output / "network_quality.json",
                        quality_progress=lambda detail: report_progress(
                            "Network acceptance", detail=detail
                        ),
                    ),
                )
                write_json(output / "network_quality.json", network.quality_report)
            with stage("sections"):
                sections = cached(
                    "sections",
                    lambda: SectionFieldGenerator(project.section_field).generate(network),
                )
                export_network_artifact(network, output / "stage_b_network.json")
                export_section_artifact(sections, output / "stage_c_sections.npz")
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

                from plume_advanced.evaluation.local_geometry import section_resolution_report
                from plume_advanced.exporters import export_target_asset
                from plume_advanced.pipeline.inspection import complete_inspection
                from plume_advanced.pipeline.recovery import (
                    build_accepted_base,
                    write_recovery_report,
                )
                from plume_advanced.run_manifest import write_run_manifest
                from plume_advanced.stages.events import GeologicalEventGenerator
                from plume_advanced.stages.floor_map import FloorMapGenerator
                from plume_advanced.stages.geometry import GeometryGenerator
                from plume_advanced.stages.surface_topology import (
                    check_closed_surface_topology,
                    component_count,
                )
                from plume_advanced.validation import GlbAsset, PortableAssetValidator

                generator = GeometryGenerator(project.geometry)
                with stage("base_geometry"):
                    accepted = cached(
                        "accepted_base",
                        lambda: build_accepted_base(project, host, network, sections,
                                                    report_path=output / "pipeline_recovery.json",
                                                    progress=progress),
                    )
                    network, sections, base = accepted.network, accepted.sections, accepted.geometry
                    generator = GeometryGenerator(base.config)
                    accepted_context = "_" + accepted.context_sha256[:16]
                    write_recovery_report(accepted, output / "pipeline_recovery.json")
                    # Earlier B/C files are provisional until D1 accepts a realization.
                    export_network_artifact(network, output / "stage_b_network.json")
                    export_section_artifact(sections, output / "stage_c_sections.npz")
                    write_json(output / "network_quality.json", network.quality_report)
                    identity.update(network=network_semantic_hash(network),
                                    sections=section_semantic_hash(sections),
                                    recovery=hashlib.sha256(json.dumps(accepted.report, sort_keys=True).encode()).hexdigest())
                    metrics.update(segment_count=len(network.segments),
                                   combined_length_m=sum(s.total_length for s in network.segments),
                                   selected_attempt=network.quality_report["selected_attempt"],
                                   repair_pass=network.quality_report["selected_repair_pass"],
                                   checks=len(accepted.report["attempts"][-1]["assessment"]["checks"]),
                                   recovery_outcome=accepted.report["outcome"],
                                   recovery_attempts=len(accepted.report["attempts"]))
                resolution = section_resolution_report(sections, base.voxel_grid.voxel_size)
                write_json(output / "resolution.json", resolution)
                if resolution["under_resolved_count"]:
                    warnings.append(
                        f"{resolution['under_resolved_count']}/{resolution['section_count']} input profiles have fewer than eight voxel samples across their smallest dimension. Inspect local mesh clearance and resolution convergence."
                    )
                floors = FloorMapGenerator(project.floor_map)
                with stage("base_floor"):
                    atlas = cached("base_floor", lambda: floors.generate(network, sections, base))
                with stage("events"):
                    events = GeologicalEventGenerator(project.events).generate(
                        sections, base, atlas, progress=progress
                    )
                with stage("final_geometry"):
                    geometry = generator.finalize(base, events, progress=progress)
                    write_json(
                        output / "surface_quality.json",
                        dict(
                            attempts=[dict(record) for record in geometry.surface_quality_records],
                            relief_scale=geometry.effective_surface_relief_scale,
                            closing_voxels=geometry.effective_density_closing_voxels,
                            opening_voxels=geometry.effective_density_opening_voxels,
                        ),
                    )
                    if (
                        geometry.effective_surface_relief_scale < 1
                        or geometry.effective_density_opening_voxels
                    ):
                        warnings.append(
                            "Surface acceptance reduced detail or removed grid-scale air bridges. Inspect surface_quality.json for the effective settings."
                        )
                    digest = hashlib.sha256()
                    for buffer in (geometry.assembled_vertices, geometry.assembled_faces):
                        array = np.ascontiguousarray(buffer)
                        digest.update(str((array.shape, array.dtype.str)).encode())
                        digest.update(memoryview(array).cast("B"))
                    identity["mesh"] = digest.hexdigest()
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
                    identity["texture_recovery"] = sha256_file(
                        next(path for path in exported.files if path.name == "texture_recovery.json")
                    )
                with stage("pipeline_inspection"):
                    inspection_files = complete_inspection(geometry, exported, resolution, output)
                    inspection = json.loads(inspection_files[0].read_text())
                    warnings = list(dict.fromkeys([*warnings, *inspection["warnings"]]))
                    metrics.update(
                        inspected_centers=inspection["export_inspection"]["visual"][
                            "inspected_centers"
                        ],
                        surface_attempts=len(geometry.surface_quality_records),
                        visual_attempts=len(inspection["export_inspection"]["visual_attempts"]),
                        texture_recovery_outcome=inspection["texture_recovery"]["outcome"],
                        texture_map_repairs=sum(bool(a["repairs"]) for a in inspection["texture_recovery"]["assets"]),
                        texture_package_attempts=len(inspection["texture_recovery"]["package_attempts"]),
                        collision_raw_fallback=inspection["export_inspection"]["collision"][
                            "used_raw_fallback"
                        ],
                    )
                    manifest = write_run_manifest(
                        project,
                        output / "run_manifest.json",
                        outputs=(
                            *exported.files,
                            *inspection_files,
                            output / "resolution.json",
                            output / "network_quality.json",
                            output / "surface_quality.json",
                            output / "pipeline_recovery.json",
                            output / "stage_b_network.json",
                            output / "stage_c_sections.npz",
                            output / "stage_c_sections.json",
                        ),
                        elapsed_seconds=sum(timings.values()),
                        source_root=Path(case.config).parent.parent,
                    )
                with stage("portable_validation"):
                    profile = (
                        "textured"
                        if all(
                            (
                                project.geometry.cave_diffuse_texture,
                                project.geometry.cave_normal_texture,
                                project.geometry.cave_roughness_texture,
                            )
                        )
                        else "neutral"
                    )
                    checks = PortableAssetValidator(
                        exported.primary_asset, run_manifest_path=manifest, material_profile=profile
                    ).validate()
                    write_json(output / "asset_checks.json", [asdict(check) for check in checks])
                    failed = [check.name for check in checks if not check.passed]
                    if failed:
                        raise AssertionError(f"Portable asset checks failed: {failed}")
                    glb = GlbAsset(exported.primary_asset)
                    _, _, primitive = glb.cave_primitive()
                    positions, inverse = np.unique(
                        glb.accessor(primitive["attributes"]["POSITION"]),
                        axis=0,
                        return_inverse=True,
                    )
                    faces = inverse[glb.accessor(primitive["indices"]).reshape((-1, 3))]
                    # Event-free expected topology is checked again on actual exported positions,
                    # rejoining UV seams exactly, never rounding narrow triangles away.
                    if (
                        not geometry.structural_event_ids
                        and geometry.expected_surface_genus is not None
                    ):
                        topology = check_closed_surface_topology(
                            positions,
                            faces,
                            component_count(faces),
                            geometry.expected_surface_genus,
                        )
                        write_json(output / "export_topology.json", topology)
            return dict(
                status="passed",
                identity=identity,
                metrics=metrics,
                timings_s=timings,
                warnings=warnings,
            )


def _latest_stage(output: Path) -> str:
    try:
        return json.loads((output / "status.json").read_text())["stage"]
    except (OSError, ValueError, KeyError, TypeError):
        return "configuration"


def _heartbeat(output: Path, stopped: Event) -> None:
    started = time.monotonic()
    while not stopped.wait(10):
        try:
            status = json.loads((output / "status.json").read_text())
            count = (
                f"{status['completed']}/{status['total']}"
                if status.get("total") is not None
                else "working"
            )
            print(
                f"  {time.monotonic() - started:.0f}s · {status['stage']} / "
                f"{status['step']} · {count} · {status['detail']}",
                flush=True,
            )
        except (OSError, ValueError, KeyError, TypeError):
            print(f"  {time.monotonic() - started:.0f}s · worker starting", flush=True)


def run_isolated(
    case: ReliabilityCase,
    output: Path,
    *,
    timeout_s: float,
    hash_seed: int = 11,
    memory_limit_mib: int = 8192,
    checkpoint_directory: Path | None = None,
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
    command = [
        sys.executable,
        "-m",
        "plume_advanced.evaluation.reliability",
        "--worker",
        str(output),
        "--memory-limit-mib",
        str(memory_limit_mib),
    ]
    if checkpoint_directory is not None:
        command.extend(["--checkpoint-directory", str(checkpoint_directory)])
    stopped = Event()
    heartbeat = Thread(target=_heartbeat, args=(output, stopped), daemon=True)
    heartbeat.start()
    try:
        with (output / "worker.log").open("w") as log:
            try:
                process = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s
                )
                try:
                    result = json.loads((output / "result.json").read_text())
                    if not isinstance(result, dict) or result.get("status") not in {
                        "passed",
                        "failed",
                    }:
                        raise ValueError("Invalid worker result")
                    if result["status"] == "passed" and not result.get("identity"):
                        raise ValueError("Passing worker has no deterministic identity")
                except (OSError, ValueError):
                    result = dict(
                        status="failed",
                        reason=f"Worker exited {process.returncode} without a valid result",
                    )
                if process.returncode != 0:
                    result["status"] = "failed"
                result["exit_code"] = process.returncode
                if result["status"] != "passed" and "diagnostic" not in result:
                    result["diagnostic"] = process_diagnostic(
                        "worker_crash",
                        result.get("reason", f"Worker exited {process.returncode}"),
                        _latest_stage(output),
                    )
            except subprocess.TimeoutExpired:
                message = f"Exceeded {timeout_s}s wall-time limit"
                result = dict(
                    status="timeout",
                    reason=message,
                    diagnostic=process_diagnostic("timeout", message, _latest_stage(output)),
                )
            except OSError as error:
                result = dict(
                    status="failed", reason=str(error), diagnostic=diagnose(error, "worker")
                )
            except KeyboardInterrupt:
                seal_result(
                    output,
                    dict(
                        status="interrupted",
                        case=asdict(case),
                        diagnostic=process_diagnostic(
                            "interrupted", "Interrupted by user", _latest_stage(output)
                        ),
                    ),
                )
                raise
    finally:
        stopped.set()
        heartbeat.join()
    result.update(
        case=asdict(case), elapsed_s=time.perf_counter() - started, python_hash_seed=hash_seed
    )
    return seal_result(output, result)


def _attempt(
    case: ReliabilityCase,
    root: Path,
    output: Path,
    *,
    report_only: bool,
    timeout_s: float,
    memory_limit_mib: int,
    hash_seed: int = 11,
) -> dict:
    attempts = sorted(root.glob("attempt_[0-9][0-9][0-9][0-9]"))
    integrity_reason = None
    if attempts:
        previous, integrity_reason = verify_result(attempts[-1])
        if previous is not None:
            previous.update(directory=attempts[-1].relative_to(output).as_posix(), reused=True)
            return previous
    if report_only:
        previous = read_result(attempts[-1]) if attempts else None
        if previous is not None and previous.get("status") != "passed":
            previous.update(directory=attempts[-1].relative_to(output).as_posix(), reused=True)
            return previous
        return dict(
            status="failed" if attempts else "pending",
            case=asdict(case),
            directory=attempts[-1].relative_to(output).as_posix() if attempts else None,
            diagnostic=process_diagnostic(
                "artifact_integrity", integrity_reason or "Case has not run"
            ),
        )
    root.mkdir(parents=True, exist_ok=True)
    number = int(attempts[-1].name.split("_")[-1]) + 1 if attempts else 0
    if number > 9999:
        raise ValueError("Too many retained attempts; use a new campaign")
    target = root / f"attempt_{number:04d}"
    result = run_isolated(
        case,
        target,
        timeout_s=timeout_s,
        hash_seed=hash_seed,
        memory_limit_mib=memory_limit_mib,
        # A replay is always cold, including after interruption.
        checkpoint_directory=root / "checkpoints" if hash_seed == 11 else None,
    )
    result.setdefault("case", asdict(case))
    result.update(directory=target.relative_to(output).as_posix(), reused=False)
    return result


def run_campaign(
    cases: list[ReliabilityCase],
    output: Path,
    *,
    timeout_s: float = 180.0,
    replay: bool = True,
    memory_limit_mib: int = 8192,
    resume: bool = False,
    report_only: bool = False,
) -> dict:
    if not cases or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("At least one case and a positive timeout are required")
    if type(memory_limit_mib) is not int or memory_limit_mib < 0:
        raise ValueError("memory limit must be a nonnegative integer")
    for case in cases:
        if type(case.seed) is not int or case.seed < 0:
            raise ValueError("seeds must be nonnegative integers")
        if case.scope not in {"sections", "full"}:
            raise ValueError("scope must be sections or full")
        if case.voxel_size is not None and (
            not math.isfinite(case.voxel_size) or case.voxel_size <= 0
        ):
            raise ValueError("voxel size must be finite and positive")
    if report_only and not resume:
        raise ValueError("--report-only requires --resume")
    source = package_source_hash()
    identity = plan_identity(cases, source=source, replay=replay)
    if resume:
        if read_plan(output) != identity:
            raise ValueError(
                "Campaign inputs, code, runtime or replay policy changed. Use a new output directory; previous evidence remains unchanged."
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    with campaign_lock(output):
        if not resume:
            write_json(output / "plan.json", identity)
        summary: dict[str, Any] = dict(
            schema="plume.seed-reliability.v2",
            complete=False,
            passed=False,
            source_sha256=source,
            runtime=runtime_identity(),
            memory_limit_mib=memory_limit_mib,
            timeout_s=timeout_s,
            planned_cases=len(cases),
            replay=replay,
            cases=[],
        )

        def publish():
            write_json(output / "summary.json", summary)
            write_report(output, summary)

        publish()  # Durable before the first worker starts, including on interruption.
        try:
            for index, case in enumerate(cases):
                print(
                    f"{index + 1}/{len(cases)} {Path(case.config).stem} {case.body or 'preset'} seed={case.seed}",
                    flush=True,
                )
                result = _attempt(
                    case,
                    output / f"case_{index:04d}",
                    output,
                    report_only=report_only,
                    timeout_s=timeout_s,
                    memory_limit_mib=memory_limit_mib,
                )
                if replay and result["status"] == "passed":
                    print("  Checking fresh-process replay", flush=True)
                    repeated = _attempt(
                        case,
                        output / f"replay_{index:04d}",
                        output,
                        report_only=report_only,
                        timeout_s=timeout_s,
                        hash_seed=37,
                        memory_limit_mib=memory_limit_mib,
                    )
                    result["replay_directory"] = repeated.get("directory")
                    result["replay_passed"] = repeated["status"] == "passed" and result[
                        "identity"
                    ] == repeated.get("identity")
                    if not result["replay_passed"]:
                        result.update(
                            status="failed",
                            reason="Fresh-process replay failed or mismatched",
                            diagnostic=process_diagnostic(
                                "replay_mismatch",
                                "Fresh-process replay failed or mismatched; inspect the replay directory",
                            ),
                        )
                        result["replay_diagnostic"] = repeated.get("diagnostic")
                summary["cases"].append(result)
                publish()
                print(
                    f"  {result['status']}"
                    + (" (verified saved result)" if result.get("reused") else ""),
                    flush=True,
                )
        except KeyboardInterrupt:
            summary["campaign_error"] = (
                "Interrupted. Resume the same command; validated checkpoints and prior attempts are retained."
            )
            publish()
            raise
        except Exception as error:
            summary["campaign_error"] = f"Campaign stopped: {type(error).__name__}: {error}"
            publish()
            raise
        summary.update(
            complete=all(r["status"] != "pending" for r in summary["cases"]),
            passed=all(r["status"] == "passed" for r in summary["cases"]),
        )
        summary["source_unchanged"] = source == package_source_hash()
        summary["inputs_unchanged"] = identity == plan_identity(cases, source=source, replay=replay)
        summary["passed"] &= summary["source_unchanged"] and summary["inputs_unchanged"]
        if not summary["source_unchanged"] or not summary["inputs_unchanged"]:
            summary["campaign_error"] = (
                "Code, runtime or input contents changed during this run. Results are not a reproducible passing campaign. Start a new campaign with fixed inputs."
            )
        publish()
        print(f"Report: {output.resolve() / 'report.html'}", flush=True)
        return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--bodies", choices=["earth", "mars", "moon"], nargs="+")
    parser.add_argument("--scope", choices=["sections", "full"])
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument(
        "--timeout", type=float, help="Seconds per worker; default 600 for sections, 3600 for full"
    )
    parser.add_argument(
        "--memory-limit-mib",
        type=int,
        default=8192,
        help="Linux worker address-space ceiling; 0 disables it. Other platforms record it as unavailable.",
    )
    parser.add_argument(
        "--no-replay", action="store_true", help="Explicitly omit cold repeatability checks"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify completed cases and resume unfinished cases without changing seeds",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="With --resume, recheck saved evidence without generation",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check configuration and input availability without generation",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-directory", type=Path, help=argparse.SUPPRESS)
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
                ReliabilityCase(**json.loads((args.worker / "case.json").read_text())),
                args.worker,
                checkpoint_directory=args.checkpoint_directory,
            )
        except Exception as error:
            traceback.print_exc()
            from plume_advanced.pipeline.inspection import record_failure

            try:
                record_failure(args.worker, error, _latest_stage(args.worker))
            except OSError:
                # Preserve the original failure if its cause prevents writing
                # another report (for example a full filesystem).
                pass
            result = dict(
                status="failed",
                reason=f"{type(error).__name__}: {error}",
                diagnostic=diagnose(error, _latest_stage(args.worker)),
            )
        result.update(
            source_sha256=worker_source,
            runtime=runtime_identity(),
            memory_limit_applied=memory_limited,
        )
        if worker_source != package_source_hash():
            result.update(status="failed", reason="Production code changed during worker execution")
        if sys.platform == "linux":
            import resource

            result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        write_json(args.worker / "result.json", result)
        return 0 if result["status"] == "passed" else 1
    try:
        if args.resume and not args.configs:
            if not args.output:
                parser.error("--resume requires --output")
            if (
                args.seeds
                or args.bodies
                or args.scope
                or args.voxel_size is not None
                or args.no_replay
            ):
                parser.error(
                    "When resuming without --configs, the saved plan supplies seeds, bodies, scope, voxel size and replay policy"
                )
            plan = read_plan(args.output)
            cases = [ReliabilityCase(**record) for record in plan["cases"]]
            replay = plan["replay"]
        else:
            if not args.configs or (not args.output and not args.preflight):
                parser.error(
                    "--configs and --output are required (output optional for --preflight)"
                )
            cases = [
                ReliabilityCase(
                    str(config.resolve()), seed, body, args.scope or "sections", args.voxel_size
                )
                for config in args.configs
                for body in (args.bodies or [None])
                for seed in sorted(
                    set(
                        args.seeds
                        if args.seeds is not None
                        else [0, 1, 2, 3, 17, 42, 20260910, 4294967295]
                    )
                )
            ]
            replay = not args.no_replay
        if args.preflight:
            report = preflight(cases, args.output)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
        timeout = (
            args.timeout
            if args.timeout is not None
            else (3600.0 if any(c.scope == "full" for c in cases) else 600.0)
        )
        result = run_campaign(
            cases,
            args.output,
            timeout_s=timeout,
            replay=replay,
            memory_limit_mib=args.memory_limit_mib,
            resume=args.resume,
            report_only=args.report_only,
        )
        return 0 if result["passed"] else 1
    except (ValueError, FileExistsError, RuntimeError) as error:
        print(f"PLUME: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("PLUME interrupted; resume with --output DIRECTORY --resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
