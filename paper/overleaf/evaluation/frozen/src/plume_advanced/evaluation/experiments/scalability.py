"""Child-process wall-time and peak-RSS scalability benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult


def run_scalability(config: EvaluationConfig, *, force: bool = False) -> dict:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("scalability requires the optional 'paper' dependencies") from error
    section = config.section("scalability")
    lengths = tuple(
        float(value) for value in section.get("route_lengths_m", (500, 1000, 2000, 5000))
    )
    modes = tuple(str(value) for value in section.get("storage_modes", ("dense", "tiled")))
    timeout_s = float(section.get("timeout_s", 7200.0))
    memory_limit_gib = float(section.get("memory_limit_gib", 12.0))
    quality = str(section.get("quality", "standard"))
    if timeout_s <= 0 or memory_limit_gib <= 0:
        raise ValueError("Benchmark time and memory limits must be positive")
    body = str(section.get("body", "earth"))
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("scalability")),
    )
    identity = str(provenance["identity_sha256"])
    store = ResultStore(config.output_root, "scalability", provenance_sha256=identity)
    for seed in config.seeds("scalability"):
        for length in lengths:
            for mode in modes:
                template = ExperimentResult(
                    experiment_name="scalability",
                    run_id=f"seed-{seed:06d}-{int(length):06d}m-{mode}",
                    condition_id=f"{int(length)}m-{mode}",
                    seed=seed,
                    status="complete",
                    git_commit=str(provenance["git_commit"]),
                    git_dirty=bool(provenance["git_dirty"]),
                    resolved_config_sha256=str(provenance["resolved_config_sha256"]),
                    provenance_sha256=identity,
                )

                def operation(seed=seed, length=length, mode=mode):
                    with tempfile.TemporaryDirectory(prefix="plume-scalability-") as directory:
                        result_path = Path(directory) / "result.json"
                        command = (
                            sys.executable,
                            "-m",
                            "plume_advanced.evaluation.experiments.scalability_worker",
                            "--project-config",
                            str(config.project_config),
                            "--body",
                            body,
                            "--seed",
                            str(seed),
                            "--route-length-m",
                            str(length),
                            "--storage-mode",
                            mode,
                            "--quality",
                            quality,
                            "--output",
                            str(result_path),
                        )
                        measurements = monitor_worker(command, Path(directory), timeout_s,
                                                      memory_limit_gib, psutil)
                        if not result_path.is_file():
                            raise RuntimeError("Worker completed without a result file")
                        payload = json.loads(result_path.read_text(encoding="utf-8"))
                        return {
                            "route_length_requested_m": length,
                            "storage_mode_requested": mode,
                            **measurements,
                            **payload,
                        }

                run_case(store, template, operation, force=force)
    rows = store.rows()
    summary = {
        "schema": "plume.scalability-summary.v1",
        "environment": provenance,
        "planned_n": len(rows),
        "complete_n": sum(row["status"] == "complete" for row in rows),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "memory_limit_gib": memory_limit_gib,
        "timeout_s": timeout_s,
        "timing_scope": "host through finalized cave mesh; excludes optional events, appearance and export",
        "conditions": summarize_conditions(rows, lengths, modes),
    }
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (store.root / "environment.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def monitor_worker(command, directory, timeout_s, memory_limit_gib, psutil):
    """Keep logs off pipes, bound RSS, and preserve failure measurements."""
    log_path = directory / "worker.log"
    started = time.perf_counter()
    peak_rss = 0
    failure = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        monitored = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    family = [monitored, *monitored.children(recursive=True)]
                    peak_rss = max(peak_rss, sum(p.memory_info().rss for p in family if p.is_running()))
                except psutil.Error:
                    pass
                if peak_rss > memory_limit_gib * 1024**3:
                    failure = "memory_limit"
                elif time.perf_counter() - started > timeout_s:
                    failure = "timeout"
                if failure:
                    for child in monitored.children(recursive=True):
                        child.kill()
                    process.kill()
                    break
                time.sleep(.05)
            process.wait()
        finally:
            if process.poll() is None:
                for child in monitored.children(recursive=True):
                    child.kill()
                process.kill()
                process.wait()
    measurements = {"peak_rss_bytes": peak_rss, "peak_rss_gib": peak_rss / 1024**3,
                    "wall_time_s": time.perf_counter()-started,
                    "memory_limit_gib": memory_limit_gib, "timeout_s": timeout_s,
                    "worker_returncode": process.returncode}
    if failure or process.returncode:
        message = f"{failure or 'worker_error'}: exit {process.returncode}; {log_path.read_text()[-2000:]}"
        error = TimeoutError(message) if failure == "timeout" else RuntimeError(message)
        setattr(error, "metrics", {**measurements, "failure_kind": failure or "worker_error"})
        raise error
    return measurements


def summarize_conditions(rows, lengths, modes):
    result = {}
    for length in lengths:
        for mode in modes:
            key = f"{int(length)}m-{mode}"
            selected = [r for r in rows if r["condition_id"] == key]
            complete = [r for r in selected if r["status"] == "complete"]
            result[key] = {"planned_n": len(selected), "complete_n": len(complete),
                "failed_n": sum(r["status"] != "complete" for r in selected),
                "median_wall_time_s": float(np.median([r["wall_time_s"] for r in complete])) if complete else None,
                "median_peak_rss_gib": float(np.median([r["peak_rss_gib"] for r in complete])) if complete else None}
    return result


__all__ = ["run_scalability"]
