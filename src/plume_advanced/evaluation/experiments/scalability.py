"""Child-process wall-time and peak-RSS scalability benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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
    body = str(section.get("body", "earth"))
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("scalability")),
    )
    store = ResultStore(config.output_root, "scalability")
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
                            "--output",
                            str(result_path),
                        )
                        started = time.perf_counter()
                        process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        monitored = psutil.Process(process.pid)
                        peak_rss = 0
                        while process.poll() is None:
                            elapsed = time.perf_counter() - started
                            if elapsed > timeout_s:
                                process.kill()
                                process.wait()
                                raise TimeoutError(f"case exceeded {timeout_s:.0f} s")
                            try:
                                family = [monitored, *monitored.children(recursive=True)]
                                peak_rss = max(
                                    peak_rss,
                                    sum(
                                        member.memory_info().rss
                                        for member in family
                                        if member.is_running()
                                    ),
                                )
                            except psutil.Error:
                                pass
                            time.sleep(0.05)
                        stdout, stderr = process.communicate()
                        wall_time = time.perf_counter() - started
                        if process.returncode != 0 or not result_path.is_file():
                            raise RuntimeError(
                                f"child exited {process.returncode}: {(stderr or stdout)[-2000:]}"
                            )
                        payload = json.loads(result_path.read_text(encoding="utf-8"))
                        return {
                            "route_length_requested_m": length,
                            "storage_mode_requested": mode,
                            "peak_rss_bytes": peak_rss,
                            "peak_rss_gib": peak_rss / (1024.0**3),
                            "wall_time_s": wall_time,
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
    }
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (store.root / "environment.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = ["run_scalability"]
