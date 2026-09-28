#!/usr/bin/env python3
"""Run one cap-free scalability case while protecting the shared workstation.

The experiment's declared software RSS limit stays disabled. This supervisor
only terminates a worker when *host-wide* RAM and swap approach exhaustion;
it writes an explicit operator-stop receipt and preserves the case record.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-available-gib", type=float, default=2.5)
    parser.add_argument("--minimum-swap-used-gib", type=float, default=30.)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = ["uv", "run", "--extra", "paper", "plume-evaluate", "--config",
               str(args.config.resolve()), "scalability"]
    started = time.monotonic()
    stop = None
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/plume-matplotlib"
    with (out / "console.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        parent = psutil.Process(process.pid)
        while process.poll() is None:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            available = memory.available / 2**30
            swap_used = swap.used / 2**30
            if (available <= 1.5 or
                    (available <= args.minimum_available_gib
                     and swap_used >= args.minimum_swap_used_gib)):
                children = parent.children(recursive=True)
                workers = []
                for child in children:
                    try:
                        workers.append((child.memory_info().rss, child))
                    except psutil.NoSuchProcess:
                        pass
                if workers:
                    worker_rss, worker = max(workers, key=lambda row: row[0])
                    stop = {
                        "schema": "plume.resource-stop.v1",
                        "reason": "Operator safety stop of cap-free worker on shared host",
                        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
                        "worker_pid": worker.pid,
                        "worker_rss_gib_at_stop": worker_rss / 2**30,
                        "host_memory_total_gib": memory.total / 2**30,
                        "host_memory_available_gib": available,
                        "host_swap_total_gib": swap.total / 2**30,
                        "host_swap_used_gib": swap_used,
                        "software_memory_limit_gib": 0,
                        "elapsed_s_at_stop": time.monotonic() - started,
                        "status": "operator_stopped_not_timing_result",
                    }
                    (out / "operator_stop.json").write_text(
                        json.dumps(stop, indent=2) + "\n")
                    worker.send_signal(signal.SIGTERM)
                    print(f"Stopped worker {worker.pid} at {worker_rss / 2**30:.2f} GiB RSS",
                          flush=True)
                    break
            time.sleep(1.)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            code = process.wait(timeout=30)
    print(json.dumps({"returncode": code, "elapsed_s": time.monotonic()-started,
                      "operator_stop": stop is not None, "console_log": str(out / "console.log")}),
          flush=True)


if __name__ == "__main__":
    main()
