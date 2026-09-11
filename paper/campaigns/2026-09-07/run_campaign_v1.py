"""Run every declared experiment against the preserved source snapshot."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[2]
FROZEN = CAMPAIGN / "frozen"
OUTPUTS = ROOT / "paper/outputs"
DATA = ROOT / "data/reference/pdc_v2/extracted/Pyroduct Digital Catalog in .txt"
COMMANDS = (
    "audit", "morphometry", "host-ablation", "scalability",
    "controllability", "sampling-ablation", "determinism", "export-consistency",
    "aggregate", "figures", "latex",
)


def main():
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(FROZEN / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "PLUME_EVALUATION_WORKERS": "4",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/plume-evaluation-mpl",
    })
    log_root = CAMPAIGN / "logs"
    log_root.mkdir(exist_ok=True)
    status_path = CAMPAIGN / "execution.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "commands": {}, "status": "running",
    }
    def save():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)
    for name in COMMANDS:
        # An unsuccessful experiment still counts as executed; its failed cases
        # remain evidence. Explicit reruns require removing this command entry.
        if name in status["commands"] and "returncode" in status["commands"][name]:
            continue
        command = [sys.executable, "-m", "plume_advanced.evaluation.cli", "--config",
                   str(FROZEN / "paper/experiments.toml"), name]
        if name == "morphometry":
            command.extend(("--data-root", str(DATA)))
        started = time.monotonic()
        status["active_command"] = name
        status["commands"][name] = {
            "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "argv": command, "log": str(log_root / f"{name}.log"),
        }
        save()
        print(f"Starting {name}", flush=True)
        with (log_root / f"{name}.log").open("a") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        status["commands"][name].update({
            "returncode": result.returncode,
            "elapsed_s": time.monotonic() - started,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        })
        save()
        print(f"Finished {name}: exit {result.returncode}", flush=True)
    status.update(status="finished", active_command=None,
                  finished_at_utc=datetime.datetime.now(datetime.UTC).isoformat())
    save()
    return int(any(item["returncode"] for item in status["commands"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
