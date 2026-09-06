from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "emplacement_backend_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("emplacement_backend_benchmark", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_benchmark = _MODULE.run_benchmark


def test_emplacement_backend_benchmark_smoke(tmp_path) -> None:
    report = run_benchmark(
        tmp_path,
        seeds=(2,),
        families=("natural", "monotonic"),
        backends=("internal",),
        flowy_executable=None,
    )

    assert report["recommendation"]["selected_default"] == "internal"
    assert len(report["cases"]) == 2
    assert all(case["success"] and case["valid"] for case in report["cases"])
    assert all(case["deterministic"] for case in report["cases"])
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "network_diagrams.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "scorecard.png").exists()
    persisted = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert persisted["protocol"]["stage"] == "B"
    assert persisted["cases"] == report["cases"]
