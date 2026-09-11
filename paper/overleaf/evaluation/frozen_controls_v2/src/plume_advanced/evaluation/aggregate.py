"""Aggregate saved raw/summary artifacts without rerunning generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plume_advanced.evaluation.provenance import semantic_hash


def aggregate_results(results_root: str | Path) -> dict[str, Any]:
    root = Path(results_root)
    summaries: dict[str, Any] = {}
    for path in sorted(root.glob("*/summary.json")):
        summaries[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "schema": "plume.paper-aggregate.v1",
        "experiment_count": len(summaries),
        "experiments": summaries,
    }
    result["semantic_sha256"] = semantic_hash(result)
    (root / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(root, result)
    return result


def _write_report(root: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# PLUME-Advanced paper results report",
        "",
        f"Aggregate artifact hash: `{aggregate['semantic_sha256']}`",
        "",
        "This report distinguishes measured experiment outputs from generator assumptions. "
        "Planetary presets remain controlled extrapolation scenarios; only terrestrial PDC "
        "sections provide an external morphometric reference.",
        "",
        "## Experiment status",
        "",
        "| Experiment | Complete | Failed/timeout | Notes |",
        "|---|---:|---:|---|",
    ]
    for name, summary in aggregate["experiments"].items():
        complete = summary.get("complete_n", summary.get("complete_worlds", "n/a"))
        failed = summary.get("failed_n", summary.get("failed_worlds", "n/a"))
        note = "Inspect summary and raw cases before making a paper claim."
        lines.append(f"| {name} | {complete} | {failed} | {note} |")
    lines.extend(
        [
            "",
            "## Scientific interpretation guardrails",
            "",
            "- Morphometry supports only claims about selected descriptor distributions, not full geological accuracy.",
            "- Host ablations show whether process-informed routing terms are behaviorally active; they do not validate lava emplacement physics.",
            "- Moon and Mars results describe parameterized scenario families, not physical validation.",
            "- Failed and invalid cases remain in each experiment's raw result index.",
            "",
            "## Unsupported claims",
            "",
            "Do not claim identical rendering, physics, or sensor behavior across engines without completed manual imports. "
            "Do not claim adaptive-sampling advantage unless the saved matched-fidelity results show it.",
            "",
        ]
    )
    (root / "PAPER_RESULTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = ["aggregate_results"]
