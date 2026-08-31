"""Traceable LaTeX macros/tables derived from saved summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def generate_latex(results_root: str | Path) -> list[Path]:
    root = Path(results_root)
    output = root / "latex"
    output.mkdir(parents=True, exist_ok=True)
    aggregate_path = root / "aggregate_summary.json"
    if not aggregate_path.is_file():
        from plume_advanced.evaluation.aggregate import aggregate_results

        aggregate_results(root)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    experiments = aggregate.get("experiments", {})
    morphology = experiments.get("morphometry", {})
    sampling = experiments.get("sampling_ablation", {})
    controllability = experiments.get("controllability", {})
    scalability = experiments.get("scalability", {})
    exports = experiments.get("export_consistency", {})
    lines = [f"% Generated from aggregate_summary.json sha256={digest}"]
    _macro(lines, "PaperMorphWorlds", morphology.get("complete_worlds"))
    _macro(lines, "PaperGeneratedSections", morphology.get("generated_sections"))
    _macro(lines, "PaperAggregateNWOne", morphology.get("aggregate_normalized_w1"), precision=3)
    adaptive = sampling.get("median_adaptive_count")
    uniform = sampling.get("median_uniform_count")
    reduction = None
    if adaptive is not None and uniform:
        reduction = 100.0 * (float(uniform) - float(adaptive)) / float(uniform)
    _macro(lines, "PaperAdaptiveReduction", reduction, precision=1, suffix="\\%")
    _macro(lines, "PaperScalabilityComplete", scalability.get("complete_n"))
    _macro(lines, "PaperExportComplete", exports.get("complete_n"))
    _macro(lines, "PaperManualImports", exports.get("manual_imports_completed"))
    for control, payload in controllability.get("controls", {}).items():
        _macro(
            lines,
            f"Paper{control.title()}Spearman",
            payload.get("spearman_rho"),
            precision=3,
        )
    metrics_path = output / "paper_metrics.tex"
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths = [metrics_path]
    metrics = morphology.get("metrics", [])
    if metrics:
        table = output / "table_morphometry.tex"
        table.write_text(_morphometry_table(metrics, digest), encoding="utf-8")
        paths.append(table)
    controls = controllability.get("controls", {})
    if controls:
        table = output / "table_controllability.tex"
        table.write_text(_controllability_table(controls, digest), encoding="utf-8")
        paths.append(table)
    return paths


def _macro(
    lines: list[str],
    name: str,
    value: Any,
    *,
    precision: int | None = None,
    suffix: str = "",
) -> None:
    if value is None:
        rendered = "NA"
    elif precision is not None:
        rendered = f"{float(value):.{precision}f}{suffix}"
    else:
        rendered = f"{value}{suffix}"
    lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")


def _morphometry_table(metrics: list[dict[str, Any]], digest: str) -> str:
    lines = [
        f"% Generated from aggregate_summary.json sha256={digest}",
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Metric & PLUME NW1 & Baseline NW1 & KS \\\\",
        "\\hline",
    ]
    for row in metrics:
        name = str(row["metric"]).replace("_", "\\_")
        lines.append(
            f"{name} & {row['generated_normalized_w1']:.3f} & "
            f"{row['baseline_normalized_w1']:.3f} & {row['generated_ks']:.3f} \\\\"
        )
    lines.extend(("\\hline", "\\end{tabular}", ""))
    return "\n".join(lines)


def _controllability_table(controls: dict[str, dict[str, Any]], digest: str) -> str:
    lines = [
        f"% Generated from aggregate_summary.json sha256={digest}",
        "\\begin{tabular}{llrr}",
        "\\hline",
        "Control & Primary metric & $n$ & Spearman $\\rho$ \\\\",
        "\\hline",
    ]
    for control, row in controls.items():
        metric = str(row.get("primary_metric", "NA")).replace("_", "\\_")
        rho = row.get("spearman_rho")
        rendered_rho = "NA" if rho is None else f"{float(rho):.3f}"
        lines.append(
            f"{control.replace('_', '\\_')} & {metric} & {row.get('n', 'NA')} & {rendered_rho} \\\\"
        )
    lines.extend(("\\hline", "\\end{tabular}", ""))
    return "\n".join(lines)


__all__ = ["generate_latex"]
