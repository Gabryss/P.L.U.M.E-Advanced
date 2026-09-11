"""Command-line interface for reproducible scientific evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.aggregate import aggregate_results
from plume_advanced.evaluation.config import load_evaluation_config
from plume_advanced.evaluation.datasets.pdc import audit_pdc
from plume_advanced.evaluation.experiments.controllability import run_controllability
from plume_advanced.evaluation.experiments.determinism import run_determinism
from plume_advanced.evaluation.experiments.export_consistency import run_export_consistency
from plume_advanced.evaluation.experiments.host_ablation import run_host_ablation
from plume_advanced.evaluation.experiments.morphometry import run_morphometry
from plume_advanced.evaluation.experiments.sampling_ablation import run_sampling_ablation
from plume_advanced.evaluation.experiments.scalability import run_scalability
from plume_advanced.evaluation.latex import generate_latex
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.runner import write_experiment_manifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "paper" / "experiments.toml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Audit evaluation readiness and resolved defaults.")
    pdc = subparsers.add_parser("pdc-audit", help="Audit a PDC v2.0 TXT tree.")
    pdc.add_argument("--data-root", type=Path, default=None)
    pdc.add_argument("--implicit-closure", action="store_true")
    for name in (
        "morphometry",
        "controllability",
        "host-ablation",
        "sampling-ablation",
        "scalability",
        "export-consistency",
        "determinism",
        "all",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--force", action="store_true")
        if name in {"morphometry", "all"}:
            command.add_argument("--data-root", type=Path, default=None)
        if name == "morphometry":
            command.add_argument(
                "--reference-partition",
                choices=("calibration", "evaluation", "all"),
                default=None,
            )
            command.add_argument("--max-seeds", type=int, default=None)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", type=Path, default=None)
    figures = subparsers.add_parser("figures")
    figures.add_argument("--results", type=Path, default=None)
    latex = subparsers.add_parser("latex")
    latex.add_argument("--results", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_evaluation_config(args.config)
    command = args.command
    if command == "audit":
        payload = _audit(config)
    elif command == "pdc-audit":
        payload = audit_pdc(
            config.pdc_root(args.data_root),
            config.output_root,
            implicit_closure=args.implicit_closure,
        )
    elif command == "morphometry":
        payload = run_morphometry(
            config,
            config.pdc_root(args.data_root),
            force=args.force,
            reference_partition=args.reference_partition,
            max_seeds=args.max_seeds,
        )
    elif command == "controllability":
        payload = run_controllability(config, force=args.force)
    elif command == "host-ablation":
        rows = run_host_ablation(config, force=args.force)
        payload = {
            "rows": len(rows),
            "complete_n": sum(row["status"] == "complete" for row in rows),
            "failed_n": sum(row["status"] != "complete" for row in rows),
        }
    elif command == "sampling-ablation":
        payload = run_sampling_ablation(config, force=args.force)
    elif command == "scalability":
        payload = run_scalability(config, force=args.force)
    elif command == "export-consistency":
        payload = run_export_consistency(config, force=args.force)
    elif command == "determinism":
        payload = run_determinism(config, force=args.force)
    elif command == "all":
        data_root = config.pdc_root(args.data_root)
        audit_pdc(data_root, config.output_root)
        run_morphometry(config, data_root, force=args.force)
        run_controllability(config, force=args.force)
        run_host_ablation(config, force=args.force)
        run_sampling_ablation(config, force=args.force)
        run_scalability(config, force=args.force)
        run_export_consistency(config, force=args.force)
        run_determinism(config, force=args.force)
        payload = aggregate_results(config.output_root)
        from plume_advanced.evaluation.plotting import generate_figures

        generate_figures(config.output_root)
        generate_latex(config.output_root)
    elif command == "aggregate":
        payload = aggregate_results(args.results or config.output_root)
    elif command == "figures":
        from plume_advanced.evaluation.plotting import generate_figures

        payload = {
            "files": [str(path) for path in generate_figures(args.results or config.output_root)]
        }
    elif command == "latex":
        payload = {
            "files": [str(path) for path in generate_latex(args.results or config.output_root)]
        }
    else:
        raise AssertionError(command)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if command in {
        "morphometry", "controllability", "host-ablation", "sampling-ablation",
        "scalability", "export-consistency", "determinism", "all",
    }:
        return int(_experiment_failed(payload))
    return 0


def _experiment_failed(summary: dict[str, Any]) -> bool:
    """Propagate failed/empty cases and failed checks to automation callers."""

    if "experiments" in summary:
        experiments = summary["experiments"]
        return not experiments or any(_experiment_failed(item) for item in experiments.values())
    if summary.get("passed") is False:
        return True
    if any(summary.get(key, 0) > 0 for key in ("failed_n", "failed_worlds", "invalid_n")):
        return True
    if any(summary.get(key) == 0 for key in ("complete_n", "complete_worlds")):
        return True
    return any(value is False for value in summary.get("checks", {}).values())


def _audit(config) -> dict:
    project = load_project_config(config.project_config)
    calibration_caves = set(config.pdc_cave_partition("calibration"))
    evaluation_caves = set(config.pdc_cave_partition("evaluation"))
    overlap = calibration_caves & evaluation_caves
    if overlap:
        raise ValueError(f"PDC cave partitions overlap: {sorted(overlap)}")
    provenance = capture_provenance(
        ROOT,
        resolved_config=project_config_manifest(project),
        inputs=(
            config.path,
            config.project_config,
            config.pdc_partition_path("calibration"),
            config.pdc_partition_path("evaluation"),
        ),
    )
    payload = {
        "schema": "plume.evaluation-audit.v1",
        "ready": True,
        "project_config": str(config.project_config),
        "experiment_config": str(config.path),
        "output_root": str(config.output_root),
        "routing_weights": project.host_field.routing_weights.resolved(),
        "sampling_policy": project.section_field.sampling_policy,
        "pdc_partition": {
            "calibration_caves": len(calibration_caves),
            "evaluation_caves": len(evaluation_caves),
            "overlap_caves": 0,
        },
        "provenance": provenance,
    }
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "evaluation_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_experiment_manifest(
        config.output_root,
        config_path=config.path,
        experiments=(
            "morphometry",
            "controllability",
            "host_ablation",
            "sampling_ablation",
            "scalability",
            "export_consistency",
            "determinism",
        ),
        provenance=provenance,
    )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
