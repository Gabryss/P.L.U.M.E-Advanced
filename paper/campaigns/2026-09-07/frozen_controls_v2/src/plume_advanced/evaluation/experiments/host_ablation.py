"""Matched-seed component-wise host routing ablation."""

from __future__ import annotations

import json
from typing import Any

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.experiments.common import (
    config_hash,
    for_seed,
    generate_network,
    with_routing_condition,
)
from plume_advanced.evaluation.metrics.network import (
    network_metrics,
    symmetric_centerline_distance,
)
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.runner import ResultStore, run_cases
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.evaluation.statistics import paired_bootstrap_difference

_BASELINE_CACHE: dict[str, Any] = {}


def _baseline(project):
    key = config_hash(project)
    if key not in _BASELINE_CACHE:
        _BASELINE_CACHE.clear()
        host, network = generate_network(with_routing_condition(project, "full"))
        _BASELINE_CACHE[key] = host, network, network_metrics(network, host)
    return _BASELINE_CACHE[key]


def run_host_ablation(config: EvaluationConfig, *, force: bool = False) -> list[dict]:
    section = config.section("host_ablation")
    conditions = tuple(
        section.get(
            "conditions",
            ["full", "no_slope", "no_cover", "no_fracture", "no_capacity", "no_stability"],
        )
    )
    base = load_project_config(
        config.project_config, world_body=section.get("body", "earth"), dev_mode=False
    )
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("host_ablation")),
    )
    identity = str(provenance["identity_sha256"])
    store = ResultStore(config.output_root, "host_ablation", provenance_sha256=identity)
    _BASELINE_CACHE.clear()
    tasks = []
    for seed in config.seeds("host_ablation"):
        seeded = for_seed(base, seed)
        for condition in conditions:
            project = with_routing_condition(seeded, str(condition))
            run_id = f"seed-{seed:06d}-{condition}"
            template = ExperimentResult(
                experiment_name="host_ablation",
                run_id=run_id,
                condition_id=str(condition),
                seed=seed,
                status="complete",
                git_commit=str(provenance["git_commit"]),
                git_dirty=bool(provenance["git_dirty"]),
                resolved_config_sha256=config_hash(project),
                provenance_sha256=identity,
            )

            def operation(project=project, condition=condition, seeded=seeded):
                full_host, full_network, full_metrics = _baseline(seeded)
                width = float(full_network.summary()["mean_segment_width"])
                if condition == "full":
                    host, network, metrics = full_host, full_network, full_metrics
                else:
                    host, network = generate_network(project)
                    metrics = network_metrics(network, host)
                distance = symmetric_centerline_distance(full_network, network)
                return {
                    **metrics,
                    "centerline_displacement_mean_m": distance["mean_m"],
                    "centerline_displacement_p95_m": distance["p95_m"],
                    "centerline_displacement_mean_passage_widths": distance["mean_m"]
                    / max(width, 1e-9),
                    "routing_weights": project.host_field.routing_weights.resolved(),
                    "routing_enabled": project.host_field.routing_weights.enabled,
                }

            tasks.append((template, operation))
    run_cases(store, tasks, force=force, chunksize=len(conditions))
    summary = _summary(store.rows(), conditions, config=config)
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return store.rows()


def _summary(rows: list[dict], conditions: tuple[str, ...], *, config=None) -> dict:
    complete = [row for row in rows if row["status"] == "complete"]
    baseline = {row["seed"]: row for row in complete if row["condition_id"] == "full"}
    paired = {}
    for index, condition in enumerate(conditions):
        matches = [r for r in complete if r["condition_id"] == condition and r["seed"] in baseline]
        comparisons = {}
        for metric in ("centerline_displacement_mean_m", "cyclomatic_number"):
            if matches:
                interval = paired_bootstrap_difference(
                    [baseline[r["seed"]][metric] for r in matches], [r[metric] for r in matches],
                    iterations=config.bootstrap_iterations if config else 2000,
                    confidence_level=config.confidence_level if config else .95,
                    seed=(config.bootstrap_seed if config else 0)+index,
                )
                comparisons[metric] = {"median_delta": interval.estimate,
                                       "ci_lower": interval.lower, "ci_upper": interval.upper}
        paired[condition] = {"valid_pairs": len(matches), "effects": comparisons}
    return {
        "schema": "plume.host-ablation-summary.v1",
        "planned_n": len(rows),
        "complete_n": len(complete),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "invalid_n": sum(row["status"] == "invalid" for row in rows),
        "paired_effects": paired,
        "conditions": {
            condition: {
                "n": sum(row["condition_id"] == condition for row in complete),
                "median_centerline_displacement_m": _median(
                    [
                        row["centerline_displacement_mean_m"]
                        for row in complete
                        if row["condition_id"] == condition
                    ]
                ),
                "median_cyclomatic_number": _median(
                    [
                        row["cyclomatic_number"]
                        for row in complete
                        if row["condition_id"] == condition
                    ]
                ),
            }
            for condition in conditions
        },
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        float(ordered[middle])
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )


__all__ = ["run_host_ablation"]
