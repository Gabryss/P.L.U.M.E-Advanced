"""Matched-seed component-wise host routing ablation."""

from __future__ import annotations

import json

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
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult


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
    store = ResultStore(config.output_root, "host_ablation")
    for seed in config.seeds("host_ablation"):
        seeded = for_seed(base, seed)
        full_project = with_routing_condition(seeded, "full")
        full_host, full_network = generate_network(full_project)
        full_metrics = network_metrics(full_network, full_host)
        width = float(full_network.summary()["mean_segment_width"])
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
            )

            def operation(project=project, condition=condition):
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

            run_case(store, template, operation, force=force)
    summary = _summary(store.rows(), conditions)
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return store.rows()


def _summary(rows: list[dict], conditions: tuple[str, ...]) -> dict:
    complete = [row for row in rows if row["status"] == "complete"]
    return {
        "schema": "plume.host-ablation-summary.v1",
        "planned_n": len(rows),
        "complete_n": len(complete),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "invalid_n": sum(row["status"] == "invalid" for row in rows),
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
