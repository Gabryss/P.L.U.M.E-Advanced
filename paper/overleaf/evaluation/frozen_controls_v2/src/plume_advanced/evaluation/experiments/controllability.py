"""Matched-seed sweeps for predeclared user-facing procedural controls."""

from __future__ import annotations

import json
import math

import numpy as np

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.experiments.common import (
    config_hash,
    for_seed,
    generate_network,
)
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.runner import ResultStore, run_cases
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.evaluation.statistics import spearman_correlation
from plume_advanced.stages.section_field import SectionFieldGenerator


def run_controllability(config: EvaluationConfig, *, force: bool = False) -> dict:
    section = config.section("controllability")
    sweeps = {
        "distributary": tuple(section.get("distributary_values", (0.2, 0.5, 0.8))),
        "duration": tuple(section.get("duration_values", (0.5, 1.0, 1.5))),
        "inflation": tuple(section.get("inflation_values", (0.2, 0.5, 0.8))),
        "supply": tuple(section.get("supply_values", (0.7, 1.0, 1.3))),
    }
    control_keys = {
        "distributary": "distributary_tendency", "duration": "duration_scale",
        "inflation": "inflation", "supply": "supply_rate_scale",
    }
    resolved_controls = {
        (control, float(value)): load_project_config(
            config.project_config, world_body=section.get("body", "earth"), dev_mode=False,
            flow_regime_overrides={control_keys[control]: float(value)},
        )
        for control, values in sweeps.items() for value in values
    }
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("controllability")),
    )
    identity = str(provenance["identity_sha256"])
    store = ResultStore(config.output_root, "controllability", provenance_sha256=identity)
    tasks = []
    for seed in config.seeds("controllability"):
        for control, values in sweeps.items():
            for value in values:
                project = for_seed(resolved_controls[(control, float(value))], seed)
                template = ExperimentResult(
                    experiment_name="controllability",
                    run_id=f"seed-{seed:06d}-{control}-{float(value):.3f}",
                    condition_id=f"{control}={float(value):.3f}",
                    seed=seed,
                    status="complete",
                    git_commit=str(provenance["git_commit"]),
                    git_dirty=bool(provenance["git_dirty"]),
                    resolved_config_sha256=config_hash(project),
                    provenance_sha256=identity,
                )

                def operation(project=project, control=control, value=float(value)):
                    host, network = generate_network(project)
                    metrics = network_metrics(network, host)
                    result = {"control": control, "control_value": value, **metrics}
                    if control == "inflation":
                        sections = SectionFieldGenerator(project.section_field).generate(network)
                        junction_widths = [
                            sample.tube_width
                            for field in sections.segment_fields
                            for sample in field.samples
                            if sample.junction_blend_weight >= 0.5
                        ]
                        other_widths = [
                            sample.tube_width
                            for field in sections.segment_fields
                            for sample in field.samples
                            if sample.junction_blend_weight < 0.2
                        ]
                        result["junction_to_passage_width_ratio"] = (
                            float(np.mean(junction_widths))
                            / max(float(np.mean(other_widths)), 1e-9)
                            if junction_widths and other_widths
                            else 1.0
                        )
                    result["host_horizontal_scale"] = project.host_field.body_spatial_scale
                    return result

                tasks.append((template, operation))
    run_cases(store, tasks, force=force)
    rows = store.rows()
    complete = [row for row in rows if row["status"] == "complete"]
    primary = {
        "distributary": "cyclomatic_number",
        "duration": "main_route_length_m",
        "inflation": "junction_to_passage_width_ratio",
        "supply": "host_horizontal_scale",
    }
    summaries = {}
    for control, metric in primary.items():
        selected = [row for row in complete if row["control"] == control and metric in row]
        rho = (
            spearman_correlation(
                [row["control_value"] for row in selected],
                [row[metric] for row in selected],
            )
            if selected
            else None
        )
        summaries[control] = {
            "primary_metric": metric,
            "n": len(selected),
            "spearman_rho": rho if rho is None or math.isfinite(rho) else None,
        }
    summary = {
        "schema": "plume.controllability-summary.v2",
        "intervention_policy": "Override the declared flow control before resolving every stage, exactly as a project TOML edit would do.",
        "planned_n": len(rows),
        "complete_n": len(complete),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "controls": summaries,
    }
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary



__all__ = ["run_controllability"]
