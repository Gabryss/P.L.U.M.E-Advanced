"""Adaptive versus uniform Stage-C sampling on the same profile field."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.experiments.common import config_hash, for_seed, generate_network
from plume_advanced.evaluation.metrics.sections import section_geometric_error
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.stages.section_field import SectionFieldGenerator


def run_sampling_ablation(config: EvaluationConfig, *, force: bool = False) -> dict:
    section = config.section("sampling_ablation")
    base = load_project_config(
        config.project_config, world_body=section.get("body", "earth"), dev_mode=False
    )
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("sampling_ablation")),
    )
    store = ResultStore(config.output_root, "sampling_ablation")
    for seed in config.seeds("sampling_ablation"):
        project = for_seed(base, seed)
        template = ExperimentResult(
            experiment_name="sampling_ablation",
            run_id=f"seed-{seed:06d}-sampling",
            condition_id="adaptive_vs_uniform",
            seed=seed,
            status="complete",
            git_commit=str(provenance["git_commit"]),
            git_dirty=bool(provenance["git_dirty"]),
            resolved_config_sha256=config_hash(project),
        )

        def operation(project=project):
            _host, network = generate_network(project)
            adaptive_config = replace(project.section_field, sampling_policy="adaptive")
            adaptive = SectionFieldGenerator(adaptive_config).generate(network)
            adaptive_count = int(adaptive.summary()["sample_count"])
            segment_count = max(len(network.segments), 1)
            total_length = sum(segment.total_length for segment in network.segments)
            matched_spacing = total_length / max(adaptive_count - segment_count, 1)
            uniform_config = replace(
                project.section_field,
                sampling_policy="uniform",
                uniform_sample_spacing=max(matched_spacing, 1e-6),
            )
            reference_config = replace(
                project.section_field,
                sampling_policy="reference",
                reference_sample_spacing=float(
                    section.get(
                        "reference_spacing_m", project.section_field.reference_sample_spacing
                    )
                ),
            )
            uniform = SectionFieldGenerator(uniform_config).generate(network)
            reference = SectionFieldGenerator(reference_config).generate(network)
            adaptive_error = section_geometric_error(adaptive, reference)
            uniform_error = section_geometric_error(uniform, reference)
            median_width = float(adaptive.summary()["mean_tube_width"])
            return {
                "adaptive_section_count": adaptive_count,
                "uniform_section_count": int(uniform.summary()["sample_count"]),
                "reference_section_count": int(reference.summary()["sample_count"]),
                "uniform_spacing_m": matched_spacing,
                "adaptive_error_mean_m": adaptive_error["mean_m"],
                "adaptive_error_p95_m": adaptive_error["p95_m"],
                "uniform_error_mean_m": uniform_error["mean_m"],
                "uniform_error_p95_m": uniform_error["p95_m"],
                "adaptive_error_mean_widths": adaptive_error["mean_m"] / max(median_width, 1e-9),
                "uniform_error_mean_widths": uniform_error["mean_m"] / max(median_width, 1e-9),
            }

        run_case(store, template, operation, force=force)
    complete = [row for row in store.rows() if row["status"] == "complete"]
    summary = {
        "schema": "plume.sampling-ablation-summary.v1",
        "planned_n": len(store.rows()),
        "complete_n": len(complete),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in store.rows()),
        "median_adaptive_count": _median([row["adaptive_section_count"] for row in complete]),
        "median_uniform_count": _median([row["uniform_section_count"] for row in complete]),
        "median_adaptive_error_m": _median([row["adaptive_error_mean_m"] for row in complete]),
        "median_uniform_error_m": _median([row["uniform_error_mean_m"] for row in complete]),
    }
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


__all__ = ["run_sampling_ablation"]
