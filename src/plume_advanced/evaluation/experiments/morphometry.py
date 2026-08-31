"""Stage-C morphology comparison against PDC and matched ellipses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.datasets.pdc import load_pdc
from plume_advanced.evaluation.experiments.common import config_hash, for_seed, generate_sections
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry, ellipse_baseline
from plume_advanced.evaluation.metrics.sections import generated_section_records
from plume_advanced.evaluation.provenance import capture_provenance, directory_identity
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.evaluation.statistics import (
    ks_statistic,
    median_iqr,
    normalized_wasserstein_distance,
    two_population_cluster_bootstrap,
    wasserstein_distance,
)

DEFAULT_METRICS = (
    "width_m",
    "height_m",
    "aspect_ratio",
    "area_m2",
    "compactness",
    "floor_residual_norm",
    "roof_asymmetry_norm",
)


def run_morphometry(
    config: EvaluationConfig,
    data_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    section = config.section("morphometry")
    pdc_sections, rejections = load_pdc(data_root)
    if not pdc_sections:
        raise ValueError("PDC audit retained no contours; inspect pdc_rejections.csv")
    reference = [
        {
            "reference_cave_id": item.reference_cave_id,
            "reference_section_id": item.reference_section_id,
            "relative_path": item.relative_path,
            **contour_morphometry(item.contour),
        }
        for item in pdc_sections
    ]
    base = load_project_config(
        config.project_config, world_body=section.get("body", "earth"), dev_mode=False
    )
    dataset = directory_identity(data_root)
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("morphometry")),
    )
    store = ResultStore(config.output_root, "morphometry")
    for seed in config.seeds("morphometry"):
        project = for_seed(base, seed)
        template = ExperimentResult(
            experiment_name="morphometry",
            run_id=f"seed-{seed:06d}-earth-sections",
            condition_id="plume_advanced",
            seed=seed,
            status="complete",
            git_commit=str(provenance["git_commit"]),
            git_dirty=bool(provenance["git_dirty"]),
            resolved_config_sha256=config_hash(project),
            input_dataset_id="pdc-v2.0",
            input_dataset_sha256_or_version=str(dataset["sha256"]),
        )

        def operation(project=project, seed=seed):
            _host, _network, sections = generate_sections(project)
            generated = generated_section_records(
                sections,
                seed=seed,
                world_id=f"earth-{seed:06d}",
            )
            baseline = []
            for record in generated:
                baseline.append(
                    {
                        **{
                            key: record[key]
                            for key in (
                                "generated_seed",
                                "generated_world_id",
                                "generated_segment_id",
                                "segment_arc_length_m",
                            )
                        },
                        **contour_morphometry(
                            ellipse_baseline(float(record["width_m"]), float(record["height_m"]))
                        ),
                    }
                )
            return {
                "section_count": len(generated),
                "generated_sections": generated,
                "baseline_sections": baseline,
            }

        run_case(store, template, operation, force=force)
    output = store.root
    complete = [row for row in store.rows() if row["status"] == "complete"]
    generated = [record for row in complete for record in row["generated_sections"]]
    baseline = [record for row in complete for record in row["baseline_sections"]]
    _write_csv(output / "raw_generated_sections.csv", generated)
    _write_csv(output / "raw_baseline_sections.csv", baseline)
    _write_csv(output / "raw_reference_sections.csv", reference)
    _write_csv(
        output / "rejections.csv",
        [
            {"relative_path": item.relative_path, "reason": item.reason, "detail": item.detail}
            for item in rejections
        ],
    )
    summary = _summarize(config, section, reference, generated, baseline, store.rows())
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "summary.csv", summary["metrics"])
    return summary


def _summarize(
    config: EvaluationConfig,
    section: dict[str, Any],
    reference: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = tuple(section.get("metrics", DEFAULT_METRICS))
    summaries: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        reference_values = [float(row[metric]) for row in reference]
        generated_values = [float(row[metric]) for row in generated]
        baseline_values = [float(row[metric]) for row in baseline]
        if not generated_values:
            continue
        ci = two_population_cluster_bootstrap(
            reference_values,
            [row["reference_cave_id"] for row in reference],
            generated_values,
            [row["generated_world_id"] for row in generated],
            lambda first, second: normalized_wasserstein_distance(first, second),
            iterations=config.bootstrap_iterations,
            confidence_level=config.confidence_level,
            seed=config.bootstrap_seed + index,
        )
        summaries.append(
            {
                "metric": metric,
                "reference": median_iqr(reference_values),
                "generated": median_iqr(generated_values),
                "baseline": median_iqr(baseline_values),
                "generated_w1": wasserstein_distance(reference_values, generated_values),
                "generated_normalized_w1": normalized_wasserstein_distance(
                    reference_values, generated_values
                ),
                "generated_ks": ks_statistic(reference_values, generated_values),
                "baseline_normalized_w1": normalized_wasserstein_distance(
                    reference_values, baseline_values
                ),
                "normalized_w1_ci_lower": ci.lower,
                "normalized_w1_ci_upper": ci.upper,
            }
        )
    aggregate_metrics = set(
        section.get(
            "aggregate_metrics",
            ("aspect_ratio", "compactness", "floor_residual_norm", "roof_asymmetry_norm"),
        )
    )
    aggregate_values = [
        row["generated_normalized_w1"] for row in summaries if row["metric"] in aggregate_metrics
    ]
    return {
        "schema": "plume.morphometry-summary.v1",
        "planned_worlds": len(rows),
        "complete_worlds": sum(row["status"] == "complete" for row in rows),
        "failed_worlds": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "reference_caves": len({row["reference_cave_id"] for row in reference}),
        "reference_sections": len(reference),
        "generated_sections": len(generated),
        "aggregate_normalized_w1": float(np.mean(aggregate_values)) if aggregate_values else None,
        "metrics": summaries,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(
        {key for row in rows for key, value in row.items() if not isinstance(value, dict)}
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


__all__ = ["run_morphometry"]
