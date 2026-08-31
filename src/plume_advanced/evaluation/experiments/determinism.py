"""Compact same-run and stage-isolation checks."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.experiments.common import config_hash, for_seed, generate_sections
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.pipeline import StageCheckpointStore


def run_determinism(config: EvaluationConfig, *, force: bool = False) -> dict:
    section = config.section("determinism")
    base = load_project_config(
        config.project_config, world_body=section.get("body", "earth"), dev_mode=True
    )
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("determinism")),
    )
    store = ResultStore(config.output_root, "determinism")
    for seed in config.seeds("determinism"):
        project = for_seed(base, seed)
        template = ExperimentResult(
            experiment_name="determinism",
            run_id=f"seed-{seed:06d}-isolation",
            condition_id="same_config_events_export_resume",
            seed=seed,
            status="complete",
            git_commit=str(provenance["git_commit"]),
            git_dirty=bool(provenance["git_dirty"]),
            resolved_config_sha256=config_hash(project),
        )

        def operation(project=project):
            first_host, first_network, first_sections = generate_sections(project)
            second_host, second_network, second_sections = generate_sections(project)
            hashes = {
                "host": (host_semantic_hash(first_host), host_semantic_hash(second_host)),
                "network": (
                    network_semantic_hash(first_network),
                    network_semantic_hash(second_network),
                ),
                "sections": (
                    section_semantic_hash(first_sections),
                    section_semantic_hash(second_sections),
                ),
            }
            events_off = replace(project, events=replace(project.events, enabled=False))
            event_host, event_network, event_sections = generate_sections(events_off)
            export_changed = replace(project, export=replace(project.export, target="neutral"))
            export_host, export_network, export_sections = generate_sections(export_changed)
            with tempfile.TemporaryDirectory(prefix="plume-eval-checkpoint-") as directory:
                store_checkpoint = StageCheckpointStore(Path(directory), config_hash(project))
                store_checkpoint.save("section_field", first_sections)
                resumed, reused = store_checkpoint.load_or_build(
                    "section_field", lambda: second_sections, resume=True
                )
            return {
                "same_run_host_equal": hashes["host"][0] == hashes["host"][1],
                "same_run_network_equal": hashes["network"][0] == hashes["network"][1],
                "same_run_sections_equal": hashes["sections"][0] == hashes["sections"][1],
                "events_off_host_equal": hashes["host"][0] == host_semantic_hash(event_host),
                "events_off_network_equal": hashes["network"][0]
                == network_semantic_hash(event_network),
                "events_off_sections_equal": hashes["sections"][0]
                == section_semantic_hash(event_sections),
                "export_changed_host_equal": hashes["host"][0] == host_semantic_hash(export_host),
                "export_changed_network_equal": hashes["network"][0]
                == network_semantic_hash(export_network),
                "export_changed_sections_equal": hashes["sections"][0]
                == section_semantic_hash(export_sections),
                "resume_checkpoint_reused": reused,
                "resume_sections_equal": hashes["sections"][0] == section_semantic_hash(resumed),
            }

        run_case(store, template, operation, force=force)
    rows = store.rows()
    complete = [row for row in rows if row["status"] == "complete"]
    boolean_fields = sorted(
        key for key in complete[0] if complete and key.endswith(("_equal", "_reused"))
    )
    summary = {
        "schema": "plume.determinism-summary.v1",
        "planned_n": len(rows),
        "complete_n": len(complete),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "checks": {field: all(bool(row[field]) for row in complete) for field in boolean_fields},
    }
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = ["run_determinism"]
