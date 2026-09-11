#!/usr/bin/env python3
"""Screen a fixed-host matrix of system counts and named seeds through Stage C."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from plume_advanced.config import load_project_config
from plume_advanced.procedural import derive_subseed
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import NetworkQualityError
from plume_advanced.stages.network_systems import system_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/earth_interacting_systems.toml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260910, 20260911, 20260912])
    args = parser.parse_args()
    if any(count < 2 or count > 8 for count in args.counts):
        parser.error("system counts must be in [2, 8]")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seed labels must be nonnegative")
    config = load_project_config(args.config)
    if not config.network.quality.enabled:
        parser.error("the campaign requires network.quality.enabled = true")
    host = HostFieldGenerator(config.host_field).generate()
    cases = []
    args.output.mkdir(parents=True, exist_ok=True)
    for count in sorted(set(args.counts)):
        for seed in sorted(set(args.seeds)):
            name = f"systems{count}_seed{seed}"
            print(name, flush=True)
            network_config = replace(
                config.network,
                random_seed=derive_subseed(seed, "systems-validation", count),
                systems=replace(config.network.systems, count=count),
            )
            case = {
                "name": name,
                "count": count,
                "seed_label": seed,
                "network_seed": network_config.random_seed,
            }
            try:
                network = CaveNetworkGenerator(network_config).generate(
                    host,
                    section_config=config.section_field,
                    quality_report_path=args.output / name / "network_quality_report.json",
                    quality_progress=lambda line: print(line, flush=True),
                )
                report = network.quality_report
                case.update(
                    accepted=True,
                    **system_summary(network),
                    selected_attempt=report["selected_attempt"],
                    selected_seed=report["selected_seed"],
                    shape_sha256=report["selected_shape_sha256"],
                    check_count=len(report["attempts"][-1]["checks"]),
                    flow_error=network.max_flow_conservation_error(),
                )
            except NetworkQualityError as error:
                case.update(accepted=False, reason=str(error))
            cases.append(case)
            summary = {
                "scope": "fixed host, network and sections; no meshes or rocks",
                "config": str(args.config.resolve()),
                "cases": cases,
                "all_accepted": all(item["accepted"] for item in cases),
                "case_count": len(cases),
                "planned_case_count": len(set(args.counts)) * len(set(args.seeds)),
                "complete": len(cases) == len(set(args.counts)) * len(set(args.seeds)),
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
    return 0 if all(case["accepted"] for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
