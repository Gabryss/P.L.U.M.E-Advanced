#!/usr/bin/env python3
"""Compare two fresh A-C runs under different Python hash seeds; never mesh."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    export_network_artifact,
    export_section_artifact,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        config = load_project_config(args.config)
        if not config.network.quality.enabled:
            parser.error("Reproducibility validation requires network.quality.enabled = true")
        host = HostFieldGenerator(config.host_field).generate()
        network = CaveNetworkGenerator(config.network).generate(
            host,
            section_config=config.section_field,
            quality_report_path=args.output / "network_quality_report.json",
            quality_progress=lambda message: print(message, flush=True),
        )
        sections = SectionFieldGenerator(config.section_field).generate(network)
        export_network_artifact(network, args.output / "stage_b_network.json")
        export_section_artifact(sections, args.output / "stage_c_sections")
        identity = {
            "network_sha256": network_semantic_hash(network),
            "sections_sha256": section_semantic_hash(sections),
            "shape_sha256": network.quality_report["selected_shape_sha256"],
        }
        (args.output / "identity.json").write_text(
            json.dumps(identity, sort_keys=True, indent=2) + "\n"
        )
        return

    children, logs = [], []
    try:
        for index, hash_seed in enumerate((11, 37), start=1):
            folder = args.output / f"run_{index}"
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "generation.log").open("w")
            logs.append(log)
            environment = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--config",
                        str(args.config.resolve()),
                        "--output",
                        str(folder.resolve()),
                        "--worker",
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [child.wait() for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for log in logs:
            log.close()
    result = {
        "schema": "plume.network-reproduction.v1",
        "process_exit_codes": codes,
        "python_hash_seeds": [11, 37],
        "scope": "host, network and sections; no mesh or rocks",
    }
    if any(codes):
        result.update(passed=False, reason="At least one run failed; inspect its report and log")
    else:
        first, second = (args.output / f"run_{i}" for i in (1, 2))
        identities = [json.loads((p / "identity.json").read_text()) for p in (first, second)]
        report_equal = (first / "network_quality_report.json").read_bytes() == (
            second / "network_quality_report.json"
        ).read_bytes()
        result.update(
            passed=identities[0] == identities[1] and report_equal,
            identities=identities,
            reports_byte_identical=report_equal,
        )
    (args.output / "reproducibility.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
