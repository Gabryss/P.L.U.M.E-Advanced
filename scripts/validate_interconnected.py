#!/usr/bin/env python3
"""Fixed-host A-C campaign with a fresh-process replay of the first case."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from plume_advanced.config import load_project_config, write_project_config_manifest
from plume_advanced.evaluation.artifacts import (
    export_network_artifact,
    export_section_artifact,
    host_semantic_hash,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.stages.host_field import HostFieldGenerator, export_host_influence_report
from plume_advanced.stages.network import CaveNetworkGenerator, export_network_report
from plume_advanced.stages.network_interconnected import spatial_metrics
from plume_advanced.stages.network_quality import NetworkQualityError
from plume_advanced.stages.section_field import SectionFieldGenerator


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def worker(args):
    config = load_project_config(args.config)
    seed = (int(config.network.random_seed) + args.offset) % 2**32
    config = replace(
        config,
        network=replace(config.network, random_seed=seed),
        stage_seeds=replace(config.stage_seeds, network=seed),
    )
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write_project_config_manifest(config, root / "resolved_project_config.json")
    shutil.copyfile(args.config, root / "input_config.toml")
    host = HostFieldGenerator(config.host_field).generate()
    before = host_semantic_hash(host)
    try:
        network = CaveNetworkGenerator(config.network).generate(
            host,
            section_config=config.section_field,
            quality_report_path=root / "network_quality_report.json",
            quality_progress=lambda s: print(s, flush=True),
        )
    except NetworkQualityError as error:
        write(
            root / "result.json",
            {"accepted": False, "offset": args.offset, "network_seed": seed, "reason": str(error)},
        )
        return 1
    sections = SectionFieldGenerator(config.section_field).generate(network)
    export_network_report(network, root / "stage_b_network_report.json")
    export_network_artifact(network, root / "stage_b_network.json")
    export_section_artifact(sections, root / "stage_c_sections")
    export_host_influence_report(host, root / "stage_a_host_report.json")
    np.savez_compressed(
        root / "stage_a_host_fields.npz",
        x=host.x_coords,
        y=host.y_coords,
        elevation=host.elevation,
        cover=host.cover_thickness,
        routing_cost=host.routing_cost,
    )
    # Local trusted checkpoints for inspecting/re-rendering this campaign.
    with (root / "stage_ac_checkpoint.pkl").open("wb") as stream:
        pickle.dump((config, host, network, sections), stream)
    metrics = spatial_metrics(network, sections)
    write(root / "interconnection_metrics.json", metrics)
    identity = dict(
        host_sha256=host_semantic_hash(host),
        network_sha256=network_semantic_hash(network),
        sections_sha256=section_semantic_hash(sections),
    )
    result = dict(
        accepted=True,
        offset=args.offset,
        network_seed=seed,
        selected_attempt=network.quality_report["selected_attempt"],
        selected_seed=network.quality_report["selected_seed"],
        repair_pass=network.quality_report["selected_repair_pass"],
        check_count=len(network.quality_report["attempts"][-1]["checks"]),
        host_unchanged=identity["host_sha256"] == before,
        identity=identity,
        metrics={k: v for k, v in metrics.items() if k not in {"station_m", "channel_count"}},
        interaction_events=network.backend_provenance["interaction_events"],
        total_passage_length_m=sum(s.total_length for s in network.segments),
        scope="host, network and cross sections; no cave mesh or rocks",
    )
    write(root / "result.json", result)
    if args.render:
        from plume_advanced.evaluation.visualization.emplacement import (
            render_emplacement_phase_activity,
        )
        from plume_advanced.visualization.host_field import HostFieldPlotter
        from plume_advanced.visualization.interconnected import render_interconnected
        from plume_advanced.visualization.section_field import SectionFieldPlotter

        HostFieldPlotter().render(host, root / "stage_a_host_field.png")
        render_interconnected(host, network, sections, root / "stage_bc_interconnected.png")
        SectionFieldPlotter().render(network, sections, root / "stage_c_sections.png")
        render_emplacement_phase_activity(network, root / "stage_b_emplacement_history.png")
    print(json.dumps(result["metrics"], sort_keys=True), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--offset", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--render", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_project_config(args.config)
    if config.network.topology.style != "interconnected" or not config.network.quality.enabled:
        parser.error("requires interconnected topology and enabled quality screening")
    if any(i < 0 for i in args.seed_offsets):
        parser.error("seed offsets must be nonnegative")
    if args.worker:
        return worker(args)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose an empty output directory; campaign results are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    summary = dict(
        scope="fixed-host stages A-C; no meshes or rocks",
        complete=False,
        cases=cases,
        all_accepted=False,
    )
    offsets = sorted(set(args.seed_offsets))
    for offset in offsets:
        name = f"seed_offset_{offset}"
        print(name, flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.resolve()),
            "--output",
            str((args.output / name).resolve()),
            "--worker",
            "--offset",
            str(offset),
        ]
        with (args.output / f"{name}.log").open("w") as log:
            code = subprocess.call(
                command + ["--render"],
                env=dict(os.environ, PYTHONHASHSEED="11"),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        path = args.output / name / "result.json"
        case = (
            json.loads(path.read_text())
            if path.exists()
            else dict(accepted=False, offset=offset, reason="worker failed; inspect log")
        )
        case.update(directory=name, exit_code=code)
        if offset == offsets[0] and code == 0:
            replay = args.output / "replay"
            replay_command = command.copy()
            replay_command[replay_command.index("--output") + 1] = str(replay.resolve())
            with (args.output / "replay.log").open("w") as log:
                replay_code = subprocess.call(
                    replay_command,
                    env=dict(os.environ, PYTHONHASHSEED="37"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            report = replay / "network_quality_report.json"
            same_report = (
                replay_code == 0
                and report.read_bytes()
                == (args.output / name / "network_quality_report.json").read_bytes()
            )
            replay_identity = (
                json.loads((replay / "result.json").read_text()).get("identity")
                if (replay / "result.json").exists()
                else None
            )
            summary["reproducibility"] = dict(
                passed=same_report and replay_identity == case["identity"],
                python_hash_seeds=[11, 37],
                reports_byte_identical=same_report,
                identities_equal=replay_identity == case["identity"],
            )
        cases.append(case)
        write(args.output / "summary.json", summary)
    summary.update(
        complete=True,
        all_accepted=all(
            c["accepted"] and c["exit_code"] == 0 and c.get("host_unchanged", False) for c in cases
        ),
        case_count=len(cases),
    )
    summary["passed"] = summary["all_accepted"] and summary.get("reproducibility", {}).get(
        "passed", False
    )
    write(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
