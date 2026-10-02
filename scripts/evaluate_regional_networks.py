#!/usr/bin/env python3
"""Bounded Stage A-B campaign for regional growth; no sections or mesh generation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.artifacts import (
    export_network_artifact,
    host_semantic_hash,
    network_semantic_hash,
)
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import (
    NetworkQualityError,
    assess_network,
    write_quality_report,
)
from plume_advanced.stages.network_systems import GenerationDomainError
from plume_advanced.visualization.network_layers import render_layered_network
from plume_advanced.visualization.network_morphology import (
    render_campaign_gallery,
    render_network_morphology,
)
from plume_advanced.visualization.regional import render_network_comparison
from plume_advanced.world import derive_stage_seeds

ROOT = Path(__file__).resolve().parents[1]


def _generate_candidate(config, host, report_path):
    """Keep one attempt's outcome separate from its reproducibility replay."""
    try:
        return CaveNetworkGenerator(config).generate(host, quality_report_path=report_path), None
    except (NetworkQualityError, GenerationDomainError) as error:
        return None, dict(
            type=type(error).__name__, message=str(error),
            quality=json.loads(report_path.read_text()) if report_path.exists() else None,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/regional-evaluation")
    parser.add_argument(
        "--config", type=Path, help="Evaluate a regional recipe instead of the default presets"
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=("short-multi", "long-multi"),
        default=None,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 17, 42])
    parser.add_argument("--source-counts", nargs="+", type=int)
    parser.add_argument("--layers", nargs="+", type=int, choices=(1, 2, 3, 4))
    parser.add_argument(
        "--replay", action="store_true", help="Regenerate every case and compare semantic hashes"
    )
    args = parser.parse_args(argv)
    if args.config and args.presets:
        parser.error("Choose --config or --presets, not both")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Campaign output must be empty; choose a new directory to preserve previous evidence")
    report: dict[str, Any] = dict(scope="network_only", cases=[], accepted=0, failed=0)
    recipes = (
        [args.config]
        if args.config
        else [
            ROOT / f"config/{preset}.toml"
            for preset in (args.presets or ["short-multi", "long-multi"])
        ]
    )
    for recipe in recipes:
        preset = recipe.stem
        for seed in args.seeds:
            c = load_project_config(recipe, seed_override=seed)
            counts = args.source_counts or [c.network.systems.count]
            layer_counts = args.layers or [
                c.network.layers.count if c.network.layers.enabled else 1
            ]
            for count, layer_count in product(counts, layer_counts):
                name = f"{preset}-sources{count}-seed{seed}"
                if layer_count > 1:
                    name += f"-layers{layer_count}"
                output = args.output / name
                output.mkdir(parents=True, exist_ok=True)
                cfg = replace(
                    c.network,
                    # Some presets pin a stage seed. This campaign explicitly
                    # varies both the host and the network across root seeds.
                    random_seed=derive_stage_seeds(seed).network,
                    topology=replace(c.network.topology, generation_mode="regional_growth"),
                    systems=replace(c.network.systems, count=count),
                    layers=replace(
                        c.network.layers, enabled=layer_count > 1, count=max(2, layer_count)
                    ),
                    quality=(
                        c.network.quality
                        if args.config
                        else replace(c.network.quality, max_attempts=4, repair_passes=2)
                    ),
                )
                write_quality_report(
                    project_config_manifest(replace(c, network=cfg)),
                    output / "resolved_config.json",
                )
                host = HostFieldGenerator(c.host_field).generate()
                before = host_semantic_hash(host)
                started = time.perf_counter()
                result = dict(case=name, requested_seed=seed, sources=count, host_sha256=before)
                print(f"Starting {name}", flush=True)
                n, error = _generate_candidate(cfg, host, output / "quality.json")
                if n is not None:
                    assessment = assess_network(n, host)
                    export_network_artifact(n, output / "network.json")
                    render_network_comparison(host, [(name, n)], output / "network.png")
                    render_network_morphology(host, n, output / "morphology.png")
                    if cfg.layers.enabled:
                        render_layered_network(host, n, output / "layers.png")
                    if cfg.target_route_length_m > 1000:
                        render_network_comparison(
                            host, [(name, n)], output / "network_windows.png", window_m=600
                        )
                    result.update(
                        accepted=assessment["accepted"],
                        selected_network_seed=n.config.random_seed,
                        network_sha256=network_semantic_hash(n),
                        summary=n.summary(),
                        accepted_branches=n.backend_provenance["accepted_branches"],
                        requested_branches=n.backend_provenance["requested_branches"],
                    )
                else:
                    result.update(accepted=False, error=error["message"], error_type=error["type"])
                if args.replay:
                    print(f"Replaying {name}", flush=True)
                    repeated, repeated_error = _generate_candidate(cfg, host, output / "replay_quality.json")
                    if n is not None and repeated is not None:
                        result["replay_matches"] = (
                            network_semantic_hash(n) == network_semantic_hash(repeated)
                            and n.backend_provenance == repeated.backend_provenance
                            and json.loads((output / "quality.json").read_text())
                            == json.loads((output / "replay_quality.json").read_text())
                        )
                    else:
                        result["replay_matches"] = n is None and repeated is None and error == repeated_error
                    if repeated_error:
                        result["replay_error"] = repeated_error["message"]
                    result["accepted"] &= result["replay_matches"]
                result.update(
                    elapsed_s=time.perf_counter() - started,
                    host_unchanged=before == host_semantic_hash(host),
                )
                result["accepted"] &= result["host_unchanged"]
                report["accepted" if result["accepted"] else "failed"] += 1
                report["cases"].append(result)
                write_quality_report(report, args.output / "campaign.json")
                print(
                    f"{name}: {'PASS' if result['accepted'] else 'FAIL'} in {result['elapsed_s']:.1f}s",
                    flush=True,
                )
    render_campaign_gallery(args.output)
    return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
