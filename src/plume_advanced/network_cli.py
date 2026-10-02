"""Generate and inspect networks without generating sections, meshes or rocks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.artifacts import export_network_artifact, host_semantic_hash
from plume_advanced.progress import TerminalProgress
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import NetworkQualityError
from plume_advanced.stages.network_systems import GenerationDomainError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/regional-network.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/regional-network"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also generate the existing independent-growth model on the SAME host",
    )
    args = parser.parse_args(argv)
    cfg = load_project_config(args.config, seed_override=args.seed)
    if not cfg.network.quality.enabled:
        parser.error("plume-network requires network.quality.enabled = true")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resolved_config.json").write_text(
        json.dumps(project_config_manifest(cfg), indent=2, sort_keys=True) + "\n"
    )
    progress = TerminalProgress(
        total_stages=4 if args.compare else 3,
        trace_path=args.output / "progress.jsonl",
        context="Stages A-B only",
    )
    try:
        progress.start("Host field")
        host = HostFieldGenerator(cfg.host_field).generate()
        progress.finish()
        progress.start("Network", "Bounded candidate search and local repair")
        network = CaveNetworkGenerator(cfg.network).generate(
            host, quality_report_path=args.output / "quality.json", quality_progress=progress.log
        )
        export_network_artifact(network, args.output / "network.json")
        progress.finish()
        comparisons = [
            (
                "Regional growth"
                if cfg.network.topology.generation_mode == "regional_growth"
                else "Network",
                network,
            )
        ]
        if args.compare:
            progress.start("Baseline comparison", "Same host and initial network seed")
            baseline_config = replace(
                cfg.network,
                topology=replace(cfg.network.topology, generation_mode="independent_growth"),
                regional=replace(cfg.network.regional, outlet_count=1),
                layers=replace(cfg.network.layers, enabled=False),
                detail=replace(cfg.network.detail, enabled=False),
            )
            try:
                baseline = CaveNetworkGenerator(baseline_config).generate(
                    host,
                    quality_report_path=args.output / "baseline_quality.json",
                    quality_progress=progress.log,
                )
                export_network_artifact(baseline, args.output / "baseline_network.json")
                comparisons.insert(0, ("Existing multi-network", baseline))
            except NetworkQualityError:
                progress.log(
                    "Baseline exhausted its search budget; see baseline_quality.json. Regional output retained."
                )
            progress.finish()
        progress.start("Network figures & 3D viewer")
        from plume_advanced.visualization.regional import render_network_comparison

        render_network_comparison(host, comparisons, args.output / "networks.png")
        if cfg.network.topology.generation_mode == "regional_growth":
            from plume_advanced.visualization.network_morphology import render_network_morphology

            render_network_morphology(host, network, args.output / "morphology.png")
        if cfg.network.layers.enabled:
            from plume_advanced.visualization.network_layers import render_layered_network

            render_layered_network(host, network, args.output / "layers.png")
        if cfg.network.target_route_length_m > 1000:
            render_network_comparison(
                host, [comparisons[-1]], args.output / "network_windows.png", window_m=600
            )
        from plume_advanced.network_viewer import export_network_viewer

        export_network_viewer([args.output / "network.json"], args.output / "viewer.html")
        (args.output / "summary.json").write_text(
            json.dumps(
                dict(
                    scope="host_and_network_only",
                    host_sha256=host_semantic_hash(host),
                    requested_seed=cfg.procedural_seed,
                    selected_network_seed=network.config.random_seed,
                    summary=network.summary(),
                    regional=network.backend_provenance,
                ),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        progress.finish()
    except (NetworkQualityError, GenerationDomainError) as error:
        progress.log(str(error))
        return 1
    finally:
        progress.close()
    print(f"Network, inspection figures and offline 3D viewer: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
