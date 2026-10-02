#!/usr/bin/env python3
"""Paired coarse/detail campaign on identical accepted graphs and immutable hosts."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.artifacts import (
    export_network_artifact,
    host_semantic_hash,
    network_payload,
    network_semantic_hash,
)
from plume_advanced.evaluation.metrics.network_detail import locality_diagnostics
from plume_advanced.network_viewer import export_network_viewer
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_detail import refine_network
from plume_advanced.stages.network_layers import segment_xyz
from plume_advanced.stages.network_quality import (
    NetworkQualityError,
    assess_network,
    write_quality_report,
)
from plume_advanced.stages.network_systems import GenerationDomainError
from plume_advanced.visualization.network_detail import render_network_detail
from plume_advanced.visualization.regional import render_network_comparison
from plume_advanced.world import derive_stage_seeds


def measurements(network, host):
    """Measure on a common 0.5 m grid, independent of stored sample density."""
    length = energy = bends = width_gradient = grade = 0.
    heading_variation = width_variation = vertical_variation = 0.
    for s in network.segments:
        xyz = segment_xyz(s, network.config.layers)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
        station = np.linspace(0, arc[-1], max(3, int(np.ceil(arc[-1]/.5))+1))
        points = np.column_stack([np.interp(station, arc, xyz[:, k]) for k in range(3)])
        widths = np.interp(station, arc, [p.width for p in s.points])
        ds = np.diff(station)
        costs = np.array([host.sample(x, y).growth_cost for x, y in points[:, :2]])
        energy += float(np.sum(.5*(costs[:-1]+costs[1:])*ds))
        tangents = np.diff(points[:, :2], axis=0)/ds[:, None]
        bends += float(np.sum(np.linalg.norm(np.diff(tangents, axis=0), axis=1)**2)/ds[0])
        width_gradient = max(width_gradient, float(np.max(abs(np.diff(widths)/ds))))
        grade = max(grade, float(np.max(abs(np.diff(points[:, 2])/ds))))
        heading_variation += float(np.sum(abs(np.diff(np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))))))
        width_variation += float(np.sum(abs(np.diff(widths))))
        slopes = np.diff(points[:, 2])/ds
        vertical_variation += float(np.sum(abs(np.diff(slopes))))
        length += arc[-1]
    return dict(length_m=float(length), host_cost_integral=energy, squared_bending_integral=bends,
                maximum_width_gradient=width_gradient, maximum_grade=grade,
                heading_variation_degrees_per_100m=float(np.degrees(heading_variation)*100/max(length, 1e-9)),
                width_total_variation_m_per_100m=width_variation*100/max(length, 1e-9),
                grade_total_variation_per_100m=vertical_variation*100/max(length, 1e-9),
                samples=sum(len(s.points) for s in network.segments))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/varied-network.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/network-detail-evaluation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 17, 23, 53])
    parser.add_argument("--layers", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 3])
    parser.add_argument("--sources", type=int, help="Override the recipe's source count")
    parser.add_argument("--strength", type=float, help="Detail amplitude in [0, 1]; defaults to the recipe")
    parser.add_argument("--previous", type=Path, help="Saved paired campaign to compare; never re-executes its old implementation")
    args = parser.parse_args(argv)
    if args.strength is not None and not 0 <= args.strength <= 1:
        parser.error("Detail strength must be between zero and one")
    if args.sources is not None and args.sources < 2:
        parser.error("Regional detail currently requires at least two source systems")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output directory to preserve earlier evidence")
    package = Path(__file__).resolve().parents[1] / "src/plume_advanced"
    source_hash = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source_hash.update(path.relative_to(package).as_posix().encode())
        source_hash.update(path.read_bytes())
    report: dict[str, Any] = dict(scope="network_only", cases=[], accepted=0, failed=0, viewer_labels={},
                                 implementation_sha256=source_hash.hexdigest(), python=platform.python_version(),
                                 numpy=np.__version__, recipe_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    artifacts = []
    for seed in args.seeds:
        cfg = load_project_config(args.config, seed_override=seed)
        for layers in args.layers:
            name = f"seed{seed}-layers{layers}"
            output = args.output / name
            output.mkdir(parents=True, exist_ok=True)
            nc = replace(cfg.network, random_seed=derive_stage_seeds(seed).network,
                         systems=replace(cfg.network.systems, count=args.sources or cfg.network.systems.count),
                         layers=replace(cfg.network.layers, enabled=layers > 1, count=max(2, layers)),
                         detail=replace(cfg.network.detail, enabled=False,
                                        strength=cfg.network.detail.strength if args.strength is None else args.strength))
            host = HostFieldGenerator(cfg.host_field).generate()
            before = host_semantic_hash(host)
            result = dict(case=f"{name}/detailed", seed=seed, layers=layers, accepted=False)
            report["cases"].append(result)
            print(f"{name}: coarse generation", flush=True)
            try:
                started = time.perf_counter()
                coarse = CaveNetworkGenerator(nc).generate(host, quality_report_path=output / "coarse/quality.json")
                result["coarse_seconds"] = time.perf_counter()-started
                dc = replace(coarse.config, detail=replace(nc.detail, enabled=True))
                print(f"{name}: detail refinement", flush=True)
                started = time.perf_counter()
                detailed = refine_network(CaveNetworkGenerator(dc), host, replace(coarse, config=dc))
                result["detail_seconds"] = time.perf_counter()-started
                print(f"{name}: cold generation replay", flush=True)
                cold_config = replace(nc, detail=dc.detail)
                replay = CaveNetworkGenerator(cold_config).generate(host, quality_report_path=output / "replay_quality.json")
                result.update(replay_matches=network_semantic_hash(replay) == network_semantic_hash(detailed),
                              host_unchanged=host_semantic_hash(host) == before,
                              before=measurements(coarse, host), after=measurements(detailed, host),
                              detail=detailed.backend_provenance["detail"],
                              locality=locality_diagnostics(network_payload(coarse), network_payload(detailed)))
                if args.previous:
                    prior_coarse = json.loads((args.previous / name / "coarse/network.json").read_text())
                    prior_detail = json.loads((args.previous / name / "detailed/network.json").read_text())
                    # Config/audit hashes can change independently of geometry.
                    if prior_coarse["nodes"] != network_payload(coarse)["nodes"] or prior_coarse["segments"] != network_payload(coarse)["segments"]:
                        raise ValueError(f"Previous campaign has different coarse geometry: {name}")
                    result["previous_locality"] = locality_diagnostics(prior_coarse, prior_detail)
                    result["previous_artifact_sha256"] = hashlib.sha256(
                        (args.previous / name / "detailed/network.json").read_bytes()).hexdigest()
                result["accepted"] = (assess_network(detailed, host)["accepted"] and result["replay_matches"]
                                      and result["host_unchanged"] and result["detail"]["original_routes_preserved"])
                for label, network in (("coarse", coarse), ("detailed", detailed)):
                    destination = output / label
                    export_network_artifact(network, destination / "network.json")
                    write_quality_report(network.quality_report, destination / "quality.json")
                    write_quality_report(project_config_manifest(replace(cfg, network=network.config)), destination / "resolved_config.json")
                    artifact_path = str((destination / "network.json").resolve())
                    artifacts.append(artifact_path)
                    report["viewer_labels"][artifact_path] = f"Seed {seed} · {layers} layer(s) · {label.capitalize()}"
                render_network_comparison(host, [("Coarse", coarse), ("Detailed", detailed)], output / "comparison.png")
                render_network_detail(host, coarse, detailed, output / "detail.png")
            except (NetworkQualityError, GenerationDomainError) as error:
                result["error"] = str(error)
            report["accepted" if result["accepted"] else "failed"] += 1
            write_quality_report(report, args.output / "campaign.json")
            print(f"{name}: {'PASS' if result['accepted'] else 'FAIL'}", flush=True)
    if artifacts:
        export_network_viewer(artifacts, args.output / "viewer.html", labels=report["viewer_labels"])
    print(json.dumps(dict(accepted=report["accepted"], failed=report["failed"])), flush=True)
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
