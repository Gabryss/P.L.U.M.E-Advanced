#!/usr/bin/env python3
"""Evaluate a frozen width fit on fresh network seeds, with baseline and replay.

Multi-layer cases are robustness tests only: no measured layer calibration is
available. No fit, seed selection or retry beyond the recipe occurs here.
"""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.artifacts import (
    export_network_artifact,
    host_semantic_hash,
    network_semantic_hash,
)
from plume_advanced.evaluation.metrics.network_calibration import weighted_quantiles
from plume_advanced.network_viewer import export_network_viewer
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import (
    NetworkQualityError,
    assess_network,
    write_quality_report,
)
from plume_advanced.stages.network_systems import GenerationDomainError
from plume_advanced.visualization.regional import render_network_comparison


def width_samples(network):
    widths: list[float] = []
    weights: list[float] = []
    for segment in network.segments:
        arc = np.array([p.arc_length for p in segment.points])
        edges = np.linspace(0., arc[-1], max(2, int(np.ceil(arc[-1]))+1))
        widths.extend(np.interp(.5*(edges[:-1]+edges[1:]), arc, [p.width for p in segment.points]))
        weights.extend(np.diff(edges))
    return np.array(widths), np.array(weights)


def statistics(samples, target):
    widths, weights = samples
    q = weighted_quantiles(widths, weights, [.05, .25, .5, .75, .95])
    return dict(quantiles_m=q.tolist(), central_log_quantile_error=float(np.mean(np.log(q[1:4]/target[1:4])**2)),
                fraction_above_width_cap=float(np.sum(weights[widths >= 9.5-1e-6])/np.sum(weights)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/earth-survey-network.toml"))
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 211, 331, 487])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 3], choices=[1, 2, 3, 4])
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh campaign directory")
    fit = json.loads(args.fit.read_text())
    recipe_digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if recipe_digest != fit["recipe_sha256"]:
        parser.error("Recipe differs from the frozen fit; do not silently evaluate modified parameters")
    args.output.mkdir(parents=True, exist_ok=True)
    target = np.array(fit["targets"]["representative_width_quantiles_m"])
    report: dict[str, Any] = dict(scope="width_calibration_and_network_robustness", fit_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
                  recipe_sha256=recipe_digest, cases=[], accepted=0, failed=0,
                  limitations=["Topology is not fitted", "Multi-layer cases test robustness only",
                               "Baseline and fitted routing may differ: widths affect planning and acceptance"])
    artifacts, labels = [], {}
    pooled: dict[tuple[int, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for seed in args.seeds:
        cfg = load_project_config(args.config, seed_override=seed)
        for layers in args.layers:
            name = f"seed{seed}-layers{layers}"
            output = args.output/name
            output.mkdir(parents=True, exist_ok=True)
            host = HostFieldGenerator(cfg.host_field).generate()
            host_hash = host_semantic_hash(host)
            nc = replace(cfg.network, layers=replace(cfg.network.layers, enabled=layers > 1, count=max(2, layers)))
            if nc.random_seed in fit["training_field_seeds"]:
                raise ValueError("Evaluation overlaps field training seeds")
            baseline = replace(nc, base_passage_radius=3.8, minimum_passage_radius=1.5,
                               regional=replace(nc.regional, width_log_sigma=0),
                               detail=replace(nc.detail, enabled=True, strength=.8))
            row: dict[str, Any] = dict(case=name, root_seed=seed, layers=layers, accepted=False, variants={})
            report["cases"].append(row)
            comparisons = []
            for kind, controls in (("baseline", baseline), ("fitted", nc)):
                destination = output/kind
                destination.mkdir(parents=True, exist_ok=True)
                result: dict[str, Any] = dict(accepted=False)
                row["variants"][kind] = result
                print(f"{name}: {kind}", flush=True)
                started = time.perf_counter()
                try:
                    network = CaveNetworkGenerator(controls).generate(host, quality_report_path=destination/"quality.json")
                    if network.config.random_seed in fit["training_field_seeds"]:
                        raise ValueError("Selected candidate overlaps field training seeds")
                    valid = assess_network(network, host)["accepted"]
                    result.update(accepted=valid, selected_seed=network.config.random_seed,
                                  seconds=time.perf_counter()-started,
                                  width=statistics(width_samples(network), target),
                                  host_unchanged=host_hash == host_semantic_hash(host),
                                  summary=network.summary())
                    if kind == "fitted":
                        print(f"{name}: cold replay", flush=True)
                        replay = CaveNetworkGenerator(controls).generate(host, quality_report_path=destination/"replay_quality.json")
                        result["replay_matches"] = network_semantic_hash(replay) == network_semantic_hash(network)
                        result["accepted"] &= result["replay_matches"] and result["host_unchanged"]
                    samples, weights = width_samples(network)
                    pooled.setdefault((layers, kind), []).append((samples, weights/weights.sum()))
                    export_network_artifact(network, destination/"network.json")
                    write_quality_report(project_config_manifest(replace(cfg, network=network.config)), destination/"resolved_config.json")
                    artifacts.append(destination/"network.json")
                    labels[str((destination/"network.json").resolve())] = f"{'Survey widths' if kind == 'fitted' else 'Previous local detail'} · seed {seed} · {layers} layer(s)"
                    comparisons.append((kind.capitalize(), network))
                except (NetworkQualityError, GenerationDomainError) as error:
                    result["error"] = str(error)
                    result["seconds"] = time.perf_counter()-started
            row["accepted"] = all(v["accepted"] for v in row["variants"].values())
            report["accepted" if row["accepted"] else "failed"] += 1
            if comparisons:
                render_network_comparison(host, comparisons, output/"comparison.png")
            write_quality_report(report, args.output/"campaign.json")
            print(f"{name}: {'PASS' if row['accepted'] else 'FAIL'}", flush=True)
    report["aggregate"] = {}
    fig, axes = plt.subplots(1, len(args.layers), figsize=(6*len(args.layers), 4), squeeze=False, layout="constrained")
    for layers, ax in zip(args.layers, axes[0]):
        ax.plot([5, 25, 50, 75, 95], target, "o-", color="black", label="Survey target")
        for kind in ("baseline", "fitted"):
            samples = pooled.get((layers, kind), [])
            if not samples:
                continue
            aggregate = statistics((np.concatenate([s for s, _ in samples]),
                                    np.concatenate([w for _, w in samples])), target)
            aggregate["completed_cases"] = len(samples)
            report["aggregate"][f"layers{layers}-{kind}"] = aggregate
            ax.plot([5, 25, 50, 75, 95], aggregate["quantiles_m"], "o-", label=kind.capitalize())
        ax.set(title=f"{layers} layer(s)" + (" · robustness only" if layers > 1 else " · Earth width scenario"),
               xlabel="Percentile", ylabel="Passage width (m)")
        ax.legend()
    fig.savefig(args.output/"width-evaluation.png", dpi=160)
    plt.close(fig)
    if artifacts:
        export_network_viewer(artifacts, args.output/"viewer.html", labels=labels)
    write_quality_report(report, args.output/"campaign.json")
    print(json.dumps(dict(accepted=report["accepted"], failed=report["failed"], aggregate=report["aggregate"])), flush=True)
    return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
