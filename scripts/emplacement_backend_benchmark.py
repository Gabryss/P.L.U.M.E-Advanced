#!/usr/bin/env python3
"""Bounded Stage-B benchmark for emplacement proposal backends.

The benchmark intentionally stops at Stage B.  It uses the same named seeds,
host fields, and network controls for each backend, records failures without
fallback, and writes a machine-readable report plus comparative figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator
from plume_advanced.stages.network import CaveNetwork, CaveNetworkConfig, CaveNetworkGenerator


BACKENDS = ("internal", "downflow_reference", "flowy")
FAMILIES = ("natural", "monotonic")
DEFAULT_SEEDS = (2, 7, 11, 17, 23)
FLOWY_DEFAULT = "/tmp/plume-flowy-evaluation/build-plume/flowy"


def _host_config(family: str, seed: int) -> HostFieldConfig:
    if family not in FAMILIES:
        raise ValueError(f"unknown terrain family: {family}")
    base = HostFieldConfig()
    return replace(
        base,
        grid=GridConfig(width=1800.0, height=1400.0, nx=60, ny=48),
        random_seed=seed,
        target_route_length_m=1000.0,
        seed_point=(0.0, 0.0),
        flow_angle_degrees=0.0,
        waves=() if family == "monotonic" else base.waves,
    )


def _network_config(backend: str, seed: int, flowy_executable: str | None) -> CaveNetworkConfig:
    return CaveNetworkConfig(
        random_seed=seed,
        target_route_length_m=1000.0,
        source_count=2,
        trace_max_steps=140,
        spur_count=1,
        network_density=1.0,
        emplacement_backend=backend,
        downflow_ensemble_size=4,
        flowy_executable=flowy_executable,
        flowy_timeout_s=30.0,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _network_signature(network: CaveNetwork, metrics: dict[str, Any]) -> str:
    """Hash topology and diagnostics, excluding timing and temp paths."""

    payload = {
        "metrics": metrics,
        "nodes": [
            [node.node_id, node.kind, round(node.x, 8), round(node.y, 8), round(node.along_position, 8)]
            for node in network.nodes
        ],
        "segments": [
            {
                "id": segment.segment_id,
                "start": segment.start_node_id,
                "end": segment.end_node_id,
                "kind": segment.kind,
                "z": segment.z_level,
                "points": [
                    [round(point.x, 8), round(point.y, 8), round(point.elevation, 8), round(point.arc_length, 8)]
                    for point in segment.points
                ],
            }
            for segment in network.segments
        ],
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _downstream_progress(network: CaveNetwork) -> float:
    proposal_progress = network.backend_provenance.get("proposal_downstream_progress_m")
    if isinstance(proposal_progress, (int, float)):
        return float(proposal_progress)
    nodes = {node.node_id: node for node in network.nodes}
    route = [nodes[node_id] for node_id in network.dominant_route_node_ids if node_id in nodes]
    if len(route) < 2:
        return 0.0
    return float(route[-1].along_position - route[0].along_position)


def _case_metrics(network: CaveNetwork, host: Any) -> dict[str, Any]:
    diagnostics = network_metrics(network, host)
    route_length = max(float(diagnostics.get("main_route_length_m", 0.0)), 1e-9)
    topology = diagnostics.get("normalized_topology", {})
    return {
        "connected_components": int(diagnostics["connected_component_count"]),
        "source_unreachable_nodes": int(diagnostics["source_unreachable_node_count"]),
        "entries_without_exit_path": int(diagnostics["entries_without_exit_path_count"]),
        "zero_flux_segments": int(diagnostics["zero_flux_segment_count"]),
        "branch_fraction": float(topology.get("branch_segment_fraction", 0.0)),
        "cyclomatic_number": int(diagnostics["cyclomatic_number"]),
        "cyclomatic_per_km": float(diagnostics["cyclomatic_number"] * 1000.0 / route_length),
        "junction_density_per_km": float(len(network.junctions) * 1000.0 / route_length),
        "stacked_share": float(diagnostics["stacked_segment_count"] / max(diagnostics["edge_count"], 1)),
        "sinuosity": float(diagnostics["length_weighted_mean_sinuosity"]),
        "downstream_progress_m": _downstream_progress(network),
        "main_route_length_m": float(diagnostics["main_route_length_m"]),
        "node_count": int(diagnostics["node_count"]),
        "edge_count": int(diagnostics["edge_count"]),
    }


def _provenance(backend: str, flowy_executable: str | None) -> dict[str, Any]:
    if backend == "internal":
        return {
            "implementation": "PLUME builtin hybrid_lobe",
            "scientific_role": "production process-informed baseline",
            "official_library": False,
            "license_burden": "none beyond PLUME",
        }
    if backend == "downflow_reference":
        return {
            "implementation": "PLUME perturbed-DEM reference adapter",
            "scientific_role": "optional experimental prior",
            "official_library": False,
            "license_burden": "none beyond PLUME; not an official DOWNFLOW package",
        }
    return {
        "implementation": "flowy-code/flowy executable",
        "scientific_role": "optional experimental prior",
        "official_library": True,
        "executable": flowy_executable,
        "license_burden": "external GPL-3.0 dependency; runtime executable required",
    }


def _run_one(
    *,
    family: str,
    backend: str,
    seed: int,
    host: Any,
    flowy_executable: str | None,
) -> tuple[dict[str, Any], CaveNetwork | None]:
    config = _network_config(backend, seed, flowy_executable)
    started = time.perf_counter()
    try:
        network = CaveNetworkGenerator(config).generate(host)
        elapsed = time.perf_counter() - started
        metrics = _case_metrics(network, host)
        valid = (
            metrics["connected_components"] == 1
            and metrics["source_unreachable_nodes"] == 0
            and metrics["entries_without_exit_path"] == 0
            and metrics["zero_flux_segments"] == 0
        )
        diagnostics = network_metrics(network, host)
        result = {
            "family": family,
            "backend": backend,
            "seed": seed,
            "success": True,
            "valid": bool(valid),
            "runtime_s": float(elapsed),
            "metrics": metrics,
            "backend_provenance": _jsonable(network.backend_provenance),
            "signature": _network_signature(network, diagnostics),
            "error": None,
        }
        return result, network
    except Exception as exc:  # benchmark records backend failures by design
        elapsed = time.perf_counter() - started
        return {
            "family": family,
            "backend": backend,
            "seed": seed,
            "success": False,
            "valid": False,
            "runtime_s": float(elapsed),
            "metrics": None,
            "backend_provenance": _provenance(backend, flowy_executable),
            "signature": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, None


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(case["family"], case["backend"])].append(case)
    output: dict[str, Any] = {}
    metric_names = (
        "runtime_s",
        "branch_fraction",
        "cyclomatic_per_km",
        "junction_density_per_km",
        "stacked_share",
        "sinuosity",
        "downstream_progress_m",
    )
    for (family, backend), rows in sorted(grouped.items()):
        successes = [row for row in rows if row["success"]]
        valid = [row for row in rows if row["valid"]]
        values: dict[str, dict[str, float | None]] = {}
        for name in metric_names:
            source_rows = rows if name == "runtime_s" else successes
            source = [
                float(row["runtime_s"] if name == "runtime_s" else row["metrics"][name])
                for row in source_rows
            ]
            values[name] = {
                "median": float(np.median(source)) if source else None,
                "mean": float(np.mean(source)) if source else None,
                "p10": float(np.percentile(source, 10.0)) if source else None,
                "p90": float(np.percentile(source, 90.0)) if source else None,
            }
        output[f"{family}/{backend}"] = {
            "family": family,
            "backend": backend,
            "runs": len(rows),
            "successes": len(successes),
            "valid": len(valid),
            "success_rate": len(successes) / max(len(rows), 1),
            "valid_rate": len(valid) / max(len(rows), 1),
            "deterministic_rate": sum(bool(row.get("deterministic")) for row in rows) / max(len(rows), 1),
            "failure_types": dict(
                sorted(
                    {
                        error_type: sum(
                            (row.get("error") or {}).get("type") == error_type for row in rows
                        )
                        for error_type in {
                            (row.get("error") or {}).get("type")
                            for row in rows
                            if row.get("error")
                        }
                    }.items()
                )
            ),
            "metrics": values,
        }
    return output


def _recommend(aggregates: dict[str, Any]) -> dict[str, Any]:
    weights = {
        "validity": 0.35,
        "determinism": 0.15,
        "runtime": 0.10,
        "integration": 0.15,
        "scientific_relevance": 0.10,
        "network_scope": 0.15,
    }
    integration_scores = {"internal": 1.0, "downflow_reference": 0.95, "flowy": 0.40}
    scientific_scores = {"internal": 0.75, "downflow_reference": 0.80, "flowy": 0.95}
    network_scope_scores = {"internal": 1.0, "downflow_reference": 0.35, "flowy": 0.40}
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for backend in BACKENDS:
        rows = [value for value in aggregates.values() if value["backend"] == backend]
        if not rows:
            continue
        valid = float(np.mean([row["valid_rate"] for row in rows]))
        deterministic = float(np.mean([row["deterministic_rate"] for row in rows]))
        runtime = [row["metrics"]["runtime_s"]["median"] for row in rows if row["metrics"]["runtime_s"]["median"] is not None]
        runtime_score = 1.0 / max(float(np.mean(runtime)) if runtime else 1e9, 1e-9)
        runtime_score = min(runtime_score, 1.0)
        components[backend] = {
            "validity": valid,
            "determinism": deterministic,
            "runtime": runtime_score,
            "integration": integration_scores[backend],
            "scientific_relevance": scientific_scores[backend],
            "network_scope": network_scope_scores[backend],
        }
        scores[backend] = sum(
            weights[name] * score for name, score in components[backend].items()
        )
    selected = max(scores, key=scores.get) if scores else "internal"
    return {
        "selected_default": selected,
        "scores": scores,
        "score_components": components,
        "weights": weights,
        "optional_experimental_priors": [backend for backend in scores if backend != selected],
        "rationale": (
            f"{selected} has the strongest weighted combination of validity, deterministic "
            "reproduction, runtime, integration burden, scientific relevance, and ability to "
            "contribute a complete buried-tube network. Other backends remain explicit priors; "
            "their failures are recorded and never trigger a silent fallback."
        ),
    }


def _plot_networks(cases: list[dict[str, Any]], networks: dict[tuple[str, str], CaveNetwork], path: Path) -> None:
    fig, axes = plt.subplots(len(FAMILIES), len(BACKENDS), figsize=(15, 8), squeeze=False)
    colors = {"backbone": "#1f77b4", "source_feeder": "#2ca02c", "spur": "#9467bd"}
    for row, family in enumerate(FAMILIES):
        for col, backend in enumerate(BACKENDS):
            axis = axes[row][col]
            network = networks.get((family, backend))
            if network is None:
                failure = next((case for case in cases if case["family"] == family and case["backend"] == backend), None)
                message = (failure or {}).get("error", {}).get("message", "no successful run")
                axis.text(0.5, 0.5, "FAILED\n" + str(message)[:100], ha="center", va="center", wrap=True)
                axis.set_axis_off()
                axis.set_title(f"{family} · {backend}")
                continue
            for segment in network.segments:
                axis.plot(
                    [point.x for point in segment.points],
                    [point.y for point in segment.points],
                    color=colors.get(segment.kind, "#7f7f7f"),
                    alpha=0.8,
                    linewidth=1.0 + 0.5 * (segment.z_level != 0),
                )
            axis.scatter([node.x for node in network.nodes], [node.y for node in network.nodes], s=5, color="black", alpha=0.45)
            axis.set_title(f"{family} · {backend}")
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
    fig.suptitle("Stage-B emplacement backend centerline comparison (seed 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metrics(cases: list[dict[str, Any]], path: Path) -> None:
    names = (
        ("branch_fraction", "branch fraction"),
        ("cyclomatic_per_km", "cyclomatic / km"),
        ("junction_density_per_km", "junctions / km"),
        ("stacked_share", "stacked share"),
        ("sinuosity", "sinuosity"),
        ("downstream_progress_m", "downstream progress (m)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    colors = {"internal": "#1f77b4", "downflow_reference": "#ff7f0e", "flowy": "#2ca02c"}
    for axis, (metric, label) in zip(axes.flat, names):
        for family_index, family in enumerate(FAMILIES):
            for backend_index, backend in enumerate(BACKENDS):
                values = [
                    float(case["metrics"][metric])
                    for case in cases
                    if case["family"] == family and case["backend"] == backend and case["success"]
                ]
                if not values:
                    continue
                x = family_index * 4.0 + backend_index
                axis.scatter(np.full(len(values), x), values, color=colors[backend], alpha=0.6, s=24)
                axis.plot([x - 0.15, x + 0.15], [np.median(values)] * 2, color=colors[backend], linewidth=3)
        axis.set_xticks([0, 1, 2, 4, 5, 6])
        axis.set_xticklabels(["N/I", "N/D", "N/F", "M/I", "M/D", "M/F"], rotation=30)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[name], label=name, markersize=7) for name in BACKENDS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3)
    fig.suptitle(
        "Stage-B backend metrics across named seeds (N=natural, M=monotonic)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_scorecard(
    aggregates: dict[str, Any],
    recommendation: dict[str, Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"internal": "#1f77b4", "downflow_reference": "#ff7f0e", "flowy": "#2ca02c"}
    available_backends = [backend for backend in BACKENDS if backend in recommendation["scores"]]
    available_families = [
        family
        for family in FAMILIES
        if any(value["family"] == family for value in aggregates.values())
    ]
    positions = np.arange(len(available_backends), dtype=float)
    width = 0.36
    for family_index, family in enumerate(available_families):
        offset = (family_index - 0.5 * (len(available_families) - 1)) * width
        success = [
            float(aggregates[f"{family}/{backend}"]["success_rate"])
            for backend in available_backends
        ]
        runtime = [
            aggregates[f"{family}/{backend}"]["metrics"]["runtime_s"]["median"]
            for backend in available_backends
        ]
        axes[0].bar(positions + offset, success, width, label=family)
        axes[1].bar(
            positions + offset,
            [float(value) if value is not None else 0.0 for value in runtime],
            width,
            label=family,
        )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("successful valid runs / seeds")
    axes[0].set_title("Validity by terrain family")
    axes[1].set_ylabel("median wall time (s)")
    axes[1].set_title("Stage-B runtime")
    axes[0].legend(frameon=False)

    scores = recommendation["scores"]
    axes[2].bar(
        positions,
        [float(scores[backend]) for backend in available_backends],
        color=[colors[backend] for backend in available_backends],
    )
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("weighted decision score")
    axes[2].set_title(f"Selected default: {recommendation['selected_default']}")
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                {"internal": "internal", "downflow_reference": "DOWNFLOW\nreference", "flowy": "Flowy"}[backend]
                for backend in available_backends
            ]
        )
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Emplacement backend validity, cost, and decision")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_benchmark(
    output_dir: str | Path,
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    families: Iterable[str] = FAMILIES,
    backends: Iterable[str] = BACKENDS,
    flowy_executable: str | None = FLOWY_DEFAULT,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    families = tuple(families)
    backends = tuple(backends)
    hosts = {
        (family, seed): HostFieldGenerator(_host_config(family, seed)).generate()
        for family in families
        for seed in seeds
    }
    cases: list[dict[str, Any]] = []
    network_for_plot: dict[tuple[str, str], CaveNetwork] = {}
    for family in families:
        for backend in backends:
            for seed in seeds:
                host = hosts[(family, seed)]
                first, network = _run_one(
                    family=family, backend=backend, seed=seed, host=host, flowy_executable=flowy_executable
                )
                second, _repeat_network = _run_one(
                    family=family, backend=backend, seed=seed, host=host, flowy_executable=flowy_executable
                )
                first["deterministic"] = bool(
                    first["success"] == second["success"]
                    and first.get("signature") == second.get("signature")
                    and (first.get("error") == second.get("error") if not first["success"] else True)
                )
                if network is not None and (family, backend) not in network_for_plot:
                    network_for_plot[(family, backend)] = network
                cases.append(first)
    aggregates = _aggregate(cases)
    report = {
        "schema_version": 1,
        "protocol": {
            "stage": "B",
            "families": list(families),
            "backends": list(backends),
            "seeds": list(seeds),
            "grid": {"width_m": 1800.0, "height_m": 1400.0, "nx": 60, "ny": 48},
            "network_controls": {
                "target_route_length_m": 1000.0,
                "source_count": 2,
                "trace_max_steps": 140,
                "spur_count": 1,
                "network_density": 1.0,
                "downflow_ensemble_size": 4,
            },
            "determinism": "each case is generated twice with identical host and backend seed",
            "flowy_executable": flowy_executable,
        },
        "backend_provenance": {backend: _provenance(backend, flowy_executable) for backend in backends},
        "cases": cases,
        "aggregates": aggregates,
        "recommendation": _recommend(aggregates),
        "artifacts": {
            "report_json": str(output / "benchmark.json"),
            "network_diagrams_png": str(output / "network_diagrams.png"),
            "metric_comparison_png": str(output / "metric_comparison.png"),
            "scorecard_png": str(output / "scorecard.png"),
        },
    }
    (output / "benchmark.json").write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_networks(cases, network_for_plot, output / "network_diagrams.png")
    _plot_metrics(cases, output / "metric_comparison.png")
    _plot_scorecard(aggregates, report["recommendation"], output / "scorecard.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/emplacement_backend_benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--flowy-executable", default=FLOWY_DEFAULT)
    args = parser.parse_args()
    report = run_benchmark(
        args.output_dir,
        seeds=args.seeds,
        families=args.families,
        backends=args.backends,
        flowy_executable=args.flowy_executable,
    )
    print(json.dumps(report["recommendation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
