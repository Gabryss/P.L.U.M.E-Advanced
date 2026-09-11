"""Paper-oriented figures generated only from saved results."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "plume-advanced-cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib.pyplot as plt
import numpy as np


def generate_figures(results_root: str | Path) -> list[Path]:
    root = Path(results_root)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    morphology = root / "morphometry"
    if (morphology / "raw_reference_sections.csv").is_file():
        paths.extend(_morphometry_figure(morphology, output))
    host = root / "host_ablation" / "raw_results.csv"
    if host.is_file():
        paths.extend(_host_figure(host, output))
    sampling = root / "sampling_ablation" / "raw_results.csv"
    if sampling.is_file():
        paths.extend(_sampling_figure(sampling, output))
    controllability = root / "controllability" / "raw_results.csv"
    if controllability.is_file():
        paths.extend(_controllability_figure(controllability, output))
    scalability = root / "scalability" / "raw_results.csv"
    if scalability.is_file():
        paths.extend(_scalability_figure(scalability, output))
    return paths


def _morphometry_figure(source: Path, output: Path) -> list[Path]:
    datasets = {
        "PDC v2.0": _read_csv(source / "raw_reference_sections.csv"),
        "PLUME-Advanced": _read_csv(source / "raw_generated_sections.csv"),
        "Matched ellipse": _read_csv(source / "raw_baseline_sections.csv"),
    }
    metrics = ("aspect_ratio", "compactness", "floor_residual_norm", "roof_asymmetry_norm")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), metrics, strict=True):
        for label, rows in datasets.items():
            values = np.sort([float(row[metric]) for row in rows if row.get(metric)])
            if values.size:
                axis.plot(values, np.linspace(0.0, 1.0, values.size), label=label)
        axis.set_xlabel(metric.replace("_", " "))
        axis.set_ylabel("ECDF")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    return _save(figure, output / "figure_morphometry_cdf")


def _host_figure(source: Path, output: Path) -> list[Path]:
    rows = [row for row in _read_csv(source) if row.get("status") == "complete"]
    conditions = sorted({row["condition_id"] for row in rows})
    values = [
        np.median(
            [
                float(row["centerline_displacement_mean_m"])
                for row in rows
                if row["condition_id"] == condition
            ]
        )
        for condition in conditions
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    axis.bar(conditions, values)
    axis.set_ylabel("Median symmetric displacement (m)")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    return _save(figure, output / "figure_host_ablation")


def _sampling_figure(source: Path, output: Path) -> list[Path]:
    rows = [row for row in _read_csv(source) if row.get("status") == "complete"]
    figure, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    for label, count_key, error_key, marker in (
        ("Adaptive", "adaptive_section_count", "adaptive_error_mean_m", "o"),
        ("Uniform (matched count)", "uniform_section_count", "uniform_error_mean_m", "s"),
    ):
        axis.scatter(
            [float(row[count_key]) for row in rows],
            [float(row[error_key]) for row in rows],
            label=label,
            marker=marker,
            alpha=0.75,
        )
    axis.set_xlabel("Section count")
    axis.set_ylabel("Symmetric mean error (m)")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(figure, output / "figure_sampling_ablation")


def _controllability_figure(source: Path, output: Path) -> list[Path]:
    rows = [row for row in _read_csv(source) if row.get("status") == "complete"]
    mappings = (
        ("distributary", "cyclomatic_number", "Cyclomatic number"),
        ("duration", "main_route_length_m", "Main-route length (m)"),
        ("inflation", "junction_to_passage_width_ratio", "Junction/passage width"),
        ("supply", "host_horizontal_scale", "Host horizontal scale"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for axis, (control, metric, label) in zip(axes.ravel(), mappings, strict=True):
        selected = [row for row in rows if row.get("control") == control and row.get(metric)]
        axis.scatter(
            [float(row["control_value"]) for row in selected],
            [float(row[metric]) for row in selected],
            alpha=0.65,
            s=18,
        )
        axis.set_xlabel(control.replace("_", " "))
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    return _save(figure, output / "figure_controllability")


def _scalability_figure(source: Path, output: Path) -> list[Path]:
    rows = _read_csv(source)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for mode, color, offset in (("dense", "#31688e", -.025), ("tiled", "#d95f02", .025)):
        for successful in (True, False):
            selected = [row for row in rows
                        if row.get("storage_mode_requested", "") == mode
                        or row.get("condition_id", "").endswith(f"m-{mode}")]
            selected = [row for row in selected if (row.get("status") == "complete") == successful]
            if not selected:
                continue
            lengths = [(float(row["route_length_requested_m"])
                        if row.get("route_length_requested_m")
                        else float(row["condition_id"].split("m-", 1)[0])) / 1000.0
                       for row in selected]
            for axis, metric in zip(axes, ("wall_time_s", "peak_rss_gib"), strict=True):
                measured = [(length * (1 + offset), float(row[metric]))
                            for length, row in zip(lengths, selected, strict=True)
                            if row.get(metric) and float(row[metric]) > 0]
                if measured:
                    x, y = zip(*measured, strict=True)
                    axis.scatter(x, y, color=color, marker=("o" if mode == "dense" else "s") if successful else "x",
                                 s=25 if successful else 42, alpha=.8,
                                 label=f"{mode.title()} {'completed' if successful else 'failed/stopped'} ({len(selected)})")
    axes[0].set(xlabel="Requested route length (km)", ylabel="Wall time (s)", yscale="log")
    axes[1].set(xlabel="Requested route length (km)", ylabel="Peak RSS (GiB)", yscale="log")
    for limit in sorted({float(row["memory_limit_gib"]) for row in rows if row.get("memory_limit_gib")}):
        axes[1].axhline(limit, color="0.5", linestyle=":", linewidth=.8)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks([.5, 1., 2., 5.], ["0.5", "1", "2", "5"])
        axis.xaxis.set_minor_formatter(plt.NullFormatter())
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=6.5)
    # Failed measurements are plotted at the resource use recorded when they
    # stopped. They are not assigned the cost or speedup of a completed mesh.
    return _save(figure, output / "figure_scalability")


def _save(figure: plt.Figure, prefix: Path) -> list[Path]:
    paths = [prefix.with_suffix(".svg"), prefix.with_suffix(".png")]
    figure.savefig(paths[0])
    figure.savefig(paths[1], dpi=300)
    plt.close(figure)
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


__all__ = ["generate_figures"]
