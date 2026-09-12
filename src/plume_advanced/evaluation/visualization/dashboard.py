"""Deterministic Stage B/C diagnostic dashboard.

The renderer consumes already-generated network and section objects. It never
invokes Stage D geometry or a production pipeline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plume_advanced.evaluation.continuity import longitudinal_continuity
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.evaluation.metrics.sections import (
    pdc_comparable_section_summary,
    section_longitudinal_continuity,
)
from plume_advanced.evaluation.serialization import diagnostics_to_json


def _provenance(network: Any, provenance: dict[str, Any] | None) -> dict[str, Any]:
    config = getattr(network, "config", None)
    inferred = {
        "seed": getattr(config, "random_seed", None),
        "world": getattr(config, "world_id", None),
        "density": getattr(config, "network_density", None),
        "config": type(config).__name__ if config is not None else None,
    }
    if provenance:
        inferred.update(provenance)
    return {key: inferred[key] for key in sorted(inferred)}


def _density_rows(density_sweep: Any) -> list[dict[str, Any]]:
    if not density_sweep:
        return []
    if isinstance(density_sweep, dict):
        items = sorted(density_sweep.items(), key=lambda item: float(item[0]))
    else:
        items = []
        for item in density_sweep:
            density = (
                item.get("density")
                if isinstance(item, dict)
                else getattr(getattr(item, "config", None), "network_density", None)
            )
            if density is None:
                raise ValueError("density sweep records require a density field or network config")
            items.append((density, item))
        items.sort(key=lambda item: float(item[0]))
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for density, item in items:
        if density is None:
            raise ValueError("density sweep records require a numeric density")
        candidates = item if isinstance(item, (list, tuple)) else [item]
        for candidate in candidates:
            metrics = (
                candidate
                if isinstance(candidate, dict) and "node_count" in candidate
                else network_metrics(candidate)
            )
            normalized = metrics.get("normalized_topology", {})
            route = max(float(metrics.get("main_route_length_m", 0.0)), 1.0e-9)
            grouped[float(density)].append(
                {
                    "branch_ratio": float(normalized.get("branch_segment_fraction", 0.0)),
                    "cyclomatic_per_km": float(metrics.get("cyclomatic_number", 0.0))
                    / route
                    * 1000.0,
                    "stacked_share": float(metrics.get("stacked_segment_count", 0))
                    / max(
                        float(metrics.get("edge_count", metrics.get("segment_count", 0)) or 0),
                        1.0,
                    ),
                    "capture_junction_density_per_km": float(
                        metrics.get("vertical_capture_count", 0)
                        + metrics.get("split_merge_region_count", 0)
                    )
                    / route
                    * 1000.0,
                    "connected": metrics.get("connected_component_count", 0) == 1,
                    "zero_flux": metrics.get("zero_flux_segment_count", 0) == 0,
                }
            )
    rows = []
    for density in sorted(grouped):
        values = grouped[density]
        row: dict[str, Any] = {"density": density, "seed_count": len(values)}
        for field in (
            "branch_ratio",
            "cyclomatic_per_km",
            "stacked_share",
            "capture_junction_density_per_km",
        ):
            array = np.asarray([float(value[field]) for value in values], dtype=float)
            row[field] = float(np.median(array))
            row[f"{field}_p10"] = float(np.percentile(array, 10.0))
            row[f"{field}_p90"] = float(np.percentile(array, 90.0))
        row["all_connected"] = all(bool(value["connected"]) for value in values)
        row["all_zero_flux_free"] = all(bool(value["zero_flux"]) for value in values)
        rows.append(row)
    return rows


def _section_records(section_field: Any) -> list[dict[str, Any]]:
    records = []
    for segment_field in getattr(section_field, "segment_fields", ()):
        for sample in segment_field.samples:
            metrics = contour_morphometry(sample.profile_points)
            records.append(
                {
                    "segment_id": segment_field.segment_id,
                    "arc_length": sample.segment_arc_length,
                    **metrics,
                    "profile": sample.profile_points,
                }
            )
    return records


def _ordered_route_segments(network: Any) -> list[tuple[Any, bool]]:
    route_nodes = tuple(getattr(network, "dominant_route_node_ids", ()))
    segments = tuple(getattr(network, "segments", ()))
    ordered: list[tuple[Any, bool]] = []
    for start, end in zip(route_nodes, route_nodes[1:]):
        match = next(
            (
                segment
                for segment in segments
                if segment.start_node_id == start and segment.end_node_id == end
            ),
            None,
        )
        reverse = False
        if match is None:
            match = next(
                (
                    segment
                    for segment in segments
                    if segment.start_node_id == end and segment.end_node_id == start
                ),
                None,
            )
            reverse = match is not None
        if match is not None:
            ordered.append((match, reverse))
    return ordered


def _route_profile_data(network: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arc: list[float] = []
    elevation: list[float] = []
    grade: list[float] = []
    cumulative = 0.0
    for segment, reverse in _ordered_route_segments(network):
        points = list(segment.points)
        if reverse:
            points.reverse()
        for index, point in enumerate(points):
            if arc and index == 0:
                continue
            if arc:
                ds = max(float(point.arc_length - points[index - 1].arc_length), 0.0)
                if ds <= 0.0:
                    ds = math.dist(
                        (point.x, point.y),
                        (points[index - 1].x, points[index - 1].y),
                    )
                cumulative += ds
            arc.append(cumulative)
            elevation.append(float(point.elevation))
            if len(arc) == 1:
                grade.append(0.0)
            else:
                ds = arc[-1] - arc[-2]
                grade.append((elevation[-1] - elevation[-2]) / max(ds, 1.0e-9))
    return np.asarray(arc), np.asarray(elevation), np.asarray(grade)


def _dominant_route_continuity(network: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = {segment.segment_id: reverse for segment, reverse in _ordered_route_segments(network)}
    cumulative_records: list[dict[str, Any]] = []
    offset = 0.0
    for segment_id in ordered:
        segment_records = [record for record in records if record["segment_id"] == segment_id]
        if not segment_records:
            continue
        reverse = ordered[segment_id]
        segment_records.sort(key=lambda record: float(record["arc_length"]), reverse=reverse)
        local_end = max(float(record["arc_length"]) for record in segment_records)
        for record in segment_records:
            local_arc = (
                local_end - float(record["arc_length"]) if reverse else float(record["arc_length"])
            )
            cumulative_records.append({**record, "arc_length": offset + local_arc})
        offset += local_end
    return longitudinal_continuity(cumulative_records) if cumulative_records else {"metrics": {}}


def dashboard_payload(
    network: Any,
    section_field: Any | None = None,
    *,
    density_sweep: Any = None,
    pdc_calibration: Any = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic data for the dashboard and optional JSON sidecar."""

    metrics = network_metrics(network)
    records = _section_records(section_field) if section_field is not None else []
    calibration = None
    pdc_label = "generated-only (no PDC calibration supplied)"
    if pdc_calibration is not None:
        if not isinstance(pdc_calibration, dict):
            raise ValueError(
                "PDC data must be wrapped as {'partition': 'calibration', 'records': ...}"
            )
        partition = str(pdc_calibration.get("partition", "")).lower()
        if partition != "calibration":
            raise ValueError(
                "dashboard accepts calibration caves only; confirmatory PDC data is forbidden"
            )
        calibration = pdc_calibration.get("records", pdc_calibration.get("sections", []))
        pdc_label = "PDC calibration caves"
    payload: dict[str, Any] = {
        "schema": "plume.evaluation-dashboard.v1",
        "provenance": _provenance(network, provenance),
        "network": metrics,
        "density_sweep": _density_rows(density_sweep),
        "sections": {
            "section_count": len(records),
            "morphometry": pdc_comparable_section_summary(records) if records else {},
            "continuity": section_longitudinal_continuity(section_field)
            if section_field is not None
            else {"metrics": {}},
            "dominant_route_continuity": _dominant_route_continuity(network, records),
            "pdc_label": pdc_label,
        },
    }
    if calibration is not None:
        payload["sections"]["pdc_calibration"] = pdc_comparable_section_summary(list(calibration))
    return payload


def render_diagnostic_dashboard(
    network: Any,
    section_field: Any | None = None,
    output_path: str | Path = "diagnostic_dashboard.png",
    *,
    density_sweep: Any = None,
    pdc_calibration: Any = None,
    provenance: dict[str, Any] | None = None,
    json_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Render a 3x3 PNG dashboard and optional canonical JSON sidecar."""

    payload = dashboard_payload(
        network,
        section_field,
        density_sweep=density_sweep,
        pdc_calibration=pdc_calibration,
        provenance=provenance,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = "PLUME diagnostic dashboard | " + ", ".join(
        f"{key}={value}" for key, value in payload["provenance"].items() if value is not None
    )
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    _plot_topology(axes[0, 0], network)
    _plot_sinuosity(axes[0, 1], payload["network"])
    _plot_uphill(axes[0, 2], payload["network"])
    _plot_route_profile(axes[1, 0], network)
    _plot_density(axes[1, 1], payload["density_sweep"])
    _plot_section_evolution(axes[1, 2], section_field)
    _plot_section_continuity(axes[2, 0], payload["sections"])
    _plot_morphospace(axes[2, 1], section_field)
    _plot_pdc(axes[2, 2], payload["sections"])
    figure.suptitle(title, fontsize=12)
    figure.savefig(output, dpi=150, metadata={"Title": title, "Software": "PLUME evaluation"})
    plt.close(figure)
    sidecar = None
    if json_path is not None:
        sidecar = Path(json_path)
    else:
        sidecar = output.with_suffix(".json")
    sidecar.write_text(diagnostics_to_json(payload, indent=2) + "\n", encoding="utf-8")
    return output, sidecar


def _plot_topology(axis, network):
    colors = {
        "backbone": "#263238",
        "source_feeder": "#1976d2",
        "anastomosis": "#ef6c00",
        "spur": "#8e24aa",
    }
    for segment in sorted(network.segments, key=lambda item: item.segment_id):
        x = [point.x for point in segment.points]
        y = [point.y for point in segment.points]
        birth_phase = segment.metadata.get(
            "birth_phase", segment.metadata.get("emplacement_birth_phase")
        )
        death_phase = segment.metadata.get(
            "death_phase", segment.metadata.get("emplacement_death_phase")
        )
        phase = float(
            birth_phase
            if birth_phase is not None
            else (death_phase if death_phase is not None else 0.0)
        )
        axis.plot(
            x,
            y,
            color=colors.get(segment.kind, "#607d8b"),
            linestyle="--" if segment.z_level else "-",
            linewidth=0.8 + 0.35 * min(phase, 5.0),
            alpha=0.8,
        )
    axis.set_title("Topology (kind color, z-level dash, birth/death phase width)")
    axis.set_aspect("equal", adjustable="datalim")
    handles = [Line2D([0], [0], color=color, label=kind) for kind, color in colors.items()]
    handles.extend(
        [
            Line2D([0], [0], color="#607d8b", label="z=0 solid"),
            Line2D([0], [0], color="#607d8b", linestyle="--", label="z>0 dashed"),
        ]
    )
    axis.legend(handles=handles, fontsize=7, loc="best")


def _plot_sinuosity(axis, metrics):
    grouped = metrics.get("sinuosity_by_kind", {})
    kinds = sorted(grouped)
    med = [grouped[k].get("median") or 0 for k in kinds]
    iqr = [grouped[k].get("iqr") or 0 for k in kinds]
    axis.bar(kinds, med, yerr=iqr, color="#546e7a", capsize=3)
    axis.set_title("Sinuosity by kind (median ± IQR)")
    axis.tick_params(axis="x", rotation=45)


def _plot_uphill(axis, metrics):
    grouped = metrics.get("sustained_uphill_by_kind", {})
    kinds = sorted(grouped)
    vals = [grouped[k]["uphill_fraction"].get("median") or 0 for k in kinds]
    axis.bar(kinds, vals, color="#d84315")
    axis.set_ylabel("fraction")
    axis.set_title("Sustained uphill fraction")
    axis.tick_params(axis="x", rotation=45)
    provenance = metrics.get("uphill_provenance_by_kind", {})
    for index, kind in enumerate(kinds):
        labels = sorted(provenance.get(kind, {"unspecified": 1}))
        label = "/".join(labels)
        axis.text(index, vals[index], label, ha="center", va="bottom", fontsize=7, rotation=45)


def _plot_route_profile(axis, network):
    arc, elevation, grade = _route_profile_data(network)
    if arc.size:
        axis.plot(arc, elevation, label="elevation")
        axis.set_ylabel("elevation")
        axis.set_xlabel("cumulative route arc (m)")
        axis.set_title("Dominant route elevation / grade")
        axis2 = axis.twinx()
        axis2.plot(arc, grade, color="#ef6c00", alpha=0.6, label="grade")
        axis2.set_ylabel("dz/ds")


def _plot_density(axis, rows):
    if not rows:
        axis.text(0.5, 0.5, "No density sweep supplied", ha="center")
        axis.set_axis_off()
        return
    x = [row["density"] for row in rows]
    for key in ("branch_ratio", "stacked_share"):
        median = np.asarray([row[key] for row in rows], dtype=float)
        low = np.asarray([row[f"{key}_p10"] for row in rows], dtype=float)
        high = np.asarray([row[f"{key}_p90"] for row in rows], dtype=float)
        axis.plot(x, median, marker="o", label=key)
        axis.fill_between(x, low, high, alpha=0.12)
    rates = axis.twinx()
    for key in ("cyclomatic_per_km", "capture_junction_density_per_km"):
        median = np.asarray([row[key] for row in rows], dtype=float)
        low = np.asarray([row[f"{key}_p10"] for row in rows], dtype=float)
        high = np.asarray([row[f"{key}_p90"] for row in rows], dtype=float)
        rates.plot(x, median, marker="s", linestyle="--", label=key)
        rates.fill_between(x, low, high, alpha=0.08)
    rates.set_ylabel("density rates (per km)")
    axis.set_title("Density sweep (median with 10–90% interval)")
    axis.set_xlabel("network density")
    axis.legend(fontsize=7, loc="upper left")
    rates.legend(fontsize=7, loc="upper right")
    axis.text(
        0.01,
        0.01,
        "✓ connected / zero-flux-free; ✗ unresolved",
        transform=axis.transAxes,
        fontsize=7,
    )


def _plot_section_evolution(axis, field):
    if field is None:
        axis.text(0.5, 0.5, "No Stage-C section field", ha="center")
        axis.set_axis_off()
        return
    for segment in field.segment_fields:
        x = [sample.segment_arc_length for sample in segment.samples]
        axis.plot(x, [sample.tube_width for sample in segment.samples], color="#1565c0")
        axis.plot(x, [sample.tube_height for sample in segment.samples], color="#ef6c00")
    axis.set_title("Width / height evolution along arc length")
    axis.set_xlabel("arc length (m)")


def _plot_section_continuity(axis, sections):
    segment_reports = sections.get("continuity", {}).get("segments", {})
    labels = ("compactness", "floor residual", "roof asymmetry")
    values = _continuity_panel_values(segment_reports)
    if labels:
        axis.bar(labels, values, color="#6a1b9a")
        axis.tick_params(axis="x", rotation=45)
    axis.set_title("Continuity: repetitive short-period score (segment median)")


def _continuity_panel_values(segment_reports):
    values = []
    for feature in ("compactness", "floor_residual", "roof_asymmetry"):
        scores = [
            report.get("metrics", {}).get(feature, {}).get("repetitive_oscillation_score")
            for report in segment_reports.values()
        ]
        finite_scores = [float(score) for score in scores if score is not None]
        values.append(float(np.median(finite_scores)) if finite_scores else 0.0)
    return values


def _plot_morphospace(axis, field):
    records = _section_records(field) if field is not None else []
    if records:
        axis.scatter(
            [record["width_m"] for record in records],
            [record["height_m"] for record in records],
            s=8,
            alpha=0.35,
            label="generated",
        )
    axis.set_xlabel("width (m)")
    axis.set_ylabel("height (m)")
    axis.set_title("Cross-section morphospace")
    if records:
        axis.legend(fontsize=7)
        inset = axis.inset_axes([0.67, 0.05, 0.30, 0.30])
        selected = records[:: max(1, len(records) // 3)][:3]
        gradient = plt.cm.viridis(np.linspace(0.15, 0.9, len(selected)))
        for color, record in zip(gradient, selected, strict=True):
            contour = np.asarray(record["profile"], dtype=float)
            inset.plot(contour[:, 0], contour[:, 1], color=color, linewidth=0.8)
        inset.set_title("contour gradient", fontsize=7)
        inset.set_xticks([])
        inset.set_yticks([])


def _plot_pdc(axis, sections):
    features = ("aspect_ratio", "compactness", "floor_residual", "roof_asymmetry")
    generated_summary = sections.get("morphometry", {})
    calibration_summary = sections.get("pdc_calibration", {})
    labels = ["generated", "PDC calibration"] if calibration_summary else ["generated"]
    positions = np.arange(len(features))
    for offset, label in enumerate(labels):
        source = calibration_summary if label.startswith("PDC") else generated_summary
        medians = [source.get(feature, {}).get("median") or 0.0 for feature in features]
        lows = [
            medians[index] - (source.get(feature, {}).get("q1") or medians[index])
            for index, feature in enumerate(features)
        ]
        highs = [
            (source.get(feature, {}).get("q3") or medians[index]) - medians[index]
            for index, feature in enumerate(features)
        ]
        axis.bar(
            positions + (offset - (len(labels) - 1) / 2.0) * 0.35,
            medians,
            width=0.35,
            yerr=[lows, highs],
            label=label,
            color="#9e9e9e" if label.startswith("PDC") else "#1565c0",
            capsize=2,
        )
    axis.set_xticks(positions, features, rotation=45, ha="right")
    axis.set_title(
        "PDC quartiles (Q1 / median / Q3)\n" + sections.get("pdc_label", "generated-only")
    )
    axis.legend(fontsize=7)


__all__ = ["dashboard_payload", "render_diagnostic_dashboard"]
