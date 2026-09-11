"""Derive paper tables and an auditable report from completed campaign records."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[2]
sys.path.insert(0, str(CAMPAIGN / "frozen/src"))
from plume_advanced.evaluation.statistics import paired_bootstrap_difference

NAMES = {
    "width_m": "Width", "height_m": "Height", "aspect_ratio": "Aspect ratio",
    "area_m2": "Area", "compactness": "Compactness",
    "floor_residual_norm": "Floor residual", "roof_asymmetry_norm": "Roof asymmetry",
}
CONDITIONS = {
    "no_slope": "No slope term", "no_cover": "No cover term",
    "no_fracture": "No fracture term", "no_capacity": "No capacity term",
    "no_stability": "No stability term", "unconditioned": "Constant cost",
}


def read(path):
    return json.loads(path.read_text())


def fmt(value):
    if value is None:
        return r"\textemdash"
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("A reported measurement is non-finite")
    if value == 0:
        return "0"
    if abs(value) >= 10000 or abs(value) < .001:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"
    return f"{value:.3g}"


def interval(lower, upper):
    return f"[{fmt(lower)}, {fmt(upper)}]"


def table(headers, rows, widths, digest):
    return "\n".join([
        f"% Source summary SHA256: {digest}",
        rf"\begin{{tabularx}}{{\columnwidth}}{{{widths}}}",
        r"\toprule", " & ".join(headers) + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabularx}", "",
    ])


def paired(a, b):
    ci = paired_bootstrap_difference(a, b, iterations=2000,
                                     confidence_level=.95, seed=20260101)
    return {"valid_pairs": len(a), "median_delta": ci.estimate,
            "ci_lower": ci.lower, "ci_upper": ci.upper}


def main():
    frozen = read(CAMPAIGN / "freeze_manifest.json")
    declaration = tomllib.loads((CAMPAIGN / "frozen/paper/experiments.toml").read_text())
    source = ROOT / "paper/outputs"
    target = ROOT / "paper/overleaf/results"
    target.mkdir(exist_ok=True)
    summary = {}
    records = {}
    inventory = {}
    for name, expected in frozen["planned_cases"].items():
        rows = [read(p) for p in sorted((source / name / "cases").glob("*.json"))]
        if len(rows) != expected:
            raise ValueError(f"{name}: {len(rows)}/{expected} case records; campaign incomplete")
        if len({r["run_id"] for r in rows}) != expected:
            raise ValueError(f"{name}: duplicate run identifiers")
        summary[name] = read(source / name / "summary.json")
        records[name] = rows
        inventory[name] = {
            "planned": expected, "statuses": dict(Counter(r["status"] for r in rows)),
            "failures": dict(Counter(
                r.get("failure_kind") or r.get("failure_reason", "unknown").split("\n")[0]
                for r in rows if r["status"] != "complete")),
            "summary_sha256": hashlib.sha256((source / name / "summary.json").read_bytes()).hexdigest(),
        }
    morphology = summary["morphometry"]
    descriptors = {row["metric"]: row for row in morphology["metrics"]}
    morph_rows = [[NAMES[r["metric"]], fmt(r["baseline_normalized_w1"]),
                   fmt(r["generated_normalized_w1"]),
                   interval(r["normalized_w1_ci_lower"], r["normalized_w1_ci_upper"])]
                  for r in morphology["metrics"]]
    morph_rows.append(["Selected aggregate", fmt(morphology["aggregate_baseline_normalized_w1"]),
                       fmt(morphology["aggregate_generated_normalized_w1"]),
                       interval(morphology["aggregate_ci_lower"], morphology["aggregate_ci_upper"])])
    (target / "table_morphometry.tex").write_text(table(
        ["Descriptor", "Ellipse", "Advanced", "95\\% CI"], morph_rows,
        "@{}Yccc@{}", inventory["morphometry"]["summary_sha256"]))
    ablation = summary["host_ablation"]["paired_effects"]
    ablation_rows = []
    for name, label in CONDITIONS.items():
        values = ablation[name]
        row = [label, str(values["valid_pairs"])]
        for metric in ("centerline_displacement_mean_m", "cyclomatic_number"):
            effect = values["effects"].get(metric)
            row.append(r"\textemdash" if effect is None else
                       r"\shortstack[r]{" + f"${effect['median_delta']:.3g}$" +
                       r"\\\scriptsize " +
                       f"$[{effect['ci_lower']:.3g}, {effect['ci_upper']:.3g}]$" + "}")
        ablation_rows.append(row)
    (target / "table_ablation.tex").write_text(table(
        ["Condition", "Pairs", "Shift (m)", r"$\Delta\beta_1$"], ablation_rows,
        "@{}Yccc@{}", inventory["host_ablation"]["summary_sha256"]))
    scale_rows = []
    for length in (500, 1000, 2000, 5000):
        for mode in ("dense", "tiled"):
            row = summary["scalability"]["conditions"][f"{length}m-{mode}"]
            scale_rows.append([f"{length/1000:g}", mode.title(),
                               f"{row['complete_n']}/{row['planned_n']}",
                               fmt(row["median_wall_time_s"]), fmt(row["median_peak_rss_gib"])])
    (target / "table_scalability.tex").write_text(table(
        ["Route (km)", "Mode", "Complete", "Time (s)", "RSS (GiB)"],
        scale_rows, "@{}Ylccc@{}", inventory["scalability"]["summary_sha256"]))
    # Supplementary effects use the same predeclared primary metrics and paired
    # median-difference estimator. No generator setting is changed here.
    supplements = {"controllability": {}, "sampling": {}, "scalability": {}}
    for control, info in summary["controllability"]["controls"].items():
        metric = info["primary_metric"]
        rows = [r for r in records["controllability"]
                if r["status"] == "complete" and r["control"] == control]
        levels = sorted(declaration["controllability"][f"{control}_values"])
        low = {r["seed"]: r[metric] for r in rows if r["control_value"] == levels[0]}
        high = {r["seed"]: r[metric] for r in rows if r["control_value"] == levels[-1]}
        seeds = sorted(low.keys() & high.keys())
        supplements["controllability"][control] = {
            "primary_metric": metric, "low_level": levels[0], "high_level": levels[-1],
            "level_medians": {str(level): (float(np.median(values)) if values else None)
                              for level in levels
                              for values in [[r[metric] for r in rows if r["control_value"] == level]]},
            "paired_high_minus_low": paired([low[s] for s in seeds], [high[s] for s in seeds])
                if seeds else None,
        }
    sampled = [r for r in records["sampling_ablation"] if r["status"] == "complete"]
    if sampled:
        for field in ("error_mean_m", "error_p95_m", "section_count"):
            supplements["sampling"][field] = paired(
                [r[f"uniform_{field}"] for r in sampled],
                [r[f"adaptive_{field}"] for r in sampled])
    completed_scale = [r for r in records["scalability"] if r["status"] == "complete"]
    if any(r["storage_mode_actual"] != r["storage_mode_requested"] for r in completed_scale):
        raise ValueError("A completed benchmark used a different storage mode than requested")
    diagnostics = {}
    for condition in summary["scalability"]["conditions"]:
        rows = [r for r in completed_scale if r["condition_id"] == condition]
        if not rows:
            diagnostics[condition] = {"completed_cases": 0}
            continue
        def extent(values):
            return {"minimum": float(min(values)), "median": float(np.median(values)),
                    "maximum": float(max(values))}
        diagnostics[condition] = {
            "completed_cases": len(rows),
            "realized_main_route_m": extent([r["network"]["dominant_route_length"] for r in rows]),
            "total_network_length_m": extent([r["network"]["total_length"] for r in rows]),
            "mesh_faces": extent([r["geometry"]["face_count"] for r in rows]),
            "mesh_components": extent([r["geometry"]["component_count"] for r in rows]),
            "void_voxel_components": extent([r["geometry"]["voxel_component_count"] for r in rows]),
            "bounding_grid_voxels": extent([r["geometry"]["voxel_count"] for r in rows]),
            "voxel_size_m": extent([r["voxel_size_m"] for r in rows]),
            "input_under_resolved_fraction": extent([
                r["section_resolution"]["under_resolved_count"] / r["section_resolution"]["section_count"]
                for r in rows]),
            "roof_screen_collapse_count": extent([r["geometry"]["stability_collapse_count"] for r in rows]),
            "scope": "Recorded mesh and volume counts plus input-profile resolution screening; not local mesh cuts or a convergence test.",
        }
    supplements["geometry_diagnostics"] = diagnostics
    export_checks = [check for row in records["export_consistency"]
                     if row["status"] == "complete" for check in row["target_checks"]]
    supplements["export_diagnostics"] = {}
    for target_name in sorted({check["target"] for check in export_checks}):
        checks = [check for check in export_checks if check["target"] == target_name]
        bounds_errors = [check["bbox_relative_extent_error"] for check in checks
                         if check.get("bbox_relative_extent_error") is not None]
        supplements["export_diagnostics"][target_name] = {
            "checked_packages": len(checks),
            "visual_present": sum(bool(check["visual_asset_present"]) for check in checks),
            "collision_present": sum(bool(check["collision_present"]) for check in checks),
            "parsed_bounds_n": len(bounds_errors),
            "maximum_relative_extent_error": max(bounds_errors) if bounds_errors else None,
            "scope": "Automatic presence and sorted-extent diagnostics comparing base mesh with prepared visual surface, including possible smoothing/displacement; no inverse-pose, materials or simulator-contact certification. USD and Gazebo package bounds are not parsed.",
        }
    for length in (500, 1000, 2000, 5000):
        rows = [r for r in completed_scale if r["route_length_requested_m"] == length]
        dense = {r["seed"]: r for r in rows if r["storage_mode_actual"] == "dense"}
        tiled = {r["seed"]: r for r in rows if r["storage_mode_actual"] == "tiled"}
        seeds = sorted(dense.keys() & tiled.keys())
        supplements["scalability"][str(length)] = {
            "valid_pairs": len(seeds),
            "median_tiled_over_dense_time": float(np.median([
                tiled[s]["wall_time_s"] / dense[s]["wall_time_s"] for s in seeds])) if seeds else None,
            "median_tiled_over_dense_rss": float(np.median([
                tiled[s]["peak_rss_gib"] / dense[s]["peak_rss_gib"] for s in seeds])) if seeds else None,
            "fidelity_scope": "Resource ratios do not establish equivalent final geometry.",
        }
    amendment_path = CAMPAIGN / "controls_amendment.json"
    report = {"schema": "plume.campaign-report.v1", "inventory": inventory,
              "source_freeze_sha256": frozen["file_manifest_sha256"],
              "control_evaluator_amendment": read(amendment_path) if amendment_path.exists() else None,
              "supplementary_effects": supplements, "summaries": summary}
    (CAMPAIGN / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Evaluation campaign results", "", f"Source freeze: `{frozen['file_manifest_sha256']}`.", "",
             "All declared cases were attempted. Completion means the experiment ran; it does not mean every scientific hypothesis or quality criterion was supported.", "",
             "| Experiment | Planned | Completed | Other outcomes |",
             "| --- | ---: | ---: | --- |"]
    for name, info in inventory.items():
        complete = info["statuses"].get("complete", 0)
        other = ", ".join(f"{k}: {v}" for k, v in info["statuses"].items() if k != "complete") or "None"
        lines.append(f"| {name} | {info['planned']} | {complete} | {other} |")
    lines.extend(["", "## Cross-section morphology", "",
                  f"Reference: {morphology['reference_sections']} usable sections from "
                  f"{morphology['reference_caves']} evaluation caves. Generated: "
                  f"{morphology['generated_sections']} sections from {morphology['complete_worlds']} worlds.", "",
                  "Normalized Wasserstein distance (lower is better); Advanced intervals resample caves and worlds.", "",
                  "| Descriptor | Ellipse | Advanced | Advanced 95% interval |",
                  "| --- | ---: | ---: | --- |"])
    for row in morphology["metrics"]:
        lines.append(f"| {NAMES[row['metric']]} | {row['baseline_normalized_w1']:.4g} | "
                     f"{row['generated_normalized_w1']:.4g} | "
                     f"[{row['normalized_w1_ci_lower']:.4g}, {row['normalized_w1_ci_upper']:.4g}] |")
    lines.extend([f"| Selected aggregate | {morphology['aggregate_baseline_normalized_w1']:.4g} | "
                  f"{morphology['aggregate_generated_normalized_w1']:.4g} | "
                  f"[{morphology['aggregate_ci_lower']:.4g}, {morphology['aggregate_ci_upper']:.4g}] |", "",
                  "The aggregate averages aspect ratio, compactness, floor residual and roof asymmetry. "
                  "Width, height and aspect ratio match the ellipse by construction. These are Stage-C comparisons, "
                  "not validation of final meshes or planetary distributions.", "",
                  f"Absolute sizes remain mismatched: generated/reference median width is "
                  f"{descriptors['width_m']['generated']['median']:.2f}/"
                  f"{descriptors['width_m']['reference']['median']:.2f} m, and median height is "
                  f"{descriptors['height_m']['generated']['median']:.2f}/"
                  f"{descriptors['height_m']['reference']['median']:.2f} m.", "",
                  "## Routing interventions", "",
                  "Effects are paired medians relative to the full model; brackets are 95% bootstrap intervals.", "",
                  "| Condition | Pairs | Mean centerline displacement (m) | Cycle-rank change |",
                  "| --- | ---: | --- | --- |"])
    for name, label in CONDITIONS.items():
        values = ablation[name]
        effects = []
        for metric in ("centerline_displacement_mean_m", "cyclomatic_number"):
            effect = values["effects"].get(metric)
            effects.append("Unavailable" if effect is None else
                           f"{effect['median_delta']:.4g} [{effect['ci_lower']:.4g}, {effect['ci_upper']:.4g}]")
        lines.append(f"| {label} | {values['valid_pairs']} | " + " | ".join(effects) + " |")
    lines.extend(["", "## Resource measurements", "",
                  "Time and RSS medians include completed cases only. Completion denominators include failures.", "",
                  "| Requested route (km) | Mode | Completed | Median time (s) | Median peak RSS (GiB) |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for length in (500, 1000, 2000, 5000):
        for mode in ("dense", "tiled"):
            item = summary["scalability"]["conditions"][f"{length}m-{mode}"]
            measured = ["Unavailable" if item[key] is None else f"{item[key]:.4g}"
                        for key in ("median_wall_time_s", "median_peak_rss_gib")]
            lines.append(f"| {length/1000:g} | {mode} | {item['complete_n']}/{item['planned_n']} | " + " | ".join(measured) + " |")
    lines.extend(["", "## Parameter response and sampling", "",
                  "High-minus-low control effects and adaptive-minus-uniform sampling effects are paired by seed. "
                  "Brackets contain pointwise 95% bootstrap intervals.", "",
                  "| Control | Primary readout | Levels (low to high) | Pairs | Median change [95% interval] |",
                  "| --- | --- | --- | ---: | --- |"])
    labels = {"cyclomatic_number": "Cycle rank", "main_route_length_m": "Main-route length (m)",
              "junction_to_passage_width_ratio": "Junction / passage width",
              "host_horizontal_scale": "Resolved host scale (input check)"}
    def describe_effect(effect):
        return ("Unavailable" if effect is None else
                f"{effect['median_delta']:.4g} [{effect['ci_lower']:.4g}, {effect['ci_upper']:.4g}]")
    for control, item in supplements["controllability"].items():
        effect = item["paired_high_minus_low"]
        lines.append(f"| {control.title()} | {labels[item['primary_metric']]} | "
                     f"{item['low_level']:g} to {item['high_level']:g} | "
                     f"{effect['valid_pairs'] if effect else 0} | {describe_effect(effect)} |")
    lines.extend(["", "The supply readout is a resolved input scale, not an independent geometry response.", "",
                  "| Sampling readout | Pairs | Adaptive minus uniform [95% interval] |",
                  "| --- | ---: | --- |"])
    for metric, label in (("error_mean_m", "Mean point discrepancy (m)"),
                          ("error_p95_m", "95th-percentile point discrepancy (m)"),
                          ("section_count", "Section count")):
        effect = supplements["sampling"].get(metric)
        lines.append(f"| {label} | {effect['valid_pairs'] if effect else 0} | {describe_effect(effect)} |")
    lines.extend(["", "Counts are approximately matched; point-cloud discrepancies do not establish continuous-surface accuracy or savings at matched error.", "",
                  f"Adaptive mean discrepancy is larger in "
                  f"{sum(r['adaptive_error_mean_m'] > r['uniform_error_mean_m'] for r in sampled)}"
                  f"/{len(sampled)} completed pairs. This comparison does not support an adaptive-sampling advantage under its point-set metric.", "",
                  "## Determinism and export checks", "",
                  "| Determinism check | Passed for every completed case |", "| --- | --- |"])
    for name, passed in summary["determinism"]["checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'Yes' if passed else 'No'} |")
    exported = summary["export_consistency"]
    lines.extend(["", f"Export experiment: {exported['complete_n']}/{exported['planned_n']} cases completed. "
                  f"Visual/collision package-presence checks: {'passed' if exported['passed'] else 'not all passed'}. "
                  f"Manual application imports: {exported['manual_imports_completed']}.", "",
                  "Completed export cases mean the experiment ran. Its pass flag checks visual/collision file presence; "
                  "it does not certify orientation, materials, final mesh identity after import, or simulator contact.", "",
                  "## Failure inventory", "", "| Experiment | Failure class | Cases |", "| --- | --- | ---: |"])
    for name, info in inventory.items():
        for reason, count in info["failures"].items():
            reason = reason.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {reason} | {count} |")
    lines.extend(["", "Tables IV-VI are generated under `paper/overleaf/results/`.", "",
                  "## Scope", "", (CAMPAIGN / "README.md").read_text().split("## Scope and counts")[1].split("## Corrections")[0],
                  "Full measurements, realized geometry ranges, supplementary paired effects and summary hashes are in `report.json`."])
    (CAMPAIGN / "REPORT.md").write_text("\n".join(lines) + "\n")
    with (CAMPAIGN / "case_inventory.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "run_id", "seed", "condition", "status", "failure_reason", "elapsed_s", "case_sha256"])
        for name, rows in records.items():
            for row in rows:
                writer.writerow([name, row["run_id"], row["seed"], row["condition_id"],
                                 row["status"], row.get("failure_reason", ""), row["elapsed_s"],
                                 hashlib.sha256((source / name / "cases" / f"{row['run_id']}.json").read_bytes()).hexdigest()])
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
