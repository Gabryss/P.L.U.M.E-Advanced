#!/usr/bin/env python3
"""Fit an Earth network-width recipe from calibration-only survey data.

Does not fit topology, layer spacing or host geology. A separate audit command
reads the reserved PDC caves only AFTER the fit has been written and frozen.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plume_advanced.evaluation.config import load_evaluation_config
from plume_advanced.evaluation.datasets.pdc import load_pdc
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry
from plume_advanced.evaluation.metrics.network_calibration import (
    common_footprint_support,
    fit_width_parameters,
    projected_width_series,
    spatial_scale,
    width_targets,
)
from plume_advanced.stages.network_width_field import log_width_field

ROOT = Path(__file__).resolve().parents[1]
PDC_URL = "https://doi.org/10.5281/zenodo.17750755"
VALENTINE_URL = "https://doi.org/10.5066/P14AC3J5"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surveyed_widths(root, partition):
    config = load_evaluation_config()
    selected = config.pdc_cave_partition(partition)
    # Filtering happens before file coordinates are opened by the loader.
    sections, rejected = load_pdc(root, cave_ids=selected)
    rows = []
    excluded_intersections = 0
    source_hash = hashlib.sha256()
    for s in sections:
        source_hash.update(s.relative_path.encode())
        source_hash.update((root/s.relative_path).read_bytes())
        if s.self_intersection_count:
            excluded_intersections += 1
            continue
        rows.append(dict(cave_id=s.reference_cave_id,
                         width_m=contour_morphometry(s.contour)["width_m"]))
    return rows, dict(partition=partition, cave_ids=sorted(selected),
                     selected_geometry_sha256=source_hash.hexdigest(),
                     split_sha256=digest(config.pdc_partition_path(partition)),
                     rejected=len(rejected), excluded_intersections=excluded_intersections,
                     source=PDC_URL)


def recipe(best, correlation):
    return f'''# Earth representative-width scenario. Network only, not a validated eruption model.
# Fitted from PDC calibration caves; spatial scale informed by Valentine LiDAR.
# Reproduce: python scripts/calibrate_network_widths.py --output outputs/width-fit
# Branch/source settings are inherited, NOT fitted to those surveys.
recipe_version = 1
preset = "short-multi"
procedural_seed = 17

[network]
base_passage_radius = {best["base_passage_radius"]:.8f}
minimum_passage_radius = 0.5
maximum_passage_radius = 5.0

[network.topology]
generation_mode = "regional_growth"

[network.regional]
branch_growth = "front"
width_log_sigma = {best["width_log_sigma"]:.8f}
width_correlation_m = {correlation:.8f}

[network.layers]
# No inter-layer survey calibration is available. Multi-source remains on.
enabled = false

[network.detail]
# Avoid fitting width variation and then adding a second uncalibrated amplitude.
enabled = false

[network.quality]
max_attempts = 6
repair_passes = 2
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdc-root", type=Path, default=ROOT/"data/reference/pdc_v2/extracted/Pyroduct Digital Catalog in .txt")
    parser.add_argument("--valentine-laz", type=Path, default=ROOT/"data/reference/valentine/Valentine_TUBE_UTM_10cm.copc.laz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-fit", type=Path, help="Frozen fit.json to audit against reserved PDC caves; never refits")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output directory to preserve prior calibration evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    methods = (
        "scripts/calibrate_network_widths.py",
        "src/plume_advanced/evaluation/metrics/network_calibration.py",
        "src/plume_advanced/evaluation/datasets/pdc.py",
        "src/plume_advanced/evaluation/metrics/morphometry.py",
        "src/plume_advanced/stages/network_width_field.py",
        "src/plume_advanced/stages/network_acceptance.py",
        "src/plume_advanced/procedural.py",
    )
    method_hashes = {p: digest(ROOT/p) for p in methods}
    if args.audit_fit:
        print("Auditing frozen parameters against reserved caves; no fitting", flush=True)
        fit = json.loads(args.audit_fit.read_text())
        rows, provenance = surveyed_widths(args.pdc_root, "evaluation")
        if set(provenance["cave_ids"]) & set(fit["pdc"]["cave_ids"]):
            raise ValueError("Calibration and audit cave partitions overlap")
        targets = width_targets(rows)
        prediction = np.array(fit["fit"]["best"]["predicted_quantiles_m"])
        relative = np.array(targets["relative_width_quantiles"])
        report = dict(fit_sha256=digest(args.audit_fit), parameters_unchanged=True,
                      survey=provenance, observed=targets,
                      calibration_prediction=fit["fit"]["best"]["predicted_quantiles_m"],
                      relative_central_log_error=float(np.mean(np.log((prediction/prediction[2])[1:4]/relative[1:4])**2)),
                      median_size_ratio_to_calibration=targets["median_cave_width_m"]/fit["targets"]["median_cave_width_m"],
                      warning="Historical whole-catalogue exploration predates the split; not a pristine unseen dataset")
        (args.output/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return 0
    print("Measuring calibration-only PDC cross-sections", flush=True)
    rows, provenance = surveyed_widths(args.pdc_root, "calibration")
    targets = width_targets(rows)
    print("Measuring Valentine projected chords at three raster resolutions", flush=True)
    import laspy

    cloud = laspy.read(args.valentine_laz)
    xy = np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y)))
    raw_sensitivity = []
    measurements = []
    series = None
    for resolution in (.2, .25, .35):
        current = projected_width_series(xy, resolution_m=resolution)
        raw_sensitivity.append(dict(resolution_m=resolution, **spatial_scale(current)))
        measurements.append(current)
        if resolution == .25:
            series = current
    common = common_footprint_support(measurements)
    sensitivity = [dict(resolution_m=raw["resolution_m"], **spatial_scale(s))
                   for raw, s in zip(raw_sensitivity, common)]
    if any(row["at_search_boundary"] for row in sensitivity):
        raise ValueError("Spatial-scale fit hit search boundary; inspect the footprint before fitting")
    scales = [row["correlation_m"] for row in sensitivity]
    if max(scales)/min(scales) > 1.25:
        raise ValueError("Spatial scale is unstable across resolutions; no recipe is written")
    correlation = float(np.median([row["correlation_m"] for row in sensitivity]))
    print(f"Fitting bounded widths; spatial correlation {correlation:.2f} m", flush=True)
    arc = np.arange(0., 400.01, 1.)
    seeds = list(range(16))
    fields = np.stack([log_width_field(np.column_stack((arc, np.zeros(len(arc)))), seed, correlation)
                       for seed in seeds])
    fit = fit_width_parameters(targets, fields, arc, minimum_m=1., maximum_m=9.5, gradient=.54)
    if fit["at_search_boundary"] or fit["best"]["loss"] > .005:
        raise ValueError("Width fit is unresolved or hits a search bound; no recipe is written")
    content = recipe(fit["best"], correlation)
    if method_hashes != {p: digest(ROOT/p) for p in methods}:
        raise RuntimeError("Calibration implementation changed during fitting; rerun from stable source")
    (args.output/"earth-survey-network.toml").write_text(content)
    report = dict(schema="plume.network-width-calibration.v1", scope="Earth representative passage-width statistics",
                  pdc=provenance, targets=targets, training_field_seeds=seeds,
                  valentine=dict(source=VALENTINE_URL, file=args.valentine_laz.name,
                                 sha256=digest(args.valentine_laz), sensitivity=sensitivity,
                                 raw_support_sensitivity=raw_sensitivity,
                                 support="intersection of observed single-interval reaches at 1 m stations",
                                 measurement="PCA-normal single-interval projected chords; not cross-sections",
                                 closing_radius_m=.5, filled_holes_under_m2=1., terminal_trim_fraction=.05),
                  fitted_correlation_m=correlation, fit=fit,
                  recipe_sha256=hashlib.sha256(content.encode()).hexdigest(),
                  not_calibrated=["branch frequency and angles", "source count", "layer geometry", "erosion or hydraulics",
                                  "extraterrestrial tubes", "height or roof stability", "population extremes"],
                  method_sha256=method_hashes, python=platform.python_version(), numpy=np.__version__)
    (args.output/"fit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    assert series is not None
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), layout="constrained")
    axes[0].scatter(*series["aligned"][::4].T, s=.2, color="#147c88")
    axes[0].set(aspect="equal", xlabel="Principal-axis distance (m)", ylabel="Lateral distance (m)",
                title="Valentine Cave · measured 10 cm LiDAR projection")
    axes[1].plot(series["stations"], series["widths"], color="#147c88")
    axes[1].set(xlabel="Principal-axis distance (m)", ylabel="Single-interval chord (m)",
                title="Gaps mark branches, missing coverage or trimmed ends; never joined across")
    p = np.array(targets["probabilities"])*100
    for row in sensitivity:
        v = row["variogram"]
        axes[2].plot([r["lag_m"] for r in v], [r["normalized_semivariance"] for r in v], "o-",
                     label=f'{row["resolution_m"]:g} m raster · common observed reaches')
    distances = np.linspace(0, 12, 100)
    axes[2].plot(distances, 1-np.exp(-.5*(distances/correlation)**2), color="black", ls="--", label="Fitted covariance model")
    axes[2].set(xlabel="Separation (m)", ylabel="Normalized semivariance", title="Spatial scale is reference-specific, not a population estimate")
    axes[2].legend()
    axes[3].plot(p, targets["representative_width_quantiles_m"], "o-", label="PDC representative-width target")
    axes[3].plot(p, fit["best"]["predicted_quantiles_m"], "o-", label="Fitted bounded field · calibration seeds")
    axes[3].axhline(9.5, color="grey", ls="--", label="Configured width cap")
    axes[3].set(xlabel="Percentile", ylabel="Width (m)", title="Fit uses the middle three quantiles; tails remain visible")
    axes[3].legend()
    fig.savefig(args.output/"calibration.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(best=fit["best"], correlation_m=correlation)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
