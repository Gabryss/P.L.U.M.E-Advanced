#!/usr/bin/env python3
"""Build README figures from current A-C generation; never imply mesh qualification.

Run from the checkout root. The seed, recipes, resolved settings and generated
network/section identities are recorded beside the figures. No external data,
texture assets, old output directories or pickle checkpoints are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.identity import package_source_hash, sha256_file
from plume_advanced.stability import RoofStabilityModel
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_topology import section_footprint
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.world import BODY_PRESETS

COLORS = ["#167d8d", "#de8435", "#715b91"]


def save(fig, output, name):
    path = output / f"{name}.png"
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def schematic(output):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set(xlim=(-0.2, 14.5), ylim=(0, 5))
    ax.axis("off")
    boxes = [
        (0.1, 3.5, "Recipe + seed", "Resolve physics & policy", COLORS[0]),
        (3.7, 3.5, "Host field", "Terrain, cover, competence", COLORS[0]),
        (7.3, 3.5, "Network + sections", "Grow • merge • split • screen", COLORS[0]),
        (7.3, 0.9, "Surface + events", "Build • inspect • revalidate floor", COLORS[0]),
        (3.7, 0.9, "Export + materials", "Visual mesh & collider gates", COLORS[0]),
        (0.1, 0.9, "Delivery", "Receipts + checked assets", COLORS[0]),
        (10.9, 0.9, "Bounded repair", "Local repair / new network", COLORS[1]),
    ]
    for x, y, title, detail, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                3.0,
                1.05,
                boxstyle="round,pad=0.08",
                facecolor="#f1f6f7",
                edgecolor=color,
                lw=1.7,
            )
        )
        ax.text(x + 1.5, y + 0.68, title, ha="center", fontsize=12, weight="bold", color="#23323e")
        ax.text(x + 1.5, y + 0.29, detail, ha="center", fontsize=9.1, color="#455967")

    def arrow(start, end, color="#667783"):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.7))

    arrow((3.2, 4.02), (3.58, 4.02))
    arrow((6.8, 4.02), (7.18, 4.02))
    arrow((8.8, 3.38), (8.8, 2.06))
    arrow((7.18, 1.42), (6.8, 1.42))
    arrow((3.58, 1.42), (3.2, 1.42))
    arrow((10.42, 1.42), (10.78, 1.42), COLORS[1])
    ax.plot([12.4, 12.4, 10.8], [2.06, 4.02, 4.02], color=COLORS[1], lw=1.7)
    arrow((10.8, 4.02), (10.42, 4.02), COLORS[1])
    ax.text(
        11.2,
        2.65,
        "On failure,\nrepair and repeat\nrequired checks",
        ha="center",
        fontsize=10,
        color=COLORS[1],
    )
    ax.text(
        7.0,
        0.15,
        "Repair budgets are finite. Exhaustion retains evidence and blocks publication.",
        ha="center",
        fontsize=11,
        color="#455967",
    )
    return save(fig, output, "workflow")


def host_plot(host, network, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    fields = [
        ("Surface elevation (m)", host.elevation),
        ("Cover thickness (m)", host.cover_thickness),
        ("Routing cost", host.growth_cost),
    ]
    for ax, (label, data) in zip(axes, fields, strict=True):
        im = ax.pcolormesh(host.x_coords, host.y_coords, data, cmap="cividis", shading="auto")
        for segment in network.segments:
            ax.plot(
                [p.x for p in segment.points], [p.y for p in segment.points], color="white", lw=0.65
            )
        ax.set(xlabel="x (m)", ylabel="y (m)", title=label)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.65)
    return save(fig, output, "current_host")


def profiles_plot(sections, output):
    samples = [
        s
        for field in sections.segment_fields
        for s in field.samples
        if s.junction_blend_weight < 0.05
    ]
    samples.sort(key=lambda s: s.tube_height)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    selected = []
    for ax, percentile in zip(axes, (0.1, 0.5, 0.9), strict=True):
        sample = samples[round(percentile * (len(samples) - 1))]
        contour = np.array(sample.profile_points)
        contour = np.vstack((contour, contour[0]))
        ax.fill(*contour.T, color="#d9ecec")
        ax.plot(*contour.T, color=COLORS[0], lw=1.8)
        ax.set(
            title=f"{sample.tube_width:.2f} m × {sample.tube_height:.2f} m",
            xlabel="Local lateral distance (m)",
            ylabel="Local height (m)",
            xlim=(-6, 6),
            ylim=(-2, 3),
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        selected.append(
            dict(
                segment=sample.segment_id,
                index=sample.index,
                width_m=sample.tube_width,
                height_m=sample.tube_height,
                profile=sample.profile_points,
            )
        )
    return save(fig, output, "current_sections"), selected


def stability_plot(output):
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    width = np.linspace(0, 40, 250)
    for (name, profile), color in zip(BODY_PRESETS.items(), COLORS, strict=True):
        model = RoofStabilityModel(gravity_m_s2=profile.gravity_m_s2)
        ax.plot(
            width,
            model.load_coefficient * width**2,
            color=color,
            lw=2,
            label=f"{name.title()} · {profile.gravity_m_s2:g} m/s²",
        )
    ax.set(xlabel="Unsupported span (m)", ylabel="Required roof thickness (m)", xlim=(0, 40))
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    return save(fig, output, "gravity_screen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("docs/figures/readme"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17])
    args = parser.parse_args()
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = [schematic(out), stability_plot(out)]
    fig, axes = plt.subplots(
        len(args.seeds), 2, figsize=(13, 3.3 * len(args.seeds)), layout="constrained", squeeze=False
    )
    evidence = []
    footprint_bounds = []
    for col, mode in enumerate(("single", "multi")):
        for row, seed in enumerate(args.seeds):
            recipe = Path(f"config/short-{mode}.toml")
            config = load_project_config(recipe, seed_override=seed)
            host = HostFieldGenerator(config.host_field).generate()
            network = CaveNetworkGenerator(config.network).generate(
                host,
                section_config=config.section_field,
                quality_progress=lambda s: print(s, flush=True),
            )
            sections = SectionFieldGenerator(config.section_field).generate(network)
            mask, x, y, spacing = section_footprint(network, sections)
            footprint_bounds.append((float(x[0]), float(x[-1]), float(y[0]), float(y[-1])))
            ax = axes[row, col]
            ax.imshow(
                mask,
                origin="lower",
                extent=[
                    x[0] - spacing / 2,
                    x[-1] + spacing / 2,
                    y[0] - spacing / 2,
                    y[-1] + spacing / 2,
                ],
                cmap=ListedColormap(["white", COLORS[col]]),
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            ax.set(
                title=f"{'One system' if col == 0 else 'Three interacting systems'} · seed {seed}",
                xlabel="Distance along dominant route (m)",
                ylabel="Lateral distance (m)",
                xlim=(-15, 425),
                ylim=(-100, 100),
            )
            ax.set_aspect("equal")
            ax.grid(alpha=0.14)
            evidence.append(
                dict(
                    recipe=recipe.as_posix(),
                    seed=seed,
                    recipe_sha256=sha256_file(recipe),
                    config=project_config_manifest(config),
                    host_sha256=host_semantic_hash(host),
                    network_sha256=network_semantic_hash(network),
                    sections_sha256=section_semantic_hash(sections),
                    accepted_attempt=network.quality_report["selected_attempt"],
                )
            )
            if col == 1 and row == 0:
                figures.append(host_plot(host, network, out))
                image, profiles = profiles_plot(sections, out)
                figures.append(image)
                evidence[-1]["shown_profiles"] = profiles
    bounds = np.array(footprint_bounds)
    for ax in axes.flat:
        ax.set_xlim(bounds[:, 0].min() - 10, bounds[:, 1].max() + 10)
        ax.set_ylim(bounds[:, 2].min() - 10, bounds[:, 3].max() + 10)
    figures.append(save(fig, out, "current_topologies"))
    provenance = dict(
        schema="plume.readme-figures.v1",
        scope="Current host/network/section generation only; no mesh, texture, collision or native-engine qualification.",
        package_sha256=package_source_hash(),
        builder_sha256=sha256_file(__file__),
        cases=evidence,
        figures={p.name: sha256_file(p) for p in figures},
        schematics=["workflow.png", "gravity_screen.png"],
        gravity_parameters=dict(
            density_kg_m3=2900, effective_strength_pa=3000000, safety_factor=1.5
        ),
    )
    (out / "current_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print("README figures written; scope is A-C only.", flush=True)


if __name__ == "__main__":
    main()
