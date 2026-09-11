#!/usr/bin/env python3
"""Build README figures from saved run artifacts; never generate a new cave mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from plume_advanced.config import load_project_config
from plume_advanced.stability import RoofStabilityModel
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.world import BODY_PRESETS

COLORS = ["#247b8b", "#d67b32", "#7560a5", "#467c59"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#94a3b0", "text.color": "#20313c", "axes.labelcolor": "#20313c",
    "xtick.color": "#415562", "ytick.color": "#415562", "savefig.facecolor": "white",
})


def save(fig, out, name, title, subtitle):
    # Captions in the README and paper already introduce these stage figures.
    # Reclaim the headline area while retaining informative panel/axis labels.
    if name in {"host_fields", "network", "sections"}:
        fig.tight_layout(rect=(.025, .025, .985, .995), h_pad=2.2, w_pad=2.7)
    else:
        fig.suptitle(title, x=.05, ha="left", y=.985, fontsize=19, weight="bold")
        fig.text(.05, .925, subtitle, ha="left", fontsize=10, color="#526671")
        fig.tight_layout(rect=(.025, .025, .985, .89), h_pad=2.2, w_pad=2.7)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def pipeline(out):
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis("off")
    steps = [
        (0, 2, "A · Host field", "Terrain, cover, competence\nand routing penalties", COLORS[0]),
        (1, 2, "B · Formation network", "Lobe growth, split / rejoin,\nflux and emplacement history", COLORS[0]),
        (2, 2, "C · Section field", "Sampled profiles, floor / roof,\nlocal frames and stability", COLORS[1]),
        (2, 1, "D1 · Base cave volume", "Profile sweeps → relief\n→ mandatory roof screen", COLORS[2]),
        (1, 1, "C2 · Base floor atlas", "Raycast placement cells\nagainst the base volume", COLORS[1]),
        (0, 1, "E · Optional geology", "Collapse / choke / infill;\ngrounded rocks if enabled", COLORS[3]),
        (0, 0, "D2 · Final cave mesh", "Apply structural events;\nmarching cubes and welding", COLORS[2]),
        (1, 0, "C3 · Final floor atlas", "Relift floor cells; invalidate\nblocked placements", COLORS[1]),
        (2, 0, "F · Surface and export", "Smoothing, UVs, optional PBR;\napplication packages", COLORS[3]),
    ]
    centers = []
    for col, row, title, detail, color in steps:
        x, y = .1 + col*4.15, .2 + row*2.35
        ax.add_patch(FancyBboxPatch((x, y), 3.4, 1.7, boxstyle="round,pad=.10",
                                   facecolor=color, edgecolor="none", alpha=.10))
        ax.text(x+.16, y+1.16, title, color=color, weight="bold", fontsize=12)
        ax.text(x+.16, y+.50, detail, fontsize=10.5, linespacing=1.6)
        centers.append(np.array([x+1.7, y+.85]))
    for start, end in zip(centers[:-1], centers[1:], strict=True):
        direction = end-start
        unit = direction/np.linalg.norm(direction)
        offset = 1.88 if abs(unit[0]) > .5 else 1.01
        ax.annotate("", end-unit*offset, start+unit*offset,
                    arrowprops={"arrowstyle": "->", "color": "#60717d", "lw": 1.7})
    return save(fig, out, "pipeline", "From physical context to an inspectable environment",
                "Execution order of plume-generate. The tube-only helper omits floor atlases, optional events and PBR packaging.")


def host_plot(host, out, label):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    x0, x1, y0, y1 = host.extent
    panels = [(host.elevation, "Terrain elevation", "m", "terrain"),
              (host.cover_thickness, "Available cover", "m", "cividis"),
              (host.roof_competence, "Roof competence proxy", "index", "viridis"),
              (host.routing_cost, "Combined routing cost", "index", "magma_r")]
    for ax, (values, title, units, cmap) in zip(axes.flat, panels, strict=True):
        im = ax.imshow(values.T, origin="lower", extent=(y0, y1, x0, x1),
                       aspect="auto", cmap=cmap)
        ax.set(title=title, xlabel="Y / nominal downflow (m)", ylabel="X (m)")
        fig.colorbar(im, ax=ax, label=units, shrink=.83)
    return save(fig, out, "host_fields", "A · A shared substrate for cave growth",
                f"{label}. Deterministically reconstructed from the saved configuration; panels show the entire host domain.")


def network_plot(network, arrays, out, label):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [1.1, 1]})
    phases = sorted({int(s["metadata"].get("birth_phase", 0)) for s in network["segments"]})
    for phase in phases:
        first = True
        for seg in network["segments"]:
            if int(seg["metadata"].get("birth_phase", 0)) != phase:
                continue
            p = np.array([[p["x"], p["y"]] for p in seg["centerline"]])
            axes[0].plot(p[:, 1], p[:, 0], color=COLORS[phase % len(COLORS)], lw=1.5,
                         label=f"Birth phase {phase}" if first else None)
            first = False
    axes[0].set(xlabel="Y / nominal downflow (m)", ylabel="X (m)", title="Formation graph in plan view")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="upper left", ncol=len(phases), fontsize=9)
    seg = max(network["segments"], key=lambda s: s["physical_length_m"])
    mask = arrays["segment_id"] == seg["segment_id"]
    s = arrays["arc_length_m"][mask]
    floor, roof = arrays["floor_world_z"][mask], arrays["roof_world_z"][mask]
    axes[1].fill_between(s, floor, roof, color=COLORS[0], alpha=.20)
    axes[1].plot(s, roof, color=COLORS[2], label="Section roof")
    axes[1].plot(s, floor, color=COLORS[1], label="Section floor")
    axes[1].scatter(s, floor, s=9, color=COLORS[1])
    axes[1].set(xlabel="Distance along segment (m)", ylabel="World Z (m)",
                title=f"Longest segment ({seg['segment_id']}) · sampled floor and roof, before 3D relief")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=.2)
    return save(fig, out, "network", "B → C · Connected routes with a continuous vertical profile",
                f"{label}. Plan view has equal metric axes; the lower panel exaggerates vertical changes for readability.")


def section_plot(arrays, out, label):
    offsets = arrays["profile_offsets"]
    profiles = [arrays["profile_points"][a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    widths = np.array([np.ptp(q[:, 0]) for q in profiles])
    heights = np.array([np.ptp(q[:, 1]) for q in profiles])
    ordinary = np.flatnonzero(arrays["junction_influence"] < .10)
    order = ordinary[np.argsort(heights[ordinary])]
    chosen = list(order[(np.linspace(.1, .9, 4)*(len(order)-1)).astype(int)])
    wide = list(np.argsort(widths)[::-1])
    for index in wide:
        if index not in chosen and all(np.linalg.norm(arrays["center_xyz_m"][index]-arrays["center_xyz_m"][j]) > 20
                                       for j in chosen[4:]):
            chosen.append(index)
            if len(chosen) == 6:
                break
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, index in zip(axes.flat, chosen, strict=True):
        q = profiles[index].copy()
        q[:, 1] -= q[:, 1].min()
        q = np.vstack((q, q[0]))
        ax.fill(q[:, 0], q[:, 1], color=COLORS[0], alpha=.12)
        ax.plot(q[:, 0], q[:, 1], color=COLORS[0], lw=1.7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="Local lateral coordinate (m)", ylabel="Height above profile floor (m)",
               title=f"Sample {index} · {widths[index]:.2f} m × {heights[index]:.2f} m")
        ax.grid(alpha=.2)
    return save(fig, out, "sections", "C · Cross-section envelopes across the network",
                f"{label}. Four low-junction sections and two wide locations. Each panel has equal metric axes; limits vary.")


def stability_plot(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    ax = axes[0]
    ax.set(xlim=(-6, 7), ylim=(-1, 8))
    ax.add_patch(Rectangle((-5.5, 0), 10.5, 6, facecolor="#d9d2c4"))
    x = np.linspace(-4, 4, 200)
    z = 2.2 + .8*np.sqrt(np.maximum(0, 1-(x/4)**2))
    ax.fill_between(x, 0, z, color="white")
    ax.plot(x, z, color=COLORS[0], lw=2)
    ax.plot([-5.5, 5], [6, 6], color="#4b5b48", lw=2)
    ax.text(-5.4, 6.35, "Local ground surface")
    ax.annotate("", (0, 6), (0, 3), arrowprops={"arrowstyle": "<->", "color": COLORS[2]})
    ax.text(.25, 4.55, "Roof thickness t = d − h", fontsize=10)
    ax.annotate("", (-4, .65), (4, .65), arrowprops={"arrowstyle": "<->", "color": COLORS[0]})
    ax.text(-1.1, .95, "Span w")
    ax.annotate("", (5.8, 0), (5.8, 6), arrowprops={"arrowstyle": "<->", "color": "#20313c"})
    ax.text(6.05, 2.5, "d", fontsize=13)
    ax.text(-1.2, 1.7, "Cavity height h", fontsize=10)
    ax.text(-5.4, -.65, "Schematic · not a generated section", fontsize=10, color="#526671")
    ax.axis("off")
    width = np.linspace(0, 40, 150)
    for (body, preset), color in zip(BODY_PRESETS.items(), COLORS, strict=False):
        model = RoofStabilityModel(gravity_m_s2=preset.gravity_m_s2)
        axes[1].plot(width, model.load_coefficient*width**2, lw=2.5, color=color,
                     label=f"{body.title()} · g = {preset.gravity_m_s2:.3g} m/s²")
    axes[1].set(xlabel="Unsupported roof span w (m)", ylabel="Required roof thickness (m)", xlim=(0, 40), ylim=(0, 18))
    axes[1].legend(loc="upper left", fontsize=10)
    axes[1].grid(alpha=.2)
    return save(fig, out, "roof_stability", "Width, height and cover share one gravity-dependent constraint",
                "Isolated gravity comparison: density 2,900 kg/m³; effective strength 3 MPa; safety factor 1.5. Not body size predictions.")


def evidence_plot(validation, out):
    source = validation / "surface_scales/summary.json"
    if not source.is_file():
        return None
    data = json.loads(source.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, radius in zip(axes, [.4, .8], strict=True):
        for i, (label, records) in enumerate(data["results"].items()):
            record = next(r for r in records if np.isclose(r["radius_m"], radius))
            residual = np.array(record["rms_plane_residual_m"])*1000
            ax.bar(i, np.median(residual), color=COLORS[i], width=.65)
            ax.text(i, np.median(residual)+1, f"{np.median(residual):.1f}", ha="center", fontsize=11)
        ax.set_xticks(range(len(data["results"])), ["Valentine\n10 cm scan", "Generated\nbefore relief", "Generated\nafter relief"])
        ax.set(title=f"Neighborhood radius: {radius:.1f} m", ylabel="Median local plane-fit RMS residual (mm)", ylim=(0, 55))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    return save(fig, out, "reference_comparison", "Measured relief moves closer to the Valentine case study",
                "Saved 7 September 2026 comparison: one isolated generated reach vs a scan. Surface coverage and sampling differ.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Input TOML; defaults to generation_config.toml in the run directory")
    parser.add_argument("--output-directory", type=Path, default=Path("docs/figures/readme"))
    parser.add_argument("--validation-directory", type=Path, default=Path("outputs/geometry_validation"))
    parser.add_argument("--relief-study", type=Path, default=Path("outputs/surface_relief_study"))
    args = parser.parse_args()
    root, out = args.run_directory, args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    config_path = args.config or root / "generation_config.toml"
    config = load_project_config(config_path)
    label = f"{config.world.body.name.title()} · seed {config.procedural_seed} · current inspection scenario"
    network = json.loads((root / "network.json").read_text())
    with np.load(root / "sections.npz") as arrays:
        figures = [pipeline(out), host_plot(HostFieldGenerator(config.host_field).generate(), out, label),
                   network_plot(network, arrays, out, label), section_plot(arrays, out, label), stability_plot(out)]
    evidence = evidence_plot(args.validation_directory, out)
    if evidence:
        figures.append(evidence)
    inputs = [config_path, root / "network.json", root / "sections.npz"]
    for name in ("lava_tube_geometry.glb", "export_checks.json", "inspection_cameras.json"):
        if (root / name).is_file():
            inputs.append(root / name)
    if evidence:
        inputs.append(args.validation_directory / "surface_scales/summary.json")
    for source, name in [(root / "inspection_views.png", "inspection_views.png"),
                         (args.relief_study / "comparison.png", "surface_relief.png")]:
        if source.is_file():
            shutil.copyfile(source, out / name)
            figures.append(out / name)
            inputs.append(source)
    provenance = {
        "body": config.world.body.name, "seed": config.procedural_seed,
        "voxel_size_m": config.geometry.voxel_size,
        "scope": "Current saved A-C artifacts and exported inspection images; historical controlled reach explicitly labeled.",
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
        "figures": [p.name for p in figures],
        "illustrations": ["pipeline.png", "roof_stability.png"],
        "figure_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Wrote {len(figures)} figures to {out}")


if __name__ == "__main__":
    main()
