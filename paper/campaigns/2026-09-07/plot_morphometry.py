"""Render the measured, section-weighted morphology distributions."""
from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plume-evaluation-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "paper/outputs/morphometry"
TARGET = ROOT / "paper/overleaf/figures/f09_morphometry.png"


def main():
    summary = json.loads((SOURCE / "summary.json").read_text())
    assert summary["planned_worlds"] == summary["complete_worlds"] == 100
    sources = [SOURCE / f"raw_{name}_sections.csv" for name in ("reference", "generated", "baseline")]
    rows = []
    for source in sources:
        with source.open() as f:
            rows.append(list(csv.DictReader(f)))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.6), layout="constrained")
    metrics = [("width_m", "Width (m)"), ("height_m", "Height (m)"),
               ("aspect_ratio", "Width / height"), ("compactness", "Compactness"),
               ("floor_residual_norm", "Normalized floor residual"),
               ("roof_asymmetry_norm", "Normalized roof asymmetry")]
    styles = [("PDC evaluation", "#303840", "-"), ("Advanced", "#b7472b", "-"),
              ("Matched ellipse", "#568b9d", "--")]
    for ax, (metric, label) in zip(axes.flat, metrics):
        # Draw the matched baseline first, so shared dimensional distributions
        # remain visible as the Advanced line. Every sampled ECDF point uses its
        # exact cumulative mass; at most 4,000 points are displayed per curve.
        for index in (2, 0, 1):
            name, color, linestyle = styles[index]
            values = np.sort([float(r[metric]) for r in rows[index]])
            keep = np.unique(np.linspace(0, len(values)-1, min(4000,len(values))).astype(int))
            ax.step(values[keep], (keep+1)/len(values), where="post", label=name,
                    color=color, linestyle=linestyle, lw=1.25)
        ax.set(xlabel=label, ylabel="Cumulative fraction", ylim=(0, 1.03))
        ax.grid(alpha=.16)
    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[0,0].legend(handles, labels, loc="lower right", fontsize=6.5, frameon=False)
    fig.savefig(TARGET, dpi=330)
    plt.close(fig)
    manifest = {"figure": str(TARGET.relative_to(ROOT)),
                "sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                "source_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "scope": "Section-weighted marginal ECDFs; at most 4,000 exact ECDF support points displayed per curve. All data used for Table IV.",
                "reference_caves": summary["reference_caves"],
                "reference_sections": summary["reference_sections"],
                "generated_worlds": summary["complete_worlds"],
                "generated_sections": summary["generated_sections"]}
    (Path(__file__).resolve().parent / "morphometry_figure.json").write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == "__main__":
    main()
