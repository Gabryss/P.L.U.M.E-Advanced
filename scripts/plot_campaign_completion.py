#!/usr/bin/env python3
"""Plot the frozen first campaign's completed and failed case counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())["experiments"]
    ordered = [
        ("Morphometry", "morphometry"),
        ("Host ablation", "host_ablation"),
        ("Control sweep", "controllability"),
        ("Sampling", "sampling_ablation"),
        ("Scale/storage", "scalability"),
        ("Export consistency", "export_consistency"),
        ("Determinism", "determinism"),
    ]
    labels = [name for name, _ in ordered]
    success = np.array([data[key].get("complete_n", data[key].get("complete_worlds"))
                        for _, key in ordered], dtype=float)
    planned = np.array([data[key].get("planned_n", data[key].get("planned_worlds"))
                        for _, key in ordered], dtype=float)
    if not np.all(np.isfinite(success)) or not np.all(np.isfinite(planned)) or np.any(planned <= 0):
        raise ValueError("Campaign counts are missing or invalid")
    fractions = 100 * success / planned
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 3.45), layout="constrained")
    ax.barh(y, fractions, color="#186c8c", height=.64, label="Completed")
    ax.barh(y, 100-fractions, left=fractions, color="#d89453", height=.64,
            label="Failed")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 114)
    ax.set_xlabel("Share of requested cases (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for position, done, total in zip(y, success.astype(int), planned.astype(int)):
        ax.text(102, position, f"{done}/{total}", va="center", fontsize=8.3)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), frameon=False, ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
