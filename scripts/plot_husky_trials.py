#!/usr/bin/env python3
"""Plot three frozen Husky Gazebo route repeats for the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("trials", type=Path, nargs="+")
    args = parser.parse_args()
    if len(args.trials) != 3:
        raise ValueError("Expected exactly three frozen route trials")
    route = json.loads(args.route.read_text())
    start = np.asarray(route["start_xyz_m"][:2])
    goal = np.asarray(route["waypoints_xy_m"][-1])
    forward = (goal - start) / np.linalg.norm(goal - start)
    lateral = np.array([-forward[1], forward[0]])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
    colors = ("#136a8a", "#b75a45", "#6a5796")
    for i, (folder, color) in enumerate(zip(args.trials, colors), 1):
        result = json.loads((folder / "result.json").read_text())
        if result["route"] != route:
            raise ValueError("Trials do not have the same frozen route")
        trace = json.loads((folder / "trajectory.json").read_text())
        xy = np.array([row["pose"][:2] for row in trace])
        along = (xy - start) @ forward
        cross = (xy - start) @ lateral
        axes[0].plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.7,
                     label=f"Trial {i}: {result['status']}")
        axes[1].plot(along, cross, color=color, linewidth=1.7, label=f"Trial {i}")
    axes[0].scatter(*start, marker="o", color="#1c1c1c", s=28, label="Start")
    axes[0].scatter(*goal, marker="*", color="#1c1c1c", s=65, label="Goal")
    tolerance = route.get("goal_tolerance_m", .5)
    axes[0].add_patch(plt.Circle(goal, tolerance, fill=False, color="#555555",
                                 linewidth=.9, linestyle=":"))
    axes[0].set(xlabel="X (m)", ylabel="Y (m)", title="Gazebo cave trajectory")
    axes[0].set_xlim(min(start[0], goal[0]) - .55, max(start[0], goal[0]) + .55)
    axes[0].set_ylim(min(start[1], goal[1]) - .25, max(start[1], goal[1]) + .3)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[1].axhline(0, color="#666666", linewidth=.9, linestyle="--")
    axes[1].set(xlabel="Progress toward goal (m)", ylabel="Lateral offset (m)",
                title="Deviation from direct route")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
