#!/usr/bin/env python3
"""Plot three slope-aware Gazebo Husky trials on the frozen cave route."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})


def main() -> None:
    root = Path("outputs/simulation_robot_demo_native")
    route = json.loads(Path(
        "outputs/simulation_robot_demo_clearance/robotics/midpoint_3m_route.json"
    ).read_text())
    floor = json.loads(Path(
        "outputs/simulation_robot_demo_clearance/robotics/midpoint_3m_floor_profile.json"
    ).read_text())["points"]
    goal = route["waypoints_xy_m"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.45), constrained_layout=True)
    colors = ("#186a9c", "#c16f20", "#59853c")
    for number, color in zip((2, 3, 4), colors, strict=True):
        case = root / f"husky_midpoint_trial{number}"
        report = json.loads((case / "result.json").read_text())
        trace = json.loads((case / "trajectory.json").read_text())
        assert report["passed"] and report["floor_profile_sha256"]
        elapsed = [row["sim_time_s"] for row in trace]
        xy = [row["pose"][:2] for row in trace]
        gaps = []
        for row in trace:
            x, y, z = row["pose"]
            nearest = min(floor, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            gaps.append(z - nearest[2])
        axes[0].plot([p[0] for p in xy], [p[1] for p in xy],
                     color=color, lw=1.8, label=f"Repeat {number - 1}")
        axes[1].plot(elapsed, [math.dist(p, goal) for p in xy],
                     color=color, lw=1.8)
        axes[2].plot(elapsed, gaps, color=color, lw=1.8)
    axes[0].scatter(*route["start_xyz_m"][:2], marker="s", color="black", s=30,
                    zorder=5)
    axes[0].scatter(*goal, marker="x", color="black", s=40, zorder=5)
    axes[0].add_patch(Circle(goal, route["goal_tolerance_m"], fill=False,
                             ls=":", color="black", lw=1.1))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("World $x$ (m)")
    axes[0].set_ylabel("World $y$ (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="lower left", fontsize=9,
                   framealpha=.94, edgecolor="#cccccc", title="Gazebo runs")
    axes[1].axhline(route["goal_tolerance_m"], color="black", lw=1, ls=":")
    axes[1].set_xlabel("Simulated time (s)")
    axes[1].set_ylabel("Distance to goal (m)")
    axes[2].axhline(0, color="black", lw=1, ls=":")
    axes[2].set_xlabel("Simulated time (s)")
    axes[2].set_ylabel("Robot origin above local floor (m)")
    for ax, tag in zip(axes, "ABC", strict=True):
        ax.text(.02, .97, tag, transform=ax.transAxes, va="top", weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75})
        ax.grid(alpha=.18)
    target = Path("paper/figures/simulation_husky_route.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220)
    print(target)


if __name__ == "__main__":
    main()
