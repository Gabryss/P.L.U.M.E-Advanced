"""Plot accepted campaign centreline graphs, without modifying run evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    repo = Path(__file__).resolve().parents[3]
    root = repo / "outputs/all_repairs_campaign_20260913/final"
    audit = json.loads((root / "audit.json").read_text())
    assert audit["passed"]
    cases = [row for row in audit["cases"] if row["case"]["scope"] == "full"]
    assert len(cases) == 6
    figure, axes = plt.subplots(3, 2, figsize=(14, 8.1), layout="constrained",
                               gridspec_kw={"height_ratios": [0.6, 0.75, 1.65]})
    for axis, case in zip(axes.flat, cases, strict=True):
        run = case["runs"][0]
        network = json.loads((root / run["directory"] / "stage_b_network.json").read_text())
        for segment in network["segments"]:
            points = segment["centerline"]
            axis.plot([point["y"] for point in points],
                      [point["x"] for point in points], color="#156a86", linewidth=2)
        entries = [node for node in network["nodes"] if node["kind"] == "entry"]
        axis.scatter([node["y"] for node in entries], [node["x"] for node in entries],
                     color="#cc6d20", s=36, zorder=3, label="Source")
        label = Path(case["case"]["config"]).stem.replace("_4k", "").replace("_", " ")
        axis.set_title(f"{label} · seed {case['case']['seed']}", loc="left", fontsize=12)
        axis.set(xlabel="Y (m)", ylabel="X (m)", aspect="equal")
        axis.margins(0.08, 0.25)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    destination = repo / "docs/reviews/assets/all_repairs_networks_20260913.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
