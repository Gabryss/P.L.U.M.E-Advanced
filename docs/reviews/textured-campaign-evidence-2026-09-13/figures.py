"""Plot accepted centreline evidence and native render contact sheets without regenerating caves."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol_v2.json").read_text())
    pairs = [
        (g, i, c) for g in ("single_250", "multi") for i, c in enumerate(protocol["groups"][g])
    ]
    output = Path(__file__).resolve().parent
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), layout="constrained")
    for group, i, case in pairs:
        ax = axes[i, int(group == "multi")]
        run = root / group / f"case_{i:04d}" / "attempt_0000"
        p = run / "stage_b_network.json"
        ax.set_title(
            f"{'Three sources' if group == 'multi' else 'Single source'} · seed {case['seed']}",
            loc="left",
            fontsize=11,
        )
        if not p.exists():
            outcome = run / "result.json"
            status = json.loads(outcome.read_text()).get("status") if outcome.exists() else None
            label = "Generation rejected" if status == "failed" else "Generation pending"
            ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center")
            continue
        n = json.loads(p.read_text())
        for segment in n["segments"]:
            xy = np.array([[p["y"], p["x"]] for p in segment["centerline"]])
            metadata = segment["metadata"]
            systems = metadata.get(
                "system_ids",
                metadata.get("contributing_system_ids", [metadata.get("source_system_id", 0)]),
            )
            color = colors[int(systems[0]) % 3] if len(systems) == 1 else "#32383e"
            ax.plot(*xy.T, lw=1.6, color=color)
        for node in n["nodes"]:
            if node["kind"] == "entry":
                ax.scatter(node["y"], node["x"], marker="o", c="#c94812", s=24, zorder=4)
        ax.set(xlabel="Y (m)", ylabel="X (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.18)
    fig.savefig(output / "networks.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    comparison = []
    for row, (engine, folder) in enumerate(
        [("Unity", "unity_project"), ("Unreal", "unreal_run_01")]
    ):
        for col, (version, label) in enumerate(
            [("native", "initial light"), ("native_v2", "bounded light")]
        ):
            path = root / version / "single_250_seed1" / folder / "interior_1.png"
            pixels = plt.imread(path)[:, :, :3]
            clipped = float(np.all(pixels >= 254 / 255, axis=2).mean())
            ax = axes[row, col]
            ax.imshow(pixels)
            ax.axis("off")
            ax.set_title(f"{engine} · {label} · {clipped:.2%} clipped", loc="left", fontsize=11)
            comparison.append(
                dict(
                    engine=engine,
                    version=version,
                    view=1,
                    clipped_fraction=clipped,
                    passes_new_clipping_gate=clipped <= 0.01,
                    path=str(path),
                )
            )
    fig.savefig(output / "exposure_comparison.png", dpi=130)
    plt.close(fig)
    (output / "exposure_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    native = json.loads((root / "native_campaign.json").read_text())
    panels = root / "review_panels"
    panels.mkdir(exist_ok=True)
    for item in native["cases"]:
        if "output" not in item:
            continue
        folder = Path(item["output"])
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
        for row, (engine, subfolder) in enumerate(
            [("Unity", "unity_project"), ("Unreal", "unreal_run_01")]
        ):
            for col in range(2):
                ax = axes[row, col]
                path = folder / subfolder / f"interior_{col + 1}.png"
                ax.axis("off")
                ax.set_title(f"{engine} · view {col + 1}", loc="left")
                if path.exists():
                    ax.imshow(plt.imread(path))
        fig.savefig(panels / f"{item['group']}_seed{item['seed']}.png", dpi=120)
        plt.close(fig)
    for engine in ("unity", "unreal"):
        fig, axes = plt.subplots(6, 2, figsize=(12, 24), layout="constrained")
        for row, (group, index, case) in enumerate(pairs):
            match = next(
                (n for n in native["cases"] if n["group"] == group and n["index"] == index), None
            )
            folder = (
                None
                if not match or "output" not in match
                else Path(match["output"])
                / ("unity_project" if engine == "unity" else "unreal_run_01")
            )
            for view in (1, 2):
                ax = axes[row, view - 1]
                ax.axis("off")
                ax.set_title(
                    f"{'Multi' if group == 'multi' else 'Single'} · {case['seed']} · view {view}",
                    fontsize=11,
                    loc="left",
                )
                path = None if folder is None else folder / f"interior_{view}.png"
                if path and path.exists():
                    ax.imshow(plt.imread(path))
                else:
                    ax.text(0.5, 0.5, "No accepted capture", transform=ax.transAxes, ha="center")
        fig.savefig(output / f"{engine}_interiors.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
