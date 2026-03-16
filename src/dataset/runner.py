from __future__ import annotations

from pathlib import Path

from src.config import RunConfig
from src.pipeline import run_simulation
from src.utils.io import ensure_directory, write_json
from src.visualization.progress import progress_iter


def run_batch(config: RunConfig, output_root: str | Path, sample_count: int | None = None) -> list[dict[str, object]]:
    batch_size = sample_count or config.dataset.sample_count
    root = ensure_directory(output_root)
    summaries: list[dict[str, object]] = []

    for offset in progress_iter(range(batch_size), enabled=config.monitoring.progress_bar_enabled, desc="batch"):
        seed = config.seed + offset
        run_name = f"{config.name}-{seed}"
        summary = run_simulation(config, root / run_name, seed=seed, run_name=run_name)
        summaries.append(summary)

    write_json({"runs": summaries}, root / "batch_summary.json")
    return summaries

