from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.pipeline import run_simulation


def test_small_pipeline_run_produces_outputs(tmp_path: Path) -> None:
    config = load_config("config/default.yaml")
    config.domain.grid_shape = (12, 16, 16)
    config.dissolution.iterations = 6
    config.monitoring.slice_interval = 3
    config.monitoring.histogram_interval = 3
    config.monitoring.log_interval = 2
    config.monitoring.progress_bar_enabled = False

    summary = run_simulation(config, tmp_path / "sample", seed=11, run_name="test-run")

    assert summary["mesh_vertex_count"] >= 0
    assert (tmp_path / "sample" / "metrics.csv").exists()
    assert (tmp_path / "sample" / "summary.json").exists()
    assert (tmp_path / "sample" / "mesh" / "cave.obj").exists()

