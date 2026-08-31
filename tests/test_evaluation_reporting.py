import csv
import json
from pathlib import Path

from plume_advanced.evaluation.latex import generate_latex
from plume_advanced.evaluation.plotting import generate_figures


def test_latex_and_figures_are_derived_from_saved_results(tmp_path: Path) -> None:
    aggregate = {
        "experiments": {
            "controllability": {
                "controls": {
                    "duration": {
                        "primary_metric": "main_route_length_m",
                        "n": 3,
                        "spearman_rho": 1.0,
                    }
                }
            },
            "export_consistency": {"complete_n": 2, "manual_imports_completed": 0},
            "scalability": {"complete_n": 2},
        }
    }
    (tmp_path / "aggregate_summary.json").write_text(json.dumps(aggregate), encoding="utf-8")
    latex_paths = generate_latex(tmp_path)
    assert {path.name for path in latex_paths} == {
        "paper_metrics.tex",
        "table_controllability.tex",
    }
    assert "PaperDurationSpearman}{1.000}" in latex_paths[0].read_text(encoding="utf-8")

    result_dir = tmp_path / "scalability"
    result_dir.mkdir()
    with (result_dir / "raw_results.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "status",
                "storage_mode_requested",
                "route_length_requested_m",
                "wall_time_s",
                "peak_rss_gib",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "status": "complete",
                    "storage_mode_requested": "dense",
                    "route_length_requested_m": 500,
                    "wall_time_s": 1.0,
                    "peak_rss_gib": 0.2,
                },
                {
                    "status": "complete",
                    "storage_mode_requested": "tiled",
                    "route_length_requested_m": 1000,
                    "wall_time_s": 2.0,
                    "peak_rss_gib": 0.3,
                },
            )
        )
    figure_paths = generate_figures(tmp_path)
    assert {path.name for path in figure_paths} == {
        "figure_scalability.png",
        "figure_scalability.svg",
    }
    assert all(path.stat().st_size > 0 for path in figure_paths)
