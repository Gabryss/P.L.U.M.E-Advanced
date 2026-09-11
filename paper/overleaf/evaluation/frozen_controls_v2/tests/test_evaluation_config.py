from pathlib import Path

from plume_advanced.evaluation.config import load_evaluation_config

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_pdc_partitions_are_disjoint_and_complete() -> None:
    config = load_evaluation_config(ROOT / "paper" / "experiments.toml")
    calibration = set(config.pdc_cave_partition("calibration"))
    evaluation = set(config.pdc_cave_partition("evaluation"))

    assert len(calibration) == 76
    assert len(evaluation) == 19
    assert calibration.isdisjoint(evaluation)
    assert len(calibration | evaluation) == 95
