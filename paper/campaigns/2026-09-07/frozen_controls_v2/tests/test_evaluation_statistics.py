import numpy as np

from plume_advanced.evaluation.statistics import (
    bootstrap_percentile_ci,
    cluster_bootstrap_ci,
    normalized_wasserstein_distance,
    wasserstein_distance,
)


def test_distribution_distances_and_zero_iqr_are_well_defined() -> None:
    assert wasserstein_distance([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert normalized_wasserstein_distance([1.0, 1.0], [2.0, 2.0]) > 0.0


def test_bootstraps_are_deterministic_and_cluster_at_group_level() -> None:
    first = bootstrap_percentile_ci([1.0, 2.0, 3.0], iterations=100, seed=7)
    second = bootstrap_percentile_ci([1.0, 2.0, 3.0], iterations=100, seed=7)
    assert first == second

    clustered = cluster_bootstrap_ci(
        [0.0, 0.0, 10.0, 10.0],
        ["cave-a", "cave-a", "cave-b", "cave-b"],
        lambda values: float(np.mean(values)),
        iterations=200,
        seed=3,
    )
    assert clustered.lower <= clustered.estimate <= clustered.upper
    assert clustered.lower == 0.0
    assert clustered.upper == 10.0
