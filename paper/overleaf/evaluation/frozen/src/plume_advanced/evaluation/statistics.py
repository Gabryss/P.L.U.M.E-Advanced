"""Predeclared statistical summaries for paper experiments."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_iterations: int


def median_iqr(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = _finite(values)
    if array.size == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"), "iqr": float("nan")}
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0])
    return {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def wasserstein_distance(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float:
    return float(stats.wasserstein_distance(_finite(first), _finite(second)))


def normalized_wasserstein_distance(
    reference: Sequence[float] | np.ndarray,
    generated: Sequence[float] | np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> float:
    scale = max(median_iqr(reference)["iqr"], epsilon)
    return wasserstein_distance(reference, generated) / scale


def ks_statistic(first: Sequence[float], second: Sequence[float]) -> float:
    return float(stats.ks_2samp(_finite(first), _finite(second)).statistic)


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    result = stats.spearmanr(_finite_pair(first, second)[0], _finite_pair(first, second)[1])
    return float(result.statistic)


def standardized_paired_effect(
    baseline: Sequence[float],
    treatment: Sequence[float],
) -> float:
    first, second = _finite_pair(baseline, treatment)
    differences = second - first
    deviation = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    if deviation <= 1e-12:
        return (
            0.0 if np.allclose(differences, 0.0) else float(np.sign(np.mean(differences)) * np.inf)
        )
    return float(np.mean(differences) / deviation)


def bootstrap_percentile_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = lambda sample: float(np.median(sample)),
    *,
    iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    array = _finite(values)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one finite value")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        estimates[index] = statistic(rng.choice(array, size=array.size, replace=True))
    return _interval(statistic(array), estimates, iterations, confidence_level)


def paired_bootstrap_difference(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = lambda sample: float(np.median(sample)),
    iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    first, second = _finite_pair(baseline, treatment)
    differences = second - first
    return bootstrap_percentile_ci(
        differences,
        statistic,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
    )


def cluster_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence[Hashable],
    statistic: Callable[[np.ndarray], float],
    *,
    iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    array = np.asarray(values, dtype=float)
    clusters = np.asarray(cluster_ids, dtype=object)
    if array.shape[0] != clusters.shape[0]:
        raise ValueError("values and cluster_ids must have equal length")
    finite = np.isfinite(array)
    array = array[finite]
    clusters = clusters[finite]
    unique = np.asarray(sorted(set(clusters.tolist()), key=str), dtype=object)
    if unique.size == 0:
        raise ValueError("cluster bootstrap requires at least one cluster")
    rows = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = rng.choice(unique, size=unique.size, replace=True)
        sample = np.concatenate([array[rows[cluster]] for cluster in selected])
        estimates[index] = statistic(sample)
    return _interval(statistic(array), estimates, iterations, confidence_level)


def two_population_cluster_bootstrap(
    reference_values: Sequence[float],
    reference_clusters: Sequence[Hashable],
    generated_values: Sequence[float],
    generated_clusters: Sequence[Hashable],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    reference, reference_ids = _finite_with_clusters(reference_values, reference_clusters)
    generated, generated_ids = _finite_with_clusters(generated_values, generated_clusters)
    ref_unique = np.asarray(sorted(set(reference_ids.tolist()), key=str), dtype=object)
    gen_unique = np.asarray(sorted(set(generated_ids.tolist()), key=str), dtype=object)
    if ref_unique.size == 0 or gen_unique.size == 0:
        raise ValueError("both populations require at least one cluster")
    ref_rows = {cluster: np.flatnonzero(reference_ids == cluster) for cluster in ref_unique}
    gen_rows = {cluster: np.flatnonzero(generated_ids == cluster) for cluster in gen_unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        ref_selected = rng.choice(ref_unique, size=ref_unique.size, replace=True)
        gen_selected = rng.choice(gen_unique, size=gen_unique.size, replace=True)
        ref_sample = np.concatenate([reference[ref_rows[cluster]] for cluster in ref_selected])
        gen_sample = np.concatenate([generated[gen_rows[cluster]] for cluster in gen_selected])
        estimates[index] = statistic(ref_sample, gen_sample)
    return _interval(statistic(reference, generated), estimates, iterations, confidence_level)


def completion_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    statuses = [str(row.get("status", "invalid")) for row in rows]
    return {
        "planned_n": len(statuses),
        "complete_n": statuses.count("complete"),
        "failed_n": statuses.count("failed") + statuses.count("timeout"),
        "excluded_invalid_n": statuses.count("invalid"),
    }


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _finite_pair(first: Sequence[float], second: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired samples must have equal shape")
    finite = np.isfinite(left) & np.isfinite(right)
    return left[finite], right[finite]


def _finite_with_clusters(
    values: Sequence[float], cluster_ids: Sequence[Hashable]
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    clusters = np.asarray(cluster_ids, dtype=object)
    if array.shape[0] != clusters.shape[0]:
        raise ValueError("values and cluster_ids must have equal length")
    finite = np.isfinite(array)
    return array[finite], clusters[finite]


def _interval(
    estimate: float,
    estimates: np.ndarray,
    iterations: int,
    confidence_level: float,
) -> ConfidenceInterval:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    alpha = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return ConfidenceInterval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        bootstrap_iterations=iterations,
    )


__all__ = [
    "ConfidenceInterval",
    "bootstrap_percentile_ci",
    "cluster_bootstrap_ci",
    "completion_counts",
    "ks_statistic",
    "median_iqr",
    "normalized_wasserstein_distance",
    "paired_bootstrap_difference",
    "spearman_correlation",
    "standardized_paired_effect",
    "two_population_cluster_bootstrap",
    "wasserstein_distance",
]
