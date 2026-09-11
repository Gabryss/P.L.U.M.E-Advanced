"""Exact cluster-resampling distances with cached sorting and joint intervals."""

from __future__ import annotations

import numpy as np

from plume_advanced.evaluation.statistics import _interval


class PreparedDistance:
    """Evaluate NW1 for integer row multiplicities without expanding/sorting them."""

    def __init__(self, reference, generated):
        self.reference = np.asarray(reference, dtype=float)
        self.generated = np.asarray(generated, dtype=float)
        if not np.isfinite(self.reference).all() or not np.isfinite(self.generated).all():
            raise ValueError("Morphometric distances require finite descriptor values")
        self.ref_order = np.argsort(self.reference)
        self.gen_order = np.argsort(self.generated)
        self.ref_sorted = self.reference[self.ref_order]
        self.gen_sorted = self.generated[self.gen_order]
        support = np.sort(np.concatenate((self.ref_sorted, self.gen_sorted)))
        self.deltas = np.diff(support)
        self.ref_indices = self.ref_sorted.searchsorted(support[:-1], side="right")
        self.gen_indices = self.gen_sorted.searchsorted(support[:-1], side="right")

    def evaluate(self, reference_counts, generated_counts):
        ref_cumulative = np.r_[0, np.cumsum(np.asarray(reference_counts)[self.ref_order])]
        gen_cumulative = np.r_[0, np.cumsum(np.asarray(generated_counts)[self.gen_order])]
        if ref_cumulative[-1] == 0 or gen_cumulative[-1] == 0:
            raise ValueError("Both resampled populations must be nonempty")
        ref_cdf = ref_cumulative[self.ref_indices] / ref_cumulative[-1]
        gen_cdf = gen_cumulative[self.gen_indices] / gen_cumulative[-1]
        distance = np.dot(np.abs(ref_cdf-gen_cdf), self.deltas)
        # np.percentile's default linear interpolation on the expanded sample.
        ranks = (ref_cumulative[-1]-1)*np.array([.25, .75])
        lower = np.floor(ranks).astype(int)
        upper = np.ceil(ranks).astype(int)
        lo_values = self.ref_sorted[np.searchsorted(ref_cumulative[1:], lower+1)]
        hi_values = self.ref_sorted[np.searchsorted(ref_cumulative[1:], upper+1)]
        quantiles = lo_values + (ranks-lower)*(hi_values-lo_values)
        return float(distance / max(quantiles[1]-quantiles[0], 1e-9))


def morphology_intervals(reference, generated, metrics, aggregate_metrics, *, iterations, confidence_level, seed):
    ref_ids, ref_inverse = np.unique([r["reference_cave_id"] for r in reference], return_inverse=True)
    gen_ids, gen_inverse = np.unique([r["generated_world_id"] for r in generated], return_inverse=True)
    prepared = [PreparedDistance([r[m] for r in reference], [r[m] for r in generated]) for m in metrics]
    estimates = np.empty((iterations, len(metrics)))
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        ref_counts = np.bincount(rng.integers(len(ref_ids), size=len(ref_ids)), minlength=len(ref_ids))[ref_inverse]
        gen_counts = np.bincount(rng.integers(len(gen_ids), size=len(gen_ids)), minlength=len(gen_ids))[gen_inverse]
        for index, distance in enumerate(prepared):
            estimates[iteration, index] = distance.evaluate(ref_counts, gen_counts)
    point = np.array([d.evaluate(np.ones(len(reference), dtype=int), np.ones(len(generated), dtype=int)) for d in prepared])
    result = {metric: _interval(point[i], estimates[:, i], iterations, confidence_level) for i, metric in enumerate(metrics)}
    indices = [i for i, metric in enumerate(metrics) if metric in aggregate_metrics]
    if not indices:
        raise ValueError("Aggregate descriptors must be present in the measured descriptor set")
    result["aggregate"] = _interval(float(point[indices].mean()), estimates[:, indices].mean(axis=1), iterations, confidence_level)
    return result
