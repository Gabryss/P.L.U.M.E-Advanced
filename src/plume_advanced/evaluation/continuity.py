"""Arc-length-aware longitudinal continuity and oscillation diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ._common import finite, quantile, value


def _records(samples: Any) -> list[Any]:
    sections = value(samples, "sections", "cross_sections", "profiles", default=None)
    if sections is not None:
        return list(sections)
    if isinstance(samples, Mapping) and "records" in samples:
        return list(samples["records"])
    if isinstance(samples, Mapping):
        columns = {
            key: item
            for key, item in samples.items()
            if isinstance(item, (list, tuple, np.ndarray))
        }
        if columns:
            count = max((len(item) for item in columns.values()), default=0)
            return [
                {key: item[index] for key, item in columns.items() if index < len(item)}
                for index in range(count)
            ]
    try:
        return list(samples)
    except TypeError:
        return []


def _series(samples: Any, field: str, arc_key: str) -> tuple[np.ndarray, np.ndarray]:
    records = _records(samples)
    pairs: list[tuple[float, float]] = []
    for index, record in enumerate(records):
        coordinate = finite(value(record, arc_key, "arc_length", "distance", "s", default=index))
        measurement = finite(value(record, field, default=None))
        if coordinate is not None and measurement is not None:
            pairs.append((coordinate, measurement))
    # Duplicate stations are averaged, making output independent of duplicate
    # row ordering and avoiding undefined interpolation behavior.
    grouped: dict[float, list[float]] = {}
    for coordinate, measurement in pairs:
        grouped.setdefault(coordinate, []).append(measurement)
    ordered = sorted((coordinate, sum(values) / len(values)) for coordinate, values in grouped.items())
    if not ordered:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.array([item[0] for item in ordered]), np.array([item[1] for item in ordered])


def _one_continuity(arcs: np.ndarray, values: np.ndarray, *, short_period_fraction: float, sample_count: int) -> dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "arc_length": 0.0,
            "value_range": 0.0,
            "low_frequency_energy_ratio": None,
            "short_period_energy_ratio": None,
            "repetitive_oscillation_score": None,
            "low_frequency_evolution_score": None,
            "roughness": None,
            "dominant_period": None,
        }
    if values.size == 1 or arcs[-1] <= arcs[0]:
        return {
            "count": int(values.size),
            "arc_length": 0.0,
            "value_range": 0.0,
            "low_frequency_energy_ratio": 1.0,
            "short_period_energy_ratio": 0.0,
            "repetitive_oscillation_score": 0.0,
            "low_frequency_evolution_score": 1.0,
            "roughness": 0.0,
            "dominant_period": None,
        }
    length = float(arcs[-1] - arcs[0])
    count = max(8, min(int(sample_count), 512))
    uniform_arc = np.linspace(arcs[0], arcs[-1], count)
    uniform_values = np.interp(uniform_arc, arcs, values)
    centered = uniform_values - np.mean(uniform_values)
    variance = float(np.mean(centered**2))
    value_range = float(np.max(values) - np.min(values))
    if variance <= 1.0e-24:
        return {
            "count": int(values.size),
            "arc_length": length,
            "value_range": value_range,
            "low_frequency_energy_ratio": 1.0,
            "short_period_energy_ratio": 0.0,
            "repetitive_oscillation_score": 0.0,
            "low_frequency_evolution_score": 1.0,
            "roughness": 0.0,
            "dominant_period": None,
        }

    # Detrending removes a healthy broad-scale slope before measuring repeated
    # short-period structure. FFT bins are tied to physical arc length.
    trend = np.polyval(np.polyfit(uniform_arc, uniform_values, 1), uniform_arc)
    residual = uniform_values - trend
    spectrum = np.abs(np.fft.rfft(residual)) ** 2
    spectrum[0] = 0.0
    total_energy = float(np.sum(spectrum))
    first_short_bin = max(1, int(math.ceil(1.0 / max(short_period_fraction, 1.0e-6))))
    short_energy = float(np.sum(spectrum[first_short_bin:])) if total_energy else 0.0
    short_ratio = short_energy / total_energy if total_energy else 0.0
    low_ratio = max(0.0, min(1.0, 1.0 - short_ratio))
    differences = np.diff(uniform_values)
    second = np.diff(uniform_values, n=2)
    roughness = float(np.mean(np.abs(second)) / (np.mean(np.abs(differences)) + 1.0e-12)) if second.size else 0.0
    # Autocorrelation peak reinforces the distinction between random roughness
    # and a genuinely repetitive short-period oscillation.
    ac = np.correlate(residual, residual, mode="full")[count - 1 :]
    ac /= max(float(ac[0]), 1.0e-12)
    peak = float(np.max(ac[first_short_bin:])) if ac.size > first_short_bin else 0.0
    oscillation = max(0.0, min(1.0, short_ratio * max(0.0, peak)))
    smoothness = 1.0 / (1.0 + max(roughness - 1.0, 0.0))
    low_score = max(0.0, min(1.0, low_ratio * smoothness))
    dominant_period = None
    if total_energy and spectrum.size > 1:
        dominant_bin = int(np.argmax(spectrum[1:]) + 1)
        dominant_period = length / dominant_bin
    return {
        "count": int(values.size),
        "arc_length": length,
        "value_range": value_range,
        "low_frequency_energy_ratio": low_ratio,
        "short_period_energy_ratio": short_ratio,
        "repetitive_oscillation_score": oscillation,
        "low_frequency_evolution_score": low_score,
        "roughness": roughness,
        "dominant_period": dominant_period,
    }


def longitudinal_continuity(
    samples: Any,
    *,
    fields: Iterable[str] = ("width", "height", "aspect_ratio", "area", "compactness", "floor_residual", "roof_asymmetry"),
    arc_key: str = "arc_length",
    sample_count: int = 128,
    short_period_fraction: float = 0.20,
) -> dict[str, Any]:
    """Evaluate longitudinal feature evolution on physical arc length.

    ``short_period_fraction`` is the largest fraction of total path length
    considered a repetitive short-period oscillation; it is a descriptor, not a
    quality threshold.
    """

    metrics: dict[str, Any] = {}
    for field in sorted(set(fields)):
        arcs, values = _series(samples, field, arc_key)
        metrics[field] = _one_continuity(
            arcs,
            values,
            short_period_fraction=short_period_fraction,
            sample_count=sample_count,
        )
    return {
        "schema_version": "1.0",
        "arc_key": arc_key,
        "short_period_fraction": float(short_period_fraction),
        "metrics": metrics,
    }


continuity_diagnostics = longitudinal_continuity
evaluate_continuity = longitudinal_continuity
