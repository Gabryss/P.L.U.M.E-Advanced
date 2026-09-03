"""Stable seed derivation for independent procedural domains."""

from __future__ import annotations

import zlib

import numpy as np


def canonical_seed(seed: int | None) -> int:
    """Resolve an omitted seed to the reproducible baseline seed."""

    return 0 if seed is None else int(seed)


def derive_subseed(seed: int | None, *labels: str | int) -> int:
    """Derive a stable seed without depending on Python's randomized hash."""

    entropy = [canonical_seed(seed)]
    for label in labels:
        encoded = f"{type(label).__name__}:{label}".encode("utf-8")
        entropy.append(zlib.crc32(encoded))
    sequence = np.random.SeedSequence(entropy)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def procedural_rng(seed: int | None, *labels: str | int) -> np.random.Generator:
    """Return an independent deterministic generator for one named domain."""

    return np.random.default_rng(derive_subseed(seed, *labels))


__all__ = ["canonical_seed", "derive_subseed", "procedural_rng"]
