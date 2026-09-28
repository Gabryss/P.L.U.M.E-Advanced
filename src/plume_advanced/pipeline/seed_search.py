"""Durable, deterministic full-generation seed search (separate from local repair)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from plume_advanced.procedural import canonical_seed, derive_subseed

SCHEMA = "plume.seed-search.v1"


def next_seed(root: int | None, tried: list[int]) -> int:
    if not tried:
        return canonical_seed(root)
    nonce = 0
    used = set(tried)
    while True:
        candidate = derive_subseed(root, SCHEMA, len(tried), nonce)
        if candidate not in used:
            return candidate
        nonce += 1


class SeedSearch:
    """Only accepted pipeline completion can mark a candidate successful.

    Pending/interrupted attempts reuse their seed on resume. Rejected seeds are
    never retried. The fingerprint binds the original request, not the winner.
    """

    def __init__(self, path: Path, *, root_seed: int | None, fingerprint: str,
                 required: bool, max_attempts: int, resume: bool = False):
        self.path = path
        self.data: dict = dict(schema=SCHEMA, root_seed=canonical_seed(root_seed),
            fingerprint=fingerprint, robot_qualification_required=required,
            max_seed_attempts=max_attempts, status="pending", accepted_seed=None, attempts=[])
        if resume and path.is_file():
            saved = json.loads(path.read_text())
            for key in ("schema", "root_seed", "fingerprint", "robot_qualification_required",
                        "max_seed_attempts"):
                if saved.get(key) != self.data[key]:
                    raise ValueError(f"Seed journal {key} does not match this request; use a new output directory")
            tried: list[int] = []
            for index, row in enumerate(saved["attempts"]):
                if (row["seed"] != next_seed(root_seed, tried) or row["attempt"] != index + 1
                        or row["status"] not in {"running", "interrupted", "failed", "rejected", "accepted"}
                        or (index < len(saved["attempts"]) - 1 and row["status"] != "rejected")):
                    raise ValueError("Invalid deterministic seed journal")
                tried.append(row["seed"])
            self.data = saved

    @property
    def seed(self) -> int | None:
        rows = self.data["attempts"]
        if rows and rows[-1]["status"] != "rejected":
            return int(rows[-1]["seed"])
        limit = self.data["max_seed_attempts"]
        if limit and len(rows) >= limit:
            return None
        return next_seed(self.data["root_seed"], [row["seed"] for row in rows])

    @property
    def attempt(self) -> int:
        rows = self.data["attempts"]
        return len(rows) + (not rows or rows[-1]["status"] == "rejected")

    def begin(self, stage_seeds: dict) -> None:
        seed, attempt = self.seed, self.attempt
        if seed is None:
            raise ValueError("Seed search budget exhausted")
        row = dict(attempt=attempt, seed=seed, stage_seeds=stage_seeds, status="running")
        if attempt > len(self.data["attempts"]):
            self.data["attempts"].append(row)
        else:
            self.data["attempts"][-1] = row
        self.data.update(status="running", accepted_seed=None)
        self.save()

    def finish(self, status: str, **details) -> None:
        self.data["attempts"][-1].update(status=status, **details)
        self.data["status"] = status
        if status == "accepted":
            self.data["accepted_seed"] = self.data["attempts"][-1]["seed"]
        else:
            self.data["accepted_seed"] = None
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as stream:
                temporary_path = Path(stream.name)
                json.dump(self.data, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def retryable_rejection(error: Exception | KeyboardInterrupt) -> bool:
    """Retry candidate-dependent defects, never arbitrary exceptions or asset errors."""
    from plume_advanced.acceptance import AcceptanceError
    from plume_advanced.exporters.errors import ExportBudgetError
    from plume_advanced.pipeline.resolution import ResolutionBudgetError
    from plume_advanced.stages.network_quality import NetworkQualityError
    from plume_advanced.stages.surface_topology import SurfaceTopologyError

    if isinstance(error, (ResolutionBudgetError, ExportBudgetError)):
        return False
    if any(row.get("error_type") == "ResolutionBudgetError"
           for row in getattr(error, "report", {}).get("attempts", [])):
        return False
    if isinstance(error, AcceptanceError):
        failed = {name for name, check in error.report["checks"].items()
                  if check["required"] and check["status"] != "passed"}
        return bool(failed) and failed <= {"clearance", "ground_routes", "relief"}
    return isinstance(error, (NetworkQualityError, SurfaceTopologyError))
