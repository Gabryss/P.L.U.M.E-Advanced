"""Resumable, append-preserving execution of declared experiment cases."""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plume_advanced.evaluation.schema import ExperimentResult


class ResultStore:
    def __init__(
        self,
        root: str | Path,
        experiment_name: str,
        *,
        provenance_sha256: str | None = None,
    ) -> None:
        self.root = Path(root) / experiment_name
        self.provenance_sha256 = provenance_sha256
        self.case_root = self.root / "cases"
        self.case_root.mkdir(parents=True, exist_ok=True)

    def case_path(self, run_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        return self.case_root / f"{safe}.json"

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self.case_path(run_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def should_skip(self, expected: ExperimentResult, *, force: bool = False) -> bool:
        previous = self.load(expected.run_id)
        return bool(
            previous
            and previous.get("status") == "complete"
            and self._same_identity(previous, expected)
            and not force
        )

    @staticmethod
    def _same_identity(previous: dict[str, Any], expected: ExperimentResult) -> bool:
        return all(
            key in previous and previous[key] == value
            for key, value in expected.resume_identity().items()
        )

    def write(self, result: ExperimentResult, *, force: bool = False) -> Path:
        path = self.case_path(result.run_id)
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "complete"
                and self._same_identity(previous, result)
                and not force
            ):
                raise FileExistsError(f"completed result already exists: {result.run_id}")
            archive = self.case_root / "attempts"
            archive.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            path.replace(archive / f"{path.stem}.{stamp}.json")
        _atomic_json(path, result.to_dict())
        self.rebuild_indexes()
        return path

    def rows(self) -> list[dict[str, Any]]:
        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.case_root.glob("*.json"))
        ]
        if self.provenance_sha256 is not None:
            rows = [
                row for row in rows
                if row.get("provenance_sha256") == self.provenance_sha256
            ]
        return rows

    def rebuild_indexes(self) -> None:
        rows = self.rows()
        _atomic_json(self.root / "raw_results.json", rows)
        fields = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
        )
        csv_path = self.root / "raw_results.csv"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=self.root,
            prefix=".raw_results.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            writer = csv.DictWriter(temporary, fieldnames=fields, extrasaction="ignore")
            if fields:
                writer.writeheader()
                writer.writerows(rows)
            temporary_path = Path(temporary.name)
        temporary_path.replace(csv_path)


def run_case(
    store: ResultStore,
    template: ExperimentResult,
    operation: Callable[[], dict[str, Any]],
    *,
    force: bool = False,
) -> ExperimentResult | None:
    if store.should_skip(template, force=force):
        return None
    result = _evaluate_case(template, operation)
    store.write(result, force=force)
    print(f"{template.experiment_name}: {template.run_id} {result.status} ({result.elapsed_s:.2f} s)", flush=True)
    return result


def _evaluate_case(template, operation):
    template = replace(template, started_at_utc=datetime.now(UTC).isoformat())
    started = time.perf_counter()
    try:
        metrics = operation()
        json.dumps(metrics, allow_nan=False)
        result = replace(
            template,
            status="complete",
            metrics=metrics,
            elapsed_s=time.perf_counter() - started,
        )
    except TimeoutError as error:
        result = replace(
            template,
            status="timeout",
            failure_reason=f"{type(error).__name__}: {error}",
            elapsed_s=time.perf_counter() - started,
            metrics=getattr(error, "metrics", {}),
        )
    except Exception as error:
        result = replace(
            template,
            status="failed",
            failure_reason=f"{type(error).__name__}: {error}",
            elapsed_s=time.perf_counter() - started,
            metrics=getattr(error, "metrics", {}),
        )
    return result


_FORK_TASKS: tuple[tuple[ExperimentResult, Callable[[], dict[str, Any]]], ...] = ()


def _fork_evaluate(index):
    return _evaluate_case(*_FORK_TASKS[index])


def run_cases(store, tasks, *, force=False, chunksize=1):
    """Parallelize independent non-benchmark cases; only the parent writes results.

    Fork inherits the prepared closures without serializing them. Platforms
    without fork retain the sequential path. Timed scalability cases do not
    call this helper, so they never compete with sibling benchmark workers.
    """
    global _FORK_TASKS
    pending = [(template, op) for template, op in tasks if not store.should_skip(template, force=force)]
    workers = max(1, int(os.environ.get("PLUME_EVALUATION_WORKERS", "1")))
    if workers == 1 or "fork" not in multiprocessing.get_all_start_methods():
        for template, operation in pending:
            run_case(store, template, operation, force=force)
        return
    _FORK_TASKS = tuple(pending)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
            for result in pool.map(_fork_evaluate, range(len(pending)), chunksize=chunksize):
                store.write(result, force=force)
                print(f"{result.experiment_name}: {result.run_id} {result.status} ({result.elapsed_s:.2f} s)", flush=True)
    finally:
        _FORK_TASKS = ()


def write_experiment_manifest(
    output_root: str | Path,
    *,
    config_path: str | Path,
    experiments: Iterable[str],
    provenance: dict[str, Any],
) -> Path:
    path = Path(output_root) / "experiment_manifest.json"
    _atomic_json(
        path,
        {
            "schema": "plume.experiment-manifest.v1",
            "config_path": str(Path(config_path).resolve()),
            "experiments": list(experiments),
            "provenance": provenance,
        },
    )
    return path


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


__all__ = ["ResultStore", "run_case", "run_cases", "write_experiment_manifest"]
