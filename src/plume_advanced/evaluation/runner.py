"""Resumable, append-preserving execution of declared experiment cases."""

from __future__ import annotations

import csv
import json
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plume_advanced.evaluation.schema import ExperimentResult


class ResultStore:
    def __init__(self, root: str | Path, experiment_name: str) -> None:
        self.root = Path(root) / experiment_name
        self.case_root = self.root / "cases"
        self.case_root.mkdir(parents=True, exist_ok=True)

    def case_path(self, run_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        return self.case_root / f"{safe}.json"

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self.case_path(run_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def should_skip(self, run_id: str, *, force: bool = False) -> bool:
        previous = self.load(run_id)
        return bool(previous and previous.get("status") == "complete" and not force)

    def write(self, result: ExperimentResult, *, force: bool = False) -> Path:
        path = self.case_path(result.run_id)
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete" and not force:
                raise FileExistsError(f"completed result already exists: {result.run_id}")
            archive = self.case_root / "attempts"
            archive.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            path.replace(archive / f"{path.stem}.{stamp}.json")
        _atomic_json(path, result.to_dict())
        self.rebuild_indexes()
        return path

    def rows(self) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.case_root.glob("*.json"))
        ]

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
    if store.should_skip(template.run_id, force=force):
        return None
    started = time.perf_counter()
    try:
        metrics = operation()
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
        )
    except Exception as error:
        result = replace(
            template,
            status="failed",
            failure_reason=f"{type(error).__name__}: {error}",
            elapsed_s=time.perf_counter() - started,
        )
    store.write(result, force=force)
    return result


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


__all__ = ["ResultStore", "run_case", "write_experiment_manifest"]
