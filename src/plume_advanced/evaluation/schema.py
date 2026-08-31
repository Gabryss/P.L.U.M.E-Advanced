"""Common raw-result schema used by every paper experiment."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from plume_advanced.evaluation import EXPERIMENT_SCHEMA_VERSION, MODEL_SCHEMA_VERSION

RunStatus = Literal["complete", "failed", "timeout", "invalid"]


@dataclass(frozen=True)
class ExperimentResult:
    experiment_name: str
    run_id: str
    condition_id: str
    seed: int
    status: RunStatus
    failure_reason: str = ""
    started_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    elapsed_s: float = 0.0
    git_commit: str = ""
    git_dirty: bool = False
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    resolved_config_sha256: str = ""
    input_dataset_id: str = ""
    input_dataset_sha256_or_version: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    experiment_schema_version: int = EXPERIMENT_SCHEMA_VERSION
    model_schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"complete", "failed", "timeout", "invalid"}:
            raise ValueError(f"Unsupported experiment status: {self.status}")
        if self.status != "complete" and not self.failure_reason:
            raise ValueError("non-complete results require failure_reason")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        metrics = payload.pop("metrics")
        payload.update(metrics)
        return payload


__all__ = ["ExperimentResult", "RunStatus"]
