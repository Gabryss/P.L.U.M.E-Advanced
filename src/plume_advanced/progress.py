"""Stage and work-unit progress without affecting procedural random state."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ProgressSink = Callable[[str, int, int | None, str], None]
_sink: ContextVar[ProgressSink | None] = ContextVar("plume_progress_sink", default=None)


def report_progress(
    step: str, completed: int = 0, total: int | None = None, detail: str = ""
) -> None:
    """Report measured work; unknown totals remain indeterminate, never a fake ETA."""
    if completed < 0 or (total is not None and (total < 0 or completed > total)):
        raise ValueError("Invalid progress count")
    sink = _sink.get()
    if sink is not None:
        sink(step, completed, total, detail)


@contextmanager
def progress_scope(sink: ProgressSink) -> Iterator[None]:
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


class TerminalProgress:
    """Overall stages plus the active work unit, with a machine-readable trace."""

    _current: ClassVar[TerminalProgress | None] = None

    def __init__(
        self,
        *,
        width: int = 32,
        total_stages: int = 11,
        trace_path: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.console = console or Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=width),
            TextColumn("{task.fields[count]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[detail]}"),
            console=self.console,
        )
        self._overall = self._progress.add_task(
            "Pipeline", total=total_stages, count=f"0/{total_stages} stages", detail=""
        )
        self._total_stages = total_stages
        self._finished = 0
        self._active_task_id: TaskID | None = None
        self._last_total: int | None = None
        self._step = ""
        self._stage = ""
        self._started = time.perf_counter()
        self._trace = None
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = trace_path.open("a", encoding="utf-8")
        self._token = _sink.set(self.substep)
        self._closed = False
        self._progress.start()
        type(self)._current = self

    def _record(self, event: str, **values) -> None:
        if self._trace is not None:
            self._trace.write(
                json.dumps(
                    dict(
                        event=event,
                        elapsed_s=time.perf_counter() - self._started,
                        stage=self._stage,
                        **values,
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
            self._trace.flush()

    def log(self, message: str) -> None:
        self.console.print(message, markup=False)
        self._record("message", message=message)

    def start(self, label: str, detail: str = "") -> None:
        if self._active_task_id is not None:
            raise RuntimeError("Finish the active stage before starting another")
        self._stage = label
        self._step = ""
        self._last_total = None
        self._active_task_id = self._progress.add_task(
            label, total=None, count="working", detail=detail
        )
        self._record("stage_start", detail=detail)

    def substep(self, step: str, completed: int, total: int | None, detail: str = "") -> None:
        if self._active_task_id is None:
            return
        if step != self._step or total != self._last_total:
            # Rich reset/update interpret None as "retain the old total".
            # A fresh task is required when work becomes indeterminate.
            self._progress.remove_task(self._active_task_id)
            self._active_task_id = self._progress.add_task(
                f"{self._stage} / {step}", total=total, completed=completed,
                count="working", detail=detail,
            )
            self._step = step
        self._last_total = total
        self._progress.update(
            self._active_task_id,
            total=total,
            completed=completed,
            description=f"{self._stage} / {step}",
            count=f"{completed:,}/{total:,}" if total is not None else "working",
            detail=detail,
        )
        self._record("work", step=step, completed=completed, total=total, detail=detail)

    def update(self, current: int, total: int, detail: str = "") -> None:
        self.substep("work", min(max(current, 0), max(total, 1)), max(total, 1), detail)

    def finish(self, detail: str = "done") -> None:
        if self._active_task_id is None:
            return
        self._progress.update(
            self._active_task_id,
            total=1,
            completed=1,
            count="done",
            detail=detail,
            description=self._stage,
        )
        self._progress.stop_task(self._active_task_id)
        self._active_task_id = None
        self._finished += 1
        self._progress.update(
            self._overall,
            completed=self._finished,
            count=f"{self._finished}/{self._total_stages} stages",
        )
        self._record("stage_finish", detail=detail)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._progress.stop()
        _sink.reset(self._token)
        if self._trace is not None:
            self._trace.close()
        if type(self)._current is self:
            type(self)._current = None

    @classmethod
    def close_active(cls) -> None:
        if cls._current is not None:
            cls._current.close()
