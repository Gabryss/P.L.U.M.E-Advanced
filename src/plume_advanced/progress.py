"""Stage and work-unit progress without affecting procedural random state."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

ProgressSink = Callable[[str, int, int | None, str], None]
_sink: ContextVar[ProgressSink | None] = ContextVar("plume_progress_sink", default=None)
_deadline: ContextVar[float | None] = ContextVar("plume_work_deadline", default=None)
_phase: ContextVar[tuple[str, ...]] = ContextVar("plume_progress_phase", default=())


class GenerationTimeBudgetError(TimeoutError):
    """A cooperative generation deadline expired; never a reason to change seed."""


def check_work_budget() -> None:
    deadline = _deadline.get()
    if deadline is not None and time.perf_counter() >= deadline:
        raise GenerationTimeBudgetError(
            "Generation attempt exceeded run.max_attempt_seconds; stopped at a work boundary. "
            "No seed retry: review the last stage and its repair report before increasing the budget.")


@contextmanager
def work_budget(seconds: float) -> Iterator[None]:
    """Check at progress boundaries; an in-flight native call cannot be preempted."""
    token = _deadline.set(time.perf_counter()+seconds if seconds else None)
    try:
        yield
    finally:
        _deadline.reset(token)


class _StageElapsedColumn(TimeElapsedColumn):
    """Retain stage time even when Rich work-unit tasks are replaced."""

    def render(self, task: Task) -> Text:
        started = task.fields.get("stage_started")
        if started is None:
            return super().render(task)
        stopped = task.fields.get("stage_stopped")
        elapsed = (time.perf_counter() if stopped is None else stopped) - started
        return Text(str(timedelta(seconds=max(0, int(elapsed)))), style="progress.elapsed")


class _WorkRemainingColumn(TimeRemainingColumn):
    def render(self, task: Task) -> Text:
        # Stage counts are not equal-cost work; they cannot predict a pipeline ETA.
        if task.fields.get("hide_eta") or task.finished:
            return Text("")
        return super().render(task)


class _DetailedProgress(Progress):
    """Keep context readable below the bars, even in a narrow terminal."""

    def get_renderables(self):
        yield from super().get_renderables()
        for task in self.tasks:
            if task.fields.get("hide_eta") and "stage_started" not in task.fields:
                if task.fields.get("detail"):
                    yield Text(task.fields["detail"], style="dim")
            if "stage_started" not in task.fields or "stage_stopped" in task.fields:
                continue
            yield Text(task.fields.get("operation", task.description), style="cyan")
            if task.fields.get("detail"):
                yield Text(task.fields["detail"], style="dim")
            now = time.perf_counter()
            started = task.fields.get("work_started", task.fields["stage_started"])
            reported = task.fields.get("reported_at", started)
            elapsed = str(timedelta(seconds=max(0, int(now - started))))
            age = str(timedelta(seconds=max(0, int(now - reported))))
            yield Text(f"Operation elapsed {elapsed} · Last work report {age} ago", style="dim")


def report_progress(
    step: str, completed: int = 0, total: int | None = None, detail: str = ""
) -> None:
    """Report measured work; unknown totals remain indeterminate, never a fake ETA."""
    check_work_budget()
    if completed < 0 or (total is not None and (total < 0 or completed > total)):
        raise ValueError("Invalid progress count")
    sink = _sink.get()
    if sink is not None:
        sink(" / ".join((*_phase.get(), step)), completed, total, detail)


@contextmanager
def progress_phase(label: str) -> Iterator[None]:
    """Keep parent operations attached to nested progress without changing RNG state."""
    token = _phase.set((*_phase.get(), label))
    try:
        report_progress("Preparing")
        yield
    finally:
        _phase.reset(token)


def progress_items(step, values, *, detail="", batch_size=65_536):
    """Report completed rows in bounded batches without copying the whole input."""
    if batch_size <= 0:
        raise ValueError("Progress batch size must be positive")
    total = len(values)
    report_progress(step, 0, total, detail)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield from values[start:end]
        report_progress(step, end, total, detail)


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
        context: str = "",
    ) -> None:
        self.console = console or Console()
        self._progress = _DetailedProgress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=width),
            TextColumn("{task.fields[count]}"),
            _StageElapsedColumn(),
            _WorkRemainingColumn(),
            console=self.console,
        )
        self._overall = self._progress.add_task(
            "Pipeline", total=total_stages, count=f"0/{total_stages} stages", detail=context, hide_eta=True
        )
        self._total_stages = total_stages
        self._finished = 0
        self._active_task_id: TaskID | None = None
        self._last_total: int | None = None
        self._step = ""
        self._stage = ""
        self._stage_started: float | None = None
        self._started = time.perf_counter()
        self._trace = None
        self._context = context
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
                        context=self._context,
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
        check_work_budget()
        if self._active_task_id is not None:
            raise RuntimeError("Finish the active stage before starting another")
        self._stage = label
        self._step = ""
        self._last_total = None
        self._stage_started = time.perf_counter()
        self._active_task_id = self._progress.add_task(
            label, total=None, count="working", detail=detail, stage_started=self._stage_started
        )
        self._record("stage_start", detail=detail)
        self._progress.refresh()

    def substep(self, step: str, completed: int, total: int | None, detail: str = "") -> None:
        check_work_budget()
        if self._active_task_id is None:
            return
        changed = step != self._step or total != self._last_total
        if changed:
            # Rich reset/update interpret None as "retain the old total".
            # A fresh task is required when work becomes indeterminate.
            self._progress.remove_task(self._active_task_id)
            self._active_task_id = self._progress.add_task(
                f"{self._stage} / {step.rsplit(' / ', 1)[-1]}", total=total, completed=completed,
                count="working", detail=detail, stage_started=self._stage_started,
                work_started=time.perf_counter(),
            )
            self._step = step
        self._last_total = total
        self._progress.update(
            self._active_task_id,
            total=total,
            completed=completed,
            description=f"{self._stage} / {step.rsplit(' / ', 1)[-1]}",
            count=f"{completed:,}/{total:,}" if total is not None else "working",
            detail=detail, operation=step, reported_at=time.perf_counter(),
        )
        self._record("work", step=step, completed=completed, total=total, detail=detail)
        # Show the upcoming operation before entering a native call which may
        # hold the GIL and prevent Rich's refresh thread from running.
        if changed:
            self._progress.refresh()

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
            stage_stopped=time.perf_counter(),
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
        if self._active_task_id is not None:
            self._progress.update(self._active_task_id, stage_stopped=time.perf_counter(), hide_eta=True)
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
