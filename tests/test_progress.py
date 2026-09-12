"""Visible work units, honest unknown totals, trace completion and RNG isolation."""

import io
import json

import numpy as np
import pytest
from rich.console import Console

from plume_advanced.progress import TerminalProgress, progress_scope, report_progress
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator


def test_nested_sinks_restore_even_after_failure():
    outer = []
    inner = []
    with progress_scope(lambda *event: outer.append(event)):
        report_progress("first")
        with pytest.raises(RuntimeError):
            with progress_scope(lambda *event: inner.append(event)):
                report_progress("inner", 1, 2)
                raise RuntimeError("synthetic")
        report_progress("last")
    report_progress("ignored")
    assert [e[0] for e in outer] == ["first", "last"]
    assert inner == [("inner", 1, 2, "")]


@pytest.mark.parametrize("completed,total", [(-1, 1), (3, 2), (0, -1)])
def test_invalid_progress_does_not_claim_completion(completed, total):
    with pytest.raises(ValueError):
        report_progress("bad", completed, total)


def test_terminal_trace_uses_measured_counts_and_unknown_work(tmp_path):
    trace = tmp_path / "progress.jsonl"
    terminal = TerminalProgress(
        console=Console(file=io.StringIO(), force_terminal=False), total_stages=1, trace_path=trace
    )
    try:
        terminal.start("Export")
        report_progress("UV charts", 0, 3)
        report_progress("UV charts", 2, 3)
        assert terminal._progress.tasks[-1].total == 3
        report_progress("Native serialization", detail="waiting for writer")
        # Rich treats total=None as "keep the previous total" in reset/update.
        # Verify the displayed task, not only the JSON trace's null value.
        assert terminal._progress.tasks[-1].total is None
        report_progress("Native serialization", 1, 3)
        assert terminal._progress.tasks[-1].total == 3
        report_progress("Native serialization", detail="waiting for writer")
        assert terminal._progress.tasks[-1].total is None
        terminal.finish()
    finally:
        terminal.close()
        terminal.close()
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows[0]["event"] == "stage_start" and rows[-1]["event"] == "stage_finish"
    assert [(r["completed"], r["total"]) for r in rows if r["event"] == "work"] == [
        (0, 3),
        (2, 3),
        (0, None),
        (1, 3),
        (0, None),
    ]
    assert [r["elapsed_s"] for r in rows] == sorted(r["elapsed_s"] for r in rows)


def test_progress_preserves_generated_host_and_numpy_state():
    generator = HostFieldGenerator(HostFieldConfig(grid=GridConfig(nx=12, ny=12), random_seed=123))
    before = np.random.get_state()
    silent = generator.generate()
    events = []
    with progress_scope(lambda *event: events.append(event)):
        observed = generator.generate()
    np.testing.assert_array_equal(silent.growth_cost, observed.growth_cost)
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    assert [e[1] for e in events] == [0, 1, 2, 3, 4]
