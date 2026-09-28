"""Visible work units, honest unknown totals, trace completion and RNG isolation."""

import io
import json

import numpy as np
import pytest


def test_cooperative_deadline_expires_without_sink_and_resets(monkeypatch):
    import plume_advanced.progress as module
    now = [10.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    with module.work_budget(2):
        module.report_progress('work')
        now[0] = 12.
        with pytest.raises(module.GenerationTimeBudgetError):
            module.report_progress('next work boundary')
    module.report_progress('budget context restored')
    with module.work_budget(0):
        now[0] = 1e9
        module.report_progress('explicit unlimited')
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


def test_stage_elapsed_survives_work_changes_and_stays_frozen_after_finish(monkeypatch):
    import plume_advanced.progress as module

    now = [100.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    terminal = TerminalProgress(console=Console(file=io.StringIO()), total_stages=2)
    elapsed = module._StageElapsedColumn()
    remaining = module._WorkRemainingColumn()
    try:
        terminal.start('Base geometry')
        now[0] += 3600
        terminal.substep('Mesh', 1, 2)
        now[0] += 60
        terminal.substep('Ground routes', 0, None)
        active = terminal._progress.tasks[-1]
        assert elapsed.render(active).plain == '1:01:00'
        assert remaining.render(terminal._progress.tasks[0]).plain == ''
        terminal.finish()
        now[0] += 120
        assert elapsed.render(active).plain == '1:01:00'
        terminal.start('Export')
        now[0] += 10
        terminal.close()
        stopped = terminal._progress.tasks[-1]
        now[0] += 50
        assert elapsed.render(stopped).plain == '0:00:10'
        assert remaining.render(stopped).plain == ''
    finally:
        terminal.close()


def test_trace_keeps_seed_attempt_context(tmp_path):
    trace = tmp_path / 'trace.jsonl'
    terminal = TerminalProgress(console=Console(file=io.StringIO()), total_stages=1,
                                trace_path=trace, context='attempt 2; seed 42')
    try:
        terminal.start('Inspection')
        report_progress('Floor support', 1, 4)
        terminal.finish()
        assert terminal._progress.tasks[0].fields['detail'] == 'attempt 2; seed 42'
    finally:
        terminal.close()
    assert all(row['context'] == 'attempt 2; seed 42'
               for row in map(json.loads, trace.read_text().splitlines()))


def test_nested_export_context_survives_substeps_and_restores_after_failure():
    from plume_advanced.progress import progress_phase

    events = []
    with progress_scope(lambda *event: events.append(event)):
        with progress_phase('blender package 1/5'):
            with pytest.raises(RuntimeError), progress_phase('rock 2/200'):
                report_progress('Tangent triangles', 320, 320)
                raise RuntimeError('synthetic')
            report_progress('GLB bytes', 10, 20)
        report_progress('Package inspection')
    assert events[-3][0] == 'blender package 1/5 / rock 2/200 / Tangent triangles'
    assert events[-2][0] == 'blender package 1/5 / GLB bytes'
    assert events[-1][0] == 'Package inspection'


def test_row_progress_counts_only_rows_the_consumer_has_processed():
    from plume_advanced.progress import progress_items

    events = []
    with progress_scope(lambda *event: events.append(event)):
        rows = progress_items('OBJ triangles', np.arange(7), batch_size=3)
        assert next(rows) == 0
        assert events[-1][1:3] == (0, 7)
        assert [next(rows), next(rows)] == [1, 2]
        assert events[-1][1] == 0  # third row has not returned to the writer yet
        assert next(rows) == 3
        assert events[-1][1] == 3
        assert list(rows) == [4, 5, 6]
    assert [event[1] for event in events] == [0, 3, 6, 7]


def test_progress_detail_is_readable_and_clock_advances_without_new_work(monkeypatch):
    import plume_advanced.progress as module

    now = [100.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    terminal = TerminalProgress(console=Console(file=io.StringIO(), width=80))
    try:
        terminal.start('Re-export')
        report_progress('blender package 1/5 / OBJ watertight check',
                        detail='plume_cave_fallback.obj: building edge adjacency; no internal counter')
        now[0] += 95
        lines = [part.plain for part in terminal._progress.get_renderables() if hasattr(part, 'plain')]
        assert 'blender package 1/5 / OBJ watertight check' in lines
        assert any('plume_cave_fallback.obj' in line for line in lines)
        assert 'Operation elapsed 0:01:35 · Last work report 0:01:35 ago' in lines
        assert terminal._progress.tasks[-1].total is None
        report_progress('Tangent triangles', 320, 320)
        assert module._WorkRemainingColumn().render(terminal._progress.tasks[-1]).plain == ''
    finally:
        terminal.close()
