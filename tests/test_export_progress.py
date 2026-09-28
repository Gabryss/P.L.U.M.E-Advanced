"""Export progress names the real operation without changing serialized geometry."""

import io

import numpy as np
from test_embedded_inspection import geometry

from plume_advanced.exporters import export_target_asset
from plume_advanced.exporters.targets import _write_collision_rows
from plume_advanced.progress import progress_scope
from plume_advanced.world import ExportConfig


def test_all_target_export_reports_files_counts_and_validation(tmp_path):
    events = []
    with progress_scope(lambda *event: events.append(event)):
        result = export_target_asset(geometry(smoothing=0),
            ExportConfig(target='all', file_format='auto'), tmp_path/'export')
    assert result.primary_asset.is_file()
    steps = [event[0] for event in events]
    for suffix in ('OBJ watertight check', 'OBJ vertices', 'OBJ triangles',
                   'Collision OBJ normal calculation', 'Collision OBJ triangles',
                   'GLB binary assembly', 'GLB bytes', 'Blender import validation',
                   'USD vertices', 'USD file write', 'Publish export'):
        assert any(step.endswith(suffix) for step in steps), suffix
    assert any(step.startswith('blender package 1/5 / ') for step in steps)
    assert any(step.startswith('omniverse package 5/5 / ') for step in steps)
    for suffix in ('OBJ vertices', 'OBJ triangles', 'Collision OBJ triangles', 'GLB bytes'):
        rows = [event for event in events if event[0].endswith(suffix)]
        assert rows[0][1] == 0
        assert rows[-1][1] == rows[-1][2] > 0
        assert all(row[3] for row in rows)  # file identity remains visible


def test_collision_batch_writer_preserves_exact_obj_text():
    faces = np.arange(65_539 * 3, dtype=np.int64).reshape(-1, 3)
    before, after = io.StringIO(), io.StringIO()
    fmt = 'f %d//%d %d//%d %d//%d'
    np.savetxt(before, np.repeat(faces + 1, 2, axis=1), fmt=fmt)
    events = []
    with progress_scope(lambda *event: events.append(event)):
        _write_collision_rows(after, faces, fmt, 'Collision OBJ triangles', 'cave.obj', faces=True)
    assert before.getvalue() == after.getvalue()
    assert [event[1] for event in events] == [0, 65_536, len(faces)]
