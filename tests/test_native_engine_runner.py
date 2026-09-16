"""Failure handling for the opt-in GPU editor checks; no editor is launched by pytest."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "native_check_runner", Path(__file__).resolve().parents[1] / "scripts/check_native_engines.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('layout', ['export', 'export_blender', 'export_all/blender'])
@pytest.mark.parametrize('stem', ['plume_cave', 'plume_cave_scene'])
def test_native_export_paths_preserve_serialization_root(tmp_path, layout, stem):
    source = tmp_path / layout / f'{stem}.glb'
    source.parent.mkdir(parents=True)
    source.touch()
    collider = source.with_name(f'{stem}_collision.obj')
    collider.touch()
    visual, collision, root = runner.native_export_paths(tmp_path)
    assert visual == source and collision == collider
    assert root == (tmp_path / 'export_all' if layout.startswith('export_all') else source.parent)
    # The selected receipt must refer to this collider, including its target directory.
    assert root / collision.relative_to(root) == collider


def test_native_export_paths_reject_stale_alternative_delivery(tmp_path):
    for layout in ('export', 'export_all/blender'):
        source = tmp_path / layout / 'plume_cave.glb'
        source.parent.mkdir(parents=True)
        source.touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        runner.native_export_paths(tmp_path)


def test_native_export_paths_report_missing_delivery(tmp_path):
    with pytest.raises(FileNotFoundError, match='No native inspection GLB'):
        runner.native_export_paths(tmp_path)


def ground_evidence():
    from test_ground_routes import corridor, inspect
    return dict(export_inspection=dict(collision=dict(inspection=dict(ground_traversal=inspect(corridor())))),
                acceptance=dict(policy=dict(require_ground_routes=True)))


@pytest.mark.parametrize('engine', ['unity', 'unreal'])
def test_native_ground_plan_preserves_full_footprint_and_rotations(engine):
    quality = ground_evidence()
    plan = runner.ground_route_plan(quality, engine=engine)
    assert plan['length_m'] == .7 and plan['stations'] == 61 and plan['sweeps'] == 60
    assert plan['floor_samples'] == 61*63
    if engine == 'unity':
        assert plan['half_extents'] == pytest.approx(dict(x=.27, y=.27, z=.37))
        assert plan['poses'][0]['forward'] == dict(x=-1, y=0, z=0)
        assert plan['poses'][0]['point']['y'] == pytest.approx(.273)
    else:
        assert plan['half_extents'] == pytest.approx([37, 27, 27])
        assert plan['poses'][0]['forward'] == [1, 0, 0]
        assert plan['poses'][0]['point'][2] == pytest.approx(27.3)


@pytest.mark.parametrize('mutation', ['missing', 'failed', 'probe', 'smaller_box', 'weights'])
def test_native_ground_plan_rejects_incomplete_or_changed_evidence(mutation):
    quality = ground_evidence()
    inspection = quality['export_inspection']['collision']['inspection']
    ground = inspection['ground_traversal']
    if mutation == 'missing':
        inspection.clear()
    elif mutation == 'failed':
        ground['passed'] = False
    elif mutation == 'probe':
        ground['paths'][0]['poses'][0]['floor_points_m'].pop()
    elif mutation == 'smaller_box':
        ground['paths'][0]['sweeps'][0]['half_extents_m'][0] = .1
    else:
        ground['support_fit_weights'][0][0] = float('nan')
    with pytest.raises(ValueError):
        runner.ground_route_plan(quality, engine='unity')


def test_native_ground_results_need_all_counts_limits_and_negative_controls():
    expected = {key: value for key, value in runner.ground_route_plan(ground_evidence(), engine='unity').items()
                if key not in ('half_extents', 'poses', 'motions', 'support')}
    result = dict(**expected, passed=True, failures=0, maximumFloorErrorM=.001,
                  maximumSlopeDeg=1., maximumStepM=.01,
                  controls=[dict(name=name, expected=hit, observed=hit, passed=True)
                            for name, hit in runner.GROUND_CONTROLS.items()])
    assert runner.validate_native_ground(dict(ground=result), expected)['passed']
    mutations = [('stations', 0), ('sweeps', 0), ('floor_samples', 0), ('length_m', .5),
                 ('maximumFloorErrorM', .003), ('maximumSlopeDeg', 21), ('maximumStepM', .11),
                 ('maximumStepM', float('nan')), ('controls', []), ('controls', None)]
    for name, value in mutations:
        changed = deepcopy(result)
        changed[name] = value
        with pytest.raises(ValueError):
            runner.validate_native_ground(dict(ground=changed), expected)
    changed = deepcopy(result)
    changed['controls'][0]['observed'] = False
    with pytest.raises(ValueError):
        runner.validate_native_ground(dict(ground=changed), expected)


def physics_controls():
    return [dict(name=name, expected=hit, observed=hit, passed=True)
            for name, hit in runner.PHYSICS_CONTROLS.items()]


@pytest.mark.parametrize("color", [0, 128, 255])
def test_blank_native_captures_are_not_accepted(tmp_path, color):
    Image.new("RGB", (960, 640), (color, color, color)).save(tmp_path / "interior_1.png")
    with pytest.raises(ValueError, match="Blank/unusable"):
        runner.validate_captures(tmp_path)


def test_both_native_views_are_required(tmp_path):
    gradient = np.linspace(0, 200, 960, dtype=np.uint8)[None, :].repeat(640, axis=0)
    image = Image.fromarray(gradient).convert("RGB")
    image.save(tmp_path / "interior_1.png")
    with pytest.raises(FileNotFoundError):
        runner.validate_captures(tmp_path)
    image.save(tmp_path / "interior_2.png")
    assert len(runner.validate_captures(tmp_path)) == 2


def test_partially_overexposed_capture_is_rejected_despite_high_contrast(tmp_path):
    pixels = np.zeros((640, 960, 3), dtype=np.uint8)
    pixels[:, :480] = 255
    Image.fromarray(pixels).save(tmp_path / "interior_1.png")
    with pytest.raises(ValueError, match="Overexposed"):
        runner.validate_captures(tmp_path)


def test_native_capture_size_is_checked(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "interior_1.png")
    with pytest.raises(ValueError, match="dimensions"):
        runner.validate_captures(tmp_path)


def test_editor_failure_preserves_log(tmp_path):
    log = tmp_path / "editor.log"
    with pytest.raises(RuntimeError, match="Editor exited 3"):
        runner.run_editor(
            [sys.executable, "-c", "print('native failure'); raise SystemExit(3)"], log, 10
        )
    assert "native failure" in log.read_text()


def test_stalled_editor_has_bounded_runtime(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_editor(
            [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "editor.log", 0.1
        )


def test_rejected_generation_cannot_be_used_as_native_evidence(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "pipeline_quality_report.json").write_text(json.dumps({"passed": False}))
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="passed pipeline"):
        runner.prepare(run, output, unity=True, unreal=True)
    assert not output.exists()


def test_existing_native_project_is_never_overwritten(tmp_path):
    output = tmp_path / "native"
    output.mkdir()
    sentinel = output / "user_file"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path / "missing", output, unity=True, unreal=True)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("unreal_passed", [False, True])
def test_editor_exit_zero_does_not_override_failed_checks(tmp_path, monkeypatch, unreal_passed):
    output = tmp_path / "native"

    def prepare_stub(run, root, **engines):
        root.mkdir()
        body = dict(stations=2, edges=1, height_m=.5, width_m=.5, margin_m=.02)
        (root / "native_input_receipt.json").write_text(json.dumps({"body_routes": {"unity": body, "unreal": body}}))
        for folder, passed in (("unity_project", True), ("unreal_run_01", unreal_passed)):
            (root / folder).mkdir(parents=True)
            (root / folder / "native_result.json").write_text(json.dumps(dict(passed=passed,
                body=dict(passed=True, overlap_control=True, sweep_control=True, **body),
                physicsControls=physics_controls(), physics_controls=physics_controls(),
                bodyPassed=True, bodyStations=2, bodyEdges=1, bodyHeightM=.5, bodyWidthM=.5,
                bodyMarginM=.02, bodyOverlapControl=True, bodySweepControl=True)))

    monkeypatch.setattr(runner, "prepare", prepare_stub)
    monkeypatch.setattr(runner, "run_editor", lambda *args: 0.1)
    monkeypatch.setattr(runner, "validate_captures", lambda directory: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_native_engines",
            str(tmp_path / "run"),
            "--output",
            str(output),
            "--unity",
            sys.executable,
            "--unreal",
            sys.executable,
        ],
    )
    assert runner.main() == int(not unreal_passed)
    result = json.loads((output / "native_summary.json").read_text())
    assert result["passed"] is unreal_passed
    assert result["checks"]["unity"]["native"]["passed"]
    if not unreal_passed:
        assert "Native checks failed" in result["failures"][0]


def test_native_clearance_baseline_is_the_exported_visual_surface():
    # Smoothing/displacement can legitimately change the final visual surface.
    # Comparing the engine against raw geometry confounds import and export changes.
    raw, visual = [{"floor_distance_m": 1.0}], [{"floor_distance_m": 1.2}]
    quality = {
        "final_mesh_inspection": {"measurements": raw},
        "export_inspection": {"passed": True, "visual": {"passed": True, "measurements": visual}},
    }
    assert runner.source_measurements(quality) == visual
    quality["export_inspection"]["visual"]["passed"] = False
    with pytest.raises(ValueError, match="visual-export inspection"):
        runner.source_measurements(quality)


def test_every_planned_native_capture_is_mandatory(tmp_path):
    directory = tmp_path / "unity_project"
    directory.mkdir()
    (tmp_path / "view_plan.json").write_text(json.dumps({"view_count": 4}))
    gradient = np.linspace(0, 200, 960, dtype=np.uint8)[None, :].repeat(640, axis=0)
    image = Image.fromarray(gradient).convert("RGB")
    for i in range(1, 4):
        image.save(directory / f"interior_{i}.png")
    with pytest.raises(FileNotFoundError):
        runner.validate_captures(directory)
    image.save(directory / "interior_4.png")
    assert len(runner.validate_captures(directory)) == 4


def body_quality():
    return dict(export_inspection=dict(collision=dict(inspection=dict(traversal=dict(
        enabled=True, passed=True, height_m=1., width_m=.5, margin_m=.02,
        paths=[dict(passed=True, segment_id=8, center_path_m=[[1., 2., 3.], [4., 5., 6.]])])))))


@pytest.mark.parametrize('engine,point,radius,axis', [
    ('unity', {'x': -1., 'y': 3., 'z': -2.}, .27, .25),
    ('unreal', [100., -200., 300.], 27., 25.)])
def test_native_body_uses_placed_collider_paths_and_correct_engine_units(engine, point, radius, axis):
    plan = runner.body_route_plan(body_quality(), engine=engine)
    assert plan['paths'][0]['points'][0] == point
    assert plan['radius'] == radius and plan['half_axis'] == axis
    assert plan['stations'] == 2 and plan['edges'] == 1


@pytest.mark.parametrize('fault', ['failed', 'empty', 'nonfinite', 'dimensions'])
def test_invalid_native_body_input_is_rejected(fault):
    quality = body_quality()
    route = quality['export_inspection']['collision']['inspection']['traversal']
    if fault == 'failed':
        route['passed'] = False
    if fault == 'empty':
        route['paths'] = []
    if fault == 'nonfinite':
        route['paths'][0]['center_path_m'][0][0] = float('nan')
    if fault == 'dimensions':
        route['width_m'] = 2.
    with pytest.raises(ValueError):
        runner.body_route_plan(quality, engine='unity')


@pytest.mark.parametrize('fault', ['missing', 'count', 'control', 'dimensions'])
def test_native_body_receipt_rejects_incomplete_engine_claims(fault):
    expected = dict(stations=2, edges=1, height_m=.5, width_m=.5, margin_m=.02)
    native = dict(body=dict(passed=True, overlap_control=True, sweep_control=True, **expected))
    if fault == 'missing':
        native.clear()
    if fault == 'count':
        native['body']['edges'] = 0
    if fault == 'control':
        native['body']['overlap_control'] = False
    if fault == 'dimensions':
        native['body']['width_m'] = .1
    with pytest.raises(ValueError, match='finite-body'):
        runner.validate_native_body(native, expected, 'unreal')


def test_nonfinite_native_body_dimensions_cannot_satisfy_tolerance():
    expected = dict(stations=2, edges=1, height_m=.5, width_m=.5, margin_m=.02)
    native = dict(body=dict(passed=True, overlap_control=True, sweep_control=True, **expected))
    native['body']['height_m'] = float('nan')
    with pytest.raises(ValueError, match='finite-body'):
        runner.validate_native_body(native, expected, 'unreal')


def test_nearly_white_surfaces_are_rejected_even_without_clipped_pixels(tmp_path):
    pixels = np.full((640, 960, 3), 240, dtype=np.uint8)
    pixels[:160] = 30  # visibly varied; no pixel reaches the 254 clipping cutoff
    Image.fromarray(pixels).save(tmp_path / 'interior_1.png')
    with pytest.raises(ValueError, match='bright surfaces hide detail'):
        runner.validate_captures(tmp_path)


@pytest.mark.parametrize('engine', ['unity', 'unreal'])
@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'always_clear', 'always_hit', 'forged_pass', 'malformed'])
def test_native_physics_controls_reject_broken_queries(engine, fault):
    rows = physics_controls()
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = rows[0]
    elif fault in ('always_clear', 'always_hit'):
        for row in rows:
            row['observed'] = fault == 'always_hit'
    elif fault == 'forged_pass':
        rows[0]['observed'] = False
    else:
        rows[0] = None
    key = 'physicsControls' if engine == 'unity' else 'physics_controls'
    with pytest.raises(ValueError, match='collision negative controls'):
        runner.validate_physics_controls({key: rows}, engine)


def test_complete_native_body_and_obstruction_controls_pass():
    expected = dict(stations=2, edges=1, height_m=.5, width_m=.5, margin_m=.02)
    native = dict(body=dict(passed=True, overlap_control=True, sweep_control=True, **expected),
                  physics_controls=physics_controls())
    assert runner.validate_native_body(native, expected, 'unreal') == dict(
        passed=True, physics_controls=14, **expected)
