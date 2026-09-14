"""Failure handling for the opt-in GPU editor checks; no editor is launched by pytest."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "native_check_runner", Path(__file__).resolve().parents[1] / "scripts/check_native_engines.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("color", [0, 128, 255])
def test_blank_native_captures_are_not_accepted(tmp_path, color):
    Image.new("RGB", (960, 640), (color, color, color)).save(tmp_path / "interior_1.png")
    with pytest.raises(ValueError, match="Blank/unusable"):
        runner.validate_captures(tmp_path)


def test_both_native_views_are_required(tmp_path):
    gradient = np.linspace(0, 255, 960, dtype=np.uint8)[None, :].repeat(640, axis=0)
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
        for folder, passed in (("unity_project", True), ("unreal_run_01", unreal_passed)):
            (root / folder).mkdir(parents=True)
            (root / folder / "native_result.json").write_text(json.dumps({"passed": passed}))

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
    gradient = np.linspace(0, 255, 960, dtype=np.uint8)[None, :].repeat(640, axis=0)
    image = Image.fromarray(gradient).convert("RGB")
    for i in range(1, 4):
        image.save(directory / f"interior_{i}.png")
    with pytest.raises(FileNotFoundError):
        runner.validate_captures(directory)
    image.save(directory / "interior_4.png")
    assert len(runner.validate_captures(directory)) == 4
