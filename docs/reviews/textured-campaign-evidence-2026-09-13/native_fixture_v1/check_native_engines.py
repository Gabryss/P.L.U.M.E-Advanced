#!/usr/bin/env python3
"""Validate an accepted, rock-free short cave in isolated Unity/Unreal projects.

This opt-in check requires installed/licensed editors, GPU access, and (on the
first Unity run) package-registry access. It never opens an existing user project.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from plume_advanced.exporters.projected_materials import write_projected_material_bundle
from plume_advanced.identity import package_source_hash, sha256_file
from plume_advanced.validation import GlbAsset

REPO = Path(__file__).resolve().parents[1]
UNITY_PACKAGES = {
    "com.unity.render-pipelines.universal": "17.6.0",
    "com.unity.cloud.gltfast": "6.20.0",
    "com.unity.modules.physics": "1.0.0",
    "com.unity.modules.imageconversion": "1.0.0",
    "com.unity.modules.screencapture": "1.0.0",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_measurements(quality: dict):
    """Native imports must match the final visual export, including surface processing."""
    exported = quality.get("export_inspection", {})
    visual = exported.get("visual", {})
    if not exported.get("passed") or not visual.get("passed"):
        raise ValueError("The input needs a passed final visual-export inspection")
    return visual["measurements"]


def prepare(run: Path, output: Path, *, unity: bool, unreal: bool):
    """Reject unsuitable inputs before creating projects or running an editor."""
    run, output = run.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    quality_path = run / "pipeline_quality_report.json"
    quality = json.loads(quality_path.read_text())
    if not quality.get("passed"):
        raise ValueError("The input must have passed pipeline inspection")
    source = run / "export/plume_cave.glb"
    glb = GlbAsset(source)
    node, mesh, primitive = glb.cave_primitive()
    if len(glb.document["meshes"]) != 1 or len(mesh["primitives"]) != 1:
        raise ValueError("Native fixture supports one rock-free cave primitive")
    if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
        raise ValueError("Native fixture requires baked, untransformed GLB coordinates")
    samples = source_measurements(quality)
    if len(samples) < 50:
        raise ValueError("Native inspection views require at least 50 passage samples")
    if any(
        not s["inside"]
        or not np.isfinite([*s["point_m"], s["floor_distance_m"], s["roof_distance_m"]]).all()
        for s in samples
    ):
        raise ValueError("All source clearance samples must be finite and inside the cave")
    positions = glb.accessor(primitive["attributes"]["POSITION"])
    triangles = len(glb.accessor(primitive["indices"])) // 3
    # Exporters have used both directory names; resolve by the unique settings receipt.
    candidates = list((run / "export").rglob("settings.json"))
    if len(candidates) != 1:
        raise ValueError("Expected one continuous-material settings receipt in the export")
    settings_path = candidates[0]
    settings = json.loads(settings_path.read_text())
    if settings["source_sha256"] != sha256_file(source):
        raise ValueError("Material receipt does not match the input GLB")
    if len(glb.document.get("images", [])) != 3:
        raise ValueError("Native fixture requires three PBR maps")
    if any(glb.embedded_image(i).size != (4096, 4096) for i in range(3)):
        raise ValueError("This native 4K inspection fixture requires three 4096 x 4096 maps")
    output.mkdir(parents=True)
    write_projected_material_bundle(
        source,
        output / "native_material",
        tile_size_m=settings["tile_size_m"],
        normal_strength=settings["normal_strength"],
    )
    common = dict(triangles=triangles, vertices=len(positions))
    write_json(
        output / "native_input_receipt.json",
        dict(
            source_glb=str(source),
            glb_sha256=sha256_file(source),
            quality_sha256=sha256_file(quality_path),
            measurements_source="export_inspection.visual",
            adapter_source=package_source_hash(),
            samples=len(samples),
            **common,
            fixture_sha256={
                name: sha256_file(REPO / "tests/fixtures" / name)
                for name in ("unity/PlumeNativeCheck.cs", "unreal/native_check.py")
            },
        ),
    )
    if unity:
        project = output / "unity_project"
        for folder in ("Assets/Cave", "Assets/Editor", "Packages", "ProjectSettings"):
            (project / folder).mkdir(parents=True)
        shutil.copy2(source, project / "Assets/Cave/plume_cave.glb")
        shutil.copytree(output / "native_material", project / "Assets/PlumeMaterial")
        shutil.copy2(REPO / "tests/fixtures/unity/PlumeNativeCheck.cs", project / "Assets/Editor")
        write_json(project / "Packages/manifest.json", {"dependencies": UNITY_PACKAGES})
        (project / "ProjectSettings/ProjectVersion.txt").write_text("m_EditorVersion: 6000.6.0f1\n")
        write_json(
            project / "expected.json",
            dict(
                **common,
                samples=[
                    dict(
                        point=dict(
                            zip(
                                ("x", "y", "z"),
                                (-s["point_m"][0], s["point_m"][2], -s["point_m"][1]),
                                strict=True,
                            )
                        ),
                        floor=s["floor_distance_m"],
                        roof=s["roof_distance_m"],
                    )
                    for s in samples
                ],
            ),
        )
        unity_positions = positions * np.array([-1, 1, 1], dtype=np.float32)
        order = np.lexsort((unity_positions[:, 2], unity_positions[:, 1], unity_positions[:, 0]))
        unity_positions[order].astype("<f4").tofile(project / "positions_unity.f32")
    if unreal:
        project = output / "unreal_project"
        (project / "Config").mkdir(parents=True)
        write_json(
            project / "PLUMENative.uproject",
            dict(
                FileVersion=3,
                EngineAssociation="5.8",
                Category="Validation",
                Plugins=[
                    dict(Name=name, Enabled=True)
                    for name in (
                        "PythonScriptPlugin",
                        "EditorScriptingUtilities",
                        "InterchangeEditor",
                    )
                ],
            ),
        )
        (project / "Config/DefaultEngine.ini").write_text(
            "[/Script/Engine.RendererSettings]\nr.RayTracing=False\nr.TextureStreaming=False\n"
            "r.DynamicGlobalIlluminationMethod=0\nr.ReflectionMethod=0\n"
            "[/Script/EngineSettings.GameMapsSettings]\n"
            "EditorStartupMap=/Game/PLUME_Run01/PLUME_Inspection\n"
            "[DevOptions.Shaders]\nNumUnusedShaderCompilingThreads=9\n"
            "NumUnusedShaderCompilingThreadsDuringGame=9\nShaderCompilerCoreCountThreshold=999\n"
        )
        ue_positions = positions[:, [0, 2, 1]].astype(np.float64) * 100
        write_json(
            output / "unreal_expected.json",
            dict(
                **common,
                bounds_min=ue_positions.min(axis=0).tolist(),
                bounds_max=ue_positions.max(axis=0).tolist(),
                samples=[
                    dict(
                        point=[
                            s["point_m"][0] * 100,
                            -s["point_m"][1] * 100,
                            s["point_m"][2] * 100,
                        ],
                        floor=s["floor_distance_m"] * 100,
                        roof=s["roof_distance_m"] * 100,
                    )
                    for s in samples
                ],
            ),
        )
        fixture = REPO / "tests/fixtures/unreal/native_check.py"
        (output / "ue_run01.py").write_text(
            f"import runpy\nrunpy.run_path({str(fixture)!r})['main']({str(output)!r}, '01')\n"
        )


def run_editor(command: list[str], log: Path, timeout: float):
    """Bound each editor invocation, retaining its complete native log."""
    print(f"Native check: {log.stem}; log: {log}", flush=True)
    start = time.monotonic()
    with log.open("w") as stream:
        with subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # These are isolated Linux editor runs. Stop their compiler workers
                # too, without touching any independently opened user editor.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
    if code:
        raise RuntimeError(f"Editor exited {code}; inspect {log}")
    return time.monotonic() - start


def validate_captures(directory: Path):
    """Reject missing or blank captures; visual correctness needs more than this guard."""
    measurements = {}
    for name in ("interior_1.png", "interior_2.png"):
        with Image.open(directory / name) as image:
            if image.size != (960, 640):
                raise ValueError(f"Unexpected native image dimensions: {name}")
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        mean, deviation = float(rgb.mean()), float(rgb.std())
        if deviation < 0.01 or not 0.005 < mean < 0.98:
            raise ValueError(f"Blank/unusable native capture: {directory / name}")
        measurements[name] = dict(mean=mean, standard_deviation=deviation)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="Accepted attempt directory with export and quality report"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unity", type=Path, help="Unity 6000.6 editor executable")
    parser.add_argument("--unreal", type=Path, help="UnrealEditor 5.8 executable")
    parser.add_argument("--timeout", type=float, default=1800, help="Seconds per editor invocation")
    args = parser.parse_args()
    if not (args.unity or args.unreal) or args.timeout <= 0:
        parser.error("Specify an editor and a positive timeout")
    for editor in (args.unity, args.unreal):
        if editor and not editor.is_file():
            parser.error(f"Editor does not exist: {editor}")
    root = args.output.resolve()
    prepare(args.run, root, unity=bool(args.unity), unreal=bool(args.unreal))
    report: dict = dict(
        passed=False,
        checks={},
        scope="One accepted 4K rock-free cave; native import/material inspection",
    )
    failures = []
    # Engines run sequentially to bound GPU memory; one failure does not hide the other result.
    for engine, binary in (("unity", args.unity), ("unreal", args.unreal)):
        if not binary:
            continue
        try:
            if engine == "unity":
                project = root / "unity_project"
                base = [
                    str(binary.resolve()),
                    "-batchmode",
                    "-force-vulkan",
                    "-projectPath",
                    str(project),
                ]
                run_editor(
                    base
                    + ["-quit", "-executeMethod", "PlumeNativeCheck.Bootstrap", "-logFile", "-"],
                    root / "unity_bootstrap.log",
                    args.timeout,
                )
                run_editor(
                    base + ["-executeMethod", "PlumeNativeCheck.Evaluate", "-logFile", "-"],
                    root / "unity_evaluation.log",
                    args.timeout,
                )
                results = project
            else:
                run_editor(
                    [
                        str(binary.resolve()),
                        str(root / "unreal_project/PLUMENative.uproject"),
                        "-unattended",
                        "-nop4",
                        "-nosound",
                        "-RenderOffscreen",
                        "-vulkan",
                        f"-ExecutePythonScript={root / 'ue_run01.py'}",
                        f"-abslog={root / 'unreal.log'}",
                    ],
                    root / "unreal_console.log",
                    args.timeout,
                )
                results = root / "unreal_run_01"
            native = json.loads((results / "native_result.json").read_text())
            if not native.get("passed"):
                raise ValueError(f"Native checks failed: {results / 'native_result.json'}")
            report["checks"][engine] = dict(native=native, images=validate_captures(results))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            failures.append(f"{engine}: {error}")
            report["checks"][engine] = dict(passed=False, failure=str(error))
        write_json(root / "native_summary.json", report)
    report.update(passed=not failures, failures=failures)
    write_json(root / "native_summary.json", report)
    print(f"Native checks {'FAILED' if failures else 'passed'}: {root / 'native_summary.json'}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
