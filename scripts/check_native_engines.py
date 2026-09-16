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
import trimesh
from PIL import Image

from plume_advanced.evaluation.native_views import plan_material_views
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


def body_route_plan(quality: dict, *, engine: str) -> dict:
    """Carry every accepted collider route, including placed heights, into native physics."""
    traversal = quality["export_inspection"]["collision"]["inspection"].get("traversal", {})
    if traversal.get("enabled") is not True or traversal.get("passed") is not True:
        raise ValueError("Native physics requires passing dedicated-collider capsule paths")
    height, width, margin = (traversal.get(key) for key in ("height_m", "width_m", "margin_m"))
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (height, width, margin))
            or not np.isfinite([height, width, margin]).all()
            or not height >= width > 0 or margin < 0):
        raise ValueError("Invalid native capsule dimensions")
    if engine == "unity":
        scale, axes, signs = 1., [0, 2, 1], [-1, 1, -1]
    elif engine == "unreal":
        scale, axes, signs = 100., [0, 1, 2], [1, -1, 1]
    else:
        raise ValueError("Unknown native engine")
    paths = []
    for row in traversal.get("paths", []):
        points = np.asarray(row.get("center_path_m", []), dtype=float)
        if (row.get("passed") is not True or points.ndim != 2 or points.shape[1] != 3
                or len(points) < 2 or not np.isfinite(points).all()):
            raise ValueError("Invalid or failed native capsule path")
        converted = points[:, axes] * signs * scale
        values = ([dict(zip(("x", "y", "z"), p, strict=True)) for p in converted]
                  if engine == "unity" else converted.tolist())
        paths.append(dict(segment_id=row["segment_id"], points=values))
    if not paths:
        raise ValueError("Native capsule paths are missing")
    return dict(radius=(width/2+margin)*scale, half_axis=(height-width)/2*scale,
                height_m=height, width_m=width, margin_m=margin, paths=paths,
                stations=sum(len(p["points"]) for p in paths),
                edges=sum(len(p["points"])-1 for p in paths))


def ground_route_plan(quality: dict, *, engine: str) -> dict:
    """Convert the collider's checked poses, floor probes and swept enclosures."""
    from plume_advanced.stages.ground_routes import GroundRobot
    ground = quality["export_inspection"]["collision"]["inspection"].get("ground_traversal")
    required = quality.get("acceptance", {}).get("policy", {}).get("require_ground_routes", False)
    if ground is None and not required:
        return dict(enabled=False)
    if not ground or ground.get("passed") is not True or not ground.get("paths"):
        raise ValueError("Native ground checks require passing dedicated-collider ground routes")
    robot = GroundRobot(**ground["robot"])
    if engine not in ("unity", "unreal"):
        raise ValueError("Unknown native engine")
    scale, axes, signs = ((1., [0, 2, 1], [-1, 1, -1]) if engine == "unity"
                          else (100., [0, 1, 2], [1, -1, 1]))
    def vector(value, *, direction=False):
        a = np.asarray(value, float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError("Nonfinite or malformed ground vector")
        a = a[axes] * signs * (1 if direction else scale)
        return dict(zip(("x", "y", "z"), a, strict=True)) if engine == "unity" else a.tolist()
    def half(value):
        a = np.asarray(value, float)
        if a.shape != (3,) or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError("Invalid ground box extents")
        a = a[[1, 2, 0]] if engine == "unity" else a
        return dict(zip(("x", "y", "z"), a, strict=True)) if engine == "unity" else (a*scale).tolist()
    offsets = np.asarray(ground["support_offsets_m"], float)
    weights = np.asarray(ground["support_fit_weights"], float)
    if (offsets.ndim != 2 or offsets.shape[1] != 2 or len(offsets) < 9
            or weights.shape != (3, len(offsets)) or not np.isfinite([*offsets.flat, *weights.flat]).all()
            or not np.allclose(weights @ np.column_stack([offsets, np.ones(len(offsets))]), np.eye(3), atol=1e-8)):
        raise ValueError("Invalid ground floor-support fitting plan")
    support = [dict(offset=dict(zip(("x", "y"), xy, strict=True)),
                    weight=dict(zip(("x", "y", "z"), w, strict=True)))
               for xy, w in zip(offsets, weights.T, strict=True)]
    poses, motions = [], []
    floor_samples = 0
    minimum_half = np.array([robot.length_m, robot.width_m, robot.height_m])/2 + robot.margin_m
    for path in [*ground["paths"], *ground.get("junctions", [])]:
        if not path.get("passed") or len(path.get("poses", [])) != path.get("samples", -1) or not path.get("sweeps"):
            raise ValueError("Missing/failed ground path witnesses")
        for pose in path["poses"]:
            probes = []
            if not pose.get("passed") or len(pose["floor_points_m"]) != len(offsets):
                raise ValueError("Missing ground floor probes")
            for p in pose["floor_points_m"]:
                distance = pose["probe_height_m"] - p[2]
                if not np.isfinite(distance) or distance <= 0:
                    raise ValueError("Invalid ground floor witness")
                probes.append(dict(point=vector([p[0], p[1], pose["probe_height_m"]]), floor=distance*scale))
            poses.append(dict(point=vector(pose["center_m"]), forward=vector(pose["forward"], direction=True),
                              up=vector(pose["up"], direction=True), probes=probes))
            floor_samples += len(probes)
        for motion in path["sweeps"]:
            if np.any(np.asarray(motion["half_extents_m"]) < minimum_half - 1e-9):
                raise ValueError("Ground sweep enclosure is smaller than the robot")
            motions.append(dict(start=vector(motion["start_m"]), end=vector(motion["end_m"]),
                forward=vector(motion["forward"], direction=True), up=vector(motion["up"], direction=True),
                half_extents=half(motion["half_extents_m"])))
    return dict(enabled=True, **ground["robot"], half_extents=half(minimum_half),
                stations=len(poses), sweeps=len(motions), floor_samples=floor_samples,
                poses=poses, motions=motions, support=support)


def native_export_paths(run: Path) -> tuple[Path, Path, Path]:
    """Locate the visual, its collider and the root used by serialization receipts."""
    run = run.resolve()
    directories = (run / "export", run / "export_blender", run / "export_all/blender")
    candidates = sorted(path for directory in directories for path in directory.glob("*.glb")
                        if path.is_file() and not path.stem.endswith("_collision"))
    if not candidates:
        raise FileNotFoundError(f"No native inspection GLB found under {run}")
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous native inspection exports: {candidates}")
    source = candidates[0]
    root = run / "export_all" if source.parent == directories[2] else source.parent
    collision = source.with_name(f"{source.stem}_collision.obj")
    return source, collision, root


def prepare(run: Path, output: Path, *, unity: bool, unreal: bool, view_spacing_m=30., max_views=256):
    """Reject unsuitable inputs before creating projects or running an editor."""
    run, output = run.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    quality_path = run / "pipeline_quality_report.json"
    quality = json.loads(quality_path.read_text())
    if not quality.get("passed"):
        raise ValueError("The input must have passed pipeline inspection")
    source, collision_source, export_root = native_export_paths(run)
    glb = GlbAsset(source)
    node, mesh, primitive = glb.cave_primitive()
    if len(glb.document["meshes"]) != 1 or len(mesh["primitives"]) != 1:
        raise ValueError("Native fixture supports one rock-free cave primitive")
    if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
        raise ValueError("Native fixture requires baked, untransformed GLB coordinates")
    samples = source_measurements(quality)
    section_meta = json.loads((run / "stage_c_sections.json").read_text())
    section_path = run / "stage_c_sections.npz"
    if section_meta["npz_sha256"] != sha256_file(section_path):
        raise ValueError("Section artifact changed before native view planning")
    with np.load(section_path, allow_pickle=False) as sections:
        plan = plan_material_views(sections["center_xyz_m"], sections["segment_id"],
                                   sections["arc_length_m"], sections["tangent"], samples,
                                   spacing_m=view_spacing_m, max_views=max_views)
    if any(
        not s["inside"]
        or not np.isfinite([*s["point_m"], s["floor_distance_m"], s["roof_distance_m"]]).all()
        for s in samples
    ):
        raise ValueError("All source clearance samples must be finite and inside the cave")
    positions = glb.accessor(primitive["attributes"]["POSITION"])
    triangles = len(glb.accessor(primitive["indices"])) // 3
    # Exporters have used both directory names; resolve by the unique settings receipt.
    candidates = list(export_root.rglob("settings.json"))
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
    collision_report = quality["export_inspection"]["collision"]
    if not collision_report.get("enabled") or not collision_report.get("inspection", {}).get("passed"):
        raise ValueError("Native inspection requires a passed dedicated collider")
    receipt = next((r for r in quality["export_inspection"]["serialized"]["files"]
                    if r["path"] == collision_source.relative_to(export_root).as_posix()), None)
    if receipt is None or not receipt["passed"] or receipt["sha256"] != sha256_file(collision_source):
        raise ValueError("Dedicated collider is missing its verified serialization receipt")
    collision = trimesh.load_mesh(collision_source, process=False)
    if not isinstance(collision, trimesh.Trimesh):
        raise ValueError("Expected one dedicated collision mesh")
    collision.vertices = np.asarray(collision.vertices)[:, [0, 2, 1]] * [1, 1, -1]
    output.mkdir(parents=True)
    # Import a textureless GLB made from the verified OBJ, keeping its topology.
    collision_glb = output / "plume_collision.glb"
    collision.export(collision_glb, file_type="glb")
    checked = GlbAsset(collision_glb)
    primitive_collision = checked.document["meshes"][0]["primitives"][0]
    np.testing.assert_array_equal(checked.accessor(primitive_collision["attributes"]["POSITION"]),
                                  np.asarray(collision.vertices, np.float32))
    np.testing.assert_array_equal(checked.accessor(primitive_collision["indices"]).reshape(-1, 3),
                                  collision.faces)
    collision_samples = collision_report["inspection"]["measurements"]
    bodies = {engine: body_route_plan(quality, engine=engine)
              for engine, enabled in (("unity", unity), ("unreal", unreal)) if enabled}
    grounds = {engine: ground_route_plan(quality, engine=engine) for engine in bodies}
    write_json(output / "view_plan.json", plan)
    write_projected_material_bundle(
        source,
        output / "native_material",
        tile_size_m=settings["tile_size_m"],
        normal_strength=settings["normal_strength"],
    )
    common = dict(triangles=triangles, vertices=len(positions), collision_triangles=len(collision.faces))
    write_json(
        output / "native_input_receipt.json",
        dict(
            source_glb=str(source),
            collision_glb=str(collision_glb),
            collision_glb_sha256=sha256_file(collision_glb),
            collision_obj_sha256=sha256_file(collision_source),
            glb_sha256=sha256_file(source),
            quality_sha256=sha256_file(quality_path),
            measurements_source="export_inspection.visual",
            adapter_source=package_source_hash(),
            samples=len(samples),
            views=plan["view_count"],
            body_routes={name: dict(stations=value["stations"], edges=value["edges"],
                                    height_m=value["height_m"], width_m=value["width_m"],
                                    margin_m=value["margin_m"]) for name, value in bodies.items()},
            ground_routes={name: {k: v for k, v in value.items()
                if k not in ("poses", "motions", "support", "half_extents")} for name, value in grounds.items()},
            view_plan_sha256=sha256_file(output / "view_plan.json"),
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
        shutil.copy2(collision_glb, project / "Assets/Cave/plume_collision.glb")
        shutil.copytree(output / "native_material", project / "Assets/PlumeMaterial")
        shutil.copy2(REPO / "tests/fixtures/unity/PlumeNativeCheck.cs", project / "Assets/Editor")
        write_json(project / "Packages/manifest.json", {"dependencies": UNITY_PACKAGES})
        (project / "ProjectSettings/ProjectVersion.txt").write_text("m_EditorVersion: 6000.6.0f1\n")
        write_json(
            project / "expected.json",
            dict(
                **common,
                body=bodies["unity"],
                ground=grounds["unity"],
                views=[dict(point=dict(zip(("x", "y", "z"), (-v["point_m"][0], v["point_m"][2], -v["point_m"][1]), strict=True)),
                            look=dict(zip(("x", "y", "z"), (-v["look_m"][0], v["look_m"][2], -v["look_m"][1]), strict=True))) for v in plan["views"]],
                collision_samples=[dict(point=dict(zip(("x", "y", "z"),
                    (-v["point_m"][0], v["point_m"][2], -v["point_m"][1]), strict=True)),
                    floor=v["floor_distance_m"], roof=v["roof_distance_m"]) for v in collision_samples],
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
                body=bodies["unreal"],
                ground=grounds["unreal"],
                bounds_min=ue_positions.min(axis=0).tolist(),
                bounds_max=ue_positions.max(axis=0).tolist(),
                views=[dict(point=[v["point_m"][0]*100, -v["point_m"][1]*100, v["point_m"][2]*100],
                            look=[v["look_m"][0]*100, -v["look_m"][1]*100, v["look_m"][2]*100]) for v in plan["views"]],
                collision_samples=[dict(point=[v["point_m"][0]*100, -v["point_m"][1]*100, v["point_m"][2]*100],
                    floor=v["floor_distance_m"]*100, roof=v["roof_distance_m"]*100) for v in collision_samples],
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
    plan_path = directory.parent / "view_plan.json"
    count = json.loads(plan_path.read_text())["view_count"] if plan_path.is_file() else 2
    for name in (f"interior_{i}.png" for i in range(1, count+1)):
        with Image.open(directory / name) as image:
            if image.size != (960, 640):
                raise ValueError(f"Unexpected native image dimensions: {name}")
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        mean, deviation = float(rgb.mean()), float(rgb.std())
        if deviation < 0.01 or not 0.005 < mean < 0.98:
            raise ValueError(f"Blank/unusable native capture: {directory / name}")
        clipped = float(np.all(rgb >= 254 / 255, axis=2).mean())
        if clipped > 0.01:
            raise ValueError(f"Overexposed native capture ({clipped:.1%} clipped): {directory / name}")
        luminance_p95 = float(np.quantile(rgb @ np.array([.2126, .7152, .0722]), .95))
        if luminance_p95 > .85:
            raise ValueError(f"Overexposed native capture (bright surfaces hide detail): {directory / name}")
        measurements[name] = dict(mean=mean, standard_deviation=deviation,
                                  clipped_fraction=clipped, luminance_p95=luminance_p95)
    return measurements


PHYSICS_CONTROLS = {
    **{f"{shape}_{name}": hit for shape in ("sphere", "capsule")
       for name, hit in (("initial_overlap", True), ("grazing_overlap", True),
                         ("clear_start", False), ("thin_wall_forward", True),
                         ("thin_wall_reverse", True), ("clear_sweep", False))},
    "capsule_axis_overlap": True, "sphere_axis_clear": False,
}

GROUND_CONTROLS = dict(box_initial_overlap=True, box_length_overlap=True, box_rotated_clear=False,
    box_clear_start=False, box_thin_wall_forward=True, box_thin_wall_reverse=True, box_clear_sweep=False,
    floor_present=True, floor_missing=False, slope_below_limit=True, slope_above_limit=False,
    step_below_limit=True, step_above_limit=False)


def validate_native_ground(native: dict, expected: dict) -> dict:
    if not expected.get("enabled"):
        return dict(enabled=False)
    ground = native.get("ground", {})
    if ground.get("enabled") is not True or ground.get("passed") is not True or ground.get("failures") != 0:
        raise ValueError("Missing or failed native ground robot evidence")
    for key, value in expected.items():
        if key in ("enabled", "stations", "sweeps", "floor_samples"):
            if ground.get(key) != value:
                raise ValueError("Incomplete native ground checks")
        elif key in ("length_m", "width_m", "height_m", "margin_m", "max_slope_deg", "max_step_m", "support_spacing_m"):
            number = ground.get(key)
            if type(number) not in (float, int) or not np.isfinite(number) or abs(number-value) > 1e-6:
                raise ValueError("Mismatched native ground robot limits")
    for key, limit in (("maximumFloorErrorM", .002), ("maximumSlopeDeg", expected['max_slope_deg']+1e-5),
                       ("maximumStepM", expected['max_step_m']+1e-5)):
        value = ground.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("Native floor support exceeds reference limits")
    rows = ground.get("controls", [])
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or len(rows) != len(GROUND_CONTROLS) or {row.get("name") for row in rows} != set(GROUND_CONTROLS)
            or any(row.get("passed") is not True or row.get("expected") is not GROUND_CONTROLS[row['name']]
                   or row.get("observed") is not GROUND_CONTROLS[row['name']] for row in rows)):
        raise ValueError("Missing or failed native ground controls")
    return dict(passed=True, **expected)


def validate_physics_controls(native: dict, engine: str) -> int:
    rows = native.get("physicsControls" if engine == "unity" else "physics_controls", [])
    if not isinstance(rows, list) or len(rows) != len(PHYSICS_CONTROLS):
        raise ValueError("Missing native collision negative controls")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed native collision negative controls")
        name = row.get("name")
        if (name not in PHYSICS_CONTROLS or name in seen
                or row.get("expected") is not PHYSICS_CONTROLS[name]
                or row.get("observed") is not PHYSICS_CONTROLS[name]
                or row.get("passed") is not True):
            raise ValueError("Failed native collision negative controls")
        seen.add(name)
    return len(rows)


def validate_native_body(native: dict, expected: dict, engine: str) -> dict:
    if engine == "unity":
        body = dict(passed=native.get("bodyPassed"), stations=native.get("bodyStations"),
                    edges=native.get("bodyEdges"), height_m=native.get("bodyHeightM"),
                    width_m=native.get("bodyWidthM"), margin_m=native.get("bodyMarginM"),
                    overlap_control=native.get("bodyOverlapControl"),
                    sweep_control=native.get("bodySweepControl"))
    else:
        body = native.get("body", {})
    if (body.get("passed") is not True or body.get("overlap_control") is not True
            or body.get("sweep_control") is not True
            or any(body.get(key) != expected[key] for key in ("stations", "edges"))
            or any(not isinstance(body.get(key), (float, int))
                   or not np.isfinite(body[key])
                   or abs(body[key]-expected[key]) > 1e-6
                   for key in ("height_m", "width_m", "margin_m"))):
        raise ValueError("Missing or mismatched native finite-body evidence")
    controls = validate_physics_controls(native, engine)
    return dict(passed=True, physics_controls=controls, **expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="Accepted attempt directory with export and quality report"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unity", type=Path, help="Unity 6000.6 editor executable")
    parser.add_argument("--unreal", type=Path, help="UnrealEditor 5.8 executable")
    parser.add_argument("--timeout", type=float, default=1800, help="Seconds per editor invocation")
    parser.add_argument("--view-spacing-m", type=float, default=30.)
    parser.add_argument("--max-views", type=int, default=256)
    args = parser.parse_args()
    if not (args.unity or args.unreal) or args.timeout <= 0:
        parser.error("Specify an editor and a positive timeout")
    if not np.isfinite(args.view_spacing_m) or args.view_spacing_m <= 0 or args.max_views < 2:
        parser.error("View spacing must be finite and positive; view budget must be at least two")
    for editor in (args.unity, args.unreal):
        if editor and not editor.is_file():
            parser.error(f"Editor does not exist: {editor}")
    root = args.output.resolve()
    prepare(args.run, root, unity=bool(args.unity), unreal=bool(args.unreal),
            view_spacing_m=args.view_spacing_m, max_views=args.max_views)
    receipt = json.loads((root / "native_input_receipt.json").read_text())
    report: dict = dict(
        passed=False,
        checks={},
        scope="One accepted 4K rock-free cave; native materials, collider overlaps and bidirectional body sweeps",
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
            body_validation = validate_native_body(native, receipt["body_routes"][engine], engine)
            ground_validation = validate_native_ground(native, receipt.get("ground_routes", {}).get(engine, {}))
            report["checks"][engine] = dict(native=native, body_validation=body_validation,
                                            ground_validation=ground_validation,
                                            images=validate_captures(results))
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
