"""Render the recorded Gazebo Husky poses in Isaac RTX for a moving GIF.

These are visual reconstructions at logged poses, not an Isaac dynamics replay.
The cave, boulder shifts, materials, camera, and light match the still plate.
"""

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--preview", action="store_true", help="Render three poses for composition review")
parser.add_argument("--updates", type=int, default=100)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
out = root / "outputs/showcase_gallery/husky_visualization"
frames_dir = out / ("animation_preview" if args.preview else "animation_frames")
frames_dir.mkdir(parents=True, exist_ok=True)
app = SimulationApp({
    "headless": False, "width": 1920, "height": 1080,
    "window_width": 1920, "window_height": 1080,
    "renderer": "RayTracedLighting", "multi_gpu": False,
})

import carb
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

trace = json.loads((root / "outputs/showcase_robotics/husky_showcase_rockfall/trajectory.json").read_text())
if args.preview:
    times = [trace[0]["sim_time_s"], 16.475, trace[-1]["sim_time_s"]]
else:
    # Briefly show the logged start, then devote frames to actual traversal.
    times = [trace[0]["sim_time_s"]] + [9.0 + (trace[-1]["sim_time_s"] - 9.0) * i / 23 for i in range(24)]
samples = [min(trace, key=lambda row: abs(row["sim_time_s"] - t)) for t in times]

stage_path = root / "outputs/showcase_export/omniverse/plume_cave.usd"
robot_path = root / "outputs/showcase_robotics/robot_sources/isaac_husky_merged/husky_a200/husky_a200.usda"
context = omni.usd.get_context()
assert context.open_stage(str(stage_path))
for _ in range(50):
    app.update()
stage = context.get_stage()
stage.SetEditTarget(stage.GetSessionLayer())

for prim in stage.Traverse():
    path = str(prim.GetPath())
    for label, dx in [("Event_0360_boulder", 1.1), ("Event_0361_boulder", -1.15)]:
        if path.endswith(label):
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(dx, 0, 0))

robot = UsdGeom.Xform.Define(stage, "/Inspection/Husky")
robot.GetPrim().GetReferences().AddReference(str(robot_path))
physics_variant = robot.GetPrim().GetVariantSets().GetVariantSet("Physics")
if physics_variant.IsValid():
    physics_variant.SetVariantSelection("none")
robot_xform = UsdGeom.Xformable(robot)
translation = robot_xform.AddTranslateOp()
rotation = robot_xform.AddRotateZOp()
robot.GetPrim().Load()
for _ in range(20):
    app.update()

tire_material = UsdShade.Material.Define(stage, "/Inspection/TireRubber")
tire_shader = UsdShade.Shader.Define(stage, "/Inspection/TireRubber/Shader")
tire_shader.CreateIdAttr("UsdPreviewSurface")
tire_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.035, .04, .04))
tire_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.88)
tire_material.CreateSurfaceOutput().ConnectToSource(tire_shader.ConnectableAPI(), "surface")
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/Inspection/Husky") and "wheel_link/outdoor" in path:
        prim.SetInstanceable(False)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/Inspection/Husky") and "wheel" in path and prim.IsA(UsdGeom.Mesh):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(tire_material)


def sphere(name, position, power, radius, color):
    light = UsdLux.SphereLight.Define(stage, "/Inspection/" + name)
    light.CreateIntensityAttr(power)
    light.CreateRadiusAttr(radius)
    light.CreateColorAttr(Gf.Vec3f(*color))
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))


sphere("NearNeutral", (4.61, -30.5, 164.95), 3000, .35, (1, .98, .96))
sphere("FrontFill", (4.01, -27.5, 164.95), 2500, .45, (1, .98, .96))
sphere("WarmDepth", (5.10, -17, 165.08), 20000, 1.1, (1, .88, .76))
sphere("MiddleRocks", (4.2, -22.5, 164.95), 2500, 1.0, (1, .95, .88))
sphere("ForegroundEdge", (5.45, -30, 164.92), 700, .9, (1, .98, .96))
sphere("RobotRim", (6.0, -26.2, 165.4), 4000, 1.2, (1, .95, .88))

settings = carb.settings.get_settings()
settings.set("/rtx/post/tonemap/op", 4)
settings.set("/rtx/post/tonemap/filmIso", 145.)
settings.set("/rtx/post/aa/op", 4)
viewport = get_active_viewport()
camera = UsdGeom.Camera.Define(stage, "/Inspection/Camera_animation")
eye, target = (6.7, -29.8, 165.1), (5.1, -25.2, 164.5)
matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
camera.CreateHorizontalApertureAttr(20.955)
camera.CreateFocalLengthAttr(16.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(.02, 1000))
viewport.set_active_camera("/Inspection/Camera_animation")

receipt = []
for i, sample in enumerate(samples):
    translation.Set(Gf.Vec3d(*sample["pose"]))
    rotation.Set(math.degrees(sample["yaw"]))
    for _ in range(300 if i == 0 else args.updates):
        app.update()
    path = frames_dir / f"frame_{i:03d}.png"
    capture_viewport_to_file(viewport, str(path))
    for _ in range(25):
        app.update()
    receipt.append({
        "frame": i, "sim_time_s": sample["sim_time_s"],
        "pose": sample["pose"], "yaw": sample["yaw"],
        "path": str(path.relative_to(root)),
    })
    print("HUSKY ANIMATION FRAME", i, sample["sim_time_s"], path, flush=True)

(frames_dir / "receipt.json").write_text(json.dumps({
    "scope": "Isaac RTX visual reconstructions of recorded Gazebo poses; no Isaac robot dynamics",
    "trajectory": "outputs/showcase_robotics/husky_showcase_rockfall/trajectory.json",
    "cave_usd": str(stage_path.relative_to(root)),
    "robot_usd": str(robot_path.relative_to(root)),
    "camera": {"eye": eye, "target": target, "focal_length": 16.0},
    "boulder_shifts_m": {"Event_0360_boulder": 1.1, "Event_0361_boulder": -1.15},
    "frames": receipt,
}, indent=2) + "\n")
app.close()
