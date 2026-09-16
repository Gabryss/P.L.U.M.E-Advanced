#!/usr/bin/env python3
"""Check an exported cave inside Isaac Sim 6.1 using its bundled python.sh.

Opens the USD, checks materials/units/static triangle collision, simulates a
falling probe and captures the RTX camera. Does not qualify robot routes.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "plume.isaac-import-check.v1", "passed": False,
        "asset_sha256": hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        "scope": "USD import, RTX texture capture and one dynamic collision probe; not robot qualification.",
    }
    app = None
    (output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({
            "headless": True, "width": 1280, "height": 720,
            "renderer": "RayTracedLighting", "multi_gpu": False,
        })
        print("PLUME: application ready; importing inspection APIs", flush=True)
        import os

        import carb
        import isaacsim.core.experimental.utils.app as app_utils
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade
        version = Path(os.environ["ISAAC_PATH"]) / "VERSION"
        report["simulator_version"] = version.read_text().strip()
        print("PLUME: opening exported USD", flush=True)
        context = omni.usd.get_context()
        if not context.open_stage(str(args.asset.resolve())):
            raise RuntimeError("Isaac Sim could not open the cave USD")
        print("PLUME: USD opened; settling stage", flush=True)
        for _ in range(20):
            app.update()
        print("PLUME: stage ready; inspecting assets", flush=True)
        stage = context.get_stage()
        # All instruments belong to the session layer; the exported USD is unchanged.
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(600)
        stage.SetTimeCodesPerSecond(60)
        report["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)
        report["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
        wall = stage.GetPrimAtPath("/PLUME_Cave/CaveWall")
        collider = stage.GetPrimAtPath("/PLUME_Cave/CaveCollision")
        report["visual_triangles"] = len(UsdGeom.Mesh(wall).GetFaceVertexCountsAttr().Get())
        report["collision_triangles"] = len(UsdGeom.Mesh(collider).GetFaceVertexCountsAttr().Get())
        report["collision_enabled"] = UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
        report["collision_approximation"] = str(UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get())
        report["static_collider"] = not collider.HasAPI(UsdPhysics.RigidBodyAPI)
        material, _ = UsdShade.MaterialBindingAPI(wall).ComputeBoundMaterial()
        report["bound_material"] = str(material.GetPath()) if material else None
        textures = []
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(prim)
                if shader.GetIdAttr().Get() == "UsdUVTexture":
                    asset = shader.GetInput("file").Get()
                    textures.append({"path": asset.path, "resolved": bool(asset.resolvedPath),
                                     "wrap_s": shader.GetInput("wrapS").Get(),
                                     "wrap_t": shader.GetInput("wrapT").Get(),
                                     "image_size": list(Image.open(asset.resolvedPath).size) if asset.resolvedPath else None})
        report["textures"] = textures

        view = json.loads(args.view.read_text())
        eye = Gf.Vec3d(*view["position"])
        direction = Gf.Vec3d(*view["direction"])
        camera = UsdGeom.Camera.Define(stage, "/Inspection/Camera")
        up = Gf.Vec3d(0, 0, 1) if abs(direction.GetNormalized()[2]) < 0.99 else Gf.Vec3d(0, 1, 0)
        matrix = Gf.Matrix4d().SetLookAt(eye, eye+direction, up).GetInverse()
        UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateFocalLengthAttr(13.8)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
        light = UsdLux.SphereLight.Define(stage, "/Inspection/Light")
        light.CreateIntensityAttr(18000)
        light.CreateRadiusAttr(0.15)
        UsdGeom.Xformable(light).AddTranslateOp().Set(eye + direction + Gf.Vec3d(0, 0, 0.15))
        settings = carb.settings.get_settings()
        settings.set("/rtx/post/tonemap/op", 4)
        settings.set("/rtx/post/tonemap/filmIso", 200.)

        sphere = UsdGeom.Cube.Define(stage, "/Inspection/Probe")
        sphere.CreateSizeAttr(0.2)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(1, 0.35, 0.04)])
        start = eye + 3*direction
        start[2] = view["floor_z"] + 0.65
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(start)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
        UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(1.)
        physics = UsdPhysics.Scene.Define(stage, "/Inspection/PhysicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics.CreateGravityMagnitudeAttr(9.81)
        SimulationManager.switch_physics_engine("physx")
        SimulationManager.setup_simulation(dt=1/120., device="cpu")
        probe = RigidPrim(paths="/Inspection/Probe")
        print("PLUME: simulating contact probe", flush=True)
        app_utils.play()
        for _ in range(120):
            app.update()
        positions, _ = probe.get_world_poses()
        position = np.asarray(positions.numpy())[0].tolist()
        report["probe_position"] = position
        report["probe_fell"] = position[2] < start[2]-0.2
        report["probe_near_floor"] = abs(position[2]-view["floor_z"]-0.1) < 0.5
        report["physics_engine"] = SimulationManager.get_active_physics_engine()
        app_utils.pause()

        print("PLUME: capturing RTX camera", flush=True)
        product = rep.create.render_product(str(camera.GetPath()), (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        for _ in range(200):
            app.update()
        report["lighting_trials"] = []
        for exposure in (0, 2, 4, 6, 8):
            # Adjust the test light in the simulator, never the captured pixels.
            light.CreateExposureAttr(exposure)
            for _ in range(10):
                app.update()
            rep.orchestrator.step(rt_subframes=8)
            pixels = np.asarray(rgb.get_data())[:, :, :3]
            report["image_mean"] = float(pixels.mean())
            report["image_stddev"] = float(pixels.std())
            report["lighting_trials"].append({"light_exposure_stops": exposure, "image_mean": report["image_mean"]})
            if 30 < report["image_mean"] < 200:
                break
        Image.fromarray(pixels).save(output / "interior.png")
        report["image_sha256"] = hashlib.sha256((output / "interior.png").read_bytes()).hexdigest()
        stage.Flatten().Export(str(output / "inspection.usda"))
        report["passed"] = bool(
            report["meters_per_unit"] == 1 and report["up_axis"] == "Z"
            and report["bound_material"] and len(textures) == 3
            and all(t["resolved"] and t["wrap_s"] == t["wrap_t"] == "repeat" for t in textures)
            and report["collision_enabled"] and report["static_collider"]
            and report["collision_approximation"] == "none"
            and report["probe_fell"] and report["probe_near_floor"]
            and 10 < report["image_mean"] < 220 and report["image_stddev"] > 5
        )
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
        faulthandler.cancel_dump_traceback_later()
        if app:
            app.close(exit_code=0 if report["passed"] else 1)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
