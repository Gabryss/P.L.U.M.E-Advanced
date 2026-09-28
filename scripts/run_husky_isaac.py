#!/usr/bin/env python3
"""Drive the pinned imported Husky USD in Isaac Sim on floor or cave collision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path


def wrap(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q) -> float:
    w, x, y, z = map(float, q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def roll_pitch_of(q) -> tuple[float, float]:
    w, x, y, z = map(float, q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return roll, math.asin(max(-1., min(1., 2 * (w * y - z * x))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--cave-usd", type=Path)
    parser.add_argument("--floor-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sim-seconds", type=float, default=40.)
    args, _ = parser.parse_known_args()
    route = json.loads(args.route.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.robot_usd.open("rb") as stream:
        robot_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(schema="plume.husky-isaac-trial.v1", passed=False, status="starting",
                  route=route, floor_control=args.floor_control,
                  robot_usd=str(args.robot_usd.resolve()), robot_usd_sha256=robot_hash,
                  cave_usd=str(args.cave_usd.resolve()) if args.cave_usd else None)
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "multi_gpu": False})
        import isaacsim.core.experimental.utils.app as app_utils
        import numpy as np
        import omni.timeline
        import omni.usd
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, UsdGeom, UsdPhysics

        context = omni.usd.get_context()
        if args.floor_control:
            context.new_stage()
            stage = context.get_stage()
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            floor = UsdGeom.Cube.Define(stage, "/Trial/Floor")
            floor.CreateSizeAttr(1.)
            UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(200, 200, .1))
            UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
            UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        else:
            if not args.cave_usd or not context.open_stage(str(args.cave_usd.resolve())):
                raise ValueError("A readable cave USD is required")
            stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode((args.max_sim_seconds + 10.) * 120.)
        stage.SetTimeCodesPerSecond(120)
        start = route["start_xyz_m"]
        robot = UsdGeom.Xform.Define(stage, "/Trial/Husky")
        robot.GetPrim().GetReferences().AddReference(str(args.robot_usd.resolve()))
        robot.GetPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        pose = UsdGeom.Xformable(robot)
        pose.AddTranslateOp().Set(Gf.Vec3d(*start))
        pose.AddRotateZOp().Set(math.degrees(route["start_yaw_rad"]))
        joints = {}
        for place in ("front", "rear"):
            for side in ("left", "right"):
                name = f"{place}_{side}_wheel_joint"
                joint = stage.GetPrimAtPath(f"/Trial/Husky/Physics/{name}")
                if not joint.IsValid():
                    raise ValueError(f"Imported joint is missing: {name}")
                drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
                drive.CreateStiffnessAttr(0.)
                drive.CreateDampingAttr(50.)
                drive.CreateMaxForceAttr(150.)
                drive.CreateTargetVelocityAttr(0.)
                joints[name] = drive
        physics = UsdPhysics.Scene.Define(stage, "/Trial/PhysicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics.CreateGravityMagnitudeAttr(9.81)
        SimulationManager.switch_physics_engine("physx")
        SimulationManager.setup_simulation(dt=1/120., device="cpu")
        body = RigidPrim(paths="/Trial/Husky/Geometry/base_link")
        timeline = omni.timeline.get_timeline_interface()
        app_utils.play()
        trajectory = []
        waypoint = 0
        status = "timeout"
        frames = int(args.max_sim_seconds * 300)
        for frame in range(frames):
            app.update()
            if frame % 12:
                continue
            sim_time = timeline.get_current_time()
            if sim_time >= args.max_sim_seconds:
                break
            positions, orientations = body.get_world_poses()
            position = np.asarray(positions.numpy())[0].tolist()
            orient = np.asarray(orientations.numpy())[0].tolist()
            yaw = yaw_of(orient)
            roll, pitch = roll_pitch_of(orient)
            trajectory.append(dict(sim_time_s=sim_time, position=position,
                                   orientation_wxyz=orient, yaw=yaw,
                                   roll=roll, pitch=pitch))
            if position[2] < start[2] - .25:
                status = "fell_through_floor"
                break
            if max(abs(roll), abs(pitch)) > route.get("max_attitude_rad", .65):
                status = "attitude_limit"
                break
            goal = route["waypoints_xy_m"][waypoint]
            distance = math.hypot(goal[0] - position[0], goal[1] - position[1])
            if distance < route.get("goal_tolerance_m", .5):
                waypoint += 1
                if waypoint == len(route["waypoints_xy_m"]):
                    status = "completed"
                    break
                goal = route["waypoints_xy_m"][waypoint]
                distance = math.hypot(goal[0] - position[0], goal[1] - position[1])
            heading = wrap(math.atan2(goal[1] - position[1], goal[0] - position[0]) - yaw)
            linear = min(.25, .5 * distance) * max(0., 1 - abs(heading) / 1.2)
            angular = max(-.6, min(.6, 1.5 * heading))
            left = (linear - .555 * angular / 2) / .1651
            right = (linear + .555 * angular / 2) / .1651
            for name, drive in joints.items():
                drive.GetTargetVelocityAttr().Set(math.degrees(left if "left" in name else right))
        for drive in joints.values():
            drive.GetTargetVelocityAttr().Set(0.)
        app_utils.pause()
        report.update(status=status, passed=status == "completed",
                      waypoint_reached=waypoint, trajectory_samples=len(trajectory),
                      final_position=trajectory[-1]["position"] if trajectory else None,
                      sim_time_s=trajectory[-1]["sim_time_s"] if trajectory else None,
                      max_abs_roll_rad=max((abs(r["roll"]) for r in trajectory), default=None),
                      max_abs_pitch_rad=max((abs(r["pitch"]) for r in trajectory), default=None),
                      physics_engine=SimulationManager.get_active_physics_engine())
        (output / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if app:
            app.close(exit_code=0 if report["passed"] else 1)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
