#!/usr/bin/env python3
"""Drive a pinned Husky A200 SDF along a frozen route in Gazebo Harmonic.

The input route JSON is written before the trial. This script publishes wheel
commands, observes the model pose and retains the full trajectory and log.
It can also run an open-floor control with ``--floor-control``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def body_angles(q) -> tuple[float, float]:
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z),
                      1 - 2 * (q.x * q.x + q.y * q.y))
    sine = max(-1., min(1., 2 * (q.w * q.y - q.z * q.x)))
    return roll, math.asin(sine)


def create_world(robot_sdf: Path, package: Path | None, route: dict,
                 floor_control: bool) -> str:
    source = ET.parse(robot_sdf).getroot().find("model")
    if source is None:
        raise ValueError("Husky SDF has no model")
    source.set("name", "husky_a200")
    x, y, z = route["start_xyz_m"]
    yaw = route["start_yaw_rad"]
    ET.SubElement(source, "pose").text = f"{x} {y} {z} 0 0 {yaw}"
    drive = ET.SubElement(source, "plugin", {
        "filename": "gz-sim-diff-drive-system",
        "name": "gz::sim::systems::DiffDrive",
    })
    for side in ("left", "right"):
        for place in ("front", "rear"):
            ET.SubElement(drive, f"{side}_joint").text = f"{place}_{side}_wheel_joint"
    ET.SubElement(drive, "wheel_separation").text = "0.555"
    ET.SubElement(drive, "wheel_radius").text = "0.1651"
    ET.SubElement(drive, "topic").text = "/model/husky_a200/cmd_vel"
    ET.SubElement(drive, "odom_publish_frequency").text = "20"
    poses = ET.SubElement(source, "plugin", {
        "filename": "gz-sim-pose-publisher-system",
        "name": "gz::sim::systems::PosePublisher",
    })
    for key, value in (("publish_model_pose", "true"), ("publish_link_pose", "false"),
                       ("use_pose_vector_msg", "true"), ("update_frequency", "20")):
        ET.SubElement(poses, key).text = value
    world = ET.Element("world", name="husky_trial")
    for filename, name in (("physics", "Physics"),
                           ("scene-broadcaster", "SceneBroadcaster"),
                           ("user-commands", "UserCommands"), ("contact", "Contact")):
        ET.SubElement(world, "plugin", {
            "filename": f"gz-sim-{filename}-system", "name": f"gz::sim::systems::{name}",
        })
    physics = ET.SubElement(world, "physics", name="physics", type="ignored")
    ET.SubElement(physics, "max_step_size").text = "0.005"
    ET.SubElement(physics, "real_time_factor").text = "1"
    ET.SubElement(world, "gravity").text = "0 0 -9.81"
    if floor_control:
        floor = ET.SubElement(world, "model", name="control_floor")
        ET.SubElement(floor, "static").text = "true"
        link = ET.SubElement(floor, "link", name="ground")
        for kind in ("collision", "visual"):
            item = ET.SubElement(link, kind, name=f"ground_{kind}")
            ET.SubElement(item, "pose").text = "0 0 -0.05 0 0 0"
            ET.SubElement(ET.SubElement(item, "geometry"), "box").append(
                ET.fromstring("<size>200 200 0.1</size>"))
    else:
        if package is None:
            raise ValueError("Cave package is required outside the floor control")
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "uri").text = package.resolve().as_uri()
    world.append(source)
    root = ET.Element("sdf", version="1.11")
    root.append(world)
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-sdf", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--floor-control", action="store_true")
    parser.add_argument("--floor-profile", type=Path,
                        help="Frozen sampled floor elevations for slope-aware fall detection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-timeout", type=float, default=180.)
    args = parser.parse_args()
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.twist_pb2 import Twist
    from gz.transport13 import Node

    route = json.loads(args.route.read_text())
    if not route.get("waypoints_xy_m") or len(route["start_xyz_m"]) != 3:
        raise ValueError("Route requires start_xyz_m and waypoints_xy_m")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    floor_profile = json.loads(args.floor_profile.read_text()) if args.floor_profile else None
    if floor_profile is not None:
        floor_points = floor_profile["points"]
        if not floor_points:
            raise ValueError("Floor profile has no points")
    world = output / "husky_trial.world.sdf"
    world.write_text(create_world(args.robot_sdf, args.package, route, args.floor_control))
    observed: dict = {"pose": None, "sim_time_s": None}
    trajectory = []

    def on_pose(message):
        for pose in message.pose:
            if pose.name == "husky_a200":
                roll, pitch = body_angles(pose.orientation)
                observed.update(pose=[pose.position.x, pose.position.y, pose.position.z],
                                yaw=yaw_of(pose.orientation), roll=roll, pitch=pitch,
                                sim_time_s=observed["sim_time_s"])
                break

    def on_clock(message):
        observed["sim_time_s"] = message.sim.sec + message.sim.nsec * 1e-9

    node = Node()
    node.subscribe(Pose_V, "/model/husky_a200/pose", on_pose)
    node.subscribe(Clock, "/world/husky_trial/clock", on_clock)
    publisher = node.advertise("/model/husky_a200/cmd_vel", Twist)
    environment = os.environ.copy()
    resources = [str(args.robot_sdf.resolve().parent / "robot_prefix" / "share")]
    if args.package:
        resources.append(str(args.package.resolve().parent))
    if environment.get("GZ_SIM_RESOURCE_PATH"):
        resources.append(environment["GZ_SIM_RESOURCE_PATH"])
    environment["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(resources)
    command = ["gz", "sim", "-s", "-r", "-v", "3", str(world)]
    start = time.monotonic()
    waypoint = 0
    status = "timeout"
    with (output / "gazebo.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            while time.monotonic() - start < args.wall_timeout:
                if process.poll() is not None:
                    status = "simulator_exit"
                    break
                if observed["pose"] is not None:
                    pose = observed["pose"]
                    goal = route["waypoints_xy_m"][waypoint]
                    distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
                    trajectory.append({k: observed[k] for k in
                                       ("sim_time_s", "pose", "yaw", "roll", "pitch")})
                    if floor_profile is not None:
                        nearest = min(floor_points,
                                      key=lambda row: (row[0]-pose[0])**2 +
                                                      (row[1]-pose[1])**2)
                        floor_lost = pose[2] < nearest[2] - .05
                    else:
                        floor_lost = pose[2] < route["start_xyz_m"][2] - .25
                    if floor_lost:
                        status = "fell_through_floor"
                        break
                    if abs(observed["roll"]) > route.get("max_attitude_rad", .65) or abs(observed["pitch"]) > route.get("max_attitude_rad", .65):
                        status = "attitude_limit"
                        break
                    if distance < route.get("goal_tolerance_m", .6):
                        waypoint += 1
                        if waypoint == len(route["waypoints_xy_m"]):
                            status = "completed"
                            break
                        goal = route["waypoints_xy_m"][waypoint]
                        distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
                    heading = wrap(math.atan2(goal[1] - pose[1], goal[0] - pose[0]) - observed["yaw"])
                    linear = min(.25, .5 * distance) * max(0., 1 - abs(heading) / 1.2)
                    angular = max(-.6, min(.6, 1.5 * heading))
                    command_msg = Twist()
                    command_msg.linear.x = linear
                    command_msg.angular.z = angular
                    publisher.publish(command_msg)
                time.sleep(.1)
        finally:
            publisher.publish(Twist())
            natural_exit = process.poll()
            if natural_exit is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if trajectory:
        motion = sum(math.dist(a["pose"][:2], b["pose"][:2])
                     for a, b in zip(trajectory, trajectory[1:]))
    else:
        motion = 0.
    report = {
        "schema": "plume.husky-gazebo-trial.v1",
        "scope": "Pinned Husky A200 dynamic route trial in Gazebo; not field validation.",
        "route": route, "floor_control": args.floor_control,
        "status": status, "passed": status == "completed", "waypoints_reached": waypoint,
        "trajectory_samples": len(trajectory), "traveled_m": motion,
        "final_pose": observed["pose"], "sim_time_s": observed["sim_time_s"],
        "wall_time_s": time.monotonic() - start,
        "robot_sdf": str(args.robot_sdf.resolve()),
        "robot_sdf_sha256": hashlib.sha256(args.robot_sdf.read_bytes()).hexdigest(),
        "cave_package": str(args.package.resolve()) if args.package else None,
        "cave_model_sha256": hashlib.sha256((args.package / "model.sdf").read_bytes()).hexdigest()
            if args.package else None,
        "floor_profile": str(args.floor_profile.resolve()) if args.floor_profile else None,
        "floor_profile_sha256": hashlib.sha256(args.floor_profile.read_bytes()).hexdigest()
            if args.floor_profile else None,
        "floor_failure_rule": "robot_origin_below_nearest_frozen_floor_minus_0.05m"
            if floor_profile is not None else "robot_origin_below_start_minus_0.25m",
        "simulator_returncode": natural_exit,
    }
    (output / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
