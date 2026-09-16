#!/usr/bin/env python3
"""Render an exported cave and drop a contact probe in native Gazebo Harmonic.

Run with the system Python providing gz.transport13, gz.msgs10 and Pillow.
This is an import smoke check, not a robot traversability qualification.
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
from xml.sax.saxutils import escape


def texture_receipts(package: Path) -> list[dict]:
    """Require all three packaged PBR maps for this textured import check."""
    package = package.resolve()
    root = ET.parse(package / "model.sdf").getroot()
    receipts = []
    for role in ("albedo_map", "normal_map", "roughness_map"):
        for element in root.findall(f".//pbr/metal/{role}"):
            uri = element.text or ""
            prefix = f"model://{package.name}/"
            if not uri.startswith(prefix):
                raise ValueError(f"Expected a packaged texture URI: {uri}")
            source = (package / uri[len(prefix):]).resolve()
            if not source.is_relative_to(package) or not source.is_file():
                raise ValueError(f"Missing or external texture: {uri}")
            receipts.append({"role": role, "uri": uri,
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    if {r["role"] for r in receipts} != {"albedo_map", "normal_map", "roughness_map"}:
        raise ValueError("This textured smoke check requires albedo, normal and roughness maps")
    return receipts


def inspection_world(package: Path, view: dict) -> str:
    """Reference the unmodified exported model and add inspection instruments."""
    x, y, z = view["position"]
    dx, dy, dz = view["direction"]
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))
    floor = float(view["floor_z"])
    plugins = "".join(
        f'<plugin filename="gz-sim-{filename}-system" name="gz::sim::systems::{name}"/>'
        for filename, name in (
            ("physics", "Physics"), ("scene-broadcaster", "SceneBroadcaster"),
            ("contact", "Contact"), ("user-commands", "UserCommands"),
        )
    )
    return f'''<sdf version="1.10"><world name="plume_import_check">
      {plugins}
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
        <render_engine>ogre2</render_engine>
      </plugin>
      <physics name="physics" type="ignored"><max_step_size>0.002</max_step_size>
        <real_time_factor>1</real_time_factor></physics>
      <gravity>0 0 -9.81</gravity>
      <scene><ambient>0.12 0.12 0.12 1</ambient><background>0.02 0.02 0.02 1</background></scene>
      <include><uri>{escape(package.resolve().as_uri())}</uri></include>
      <light name="inspection_light" type="point">
        <pose>{x+dx} {y+dy} {z+0.2} 0 0 0</pose>
        <intensity>4</intensity><diffuse>1 0.96 0.9 1</diffuse><specular>0.3 0.3 0.3 1</specular>
        <attenuation><range>40</range><constant>1</constant><linear>0.01</linear>
          <quadratic>0.002</quadratic></attenuation><cast_shadows>true</cast_shadows>
      </light>
      <model name="inspection_camera"><static>true</static>
        <pose>{x} {y} {z} 0 {pitch} {yaw}</pose><link name="camera">
        <sensor name="interior" type="camera"><always_on>true</always_on>
          <update_rate>5</update_rate><topic>/plume/check/image</topic>
          <camera><horizontal_fov>1.3</horizontal_fov>
            <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
            <clip><near>0.05</near><far>500</far></clip>
          </camera></sensor></link></model>
      <model name="plume_probe"><pose>{x+3*dx} {y+3*dy} {floor+0.65} 0 0 0</pose>
        <link name="body"><inertial><mass>1</mass><inertia>
          <ixx>0.006667</ixx><iyy>0.006667</iyy><izz>0.006667</izz></inertia></inertial>
          <collision name="probe_collision"><geometry><box><size>0.2 0.2 0.2</size></box></geometry></collision>
          <visual name="probe_visual"><geometry><box><size>0.2 0.2 0.2</size></box></geometry>
            <material><diffuse>1 0.35 0.04 1</diffuse></material></visual>
          <sensor name="contact" type="contact"><always_on>true</always_on>
            <contact><collision>probe_collision</collision><topic>/plume/check/contacts</topic></contact>
          </sensor>
        </link>
        <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
          <publish_model_pose>true</publish_model_pose><publish_link_pose>false</publish_link_pose>
          <use_pose_vector_msg>true</use_pose_vector_msg><update_frequency>20</update_frequency>
        </plugin>
      </model>
    </world></sdf>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Directory containing model.sdf")
    parser.add_argument("--view", type=Path, required=True, help="position, direction, floor_z in metres/Z-up")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    # Import here so world construction can be unit tested without native Gazebo.
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.msgs10.image_pb2 import Image as GzImage
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node
    from PIL import Image, ImageStat

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text('{"passed": false, "status": "starting"}\n')
    (output / "interior.png").unlink(missing_ok=True)
    textures = texture_receipts(args.package)
    view = json.loads(args.view.read_text())
    world = output / "inspection.world.sdf"
    world.write_text(inspection_world(args.package, view))
    observed = {"images": 0, "contacts": 0, "cave_contacts": 0, "pose": None}

    def on_image(message):
        if message.width and message.height and message.step == message.width * 3:
            observed["frame"] = (message.width, message.height, bytes(message.data))
            observed["images"] += 1

    def on_contacts(message):
        for contact in message.contact:
            observed["contacts"] += 1
            if "cave" in contact.collision1.name or "cave" in contact.collision2.name:
                observed["cave_contacts"] += 1

    def on_pose(message):
        for pose in message.pose:
            if pose.name == "plume_probe":
                observed["pose"] = [pose.position.x, pose.position.y, pose.position.z]

    node = Node()
    node.subscribe(GzImage, "/plume/check/image", on_image)
    node.subscribe(Contacts, "/plume/check/contacts", on_contacts)
    node.subscribe(Pose_V, "/model/plume_probe/pose", on_pose)
    environment = os.environ.copy()
    environment["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(filter(None, (
        str(args.package.resolve().parent), environment.get("GZ_SIM_RESOURCE_PATH", ""),
    )))
    command = ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", str(world)]
    started = time.monotonic()
    with (output / "gazebo.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None and time.monotonic() - started < args.timeout:
                if observed["images"] >= 25 and observed["cave_contacts"] and observed["pose"]:
                    break
                time.sleep(0.2)
        finally:
            natural_exit = process.poll()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    frame = observed.pop("frame", None)
    if frame:
        width, height, data = frame
        image = Image.frombytes("RGB", (width, height), data)
        image.save(output / "interior.png")
        observed["image_channel_stddev"] = ImageStat.Stat(image).stddev
        observed["image_mean"] = sum(ImageStat.Stat(image).mean) / 3
        observed["image_sha256"] = hashlib.sha256((output / "interior.png").read_bytes()).hexdigest()
    pose = observed["pose"]
    observed["probe_near_floor"] = bool(pose and abs(pose[2] - view["floor_z"] - 0.1) < 0.5)
    observed["passed"] = bool(
        frame and observed["cave_contacts"] and observed["probe_near_floor"]
        and natural_exit in (None, 0) and 10 < observed["image_mean"] < 220
    )
    observed.update(
        schema="plume.gazebo-import-check.v1",
        simulator_version=subprocess.check_output(["gz", "sim", "--versions"], text=True).strip(),
        model_sdf_sha256=hashlib.sha256((args.package / "model.sdf").read_bytes()).hexdigest(),
        textures=textures,
        elapsed_seconds=round(time.monotonic()-started, 2),
        scope="Native textured camera capture and dynamic probe contact; not route or vehicle qualification.",
    )
    (output / "result.json").write_text(json.dumps(observed, indent=2)+"\n")
    print(json.dumps(observed, indent=2))
    raise SystemExit(0 if observed["passed"] else 1)


if __name__ == "__main__":
    main()
