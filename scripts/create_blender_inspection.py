#!/usr/bin/env python3
"""Run with Blender --background --python this_file.py -- RUN_DIRECTORY.

Import the completed portable scene, verify scale/counts, and save a native
inspection scene with an overview and interior cameras. No geometry is added.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def main() -> None:
    root = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output = root / "export_blender"
    source = next(output.glob("*.glb"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    assert len(meshes) == 1 and meshes[0].name == "cave_wall", [m.name for m in meshes]
    cave = meshes[0]
    bounds = np.array([tuple(cave.matrix_world @ Vector(corner)) for corner in cave.bound_box])
    minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
    with source.open("rb") as file:
        file.read(12)
        json_length, _ = struct.unpack("<II", file.read(8))
        gltf = json.loads(file.read(json_length))
    primitive = gltf["meshes"][0]["primitives"][0]
    position_accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
    expected_bounds = np.array([position_accessor["min"], position_accessor["max"]])
    expected_bounds = expected_bounds[:, [0, 2, 1]] * np.array([1, -1, 1])
    assert np.allclose([minimum, maximum],
                       [expected_bounds.min(axis=0), expected_bounds.max(axis=0)], atol=0.002, rtol=0.0)
    expected_triangles = gltf["accessors"][primitive["indices"]]["count"] // 3
    actual_triangles = sum(len(poly.vertices) - 2 for poly in cave.data.polygons)
    assert actual_triangles == expected_triangles
    sections = np.load(root / "stage_c_sections.npz")
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("Inspection ambient")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    cameras = []

    def add_camera(name, eye, target, *, orthographic=False):
        data = bpy.data.cameras.new(name)
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = Vector(eye)
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
        data.clip_start = 0.02
        data.clip_end = 20000.0
        data.lens = 24.0
        if orthographic:
            data.type = "ORTHO"
            data.ortho_scale = float(max(maximum - minimum) * 1.17
                                    * max(scene.render.resolution_x / scene.render.resolution_y, 1.0))
        else:
            light_data = bpy.data.lights.new(name + " torch", "POINT")
            light_data.energy = 60.0
            light_data.shadow_soft_size = 0.06
            light = bpy.data.objects.new(name + " torch", light_data)
            scene.collection.objects.link(light)
            light.parent = obj
            light.location = (0.0, 0.0, 0.03)
        cameras.append({"name": name, "eye_blender_m": list(eye), "target_blender_m": list(target)})
        return obj

    center = (minimum + maximum) * 0.5
    span = float(max(maximum - minimum))
    overview = add_camera("Overview", center + np.array([span * 0.23, -span * 0.18, span]),
                          center, orthographic=True)
    # Orient the plan along the gallery, so a compact long passage fills a
    # landscape image instead of appearing as a narrow vertical strip.
    plan = add_camera("Top down", center + np.array([0.0, 0.0, span]), center,
                      orthographic=True)
    _, directions = np.linalg.eigh(np.cov(sections["center_xyz_m"][:, :2].T))
    right = Vector((*directions[:, -1], 0.0)).normalized()
    up = Vector((0.0, 0.0, 1.0)).cross(right)
    plan.rotation_euler = Matrix((right, up, Vector((0.0, 0.0, 1.0)))).transposed().to_euler()
    plan.data.ortho_scale = span * 1.15
    sun_data = bpy.data.lights.new("Overview sun", "SUN")
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("Overview sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.35, -0.4, -0.35)

    # Select from actual sampled reaches; verify both roof and floor ray hits
    # before accepting a viewpoint, since relief can change profile clearance.
    eligible = np.flatnonzero((sections["height_m"] > 1.2)
                              & np.isfinite(sections["center_xyz_m"]).all(axis=1))
    # Prefer ordinary reaches, but compact galleries can have overlapping
    # junction influence everywhere. Actual mesh ray hits decide validity.
    ordinary = eligible[sections["junction_influence"][eligible] < 0.1]
    eligible = np.r_[ordinary, eligible[~np.isin(eligible, ordinary)]]
    interiors = []
    accepted_indices = []
    inverse_world = cave.matrix_world.inverted()

    def vertical_hit(eye, sign):
        direction = inverse_world.to_3x3() @ Vector((0, 0, sign))
        hit, location, normal, _ = cave.ray_cast(inverse_world @ Vector(eye), direction.normalized(),
                                                distance=15.0)
        return hit, cave.matrix_world @ location, (inverse_world.transposed().to_3x3() @ normal).normalized()

    for fraction in (0.25, 0.70):
        desired = int(len(eligible) * fraction)
        for index in eligible[np.argsort(np.abs(np.arange(len(eligible)) - desired), kind="stable")]:
            index = int(index)
            eye = sections["center_xyz_m"][index].copy()
            if any(np.linalg.norm(eye - sections["center_xyz_m"][i]) < min(80.0, 0.2 * span)
                   for i in accepted_indices):
                continue
            eye[2] = 0.5 * (sections["floor_world_z"][index] + sections["roof_world_z"][index])
            hit_up, roof, normal_up = vertical_hit(eye, 1)
            hit_down, floor, normal_down = vertical_hit(eye, -1)
            if not (hit_up and hit_down and normal_up.z < 0 and normal_down.z > 0
                    and roof.z - floor.z > 1.2):
                continue
            eye[2] = floor.z + (roof.z - floor.z) * 0.48
            target = eye + sections["tangent"][index] * 6.0
            cam = add_camera(f"Interior {len(interiors) + 1}", eye, target)
            interiors.append(cam)
            accepted_indices.append(index)
            cameras[-1].update(section_index=index, segment_id=int(sections["segment_id"][index]),
                               measured_vertical_clearance_m=float(roof.z - floor.z))
            break
    assert len(interiors) == 2, "Could not locate two verified interior viewpoints"
    interior_records = [record for record in cameras if "measured_vertical_clearance_m" in record]
    scene.camera = interiors[max(range(len(interiors)),
                                 key=lambda i: interior_records[i]["measured_vertical_clearance_m"])]
    sun.hide_render = True
    for obj in scene.objects:
        obj.select_set(False)
    cave.select_set(True)
    bpy.context.view_layer.objects.active = cave
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.clip_start = 0.02
                area.spaces.active.clip_end = 20000.0
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    native = output / "plume_full_inspection.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(native))
    report = {
        "blender_version": bpy.app.version_string,
        "source_asset": str(source.relative_to(root)),
        "native_scene": str(native.relative_to(root)),
        "mesh_objects": [obj.name for obj in meshes],
        "vertices": len(cave.data.vertices),
        "triangles": actual_triangles,
        "triangle_count_preserved": True,
        "bounds_preserved_within_2_mm": True,
        "bounds_blender_z_up_m": [minimum.tolist(), maximum.tolist()],
        "scale": list(cave.scale),
        "cameras": cameras,
        "initial_camera": scene.camera.name,
        "scope": "Actual Blender import and two interior roof/floor ray checks; no Unity or UE import test.",
    }
    (root / "blender_import_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    # Save honest images of the imported file, using the same geometry and
    # neutral material as delivered to the other applications.
    previews = root / "previews"
    previews.mkdir(exist_ok=True)
    for camera in (overview, plan, *interiors):
        scene.camera = camera
        sun.hide_render = camera not in (overview, plan)
        scene.render.filepath = str(previews / (camera.name.lower().replace(" ", "_") + ".png"))
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
