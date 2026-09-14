#!/usr/bin/env python3
"""Run with Blender --background --python this_file.py -- RUN_DIRECTORY.

Import the completed portable scene, verify scale/counts, and save a native
inspection scene with an overview and interior cameras. No geometry is added.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def configure_inspection_view(*, textured: bool, preview_samples: int = 64) -> str:
    """Make the saved startup view show the material and distinguish neutral files."""
    scene = bpy.context.scene
    scene.cycles.preview_samples = preview_samples
    scene.cycles.use_preview_denoising = True
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.clip_start = 0.02
                area.spaces.active.clip_end = 20000.0
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                if screen.name == "Layout":
                    # Rendered shading resets to Solid when Blender reopens a file.
                    # Material Preview persists and displays packed maps immediately.
                    area.spaces.active.shading.type = "MATERIAL" if textured else "SOLID"
    return "plume_textured_inspection.blend" if textured else "plume_full_inspection.blend"


def choose_inspection_target(cave, sections, index, eye):
    """Aim along sampled passage geometry and verify the sight line stays in air."""
    ids = np.flatnonzero(sections["segment_id"] == sections["segment_id"][index])
    arc = sections["arc_length_m"]
    inverse = cave.matrix_world.inverted()
    for sign in (1., -1.):
        for reach in (6., 4., 2.):
            candidate = int(ids[np.argmin(abs(arc[ids] - (arc[index] + sign * reach)))])
            target = sections["center_xyz_m"][candidate].copy()
            direction = Vector(target) - Vector(eye)
            distance = direction.length
            if candidate == index or distance < 1.:
                continue
            hit, location, _, _ = cave.ray_cast(
                inverse @ Vector(eye),
                (inverse.to_3x3() @ direction).normalized(), distance=distance)
            if not hit or (cave.matrix_world @ location - Vector(eye)).length >= .98 * distance:
                return target
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--quality", choices=("preview", "standard", "high"), default="standard")
    parser.add_argument("--mapping", choices=("uv", "triplanar"), default="uv",
                        help="UV reproduces the portable GLB; triplanar uses the native continuous shader")
    parser.add_argument("--survey", action="store_true",
                        help="Measure roof/floor ray hits at every saved section centre")
    parser.add_argument("--interior-only", action="store_true",
                        help="Refresh interior previews; retain existing overview figures")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = args.run_directory.resolve()
    samples, threshold = {"preview": (32, 0.1), "standard": (256, 0.02), "high": (1024, 0.005)}[args.quality]
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
    # Imported GLB images must survive moving the native scene to another machine.
    expected_images = len(gltf.get("images", []))
    material_images = {
        node.image.name: node.image
        for material in cave.data.materials if material and material.use_nodes
        for node in material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image
    }
    assert len(material_images) == expected_images, "GLB material images were lost during import"
    material_bindings = {}
    imported_material = cave.data.materials[0]
    shader = next(node for node in imported_material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    gltf_material = gltf["materials"][primitive["material"]]
    pbr = gltf_material.get("pbrMetallicRoughness", {})
    for socket, expected in (("Base Color", "baseColorTexture" in pbr),
                             ("Roughness", "metallicRoughnessTexture" in pbr),
                             ("Normal", "normalTexture" in gltf_material)):
        material_bindings[socket] = shader.inputs[socket].is_linked
        assert not expected or material_bindings[socket], f"Unconnected material input: {socket}"
    bpy.ops.file.pack_all()
    assert all(image.packed_file for image in material_images.values())
    if args.mapping == "triplanar":
        helper = Path(__file__).resolve().parents[1] / "src/plume_advanced/material_assets/blender_materials.py"
        spec = importlib.util.spec_from_file_location("plume_blender_materials", helper)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load material helper: {helper}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata = json.loads(source.with_suffix(".manifest.json").read_text())["cave"]["material"]
        cave.data.materials[0] = module.build_triplanar_material(
            imported_material, tile_size_m=metadata["uv_scale_m"],
            normal_strength=metadata["normal_scale"],
        )
    sections = np.load(root / "stage_c_sections.npz")
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.adaptive_threshold = threshold
    scene.cycles.use_denoising = True
    scene.view_settings.exposure = 3.0
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
            light_data.energy = 600.0
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

    if args.survey:
        rows = []
        for index, eye in enumerate(sections["center_xyz_m"]):
            hit_up, roof, normal_up = vertical_hit(eye, 1)
            hit_down, floor, normal_down = vertical_hit(eye, -1)
            inside = bool(hit_up and hit_down and normal_up.z < 0 and normal_down.z > 0
                          and roof.z > eye[2] > floor.z)
            rows.append(dict(sample_index=index, segment_id=int(sections["segment_id"][index]),
                             inside=inside, input_height_m=float(sections["height_m"][index]),
                             mesh_vertical_clearance_m=float(roof.z-floor.z) if inside else None))
        (root/"blender_section_survey.json").write_text(json.dumps(dict(
            passed=all(row["inside"] for row in rows), samples=len(rows), rows=rows,
            scope="Vertical roof/floor rays at every saved section centre on the imported smoothed mesh; not a continuous traversability proof.",
        ), indent=2)+"\n")

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
            target = choose_inspection_target(cave, sections, index, eye)
            if target is None:
                continue
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
    native = output / configure_inspection_view(textured=expected_images > 0)
    if args.mapping == "triplanar":
        native = output / "plume_continuous_inspection.blend"
    previews = root / "previews"
    if args.mapping == "triplanar":
        previews = previews / "continuous"
    previews.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(previews / "inspection_render.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(native))
    report = {
        "blender_version": bpy.app.version_string,
        "source_asset": str(source.relative_to(root)),
        "native_scene": str(native.relative_to(root)),
        "material_mapping": args.mapping,
        "material_matches_portable_glb": args.mapping == "uv",
        "mesh_objects": [obj.name for obj in meshes],
        "vertices": len(cave.data.vertices),
        "triangles": actual_triangles,
        "triangle_count_preserved": True,
        "embedded_images_expected": expected_images,
        "material_inputs_linked": material_bindings,
        "material_images": [
            {"name": image.name, "size": list(image.size),
             "color_space": image.colorspace_settings.name, "packed": bool(image.packed_file)}
            for image in material_images.values()
        ],
        "bounds_preserved_within_2_mm": True,
        "bounds_blender_z_up_m": [minimum.tolist(), maximum.tolist()],
        "scale": list(cave.scale),
        "inspection_torch_power_w": 600.0,
        "render_quality": args.quality,
        "interior_exposure_ev": 3.0,
        "render_samples": samples,
        "render_noise_threshold": threshold,
        "cameras": cameras,
        "initial_camera": scene.camera.name,
        "startup_shading": "MATERIAL" if expected_images else "SOLID",
        "scope": "Actual Blender import and two interior roof/floor ray checks; no Unity or UE import test.",
    }
    report_name = "blender_continuous_check.json" if args.mapping == "triplanar" else "blender_import_check.json"
    (root / report_name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    # UV previews reproduce the GLB; native projected previews have their own path.
    for camera in (interiors if args.interior_only else (overview, plan, *interiors)):
        scene.view_settings.exposure = 0.0 if camera in (overview, plan) else 3.0
        scene.camera = camera
        sun.hide_render = camera not in (overview, plan)
        scene.render.filepath = str(previews / (camera.name.lower().replace(" ", "_") + ".png"))
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
