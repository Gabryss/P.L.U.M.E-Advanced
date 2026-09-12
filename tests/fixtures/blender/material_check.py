"""Native Blender material regression fixture, invoked in an isolated process."""

import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(root / "textured.glb"))
scene = bpy.context.scene
mesh = next(obj for obj in scene.objects if obj.type == "MESH")
material = mesh.data.materials[0]
shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
bindings = {key: shader.inputs[key].is_linked for key in ("Base Color", "Normal", "Roughness")}
images = [n.image for n in material.node_tree.nodes if n.type == "TEX_IMAGE"]
bpy.ops.file.pack_all()
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.render.resolution_x = 192
scene.render.resolution_y = 192
scene.render.resolution_percentage = 100
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3
camera_data = bpy.data.cameras.new("camera")
camera = bpy.data.objects.new("camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0, 0, 4)
camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.5
scene.camera = camera
light_data = bpy.data.lights.new("area", "AREA")
light_data.energy = 120
light_data.shape = "DISK"
light_data.size = 3
light = bpy.data.objects.new("area", light_data)
scene.collection.objects.link(light)
light.location = (0, 0, 3)
scene.render.filepath = str(root / "render.png")
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(root / "checked.blend"))
(root / "native_result.json").write_text(
    json.dumps(
        {
            "bindings": bindings,
            "images": len(images),
            "packed": all(i.packed_file for i in images),
            "linear_data_maps": sum(i.colorspace_settings.name == "Non-Color" for i in images),
            "triangles": sum(len(p.vertices) - 2 for p in mesh.data.polygons),
        }
    )
)
sys.stdout.flush()
# Some Flatpak Blender builds hang during audio teardown after headless work.
# This fixture owns no interactive scene and has already saved every artifact.
os._exit(0)
