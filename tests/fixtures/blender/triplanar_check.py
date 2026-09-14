"""Native shader regression; no network generation. Run through pytest wrapper."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

out = Path(sys.argv[sys.argv.index("--") + 1])
helper = Path(sys.argv[sys.argv.index("--") + 2])
spec = importlib.util.spec_from_file_location("plume_blender_materials", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.world = bpy.data.worlds.new("Black")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
source = bpy.data.materials.new("Source")
source.use_nodes = True
shader = source.node_tree.nodes.get("Principled BSDF")
images = {}
for name, rgba in [
    ("Base Color", (0.3, 0.2, 0.1, 1)),
    ("Normal", (0.65, 0.4, 0.95, 1)),
    ("Roughness", (1, 0.8, 0, 1)),
]:
    im = bpy.data.images.new(name, width=4, height=4, float_buffer=True)
    im.colorspace_settings.name = "sRGB" if name == "Base Color" else "Non-Color"
    im.pixels[:] = list(rgba) * 16
    node = source.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = im
    if name == "Normal":
        normal = source.node_tree.nodes.new("ShaderNodeNormalMap")
        source.node_tree.links.new(node.outputs["Color"], normal.inputs["Color"])
        source.node_tree.links.new(normal.outputs["Normal"], shader.inputs[name])
    elif name == "Roughness":
        separate = source.node_tree.nodes.new("ShaderNodeSeparateXYZ")
        source.node_tree.links.new(node.outputs["Color"], separate.inputs[0])
        source.node_tree.links.new(separate.outputs["Y"], shader.inputs[name])
    else:
        source.node_tree.links.new(node.outputs["Color"], shader.inputs[name])
    images[name] = im
cam_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(camera)
scene.camera = camera
cam_data.type = "ORTHO"
cam_data.ortho_scale = 1
mesh = bpy.data.meshes.new("Plane")
obj = bpy.data.objects.new("Plane", mesh)
scene.collection.objects.link(obj)
mesh.from_pydata([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], [], [(0, 1, 2), (0, 2, 3)])
mesh.update()
uv = mesh.uv_layers.new(name="UVMap")
# Deliberately unrelated coordinates on either triangle.
for i, loop in enumerate(uv.data):
    loop.uv = (i * 7.314, i * 2.182)
obj.data.materials.append(source)
results = []


def sample(material, mode):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    socket = bsdf.inputs[mode].links[0].from_socket
    if mode == "Normal":
        mul = nodes.new("ShaderNodeVectorMath")
        mul.operation = "SCALE"
        mul.inputs[3].default_value = 0.5
        links.new(socket, mul.inputs[0])
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (0.5, 0.5, 0.5)
        links.new(mul.outputs[0], add.inputs[0])
        socket = add.outputs[0]
    em = nodes.new("ShaderNodeEmission")
    links.new(socket, em.inputs["Color"])
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    links.new(em.outputs[0], output.inputs["Surface"])
    obj.data.materials[0] = material
    path = out / "sample.exr"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.array(image.pixels[:]).reshape(16, 16, 4)
    bpy.data.images.remove(image)
    value = pixels[6:10, 6:10, :3].mean((0, 1))
    return value


normals = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 1),
    (-1, 2, -3),
    (1, 0.00001, 1),
]
for strength in (0.0, 1.0):
    for normal in normals:
        n = np.asarray(normal, dtype=float)
        n /= np.linalg.norm(n)
        obj.rotation_euler = Vector(n).to_track_quat("Z", "Y").to_euler()
        # Object coordinates remain Z-up: rotate camera/mesh together to test world normal transform.
        camera.location = Vector(n) * 3
        camera.rotation_euler = (-Vector(n)).to_track_quat("-Z", "Y").to_euler()
        material = module.build_triplanar_material(source, normal_strength=strength)
        actual = sample(material, "Normal") * 2 - 1
        local = np.array([strength * 0.3 / 0.9, strength * (-0.2) / 0.9, 1.0])
        local /= np.linalg.norm(local)
        expected = np.array(obj.rotation_euler.to_matrix() @ Vector(local))
        assert np.allclose(actual, expected, atol=2e-4), (strength, normal, actual, expected)
        results.append(
            {
                "normal": normal,
                "strength": strength,
                "max_error": float(abs(actual - expected).max()),
            }
        )
# Axis-transition blending must be exercised in object space, not only through transforms.
for normal in normals:
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    rotation = Vector(n).to_track_quat("Z", "Y").to_matrix()
    obj.rotation_euler = (0, 0, 0)
    for vertex, co in zip(
        mesh.vertices, [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], strict=True
    ):
        vertex.co = rotation @ Vector(co)
    mesh.update()
    camera.location = Vector(n) * 3
    camera.rotation_euler = (-Vector(n)).to_track_quat("-Z", "Y").to_euler()
    material = module.build_triplanar_material(source)
    actual = sample(material, "Normal") * 2 - 1
    w = abs(n) ** 4
    w /= sum(w)
    sign = np.where(n > 0, 1.0, -1.0)
    g = (
        np.array([0, 0.3 / 0.9, -0.2 / 0.9 * sign[0]]) * w[0]
        + np.array([-0.2 / 0.9 * sign[1], 0, 0.3 / 0.9]) * w[1]
        + np.array([0.3 / 0.9, -0.2 / 0.9 * sign[2], 0]) * w[2]
    )
    g -= n * np.dot(n, g)
    expected = n + g
    expected /= np.linalg.norm(expected)
    assert np.allclose(actual, expected, atol=2e-4), (normal, actual, expected)
    results.append({"local_normal": normal, "max_error": float(abs(actual - expected).max())})
material = module.build_triplanar_material(source)
rough = sample(material, "Roughness")
assert np.allclose(rough, 0.8, atol=2e-4), rough
# A nonconstant tile makes this a real seam regression, not a uniform-color test.
pixels = []
for y in range(4):
    for x in range(4):
        pixels.extend((x / 3, y / 3, (x + y) / 6, 1))
images["Base Color"].pixels[:] = pixels
# Render invariance to arbitrary atlas UV changes; prove the old UV shader changes.
first = sample(module.build_triplanar_material(source), "Base Color")
old_first = sample(source.copy(), "Base Color")
for i, loop in enumerate(uv.data):
    loop.uv = (0.125, 0.125)
second = sample(module.build_triplanar_material(source), "Base Color")
old_second = sample(source.copy(), "Base Color")
assert np.array_equal(first, second), (first, second)
assert np.max(abs(old_first - old_second)) > 0.02, (old_first, old_second)
assert (
    len({node.image.as_pointer() for node in material.node_tree.nodes if node.type == "TEX_IMAGE"})
    == 3
)
(out / "native_triplanar.json").write_text(
    json.dumps(
        {
            "normal_cases": results,
            "roughness_green": rough.tolist(),
            "uv_invariant": True,
            "images_reused": True,
        },
        indent=2,
    )
)
sys.stdout.flush()
os._exit(0)
