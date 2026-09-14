"""Native Blender materials. Import inside Blender; no PLUME dependencies needed.

The portable GLB keeps its UV material. This optional native material samples the
same three images in object metres and does not change the mesh or its UVs.
"""

from __future__ import annotations

import math


def _image_upstream(socket):
    """Require one unambiguous image per imported glTF material channel."""
    images = {}
    seen = set()
    pending = [link.from_node for link in socket.links]
    while pending:
        node = pending.pop()
        if node.as_pointer() in seen:
            continue
        seen.add(node.as_pointer())
        if node.type == "TEX_IMAGE" and node.image:
            images[node.image.as_pointer()] = node.image
        pending.extend(link.from_node for inp in node.inputs for link in inp.links)
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image upstream of {socket.name}")
    return next(iter(images.values()))


def build_triplanar_material(source, *, tile_size_m=4.0, normal_strength=1.0, blend_exponent=4.0):
    """Blend object-space projections with correctly oriented normal gradients.

    UVs for X/Y/Z are (y, sign(nx)*z), (z, sign(ny)*x),
    (x, sign(nz)*y). Their tangent frames have the matching outward axis normal.
    Weights are abs(N)**exponent, normalized. Projected normal-map slopes are
    blended in object space and projected into the base normal's tangent plane.
    A flat map therefore preserves N exactly, including at axis transitions.
    Images are shared, never copied or baked. No UV tangent frame is involved.

    Object coordinates preserve phase when moving/rotating a cave. One object
    unit is a metre at export; apply scale before choosing the physical tile size.
    This shader is native to Blender, not a portable glTF material extension.
    """
    for name, value, minimum, maximum in (
        ("tile_size_m", tile_size_m, 1e-6, 1e6),
        ("normal_strength", normal_strength, 0.0, 10.0),
        ("blend_exponent", blend_exponent, 1.0, 16.0),
    ):
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    if not source or not source.use_nodes:
        raise ValueError("Expected a textured glTF material")
    shaders = [n for n in source.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]
    if len(shaders) != 1:
        raise ValueError("Expected one Principled shader")
    imported = shaders[0]
    images = {
        key: _image_upstream(imported.inputs[key]) for key in ("Base Color", "Roughness", "Normal")
    }
    if images["Base Color"].colorspace_settings.name != "sRGB":
        raise ValueError("Base color must be sRGB")
    if any(images[key].colorspace_settings.name != "Non-Color" for key in ("Roughness", "Normal")):
        raise ValueError("Roughness and normal images must be Non-Color")
    if any(
        n.type == "OUTPUT_MATERIAL" and n.inputs["Displacement"].is_linked
        for n in source.node_tree.nodes
    ):
        raise ValueError("Cannot replace a material with shader displacement")

    material = source.copy()
    material.name = "PLUME continuous rock"
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()

    def node(kind, label):
        result = nodes.new(kind)
        result.label = label
        result.name = label
        return result

    def connect(value, socket):
        if isinstance(value, (int, float, tuple)):
            socket.default_value = value
        else:
            links.new(value, socket)

    def scalar(operation, a, b=None):
        result = node("ShaderNodeMath", operation)
        result.operation = operation
        connect(a, result.inputs[0])
        if b is not None:
            connect(b, result.inputs[1])
        return result.outputs[0]

    def vector(operation, a, b=None):
        result = node("ShaderNodeVectorMath", operation)
        result.operation = operation
        connect(a, result.inputs[0])
        if b is not None:
            connect(b, result.inputs[3 if operation == "SCALE" else 1])
        return result.outputs[1 if operation == "DOT_PRODUCT" else 0]

    def components(value):
        result = node("ShaderNodeSeparateXYZ", "Components")
        connect(value, result.inputs[0])
        return list(result.outputs)

    def combine(values):
        result = node("ShaderNodeCombineXYZ", "Vector")
        for value, socket in zip(values, result.inputs, strict=True):
            connect(value, socket)
        return result.outputs[0]

    def transform(value, kind, source_space, target_space):
        result = node("ShaderNodeVectorTransform", f"{kind}: {source_space} to {target_space}")
        result.vector_type = kind
        result.convert_from = source_space
        result.convert_to = target_space
        connect(value, result.inputs[0])
        return result.outputs[0]

    geometry = node("ShaderNodeNewGeometry", "Unperturbed surface")
    position = transform(geometry.outputs["Position"], "POINT", "WORLD", "OBJECT")
    position = components(vector("SCALE", position, 1.0 / tile_size_m))
    normal = vector("NORMALIZE", transform(geometry.outputs["Normal"], "NORMAL", "WORLD", "OBJECT"))
    normal_components = components(normal)
    signs = [
        scalar("SUBTRACT", scalar("MULTIPLY", scalar("GREATER_THAN", n, 0.0), 2.0), 1.0)
        for n in normal_components
    ]
    weights = [scalar("POWER", scalar("ABSOLUTE", n), blend_exponent) for n in normal_components]
    total = scalar("ADD", scalar("ADD", weights[0], weights[1]), weights[2])
    weights = [scalar("DIVIDE", weight, total) for weight in weights]
    colors, roughness, gradients = [], [], []
    # U/V are cyclic axes, so U cross V equals the signed projection normal.
    for axis, (u, v) in enumerate(((1, 2), (2, 0), (0, 1))):
        uv = combine((position[u], scalar("MULTIPLY", position[v], signs[axis]), 0.0))
        samples = {}
        for role, image in images.items():
            texture = node("ShaderNodeTexImage", f"{'XYZ'[axis]} projection / {role}")
            texture.image = image
            texture.extension = "REPEAT"
            texture.interpolation = "Linear"
            connect(uv, texture.inputs["Vector"])
            samples[role] = texture.outputs["Color"]
        colors.append(vector("SCALE", samples["Base Color"], weights[axis]))
        # glTF metallic/roughness packs roughness in G, not luminance or R.
        roughness.append(scalar("MULTIPLY", components(samples["Roughness"])[1], weights[axis]))
        decoded = components(
            vector("SUBTRACT", vector("SCALE", samples["Normal"], 2.0), (1.0, 1.0, 1.0))
        )
        divisor = scalar("MAXIMUM", decoded[2], 0.1)
        slope_u = scalar("DIVIDE", decoded[0], divisor)
        slope_v = scalar("MULTIPLY", scalar("DIVIDE", decoded[1], divisor), signs[axis])
        slope = [0.0, 0.0, 0.0]
        slope[u], slope[v] = slope_u, slope_v
        gradients.append(vector("SCALE", combine(slope), weights[axis]))
    color = vector("ADD", vector("ADD", colors[0], colors[1]), colors[2])
    rough = scalar("ADD", scalar("ADD", roughness[0], roughness[1]), roughness[2])
    gradient = vector("ADD", vector("ADD", gradients[0], gradients[1]), gradients[2])
    gradient = vector(
        "SUBTRACT", gradient, vector("SCALE", normal, vector("DOT_PRODUCT", normal, gradient))
    )
    perturbed = vector(
        "NORMALIZE", vector("ADD", normal, vector("SCALE", gradient, normal_strength))
    )
    world_normal = vector("NORMALIZE", transform(perturbed, "NORMAL", "OBJECT", "WORLD"))
    shader = node("ShaderNodeBsdfPrincipled", "Rock surface")
    shader.inputs["Metallic"].default_value = 0.0
    connect(color, shader.inputs["Base Color"])
    connect(rough, shader.inputs["Roughness"])
    connect(world_normal, shader.inputs["Normal"])
    output = node("ShaderNodeOutputMaterial", "Surface output")
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    # Organize the acyclic node graph by dependency depth for manual inspection.
    depths: dict[str, int] = {}
    for item in nodes:
        depths[item.name] = 1 + max(
            (depths.get(link.from_node.name, 0) for inp in item.inputs for link in inp.links),
            default=-1,
        )
    rows: dict[int, int] = {}
    for item in nodes:
        depth = depths[item.name]
        row = rows.get(depth, 0)
        item.location = (depth * 240, -row * 230)
        rows[depth] = row + 1
    material["plume_mapping"] = "object_metric_triplanar_v1"
    material["plume_tile_size_m"] = tile_size_m
    material["plume_normal_strength"] = normal_strength
    material["plume_blend_exponent"] = blend_exponent
    return material
