"""Run in Blender's Scripting workspace, or headlessly with --source/--output.

Applies only to cave_wall. A source/output pair saves a NEW scene and an audit.
Running interactively applies to the current scene without saving over your file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def load_builder():
    path = Path(__file__).with_name("blender_materials.py")
    spec = importlib.util.spec_from_file_location("plume_blender_materials", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the material helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_triplanar_material


def geometry_fingerprint(cave):
    from array import array

    sha = hashlib.sha256()
    for collection, attribute, kind, size in (
        (cave.data.vertices, "co", "f", 3),
        (cave.data.loops, "vertex_index", "i", 1),
        (cave.data.corner_normals, "vector", "f", 3),
    ):
        values = array(kind, [0]) * (len(collection) * size)
        collection.foreach_get(attribute, values)
        sha.update(values.tobytes())
    for layer in cave.data.uv_layers:
        uv_values = array("f", [0]) * (len(layer.data) * 2)
        layer.data.foreach_get("uv", uv_values)
        sha.update(uv_values.tobytes())
    sha.update(repr(tuple(tuple(row) for row in cave.matrix_world)).encode())
    return sha.hexdigest()


def apply_to_cave(*, tile_size_m=4.0, normal_strength=1.0, blend_exponent=4.0):
    import bpy

    cave = bpy.data.objects.get("cave_wall")
    if cave is None or cave.type != "MESH" or len(cave.data.materials) != 1:
        raise ValueError("Expected cave_wall with one material slot")
    before = geometry_fingerprint(cave)
    original = cave.data.materials[0]
    material = load_builder()(
        original,
        tile_size_m=tile_size_m,
        normal_strength=normal_strength,
        blend_exponent=blend_exponent,
    )
    cave.data.materials[0] = material
    bpy.ops.file.pack_all()
    after = geometry_fingerprint(cave)
    if before != after:
        raise RuntimeError("Material revision unexpectedly changed geometry")
    images = {
        node.image.name: node.image for node in material.node_tree.nodes if node.type == "TEX_IMAGE"
    }
    assert len(images) == 3 and all(image.packed_file for image in images.values())
    return {
        "schema": "plume.native-triplanar.v1",
        "blender_version": bpy.app.version_string,
        "geometry_uvs_normals_transform_sha256": before,
        "geometry_unchanged": True,
        "triangles": sum(len(poly.vertices) - 2 for poly in cave.data.polygons),
        "material_settings": dict(material.items()),
        "images": [
            {
                "name": im.name,
                "size": list(im.size),
                "colorspace": im.colorspace_settings.name,
                "packed": bool(im.packed_file),
            }
            for im in images.values()
        ],
        "portable_glb_changed": False,
        "scope": "Native shader only. Network and geometry validation are unchanged.",
    }


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tile-size-m", type=float)
    parser.add_argument("--normal-strength", type=float)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    if bool(args.source) != bool(args.output):
        parser.error("--source and --output must be used together")
    if args.output and (args.output.exists() or args.output.with_suffix(".material.json").exists()):
        raise FileExistsError("Choose a new output scene and report path")
    source_hash = None
    if args.source:
        with args.source.open("rb") as file:
            source_hash = hashlib.file_digest(file, "sha256").hexdigest()
        bpy.ops.wm.open_mainfile(filepath=str(args.source.resolve()))
    settings_file = Path(__file__).with_name("settings.json")
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    report = apply_to_cave(
        tile_size_m=args.tile_size_m
        if args.tile_size_m is not None
        else settings.get("tile_size_m", 4.0),
        normal_strength=args.normal_strength
        if args.normal_strength is not None
        else settings.get("normal_strength", 1.0),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
        with args.source.open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == source_hash
        with args.output.open("rb") as file:
            output_hash = hashlib.file_digest(file, "sha256").hexdigest()
        report.update(
            source_scene=str(args.source.resolve()),
            source_sha256=source_hash,
            output_scene=str(args.output.resolve()),
            output_sha256=output_hash,
        )
        args.output.with_suffix(".material.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
