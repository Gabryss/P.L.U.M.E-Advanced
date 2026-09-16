#!/usr/bin/env python3
"""Render neutral geometry previews from a saved Blender inspection scene.

Run inside Blender, passing the case directory after ``--``. The source scene
is never overwritten; only PNGs and their source identity are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Keep the helper usable in Blender's independent Python runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/plume_advanced"))
from asset_paths import find_export_asset


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--exterior-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    root = args.directory.resolve()
    source = find_export_asset(root, filename='plume_continuous_inspection.blend')
    bpy.ops.wm.open_mainfile(filepath=str(source))
    scene = bpy.context.scene
    scene.cycles.samples = 48
    scene.cycles.adaptive_threshold = .025
    scene.cycles.use_denoising = True
    scene.view_settings.exposure = 0
    material = bpy.data.materials.new('Geometry inspection only')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.3, .3, .3, 1)
    shader.inputs['Roughness'].default_value = .85
    scene.view_layers[0].material_override = material
    output = root / 'previews/clay'
    output.mkdir(exist_ok=True)
    names = ('Overview', 'Top down')
    if not args.exterior_only:
        names += ('Interior 1', 'Interior 2')
    for name in names:
        interior = name.startswith('Interior')
        scene.camera = bpy.data.objects[name]
        scene.render.resolution_percentage = 50 if interior else 100
        for obj in scene.objects:
            if obj.type == 'LIGHT':
                obj.hide_render = (obj.parent != scene.camera) if interior else obj.data.type != 'SUN'
        scene.render.filepath = str(output / (name.lower().replace(' ', '_') + '.png'))
        bpy.ops.render.render(write_still=True)
    (output / 'render_source.json').write_text(json.dumps(dict(
        source=str(source.relative_to(root)),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mapping='neutral material override, unchanged imported geometry',
        samples=48,
        exposure_ev=0,
        lighting='active camera torch for interiors; inspection sun for exteriors',
        rendered_cameras=names,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
