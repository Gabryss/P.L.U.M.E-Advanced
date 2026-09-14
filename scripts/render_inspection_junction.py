#!/usr/bin/env python3
"""Inspect a large junction/chamber in a saved Blender scene, without editing it."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main():
    import bpy
    import numpy as np
    from mathutils import Vector

    root = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    source = root / 'export_blender/plume_continuous_inspection.blend'
    bpy.ops.wm.open_mainfile(filepath=str(source))
    cave = bpy.data.objects['cave_wall']
    inverse = cave.matrix_world.inverted()
    network = json.loads((root / 'stage_b_network.json').read_text())
    junction = max(network['junctions'], key=lambda j: (
        j['kind'] == 'chamber', len(j['segment_ids']), j['blend_length']))
    with np.load(root / 'stage_c_sections.npz') as saved:
        centers = saved['center_xyz_m'].copy()
        segments = saved['segment_id'].copy()
    eligible = np.flatnonzero(np.isin(segments, junction['segment_ids']))
    distance = np.linalg.norm(centers[eligible, :2] - [junction['center_x'], junction['center_y']], axis=1)
    target = centers[eligible[np.argmin(distance)]].copy()

    def ray(eye, direction, maximum=20.):
        direction = Vector(direction).normalized()
        hit, location, normal, _ = cave.ray_cast(inverse @ Vector(eye),
            (inverse.to_3x3() @ direction).normalized(), distance=maximum)
        return hit, cave.matrix_world @ location, (inverse.transposed().to_3x3() @ normal).normalized()

    accepted = None
    for index in eligible[np.argsort(abs(distance - 8.), kind='stable')]:
        eye = centers[index].copy()
        up, roof, roof_normal = ray(eye, (0, 0, 1))
        down, floor, floor_normal = ray(eye, (0, 0, -1))
        clearance = roof.z - floor.z
        if not (up and down and roof_normal.z < 0 and floor_normal.z > 0 and clearance > .65):
            continue
        eye[2] = floor.z + .5 * clearance
        length = float(np.linalg.norm(target - eye))
        if length < 2.:
            continue
        hit, obstruction, _ = ray(eye, target - eye, length)
        if hit and np.linalg.norm(np.asarray(obstruction) - eye) < .95 * length:
            continue
        accepted = dict(section_index=int(index), segment_id=int(segments[index]),
                        eye_m=eye.tolist(), target_m=target.tolist(), clearance_m=float(clearance))
        break
    if accepted is None:
        raise RuntimeError('No verified view toward the selected junction')
    scene = bpy.context.scene
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True
    camera_data = bpy.data.cameras.new('Junction inspection')
    camera_data.lens = 20.
    camera_data.clip_start = .02
    camera = bpy.data.objects.new('Junction inspection', camera_data)
    scene.collection.objects.link(camera)
    camera.location = Vector(accepted['eye_m'])
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    light_data = bpy.data.lights.new('Junction inspection torch', 'POINT')
    light_data.energy = 600.
    light_data.shadow_soft_size = .06
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)
    light.location = camera.location
    scene.cycles.samples = 64
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.adaptive_threshold = .04
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 520
    scene.render.resolution_percentage = 100
    output = root / 'previews/junction'
    output.mkdir(exist_ok=True)
    scene.view_settings.exposure = 3.
    scene.render.filepath = str(output / 'textured.png')
    bpy.ops.render.render(write_still=True)
    material = bpy.data.materials.new('Neutral junction inspection')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.3, .3, .3, 1)
    shader.inputs['Roughness'].default_value = .85
    scene.view_layers[0].material_override = material
    scene.view_settings.exposure = 0.
    scene.render.filepath = str(output / 'clay.png')
    bpy.ops.render.render(write_still=True)
    (output / 'view.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        junction_id=junction['junction_id'], kind=junction['kind'], camera=accepted,
        samples=scene.cycles.samples, maximum_bounces=scene.cycles.max_bounces,
        scope='One large chamber/junction selected from network metadata; vertical clearance and sight line checked on the imported mesh. Source scene unchanged.',
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
