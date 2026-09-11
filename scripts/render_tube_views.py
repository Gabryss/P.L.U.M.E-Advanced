#!/usr/bin/env python3
"""Render honest, neutral-lit inspection views of the exported GLB on the CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def render(mesh, eye, target, path, title, *, far=65., width=960, height=600, fov=75.,
           annotate=True, footer="Exported mesh | neutral inspection lighting | no rocks or textures",
           vertex_colors=None):
    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.column_stack((right, up, forward))
    vertices = np.asarray(mesh.vertices)
    relative = (vertices - eye) @ basis
    faces = np.asarray(mesh.faces)
    z = relative[:, 2]
    keep = (z[faces].min(axis=1) > .04) & (z[faces].min(axis=1) < far)
    faces = faces[keep]
    scale = .5 * width / np.tan(.5 * np.radians(fov))
    projected = np.column_stack((width/2 + scale*relative[:, 0]/np.maximum(z, .04),
                                 height/2 - scale*relative[:, 1]/np.maximum(z, .04)))
    tri = projected[faces]
    keep = (tri[:, :, 0].max(axis=1) >= 0) & (tri[:, :, 0].min(axis=1) < width)
    keep &= (tri[:, :, 1].max(axis=1) >= 0) & (tri[:, :, 1].min(axis=1) < height)
    faces, tri = faces[keep], tri[keep]
    order = np.argsort(z[faces].min(axis=1))
    faces, tri = faces[order], tri[order]
    normals = np.asarray(mesh.vertex_normals)
    depth = np.full((height, width), np.inf)
    color = np.full((height, width, 3), [30., 31., 33.], dtype=float)
    xx, yy = np.meshgrid(np.arange(width)+.5, np.arange(height)+.5)
    rays = np.stack(((xx-width/2)/scale, -(yy-height/2)/scale, np.ones_like(xx)), axis=-1) @ basis.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    for face, points in zip(faces, tri, strict=True):
        xmin, ymin = np.maximum(np.floor(points.min(axis=0)).astype(int), [0, 0])
        xmax, ymax = np.minimum(np.ceil(points.max(axis=0)).astype(int), [width-1, height-1])
        if xmax < xmin or ymax < ymin:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax+1)+.5, np.arange(ymin, ymax+1)+.5)
        a, b, c = points
        denom = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denom) < 1e-10:
            continue
        w0 = ((b[1]-c[1])*(xs-c[0])+(c[0]-b[0])*(ys-c[1]))/denom
        w1 = ((c[1]-a[1])*(xs-c[0])+(a[0]-c[0])*(ys-c[1]))/denom
        w2 = 1.-w0-w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        weights = np.stack((w0, w1, w2), axis=-1) / z[face]
        inverse_depth = weights.sum(axis=-1)
        local_depth = 1. / np.maximum(inverse_depth, 1e-12)
        region = depth[ymin:ymax+1, xmin:xmax+1]
        visible = inside & (local_depth < region) & (local_depth < far)
        if not np.any(visible):
            continue
        weights /= np.maximum(inverse_depth[..., None], 1e-12)
        normal = weights @ normals[face]
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        lambert = np.abs(np.sum(normal * rays[ymin:ymax+1, xmin:xmax+1], axis=-1))
        intensity = (.23 + .77*lambert) / (1.+.016*local_depth)
        albedo = np.array([210., 204., 193.]) if vertex_colors is None else weights @ vertex_colors[face]
        shade = intensity[..., None]*albedo
        color[ymin:ymax+1, xmin:xmax+1][visible] = shade[visible]
        region[visible] = local_depth[visible]
    result = Image.fromarray(np.uint8(np.clip(color, 0, 255)))
    if annotate:
        draw = ImageDraw.Draw(result)
        draw.rectangle((0, 0, width, 39), fill=(22, 24, 27))
        draw.text((15, 13), title, fill=(235, 235, 230))
        draw.text((15, height-24), footer, fill=(225, 225, 220))
    result.save(path)
    return {"image":path.name, "eye":list(map(float, eye)), "target":list(map(float, target)),
            "visible_fraction":float(np.mean(np.isfinite(depth))), "rasterized_triangles":len(faces)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cameras", type=Path, help="Reuse camera definitions for a direct comparison")
    args = parser.parse_args()
    root = args.directory
    scene = trimesh.load(root / "lava_tube_geometry.glb", force="scene", process=False)
    # Imported glTF normals live in Trimesh's cache; preserve them instead of
    # silently recomputing triangle normals and misrepresenting the export.
    mesh = next(iter(scene.geometry.values())).copy(include_cache=True)
    # glTF Y-up back to the generator's Z-up metre coordinates.
    mesh.apply_transform(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]))
    p = np.load(root / "sections.npz")
    centers = p["center_xyz_m"]
    cameras = []
    if args.cameras:
        cameras = json.loads(args.cameras.read_text())
    else:
        pool_index = int(np.argmax(p["width_m"]))
        eligible = np.flatnonzero((p["height_m"] > 1.7) & (p["height_m"] < 2.8) & (p["junction_influence"] < .15))
        chosen = [int(eligible[len(eligible)//3]), int(eligible[2*len(eligible)//3]), pool_index]
        for number, index in enumerate(chosen):
            same = np.flatnonzero(p["segment_id"] == p["segment_id"][index])
            downstream = same[p["arc_length_m"][same] > p["arc_length_m"][index]+4]
            if not len(downstream):
                downstream = same[p["arc_length_m"][same] < p["arc_length_m"][index]-4][::-1]
            other = int(downstream[0]) if len(downstream) else index
            eye = centers[index].copy()
            eye[2] = p["floor_world_z"][index] + .55*(p["roof_world_z"][index]-p["floor_world_z"][index])
            target = centers[other].copy()
            target[2] = p["floor_world_z"][other] + .55*(p["roof_world_z"][other]-p["floor_world_z"][other])
            if other == index:
                target = eye + p["tangent"][index]*5
            name = ["gallery_interior", "branch_interior", "chamber_interior"][number]
            cameras.append({"name":name, "eye":eye.tolist(), "target":target.tolist(), "title":name.replace("_", " ").title(),
                            "section_index":index, "segment_id":int(p["segment_id"][index])})
        center = centers[pool_index]
        cameras.append({"name":"chamber_exterior", "eye":(center+np.array([29., -36., 27.])).tolist(),
                        "target":center.tolist(), "title":"Chamber and adjoining passages", "section_index":pool_index})
    reports = []
    for camera in cameras:
        print("Rendering", camera["name"], flush=True)
        reports.append(render(mesh, np.array(camera["eye"], dtype=float), np.array(camera["target"], dtype=float),
                              root / (camera["name"]+".png"), camera["title"], far=100.))
    (root / "inspection_cameras.json").write_text(json.dumps(cameras, indent=2)+"\n")
    (root / "inspection_renders.json").write_text(json.dumps(reports, indent=2)+"\n")
    contact = Image.new("RGB", (1920,1200))
    for i, camera in enumerate(cameras):
        contact.paste(Image.open(root/(camera["name"]+".png")), ((i%2)*960, (i//2)*600))
    contact.save(root / "inspection_views.png")


if __name__ == "__main__":
    main()
