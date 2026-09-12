"""Apply portable PBR maps to a completed, neutral cave without remeshing."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np

from plume_advanced.stages.geometry_export import _StrictGlbBuilder, add_cave_pbr_material
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.validation import GlbAsset


def apply_cave_material(
    source: Path, output: Path, config: GeometryConfig, *, source_tile_size_m: float,
) -> dict:
    """Create a material revision; preserve all geometry attributes except UV scale.

    Limited to PLUME's neutral single-cave inspection assets. The original is
    never overwritten. Existing material revisions must be rebuilt from their
    original, avoiding duplicate embedded textures on repeated edits.
    """
    if source.resolve() == output.resolve():
        raise ValueError("Write the material revision to a different file")
    if output.exists():
        raise FileExistsError(output)
    if not all(math.isfinite(v) and v > 0 for v in (source_tile_size_m, config.cave_texture_scale_m)):
        raise ValueError("Texture tile sizes must be finite and positive")
    if config.embedded_texture_max_size <= 0:
        raise ValueError("Texture size must be positive")
    if not math.isfinite(config.cave_normal_scale) or not 0 <= config.cave_normal_scale <= 10:
        raise ValueError("Normal scale must be in [0, 10]")
    if config.cave_displacement_texture or config.cave_displacement_scale_m != 0:
        raise ValueError("Material-only export cannot apply geometry displacement")
    asset = GlbAsset(source)
    if asset.version != 2 or asset.declared_length != len(asset.data):
        raise ValueError("Expected a complete GLB 2.0 asset")
    doc = deepcopy(asset.document)
    if doc.get('images') or doc.get('textures'):
        raise ValueError("Use the original neutral GLB as the material revision source")
    nodes = [node for node in doc['nodes'] if 'mesh' in node]
    if len(nodes) != 1 or nodes[0].get('name') != 'cave_wall':
        raise ValueError("Expected one cave_wall mesh and no props")
    mesh = doc['meshes'][nodes[0]['mesh']]
    if len(mesh['primitives']) != 1 or len(doc['buffers']) != 1:
        raise ValueError("Expected one cave primitive and one embedded buffer")
    primitive = mesh['primitives'][0]
    attributes = primitive['attributes']
    if not {'POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0'} <= attributes.keys():
        raise ValueError("Material revision needs existing metric UVs, normals and tangents")
    # Use exactly the same material construction as the full generator.
    builder = _StrictGlbBuilder()
    add_cave_pbr_material(config, builder=builder, image_cache={})
    bundle = builder.to_glb()
    json_size = struct.unpack_from('<I', bundle, 12)[0]
    material_doc = json.loads(bundle[20:20 + json_size])
    material_binary = bundle[28 + json_size:]
    material = material_doc['materials'][0]
    pbr = material['pbrMetallicRoughness']
    if not ('baseColorTexture' in pbr and 'metallicRoughnessTexture' in pbr
            and 'normalTexture' in material):
        raise ValueError("Provide all three color, normal and roughness maps")
    binary = bytearray(asset.binary)
    binary.extend(b'\0' * (-len(binary) % 4))
    offset = len(binary)
    view_offset = len(doc['bufferViews'])
    for view in material_doc['bufferViews']:
        view['byteOffset'] = view.get('byteOffset', 0) + offset
        doc['bufferViews'].append(view)
    binary.extend(material_binary)
    doc['images'] = material_doc['images']
    for image in doc['images']:
        image['bufferView'] += view_offset
    doc['textures'] = material_doc['textures']
    doc['samplers'] = [{'wrapS': 10497, 'wrapT': 10497, 'magFilter': 9729, 'minFilter': 9987}]
    for texture in doc['textures']:
        texture['sampler'] = 0
    doc['materials'] = material_doc['materials']
    primitive['material'] = 0
    # Uniform positive UV scaling preserves the tangent basis and mesh shape.
    uv_index = attributes['TEXCOORD_0']
    uv_accessor = doc['accessors'][uv_index]
    uvs = asset.accessor(uv_index).astype('<f4') * (source_tile_size_m / config.cave_texture_scale_m)
    uv_bytes = uvs.tobytes()
    uv_accessor.update(bufferView=len(doc['bufferViews']), byteOffset=0,
                       min=uvs.min(axis=0).tolist(), max=uvs.max(axis=0).tolist())
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary),
                               'byteLength': len(uv_bytes), 'target': 34962})
    binary.extend(uv_bytes)
    doc['buffers'][0]['byteLength'] = len(binary)
    doc['asset'].setdefault('extras', {})['material_revision'] = 'plume.tiled-pbr.v1'
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as file:
        file.write(struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(binary)))
        file.write(struct.pack('<II', len(encoded), 0x4E4F534A))
        file.write(encoded)
        file.write(struct.pack('<II', len(binary), 0x004E4942))
        file.write(binary)
    revised = GlbAsset(output)
    checks = {}
    for name, index in {**attributes, 'indices': primitive['indices']}.items():
        if name == 'TEXCOORD_0':
            continue
        original = asset.accessor(index)
        actual = revised.accessor(index)
        if not np.array_equal(original, actual):
            raise RuntimeError(f'Material revision altered {name}')
        checks[name] = hashlib.sha256(original.tobytes()).hexdigest()
    roles = {
        'base_color_srgb': pbr['baseColorTexture'],
        'normal_linear_opengl': material['normalTexture'],
        'roughness_green_metallic_blue_linear': pbr['metallicRoughnessTexture'],
    }
    image_indices = [doc['textures'][binding['index']]['source'] for binding in roles.values()]
    images = [revised.embedded_image(i) for i in image_indices]
    return {
        'schema': 'plume.material-revision.v1',
        'source_asset': str(source.resolve()),
        'source_sha256': hashlib.sha256(asset.data).hexdigest(),
        'asset_sha256': hashlib.sha256(revised.data).hexdigest(),
        'geometry_attributes_preserved': checks,
        'source_tile_size_m': source_tile_size_m,
        'tile_size_m': config.cave_texture_scale_m,
        'normal_scale': config.cave_normal_scale,
        'texture_paths': {'color': config.cave_diffuse_texture, 'normal': config.cave_normal_texture,
                          'roughness': config.cave_roughness_texture},
        'image_dimensions': [list(image.size) for image in images],
        'image_roles': list(roles),
        'image_indices': image_indices,
        'texture_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for name, path in [('color', config.cave_diffuse_texture),
                                              ('normal', config.cave_normal_texture),
                                              ('roughness', config.cave_roughness_texture)]},
        'scope': 'Material and uniform UV scale only; generation and clearance results are unchanged.',
    }
