"""Material portability and unchanged geometry on existing inspection assets."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plume_advanced.config import load_project_config
from plume_advanced.exporters.materials import apply_cave_material
from plume_advanced.stages.geometry_export import _StrictGlbBuilder
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.validation import GlbAsset


@pytest.fixture
def material_case(tmp_path):
    paths = []
    for name, value in [('color', (100, 80, 60)), ('normal', (128, 128, 255)),
                        ('roughness', (128, 128, 128))]:
        path = tmp_path / f'{name}.png'
        Image.new('RGB', (8, 8), value).save(path)
        paths.append(str(path))
    config = GeometryConfig(cave_diffuse_texture=paths[0], cave_normal_texture=paths[1],
                            cave_roughness_texture=paths[2], cave_displacement_texture='',
                            cave_displacement_scale_m=0, cave_texture_scale_m=4,
                            embedded_texture_max_size=4, cave_normal_scale=1)
    builder = _StrictGlbBuilder()
    material = builder.material(name='neutral')
    builder.mesh_node(name='cave_wall', positions=np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0]]),
                      faces=np.array([[0, 1, 2]]), material_index=material,
                      texcoords=np.array([[0, 0], [1, 0], [0, 1]]),
                      normals=np.array([[0, 0, 1]] * 3), tangents=np.array([[1, 0, 0, 1]] * 3))
    source = tmp_path / 'neutral.glb'
    source.write_bytes(builder.to_glb())
    return source, config


def test_revision_embeds_linear_pbr_and_preserves_geometry(material_case, tmp_path):
    source, config = material_case
    before = source.read_bytes()
    output = tmp_path / 'textured.glb'
    report = apply_cave_material(source, output, config, source_tile_size_m=8)
    asset = GlbAsset(output)
    _, _, primitive = asset.cave_primitive()
    material = asset.document['materials'][primitive['material']]
    assert material['pbrMetallicRoughness']['baseColorFactor'] == [1, 1, 1, 1]
    assert material['pbrMetallicRoughness']['roughnessFactor'] == 1
    assert material['normalTexture']['scale'] == 1
    assert len(asset.document['images']) == 3
    assert all('uri' not in image for image in asset.document['images'])
    assert report['image_dimensions'] == [[4, 4]] * 3
    assert asset.embedded_image(report['image_indices'][1]).getpixel((0, 0))[:3] == (128, 128, 255)
    assert asset.embedded_image(report['image_indices'][2]).getpixel((0, 0))[:3] == (255, 128, 0)
    assert asset.document['samplers'][0]['wrapS'] == 10497
    assert asset.document['samplers'][0]['wrapT'] == 10497
    uv_index = primitive['attributes']['TEXCOORD_0']
    np.testing.assert_array_equal(asset.accessor(uv_index), [[0, 0], [2, 0], [0, 2]])
    assert set(report['geometry_attributes_preserved']) == {'POSITION', 'NORMAL', 'TANGENT', 'indices'}
    assert source.read_bytes() == before
    duplicate = tmp_path / 'repeat.glb'
    apply_cave_material(source, duplicate, config, source_tile_size_m=8)
    assert duplicate.read_bytes() == output.read_bytes()


def test_revision_rejects_missing_maps_and_geometry_edits(material_case, tmp_path):
    source, config = material_case
    for invalid, error in [(replace(config, cave_normal_texture='missing.png'), RuntimeError),
                            (replace(config, cave_diffuse_texture=''), ValueError),
                            (replace(config, cave_displacement_scale_m=0.1), ValueError),
                            (replace(config, cave_texture_scale_m=float('nan')), ValueError)]:
        output = tmp_path / 'invalid.glb'
        with pytest.raises(error):
            apply_cave_material(source, output, invalid, source_tile_size_m=8)
        assert not output.exists()
    with pytest.raises(ValueError, match='different file'):
        apply_cave_material(source, source, config, source_tile_size_m=8)


def test_full_inspection_presets_request_portable_materials():
    root = Path(__file__).resolve().parents[1]
    for name in ('earth_short_interconnected_full', 'earth_long_interconnected_full'):
        config = load_project_config(root / 'config' / f'{name}.toml').geometry
        assert config.embedded_texture_max_size == 4096
        assert config.cave_texture_scale_m == 4.0
        for path in (config.cave_diffuse_texture, config.cave_normal_texture,
                     config.cave_roughness_texture):
            assert Path(path).parent == root / 'texture/dark_rock_8k/textures'
        assert config.cave_displacement_scale_m == 0
        assert config.cave_displacement_texture == ''


def test_material_revision_reuses_uv_storage(material_case, tmp_path):
    source, config = material_case
    original = GlbAsset(source)
    _, _, primitive = original.cave_primitive()
    uv_index = primitive['attributes']['TEXCOORD_0']
    original_uv = original.document['accessors'][uv_index]
    output = tmp_path / 'material.glb'
    apply_cave_material(source, output, config, source_tile_size_m=8)
    revised = GlbAsset(output)
    revised_uv = revised.document['accessors'][uv_index]
    assert revised_uv['bufferView'] == original_uv['bufferView']
    assert revised_uv.get('byteOffset', 0) == original_uv.get('byteOffset', 0)
    # Only the three embedded images need new buffer views, not another UV copy.
    assert len(revised.document['bufferViews']) == len(original.document['bufferViews']) + 3
