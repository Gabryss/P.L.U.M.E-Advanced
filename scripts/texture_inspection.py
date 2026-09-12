#!/usr/bin/env python3
"""Add a tiled PBR material to an existing neutral inspection run, without generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from plume_advanced.config import load_project_config
from plume_advanced.exporters.atomic import atomic_output_directory
from plume_advanced.exporters.materials import apply_cave_material
from plume_advanced.exporters.targets import (
    ExportResult,
    _write_engine_import_guide,
    _write_target_descriptor,
)
from plume_advanced.world import ExportConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--config', type=Path, required=True, help='Current project config with PBR maps')
    parser.add_argument('--output', type=Path, required=True, help='New material revision directory')
    args = parser.parse_args()
    root, output = args.run_directory.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new material revision directory: {output}')
    config = load_project_config(args.config).geometry
    source = root / 'export_blender/plume_cave_scene.glb'
    manifest = json.loads(source.with_suffix('.manifest.json').read_text())
    tile_size = manifest['cave']['material']['uv_scale_m']
    with atomic_output_directory(output) as staging:
        target = staging / 'export_blender' / source.name
        report = apply_cave_material(source, target, config, source_tile_size_m=tile_size)
        manifest['cave']['material'].update(
            diffuse=config.cave_diffuse_texture, normal=config.cave_normal_texture,
            roughness=config.cave_roughness_texture, uv_scale_m=config.cave_texture_scale_m,
            normal_scale=config.cave_normal_scale,
            embedded_texture_max_size=config.embedded_texture_max_size,
        )
        manifest['material_revision'] = 'See ../material_revision.json; original generation is unchanged.'
        target.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for engine in ('unity', 'ue5'):
            folder = staging / f'export_{engine}'
            folder.mkdir()
            asset = folder / source.name
            shutil.copy2(target, asset)
            shutil.copy2(target.with_suffix('.manifest.json'), asset.with_suffix('.manifest.json'))
            export = ExportConfig(target=engine, file_format='glb', generate_collision=False)
            result = ExportResult(target=engine, primary_asset=asset, files=(asset,))
            _write_target_descriptor(result, export, folder, source.stem)
            _write_engine_import_guide(engine, folder, asset, collision_asset=None)
            with asset.open('rb') as file:
                assert hashlib.file_digest(file, 'sha256').hexdigest() == report['asset_sha256']
        shutil.copy2(root / 'stage_c_sections.npz', staging / 'stage_c_sections.npz')
        shutil.copy2(args.config, staging / 'material_config.toml')
        report.update(source_run=str(root), generated_networks=0, added_rocks=0,
                      config_source=str(args.config.resolve()),
                      config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                      application_assets_byte_identical=True,
                      native_application_validation='See blender_import_check.json after running Blender.')
        (staging / 'material_revision.json').write_text(json.dumps(report, indent=2) + '\n')
        (staging / 'README.md').write_text(
            '# Textured cave inspection\n\n'
            'Material revision of the existing cave. No network, geometry or rocks were generated.\n\n'
            '[Blender scene](export_blender/plume_textured_inspection.blend) · '
            '[Blender GLB](export_blender/plume_cave_scene.glb) · '
            '[Unity GLB](export_unity/plume_cave_scene.glb) · '
            '[Unreal GLB](export_ue5/plume_cave_scene.glb)\n\n'
            'The GLBs are identical and embed three PBR images: sRGB base color, linear OpenGL '
            'normal, and linear metallic/roughness (roughness in green, metallic in blue). '
            'Use a glTF importer in Unity; do not assign the packed roughness map directly '
            'to a Standard/Lit metallic-smoothness slot.\n\n'
            f'Tile scale: {config.cave_texture_scale_m:g} metres. '
            f'Image budget: {config.embedded_texture_max_size} pixels per side.\n\n'
            '[Material and geometry checks](material_revision.json) · '
            '[Blender import checks](blender_import_check.json) · '
            '[Original stage figures](../STAGE_FIGURES.md)\n'
        )
    print(json.dumps({'output': str(output), **report}, indent=2))


if __name__ == '__main__':
    main()
