"""Portable GLB validation regressions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from plume_advanced.stages.geometry_export import export_geometry_glb
from plume_advanced.stages.geometry_types import (
    CaveGeometry,
    GeometryConfig,
    SurfaceTextureFrame,
    VoxelGrid,
)
from plume_advanced.validation import (
    PortableAssetValidator,
    write_validation_reports,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_portable_validator_checks_embedded_materials_and_displacement(
    tmp_path: Path,
) -> None:
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    diffuse = texture_dir / "diffuse.png"
    normal = texture_dir / "normal.png"
    roughness = texture_dir / "roughness.png"
    displacement = texture_dir / "displacement.png"
    Image.new("RGB", (4, 4), (82, 76, 68)).save(diffuse)
    Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
    Image.new("L", (4, 4), 210).save(roughness)
    height = Image.new("L", (4, 4))
    height.putdata(
        (
            0,
            64,
            128,
            255,
            32,
            96,
            160,
            224,
            64,
            128,
            192,
            255,
            0,
            80,
            176,
            240,
        )
    )
    height.save(displacement)

    frame = SurfaceTextureFrame(
        segment_id=1,
        center=(0.25, 0.25, 0.25),
        tangent=(0.0, 1.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        binormal=(0.0, 0.0, 1.0),
        longitudinal_m=0.0,
    )
    geometry = CaveGeometry(
        config=GeometryConfig(
            cave_diffuse_texture=str(diffuse),
            cave_normal_texture=str(normal),
            cave_roughness_texture=str(roughness),
            cave_displacement_texture=str(displacement),
            cave_displacement_scale_m=0.05,
            cave_smoothing_iterations=0,
            embedded_texture_max_size=16,
        ),
        voxel_grid=VoxelGrid(
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            density=np.zeros((2, 2, 2), dtype=np.float32),
            iso_level=0.0,
        ),
        chunk_meshes=(),
        assembled_vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        assembled_faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
        component_count=1,
        stamped_sample_count=1,
        stamped_segment_ids=(1,),
        surface_texture_frames=(frame,),
    )
    export_dir = tmp_path / "export_neutral"
    asset = export_geometry_glb(geometry, export_dir / "plume_cave_scene.glb")
    collision = asset.with_name(f"{asset.stem}_collision.obj")
    trimesh.Trimesh(
        vertices=np.asarray(geometry.assembled_vertices),
        faces=np.asarray(geometry.assembled_faces),
        process=False,
    ).export(collision)
    manifest = asset.with_suffix(".manifest.json")
    records = []
    for path in (asset, manifest, collision):
        records.append(
            {
                "path": str(path.relative_to(tmp_path)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "plume.run-manifest.v1",
                "status": "complete",
                "outputs": records,
            }
        ),
        encoding="utf-8",
    )

    validator = PortableAssetValidator(asset)
    updates: list[tuple[int, int, str]] = []
    checks = validator.validate(
        progress=lambda current, total, detail: updates.append(
            (current, total, detail)
        )
    )
    failed = [check for check in checks if not check.passed]
    assert not failed, [(check.name, check.detail) for check in failed]
    assert len(updates) == len(validator.PHASES)
    assert updates[-1] == (
        len(validator.PHASES),
        len(validator.PHASES),
        "Reproducibility",
    )

    json_report, markdown_report = write_validation_reports(
        asset,
        checks,
        tmp_path / "validation",
    )
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["valid"]
    assert "Baked displacement" not in markdown_report.read_text(encoding="utf-8")
    assert "Portable displacement contract" in markdown_report.read_text(
        encoding="utf-8"
    )
