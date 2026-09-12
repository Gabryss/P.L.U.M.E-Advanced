"""Optional native CPU render; enable with PLUME_BLENDER_BINARY."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plume_advanced.exporters.materials import apply_cave_material
from plume_advanced.stages.geometry_export import _StrictGlbBuilder
from plume_advanced.stages.geometry_types import GeometryConfig


@pytest.mark.integration
def test_native_blender_import_and_render(tmp_path):
    binary = os.environ.get("PLUME_BLENDER_BINARY")
    if not binary:
        pytest.skip("Set PLUME_BLENDER_BINARY to run the native Blender regression")
    checker = np.zeros((32, 32, 3), dtype=np.uint8)
    checker[:, :16] = (160, 55, 20)
    checker[:, 16:] = (30, 100, 170)
    Image.fromarray(checker).save(tmp_path / "color.png")
    Image.new("RGB", (32, 32), (128, 128, 255)).save(tmp_path / "normal.png")
    Image.new("L", (32, 32), 220).save(tmp_path / "roughness.png")
    builder = _StrictGlbBuilder()
    material = builder.material(name="neutral")
    builder.mesh_node(
        name="cave_wall",
        positions=np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]),
        faces=np.array([[0, 2, 1], [0, 3, 2]]),
        material_index=material,
        normals=np.array([[0, 1, 0]] * 4),
        texcoords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        tangents=np.array([[1, 0, 0, 1]] * 4),
    )
    source = tmp_path / "source.glb"
    source.write_bytes(builder.to_glb())
    config = GeometryConfig(
        cave_diffuse_texture=str(tmp_path / "color.png"),
        cave_normal_texture=str(tmp_path / "normal.png"),
        cave_roughness_texture=str(tmp_path / "roughness.png"),
        cave_displacement_texture="",
        cave_displacement_scale_m=0,
        cave_texture_scale_m=2,
        embedded_texture_max_size=32,
    )
    apply_cave_material(source, tmp_path / "textured.glb", config, source_tile_size_m=2)
    fixture = Path(__file__).parent / "fixtures/blender/material_check.py"
    with (tmp_path / "blender.log").open("w") as log:
        result = subprocess.run(
            [
                binary,
                "--background",
                "--factory-startup",
                "-noaudio",
                "--python",
                str(fixture),
                "--",
                str(tmp_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
    assert result.returncode == 0, (tmp_path / "blender.log").read_text()[-4000:]
    report = json.loads((tmp_path / "native_result.json").read_text())
    assert all(report["bindings"].values()) and report["packed"]
    assert report["images"] == 3 and report["linear_data_maps"] == 2 and report["triangles"] == 2
    image = np.asarray(Image.open(tmp_path / "render.png").convert("RGB"), dtype=float)
    # Two material colors must survive import and rendering; grey/pink/black
    # placeholder renders cannot satisfy both color populations.
    assert np.mean((image[:, :, 0] - image[:, :, 2]) > 25) > 0.1
    assert np.mean((image[:, :, 2] - image[:, :, 0]) > 25) > 0.1
