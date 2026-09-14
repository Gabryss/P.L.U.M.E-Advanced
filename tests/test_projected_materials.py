"""Regression coverage for native projection packages and shader compilation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plume_advanced.exporters.projected_materials import write_projected_material_bundle
from plume_advanced.stages.geometry_export import _StrictGlbBuilder
from plume_advanced.validation import GlbAsset

ASSETS = Path(__file__).resolve().parents[1] / "src/plume_advanced/material_assets"


def source_asset(path, *, textured=True):
    builder = _StrictGlbBuilder()
    kwargs = {}
    if textured:
        kwargs = dict(
            base_color_texture=Image.new("RGBA", (4, 4), (70, 30, 10, 255)),
            normal_texture=Image.new("RGBA", (4, 4), (128, 128, 255, 255)),
            metallic_roughness_texture=Image.new("RGBA", (4, 4), (255, 220, 0, 255)),
        )
    material = builder.material(name="rock", metallic_factor=0, roughness_factor=1, **kwargs)
    builder.mesh_node(
        name="cave_wall",
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]),
        faces=np.array([[0, 2, 1]]),
        normals=np.array([[0, 1, 0]] * 3),
        texcoords=np.array([[0, 0], [1, 0], [0, 1]]),
        material_index=material,
    )
    path.write_bytes(builder.to_glb())
    return path


def test_bundle_preserves_glb_and_embedded_images(tmp_path):
    source = source_asset(tmp_path / "cave.glb")
    before = source.read_bytes()
    output = tmp_path / "bundle"
    paths = write_projected_material_bundle(source, output, tile_size_m=4, normal_strength=0.8)
    assert source.read_bytes() == before and not any(p.suffix == ".glb" for p in paths)
    report = json.loads((output / "settings.json").read_text())
    assert report["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["tile_size_m"] == 4 and report["normal_strength"] == 0.8
    asset = GlbAsset(source)
    for name, image_index in zip(
        ("cave_base_color.png", "cave_normal.png", "cave_metallic_roughness.png"),
        (0, 2, 1),
        strict=True,
    ):
        image = asset.document["images"][image_index]
        view = asset.document["bufferViews"][image["bufferView"]]
        data = asset.binary[
            view.get("byteOffset", 0) : view.get("byteOffset", 0) + view["byteLength"]
        ]
        assert (output / "textures" / name).read_bytes() == data
    assert len(list((output / "textures").iterdir())) == 3
    second = tmp_path / "second"
    write_projected_material_bundle(source, second, tile_size_m=4, normal_strength=0.8)
    for path in paths:
        assert path.read_bytes() == (second / path.relative_to(output)).read_bytes()


def test_neutral_bundle_is_skipped(tmp_path):
    source = source_asset(tmp_path / "cave.glb", textured=False)
    assert (
        write_projected_material_bundle(
            source, tmp_path / "bundle", tile_size_m=4, normal_strength=1
        )
        == ()
    )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "tile,strength",
    [(0, 1), (float("nan"), 1), (float("inf"), 1), (4, -1), (4, float("nan")), (4, 11)],
)
def test_invalid_projection_settings_fail_before_io(tmp_path, tile, strength):
    with pytest.raises(ValueError):
        write_projected_material_bundle(
            tmp_path / "missing.glb",
            tmp_path / "output",
            tile_size_m=tile,
            normal_strength=strength,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.integration
def test_native_projection_frames_and_channels(tmp_path):
    binary = os.environ.get("PLUME_BLENDER_BINARY")
    if not binary:
        pytest.skip("Set PLUME_BLENDER_BINARY for native material tests")
    script = Path(__file__).parent / "fixtures/blender/triplanar_check.py"
    wrapper = (
        "import runpy, os, sys, traceback\ntry:\n"
        f"    runpy.run_path({str(script.resolve())!r}, run_name='__main__')\n"
        "except BaseException:\n    traceback.print_exc(); sys.stderr.flush(); os._exit(1)\n"
    )
    with (tmp_path / "blender.log").open("w") as log:
        result = subprocess.run(
            [
                binary,
                "--background",
                "--threads",
                "2",
                "--python-exit-code",
                "1",
                "--python-expr",
                wrapper,
                "--",
                str(tmp_path),
                str(ASSETS / "blender_materials.py"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    assert result.returncode == 0, (tmp_path / "blender.log").read_text()[-4000:]
    report = json.loads((tmp_path / "native_triplanar.json").read_text())
    assert len(report["normal_cases"]) == 27
    assert report["uv_invariant"] and report["images_reused"]


@pytest.mark.parametrize("origin_top", [0, 1])
def test_shared_hlsl_compiles(tmp_path, origin_top):
    compiler = os.environ.get("PLUME_GLSLANG_BINARY")
    if not compiler:
        pytest.skip("Set PLUME_GLSLANG_BINARY for HLSL compiler checks")
    body = (ASSETS / "plume_triplanar_body.hlsl").read_text()
    wrapper = """Texture2D ColorMap, NormalMap, RoughnessMap;
SamplerState ColorMapSampler, NormalMapSampler, RoughnessMapSampler;
#define Texture2DSample(T,S,UV) T.Sample(S,UV)
float4 main(float3 PositionM : TEXCOORD0, float3 BaseNormal : TEXCOORD1) : SV_Target {
float TileSize=4, NormalStrength=1, BlendExponent=4;
float UVOriginTop=ORIGIN_TOP;
float3 BaseColor, NormalObject; float Roughness;
BODY
return float4(BaseColor + NormalObject, Roughness);
}""".replace("ORIGIN_TOP", str(origin_top)).replace("BODY", body)
    source = tmp_path / "shader.frag.hlsl"
    source.write_text(wrapper)
    result = subprocess.run(
        [
            compiler,
            "-D",
            "-V",
            "-S",
            "frag",
            "-e",
            "main",
            str(source),
            "-o",
            str(tmp_path / "shader.spv"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "shader.spv").stat().st_size > 100


@pytest.mark.parametrize("block_index", [0, 1, 2])
@pytest.mark.parametrize("variant", ["base", "shadows", "instanced_fog", "screen_shadows", "forward_plus"])
def test_unity_urp_shader_compiles_against_unity_headers(tmp_path, block_index, variant):
    """Compile real forward/depth/shadow passes, without pretending this launches Unity."""
    import re

    compiler = os.environ.get("PLUME_DXC_BINARY")
    headers = os.environ.get("PLUME_UNITY_HEADERS")
    if not compiler or not headers:
        pytest.skip("Set PLUME_DXC_BINARY and PLUME_UNITY_HEADERS for URP compile checks")
    shader = (ASSETS / "unity/PlumeContinuousRock.shader").read_text()
    common = re.search(r"HLSLINCLUDE(.*?)ENDHLSL", shader, re.S).group(1)
    blocks = re.findall(r"HLSLPROGRAM(.*?)ENDHLSL", shader, re.S)
    defines = {
        "base": [],
        "shadows": [
            "_MAIN_LIGHT_SHADOWS",
            "_ADDITIONAL_LIGHTS",
            "_ADDITIONAL_LIGHT_SHADOWS",
            "_SHADOWS_SOFT",
            "_CASTING_PUNCTUAL_LIGHT_SHADOW",
        ],
        "instanced_fog": ["INSTANCING_ON", "FOG_EXP2", "_ADDITIONAL_LIGHTS_VERTEX"],
        "screen_shadows": ["_MAIN_LIGHT_SHADOWS_SCREEN"],
        "forward_plus": ["_FORWARD_PLUS", "_ADDITIONAL_LIGHTS"],
    }[variant]
    preamble = (
        "#define SHADER_API_D3D11 1\n#define SHADER_TARGET 45\n"
        "#define UNITY_VERSION 202230\n#define UNITY_COMPILER_DXC 1\n"
        "#define UNITY_UNIFIED_SHADER_PRECISION_MODEL 1\n"
    )
    preamble += "".join(f"#define {name} 1\n" for name in defines)
    path = tmp_path / "urp.hlsl"
    path.write_text(preamble + common + blocks[block_index])
    entries = [("Vert", "Frag"), ("ShadowVert", "ShadowFrag"), ("DepthVert", "DepthFrag")][
        block_index
    ]
    for entry, target in zip(entries, ("vs_6_0", "ps_6_0"), strict=True):
        output = tmp_path / f"{entry}.dxil"
        result = subprocess.run(
            [
                compiler,
                "-flegacy-macro-expansion",
                "-HV",
                "2018",
                "-T",
                target,
                "-E",
                entry,
                "-I",
                headers,
                "-I",
                str(ASSETS / "unity"),
                str(path),
                "-Fo",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.stat().st_size > 100
