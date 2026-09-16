"""Optional native continuous materials alongside an unchanged portable GLB."""

from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path

from plume_advanced.progress import report_progress
from plume_advanced.validation import GlbAsset

from .atomic import atomic_output_directory

ASSET_FILES = (
    "blender_materials.py",
    "apply_blender_material.py",
    "SETUP.txt",
    "plume_triplanar_body.hlsl",
    "unity/PlumeTriplanar.hlsl",
    "unity/PlumeContinuousRock.shader",
    "unity/Editor/PlumeMaterialInstaller.cs",
    "unreal/create_material.py",
)


def write_projected_material_bundle(
    source: Path,
    output: Path,
    *,
    tile_size_m: float,
    normal_strength: float,
) -> tuple[Path, ...]:
    """Extract images byte-for-byte and ship native adapters, never duplicate geometry.

    Neutral/partial materials return no bundle. An invalid complete material
    raises before publication. The new directory is committed atomically.
    """
    if not math.isfinite(tile_size_m) or not 1e-6 <= tile_size_m <= 1e6:
        raise ValueError("Texture tile size must be finite and positive")
    if not math.isfinite(normal_strength) or not 0 <= normal_strength <= 10:
        raise ValueError("Normal strength must be in [0, 10]")
    asset = GlbAsset(source)
    _, _, primitive = asset.cave_primitive()
    material = asset.document["materials"][primitive["material"]]
    pbr = material.get("pbrMetallicRoughness", {})
    if (
        not {"baseColorTexture", "metallicRoughnessTexture"} <= pbr.keys()
        or "normalTexture" not in material
    ):
        return ()
    if pbr.get("metallicFactor", 1) != 0 or pbr.get("baseColorFactor", [1] * 4) != [1] * 4:
        raise ValueError("Continuous rock material expects white base factor and nonmetallic rock")
    if pbr.get("roughnessFactor", 1) != 1:
        raise ValueError("Continuous rock material expects a unit roughness factor")
    report_progress("Continuous materials", 0, 3, "extracting the embedded PBR tile")
    image_roles = {
        "cave_base_color.png": pbr["baseColorTexture"],
        "cave_normal.png": material["normalTexture"],
        "cave_metallic_roughness.png": pbr["metallicRoughnessTexture"],
    }
    resource_root = files("plume_advanced").joinpath("material_assets")
    hashes = {}
    with atomic_output_directory(output) as staging:
        for name in ASSET_FILES:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resource_root.joinpath(name).read_bytes())
        (staging / "textures").mkdir()
        for index, (name, binding) in enumerate(image_roles.items()):
            texture = asset.document["textures"][binding["index"]]
            image = asset.document["images"][texture["source"]]
            if image.get("mimeType") != "image/png" or "bufferView" not in image:
                raise ValueError("Expected embedded PNG maps")
            view = asset.document["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            data = asset.binary[start : start + view["byteLength"]]
            if len(data) != view["byteLength"] or bytes(data[:8]) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Invalid embedded PNG buffer")
            (staging / "textures" / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
            report_progress("Continuous materials", index + 1, 3, name)
        (staging / "settings.json").write_text(
            json.dumps(
                {
                    "schema": "plume.continuous-materials.v1",
                    "tile_size_m": tile_size_m,
                    "normal_strength": normal_strength,
                    "blend_exponent": 4.0,
                    "source_asset_name": source.name,
                    "source_sha256": hashlib.sha256(asset.data).hexdigest(),
                    "texture_sha256": hashes,
                    "adapter_sha256": {
                        name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                        for name in ASSET_FILES
                    },
                    "source_asset_modified": False,
                    "native_engine_validation": "Not implied by packaging; see separate validation reports.",
                },
                indent=2,
            )
            + "\n"
        )
        names = sorted(path.relative_to(staging) for path in staging.rglob("*") if path.is_file())
    return tuple(output / name for name in names)
