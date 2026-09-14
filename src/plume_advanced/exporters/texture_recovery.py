"""Deterministic map preparation and bounded texture-package recovery gates.

Only recover information that is present: normalize usable normal vectors and
honour an explicitly declared convention. Never guess missing maps or replace
undefined vectors with invented surface detail. Source assets remain untouched.
"""

from __future__ import annotations

import errno
import json
import shlex
from dataclasses import replace
from importlib.resources import files as resources
from pathlib import Path
from typing import NoReturn

import numpy as np
from PIL import Image

from plume_advanced.identity import sha256_file
from plume_advanced.progress import report_progress
from plume_advanced.stages.geometry_export import _convert_exr_to_image, _load_displacement_image
from plume_advanced.validation import GlbAsset

from .inspection import ExportInspectionError
from .projected_materials import ASSET_FILES

ROLES = ("diffuse", "normal", "roughness", "displacement")
VERSION = "plume.texture-recovery.v1"
RESOURCE_ERRORS = {errno.ENOSPC, errno.ENOMEM, errno.EIO, errno.EMFILE, errno.ENFILE}


class TextureRecoveryError(ExportInspectionError):
    """A diagnosed material failure, distinct from geometry/programming errors."""

    def __init__(self, report: dict):
        super().__init__(report)


def new_report() -> dict:
    return dict(
        schema=VERSION,
        passed=False,
        outcome="not_requested",
        assets=[],
        failures=[],
        package_attempts=[],
        scope=(
            "Source-map decoding, bounded normal repair, portable map copies and serialized "
            "bindings/bundle integrity. Does not certify tile seamlessness, geological "
            "appearance, lighting, native shader compilation or native application import."
        ),
    )


def _fail(report: dict, message: str) -> NoReturn:
    report["passed"] = False
    report["failures"].append(message)
    raise TextureRecoveryError(report)


def _normal_stats(image: Image.Image) -> dict:
    """Bound scratch memory independently of a 4K/8K tile's total pixel count."""
    invalid = nonunit = 0
    maximum = 0.0
    for y in range(0, image.height, 128):
        vectors = (
            np.asarray(
                image.crop((0, y, image.width, min(y + 128, image.height))), dtype=np.float32
            )[..., :3]
            / 127.5
            - 1.0
        )
        lengths = np.linalg.norm(vectors, axis=-1)
        invalid += int(np.count_nonzero(lengths < 0.1))
        error = np.abs(lengths - 1.0)
        nonunit += int(np.count_nonzero(error > 0.02))
        maximum = max(maximum, float(error.max()))
    return dict(undefined_vectors=invalid, nonunit_vectors=nonunit, max_length_error=maximum)


def _repair_normal(image: Image.Image, *, flip_green: bool) -> Image.Image:
    result = image.copy()
    for y in range(0, image.height, 128):
        stop = min(y + 128, image.height)
        vectors = (
            np.asarray(image.crop((0, y, image.width, stop)), dtype=np.float32)[..., :3] / 127.5
            - 1.0
        )
        if flip_green:
            vectors[..., 1] *= -1
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        pixels = np.rint(np.clip((vectors + 1.0) * 127.5, 0, 255)).astype(np.uint8)
        result.paste(Image.fromarray(pixels), (0, y))
    return result


def _decode(path: Path, role: str, maximum: int) -> tuple[Image.Image, tuple[int, int], str]:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".exr"}:
        raise ValueError("Unsupported texture format; use PNG, JPEG or EXR")
    if path.suffix.lower() == ".exr":
        image = _convert_exr_to_image(path, max_size=maximum)
        if image is None:
            raise ValueError("EXR decoding failed; check ImageMagick and the source image")
        # EXR conversion already enforces the configured size; do not invent
        # dimensions of the original master without an EXR header reader.
        original, mode = image.size, "EXR (converted within size limit)"
    else:
        with Image.open(path) as source:
            source.load()  # Force truncated/corrupt files to fail before meshing/export.
            original, mode = source.size, source.mode
            if role == "normal" and source.mode not in {"RGB", "RGBA"}:
                raise ValueError("Normal map must contain three encoded vector channels")
            if role == "roughness" and source.mode in {"I", "I;16", "I;16B", "I;16L"}:
                # Use storage depth, not observed maximum (dark 16-bit maps are valid).
                pixels = np.asarray(source, dtype=np.float32)
                if np.any((pixels < 0) | (pixels > 65535)):
                    raise ValueError("Roughness pixels are outside the 16-bit range")
                image = Image.fromarray(np.rint(pixels / 257.0).astype(np.uint8))
            else:
                image = source.convert("L" if role == "roughness" else "RGB")
    image = image.convert("L" if role == "roughness" else "RGB")
    image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    return image, original, mode


def prepare_texture_assets(geometry, staging: Path, report: dict):
    """Prepare shared immutable maps once; return a new effective geometry.

    Displacement precision and processing stay with the existing geometric
    acceptance loop. Its source is decoded and hashed here, never quantized.
    Partial materials remain partial. Identical source/role pairs share a file.
    """
    config = geometry.config
    maximum = config.embedded_texture_max_size
    if type(maximum) is not int or maximum < 1:
        _fail(report, "embedded_texture_max_size must be a positive integer")
    cache: dict[tuple, str] = {}
    source_hashes: dict[Path, str] = {}

    def prepare(role: str, source: str, owner: str) -> str:
        if not source:
            return source
        if role not in ROLES:
            _fail(report, f"{owner}: unsupported material role {role}")
        path = Path(source)
        try:
            before_hash = sha256_file(path)
        except OSError as error:
            if error.errno in RESOURCE_ERRORS:
                raise
            _fail(report, f"{owner}/{role}: source unavailable: {path.name}: {error.strerror}")
        if path in source_hashes and source_hashes[path] != before_hash:
            _fail(report, f"Source changed during texture preparation: {path.name}")
        source_hashes[path] = before_hash
        convention = config.cave_normal_convention if owner == "cave" else "opengl"
        key = (before_hash, role, convention if role == "normal" else "")
        if key in cache:
            record = next(a for a in report["assets"] if a["key"] == list(key))
            record["users"].append(owner)
            return cache[key]
        record = dict(
            key=list(key),
            role=role,
            users=[owner],
            source_name=path.name,
            source_sha256=before_hash,
            repairs=[],
            passed=False,
        )
        report["assets"].append(record)
        report_progress("Texture source inspection", detail=f"{owner}/{role}: {path.name}")
        try:
            if role == "displacement":
                image = _load_displacement_image(source, {}, max_size=maximum)
                if image is None or not np.isfinite(np.asarray(image)).all():
                    raise ValueError("Displacement map cannot be decoded to finite heights")
                record["processing"] = (
                    "Precision preserved; displacement repair uses visual acceptance"
                )
                result = source
            else:
                image, original, mode = _decode(path, role, maximum)
                record.update(
                    source_size=list(original),
                    source_mode=mode,
                    prepared_size=list(image.size),
                    color_space="sRGB" if role == "diffuse" else "linear data",
                )
                if role == "normal":
                    record["before"] = _normal_stats(image)
                    if record["before"]["undefined_vectors"]:
                        raise ValueError(
                            "Normal map has undefined vectors; cannot infer missing directions"
                        )
                    flip = convention == "directx"
                    if record["before"]["nonunit_vectors"] or flip:
                        if not config.texture_repair_attempts:
                            raise ValueError(
                                "Normal map needs repair but texture_repair_attempts is zero"
                            )
                        report_progress(
                            "Texture map repair",
                            0,
                            1,
                            f"{owner}: normalize vectors"
                            + (" and flip declared DirectX Y" if flip else ""),
                        )
                        image = _repair_normal(image, flip_green=flip)
                        record["repairs"].append("normalize_normal_vectors")
                        if flip:
                            record["repairs"].append("directx_to_opengl")
                        report_progress("Texture map repair", 1, 1, "rechecking repaired map")
                    record["after"] = _normal_stats(image)
                    if record["after"]["undefined_vectors"] or record["after"]["nonunit_vectors"]:
                        raise ValueError("Normal map remains invalid after bounded repair")
                directory = staging / "texture_assets"
                directory.mkdir(exist_ok=True)
                target = directory / f"{role}_{before_hash}_{key[2] or 'data'}.png"
                image.save(target, format="PNG")
                with Image.open(target) as decoded:
                    decoded.load()
                    if not np.array_equal(np.asarray(decoded), np.asarray(image)):
                        raise ValueError("Prepared PNG failed its pixel round trip")
                record.update(
                    path=target.relative_to(staging).as_posix(),
                    sha256=sha256_file(target),
                    prepared_pixel_sha256=_pixel_hash(image),
                )
                result = str(target)
            if sha256_file(path) != before_hash:
                raise ValueError("Source changed during texture preparation")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            if isinstance(error, OSError) and error.errno in RESOURCE_ERRORS:
                raise
            _fail(report, f"{owner}/{role}: {path.name}: {error}")
        record["passed"] = True
        cache[key] = result
        return result

    cave_maps = {
        f"cave_{role}_texture": prepare(role, getattr(config, f"cave_{role}_texture"), "cave")
        for role in ROLES
    }
    events = tuple(
        replace(
            event,
            material_maps=tuple(
                (role, prepare(role, path, f"event_{event.event_id}"))
                for role, path in event.material_maps
            ),
        )
        for event in geometry.event_meshes
    )
    for path, digest in source_hashes.items():
        if sha256_file(path) != digest:
            _fail(report, f"Source changed during texture preparation: {path.name}")
    report["outcome"] = (
        "repaired"
        if any(a["repairs"] for a in report["assets"])
        else "unchanged"
        if report["assets"]
        else "not_requested"
    )
    return replace(
        geometry,
        config=replace(config, **cave_maps, cave_normal_convention="opengl"),
        event_meshes=events,
    )


def _pixel_hash(image: Image.Image) -> str:
    import hashlib

    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def inspect_texture_package(scene, paths: tuple[Path, ...], staging: Path, report: dict) -> dict:
    """Verify serialized texture content independently of source preparation.

    A failure here can trigger one reserialization from the same prepared scene.
    The ordinary portable/geometry validator still runs afterwards; it is never
    bypassed or converted into a texture retry.
    """
    result: dict = dict(passed=False, failures=[], checked_files=[])
    try:
        for record in report["assets"]:
            if "path" in record:
                path = staging / record["path"]
                if sha256_file(path) != record["sha256"]:
                    raise ValueError(f"Prepared texture changed: {record['path']}")
                result["checked_files"].append(dict(path=record["path"], sha256=record["sha256"]))
        glbs = [p for p in paths if p.suffix == ".glb"]
        for mtl in (p for p in paths if p.suffix == ".mtl"):
            owner = ""
            found: set[tuple[str, str]] = set()
            for line in mtl.read_text().splitlines():
                tokens = shlex.split(line, comments=True)
                if not tokens:
                    continue
                if tokens[0] == "newmtl":
                    owner = (
                        "cave"
                        if tokens[1] == "cave_wall_material"
                        else (
                            f"event_{int(tokens[1].split('_')[1])}"
                            if tokens[1].startswith("event_")
                            else ""
                        )
                    )
                role = {
                    "map_Kd": "diffuse",
                    "norm": "normal",
                    "map_Bump": "normal",
                    "map_Pr": "roughness",
                }.get(tokens[0])
                if not role or not owner:
                    continue
                expected = next(
                    (a for a in report["assets"] if a["role"] == role and owner in a["users"]), None
                )
                path = (mtl.parent / tokens[-1]).resolve()
                if (
                    expected is None
                    or not path.is_relative_to(staging.resolve())
                    or sha256_file(path) != expected["sha256"]
                ):
                    raise ValueError(f"Invalid portable OBJ {owner}/{role} reference in {mtl.name}")
                found.add((owner, role))
                result["checked_files"].append(
                    dict(path=path.relative_to(staging).as_posix(), sha256=sha256_file(path))
                )
            for a in report["assets"]:
                if (
                    a["role"] != "displacement"
                    and "cave" in a["users"]
                    and ("cave", a["role"]) not in found
                ):
                    raise ValueError(f"Missing OBJ cave {a['role']} binding in {mtl.name}")
            result["checked_files"].append(
                dict(path=mtl.relative_to(staging).as_posix(), sha256=sha256_file(mtl))
            )
        for path in glbs:
            report_progress("Texture binding inspection", detail=path.name)
            asset = GlbAsset(path)
            # Check every authored material, including shared Rocky maps.
            for node in asset.document["nodes"]:
                name = node.get("name", "")
                if name == "cave_wall":
                    owner = "cave"
                elif name.startswith("event_") and "mesh" in node:
                    owner = f"event_{int(name.split('_')[1])}"
                else:
                    continue
                primitive = asset.document["meshes"][node["mesh"]]["primitives"][0]
                material = asset.document["materials"][primitive["material"]]
                pbr = material.get("pbrMetallicRoughness", {})
                for role, slot in (
                    ("diffuse", "baseColorTexture"),
                    ("normal", "normalTexture"),
                    ("roughness", "metallicRoughnessTexture"),
                ):
                    expected = next(
                        (a for a in report["assets"] if a["role"] == role and owner in a["users"]),
                        None,
                    )
                    info = (material if role == "normal" else pbr).get(slot)
                    if expected is None:
                        if info is not None:
                            raise ValueError(f"Unrequested {owner} {slot}")
                        continue
                    if info is None or info.get("texCoord", 0) != 0:
                        raise ValueError(f"Missing or misbound {owner} {slot}")
                    texture = asset.document["textures"][info["index"]]
                    sampler = asset.document["samplers"][texture["sampler"]]
                    if any(sampler.get(axis, 10497) != 10497 for axis in ("wrapS", "wrapT")):
                        raise ValueError(f"{owner} {slot} must repeat its tile")
                    decoded = asset.embedded_image(texture["source"]).convert("RGB")
                    if list(decoded.size) != expected["prepared_size"]:
                        raise ValueError(f"{owner} {slot} dimensions changed")
                    if role == "roughness":
                        red, green, blue = decoded.split()
                        if red.getextrema() != (255, 255) or blue.getextrema() != (0, 0):
                            raise ValueError("Invalid metallic/roughness channel packing")
                        decoded = green.convert("RGB")
                    if _pixel_hash(decoded) != expected["prepared_pixel_sha256"]:
                        raise ValueError(f"{owner} {slot} pixels differ from accepted source")
                    if role == "normal" and info.get("scale", 1) != (
                        scene.geometry.config.cave_normal_scale if owner == "cave" else 1
                    ):
                        raise ValueError(f"{owner} normal strength changed")
            result["checked_files"].append(
                dict(path=path.relative_to(staging).as_posix(), sha256=sha256_file(path))
            )
        complete = all(
            getattr(scene.geometry.config, f"cave_{role}_texture")
            for role in ("diffuse", "normal", "roughness")
        )
        if complete and glbs:
            bundle = staging / "continuous_material"
            settings_path = bundle / "settings.json"
            settings = json.loads(settings_path.read_text())
            if settings["source_sha256"] != sha256_file(glbs[0]):
                raise ValueError("Continuous material refers to a different GLB")
            for key, value in (
                ("tile_size_m", scene.geometry.config.cave_texture_scale_m),
                ("normal_strength", scene.geometry.config.cave_normal_scale),
                ("blend_exponent", 4.0),
            ):
                if settings[key] != value:
                    raise ValueError(f"Continuous material {key} differs from accepted settings")
            expected_names = {
                "cave_base_color.png",
                "cave_normal.png",
                "cave_metallic_roughness.png",
            }
            if set(settings["texture_sha256"]) != expected_names or set(
                settings["adapter_sha256"]
            ) != set(ASSET_FILES):
                raise ValueError("Continuous material bundle is incomplete")
            # Check against the actual embedded bytes, not only editable settings.
            glb = GlbAsset(glbs[0])
            _, _, prim = glb.cave_primitive()
            mat = glb.document["materials"][prim["material"]]
            for name, binding in (
                ("cave_base_color.png", mat["pbrMetallicRoughness"]["baseColorTexture"]),
                ("cave_normal.png", mat["normalTexture"]),
                (
                    "cave_metallic_roughness.png",
                    mat["pbrMetallicRoughness"]["metallicRoughnessTexture"],
                ),
            ):
                import hashlib

                texture = glb.document["textures"][binding["index"]]
                image = glb.document["images"][texture["source"]]
                view = glb.document["bufferViews"][image["bufferView"]]
                start = view.get("byteOffset", 0)
                digest = hashlib.sha256(glb.binary[start : start + view["byteLength"]]).hexdigest()
                member = bundle / "textures" / name
                if settings["texture_sha256"][name] != digest or sha256_file(member) != digest:
                    raise ValueError(f"Continuous texture missing or stale: {name}")
            for name in ASSET_FILES:
                reference = (
                    resources("plume_advanced").joinpath("material_assets", name).read_bytes()
                )
                member = bundle / name
                if (
                    member.read_bytes() != reference
                    or sha256_file(member) != settings["adapter_sha256"][name]
                ):
                    raise ValueError(f"Continuous adapter missing or stale: {name}")
            for member in sorted(p for p in bundle.rglob("*") if p.is_file()):
                result["checked_files"].append(
                    dict(path=member.relative_to(staging).as_posix(), sha256=sha256_file(member))
                )
    except (OSError, ValueError, KeyError, IndexError) as error:
        if isinstance(error, OSError) and error.errno in RESOURCE_ERRORS:
            raise
        result["failures"].append(str(error).replace(str(staging), "<package>"))
    result["passed"] = not result["failures"]
    return result
