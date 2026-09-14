"""Inspect serialized packages while they are still in atomic staging."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from plume_advanced.identity import sha256_file
from plume_advanced.progress import report_progress
from plume_advanced.validation import PortableAssetValidator


class ExportInspectionError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("Export inspection failed: " + "; ".join(report["failures"]))


def _compare(actual, expected, name, *, tolerance=0.0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=tolerance):
        raise ValueError(f"Serialized {name} differs from the inspected surface")


def _inspect_texture_bindings(validator, required: dict[str, bool]) -> list[str]:
    document = validator.glb.document
    _, _, primitive = validator.glb.cave_primitive()
    material = document["materials"][primitive["material"]]
    pbr = material.get("pbrMetallicRoughness", {})
    bound = []
    for slot, enabled in required.items():
        if not enabled:
            continue
        info = (material if slot == "normalTexture" else pbr).get(slot, {})
        index = info.get("index")
        textures = document.get("textures", [])
        if type(index) is not int or not 0 <= index < len(textures):
            raise ValueError(f"Configured {slot} is missing or has an invalid texture index")
        source = textures[index].get("source")
        if type(source) is not int or not 0 <= source < len(document.get("images", [])):
            raise ValueError(f"Configured {slot} has an invalid image source")
        if info.get("texCoord", 0) != 0:
            raise ValueError(f"Configured {slot} references unavailable texture coordinates")
        if slot == "normalTexture":
            scale = float(info.get("scale", 1.0))
            if not np.isfinite(scale) or scale < 0:
                raise ValueError("Configured normalTexture has an invalid strength")
        bound.append(slot)
    return bound


def _inspect_obj(path, visual, *, object_name="cave_wall", position_tolerance=5.1e-10):
    counts = {"v": 0, "f": 0}
    buffers: dict[str, list] = {"v": [], "f": []}
    expected = {"v": visual["positions"], "f": visual["faces"]}
    found = False

    def flush(kind):
        if buffers[kind]:
            start = counts[kind]
            stop = start + len(buffers[kind])
            _compare(
                buffers[kind],
                expected[kind][start:stop],
                "OBJ " + kind,
                tolerance=position_tolerance if kind == "v" else 0.0,
            )
            counts[kind] = stop
            buffers[kind].clear()

    with path.open(encoding="utf-8") as source:
        for line in source:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "o":
                if found:
                    break
                found = tokens[1:] == [object_name]
            if not found:
                continue
            kind = tokens[0]
            if kind == "v":
                buffers[kind].append([float(v) for v in tokens[1:]])
            elif kind == "f":
                buffers[kind].append([int(v.split("/")[0]) - 1 for v in tokens[1:]])
            else:
                continue
            if len(buffers[kind]) == 8192:
                flush(kind)
    for kind in buffers:
        flush(kind)
    if counts != {"v": len(visual["positions"]), "f": len(visual["faces"])}:
        raise ValueError("OBJ is missing cave geometry")


def _inspect_usda(path, visual, *, object_name="CaveWall"):
    # This reads PLUME's own ASCII schema, not arbitrary third-party USD.
    text = path.read_text(encoding="utf-8")
    match = re.search(r'def Mesh "' + re.escape(object_name) + r'".*?\{(.*?)\n\s*\}', text, flags=re.S)
    if not match:
        raise ValueError(f"USD is missing {object_name}")
    body = match[1]
    for name, expected in (
        ("points", visual["positions"]),
        ("faceVertexIndices", visual["faces"]),
        ("faceVertexCounts", np.full(len(visual["faces"]), 3)),
    ):
        record = re.search(rf"\b{name} = \[(.*?)\]", body, flags=re.S)
        if not record:
            raise ValueError(f"USD is missing {name}")
        values = np.fromstring(record[1].translate(str.maketrans("(),", "   ")), sep=" ")
        if name == "points":
            # Read at the declared point3f precision, as a USD consumer does.
            # Fixed decimal tolerances either reject correct large coordinates
            # or conceal changed small coordinates. Require the inspected bits.
            values = values.astype(np.float32)
            expected = np.asarray(expected, dtype=np.float32)
        _compare(
            values,
            np.asarray(expected).ravel(),
            "USD " + name,
        )


def inspect_package(scene, files: tuple[Path, ...], staging: Path) -> dict:
    """GLB content checks and exact/tolerance-bounded geometry round trips.

    Run provenance is finalized later by the pipeline. It is deliberately not
    claimed by this package-only check. Native engine/shader compilation is not
    available inside this portable generation process.
    """
    report: dict = dict(
        schema="plume.serialized-inspection.v1",
        passed=False,
        failures=[],
        files=[],
        scope="GLB portable content plus GLB/OBJ/PLUME-USDA geometry round trip; native application import and final run provenance are separate",
    )
    assets = [
        p
        for p in files
        if p.suffix in {".glb", ".obj", ".usda", ".usd"}
        and "continuous_material" not in p.parts
    ]
    if not assets:
        report["failures"].append("No inspectable visual asset in package")
        raise ExportInspectionError(report)
    seen: dict = {}
    for index, path in enumerate(assets):
        report_progress("Serialized asset inspection", index, len(assets), path.name)
        record: dict = dict(path=path.relative_to(staging).as_posix(), sha256=sha256_file(path))
        try:
            key = (path.suffix, record["sha256"])
            if key in seen and path.suffix != ".glb":
                record["equivalent_to"] = seen[key]
            elif path.suffix == ".glb":
                config = scene.geometry.config
                required = {
                    "baseColorTexture": bool(config.cave_diffuse_texture),
                    "normalTexture": bool(config.cave_normal_texture),
                    "metallicRoughnessTexture": bool(config.cave_roughness_texture),
                }
                # Partial PBR materials are supported: do not require maps the
                # configuration intentionally omitted. Still validate every
                # configured binding, including normal/roughness-only materials.
                profile = "textured" if all(required.values()) else "neutral"
                validator = PortableAssetValidator(
                    path,
                    material_profile=profile,
                    expected_collision=bool(len(scene.collision_faces)),
                    run_manifest_path=staging / ".not-finalized.json",
                )
                checks = validator.validate(
                    include_run_provenance=False,
                    progress=lambda n, total, phase: report_progress(
                        "Portable asset checks", n, total, phase
                    ),
                )
                record["checks"] = [asdict(check) for check in checks]
                failed = [check.name for check in checks if not check.passed]
                if failed:
                    raise ValueError(", ".join(failed))
                record["configured_texture_bindings"] = _inspect_texture_bindings(
                    validator, required
                )
                node, _, primitive = validator.glb.cave_primitive()
                for name, key_name in (
                    ("POSITION", "positions"),
                    ("NORMAL", "normals"),
                    ("TANGENT", "tangents"),
                    ("TEXCOORD_0", "texcoords"),
                ):
                    _compare(
                        validator.glb.accessor(primitive["attributes"][name]),
                        scene.gltf_visual[key_name],
                        "GLB " + name,
                    )
                _compare(
                    validator.glb.accessor(primitive["indices"]).reshape(-1, 3),
                    scene.gltf_visual["faces"],
                    "GLB faces",
                )
                for name, default in (
                    ("translation", [0, 0, 0]),
                    ("scale", [1, 1, 1]),
                    ("rotation", [0, 0, 0, 1]),
                    ("matrix", np.eye(4).ravel().tolist()),
                ):
                    _compare(node.get(name, default), default, "GLB node " + name)
            elif path.suffix == ".obj":
                if path.name.endswith("_collision.obj"):
                    _inspect_obj(path, dict(positions=scene.collision_vertices, faces=scene.collision_faces),
                                 object_name="cave_collision", position_tolerance=0.)
                else:
                    _inspect_obj(path, scene.canonical_visual)
            else:
                _inspect_usda(path, scene.canonical_visual)
                if len(scene.collision_faces):
                    _inspect_usda(path, dict(positions=scene.collision_vertices,
                                            faces=scene.collision_faces), object_name="CaveCollision")
            seen[key] = record["path"]
            record["passed"] = True
        except (ValueError, KeyError, IndexError) as error:
            record.update(passed=False, error=str(error))
            report["failures"].append(f"{record['path']}: {error}")
        report["files"].append(record)
    report["passed"] = not report["failures"]
    if not report["passed"]:
        raise ExportInspectionError(report)
    report_progress(
        "Serialized asset inspection", len(assets), len(assets), "package content verified"
    )
    return report
