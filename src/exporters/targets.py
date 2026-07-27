"""Target packages built from the canonical PLUME cave geometry.

The adapters intentionally use open interchange formats and Python code.  They
do not invoke Blender or require a GUI application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np
import trimesh

from stages.geometry_export import export_geometry_glb, export_geometry_obj
from stages.geometry_types import CaveGeometry
from world import ExportConfig


@dataclass(frozen=True)
class ExportResult:
    """Files produced by one target adapter."""

    target: str
    primary_asset: Path
    files: tuple[Path, ...]
    warnings: tuple[str, ...] = ()


def export_target_asset(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output_root: str | Path,
    *,
    asset_name: str = "plume_cave",
) -> ExportResult:
    """Export a canonical cave using the selected target package."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_asset_name(asset_name)
    target = export_config.target

    if target in {"neutral", "blender", "ue5", "unity"}:
        result = _export_glb_or_obj(
            cave_geometry,
            export_config,
            output,
            safe_name,
        )
        descriptor = _write_target_descriptor(result, export_config, output, safe_name)
        return ExportResult(
            target=target,
            primary_asset=result.primary_asset,
            files=result.files + (descriptor,),
            warnings=result.warnings,
        )
    if target == "gazebo":
        return _export_gazebo(cave_geometry, export_config, output, safe_name)
    if target == "omniverse":
        return _export_omniverse(cave_geometry, export_config, output, safe_name)
    raise ValueError(f"Unsupported export target {target!r}")


def _export_glb_or_obj(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
) -> ExportResult:
    file_format = export_config.file_format
    if file_format == "glb":
        asset = export_geometry_glb(cave_geometry, output / f"{asset_name}.glb")
        files = (asset, asset.with_suffix(".manifest.json"))
    elif file_format == "obj":
        asset = export_geometry_obj(cave_geometry, output / f"{asset_name}.obj")
        files = tuple(
            path
            for path in (asset, asset.with_suffix(".mtl"))
            if path.exists()
        )
    else:
        raise ValueError(
            f"Target {export_config.target!r} currently supports glb or obj; "
            f"got {file_format!r}. USD is supported by the Omniverse adapter."
        )
    return ExportResult(
        target=export_config.target,
        primary_asset=asset,
        files=files,
        warnings=_pending_feature_warnings(export_config),
    )


def _write_target_descriptor(
    result: ExportResult,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
) -> Path:
    conventions = {
        "neutral": {
            "application_coordinates": "format-defined",
            "application_length_unit": "format-defined",
            "recommended_import_uniform_scale": 1.0,
        },
        "blender": {
            "application_coordinates": "right-handed Z-up",
            "application_length_unit": "metre",
            "application_units_per_asset_metre": 1.0,
            "recommended_import_uniform_scale": 1.0,
        },
        "ue5": {
            "application_coordinates": "left-handed Z-up",
            "application_length_unit": "centimetre",
            "application_units_per_asset_metre": 100.0,
            "recommended_import_uniform_scale": 1.0,
            "note": "Use UE Interchange glTF import; glTF metres convert to UE centimetres.",
        },
        "unity": {
            "application_coordinates": "left-handed Y-up",
            "application_length_unit": "metre",
            "application_units_per_asset_metre": 1.0,
            "recommended_import_uniform_scale": 1.0,
            "note": "Use a glTF importer that preserves supplied normals and tangents.",
        },
    }[export_config.target]
    descriptor = output / f"{asset_name}.{export_config.target}.json"
    payload = {
        "schema": "plume.target_export.v1",
        "target": export_config.target,
        "primary_asset": result.primary_asset.name,
        "source_coordinates": "right-handed Z-up metres",
        "asset_coordinates": (
            "glTF right-handed Y-up metres"
            if result.primary_asset.suffix.lower() == ".glb"
            else "right-handed Z-up metres"
        ),
        "target_conventions": conventions,
        "requirements": asdict(export_config),
        "warnings": list(result.warnings),
    }
    descriptor.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return descriptor


def _export_gazebo(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
) -> ExportResult:
    if export_config.file_format not in {"obj", "dae"}:
        raise ValueError("Gazebo export.format must be 'obj' or 'dae'")
    if export_config.file_format == "dae":
        raise ValueError(
            "DAE output is planned but not yet available without an additional "
            "Collada library; select export.format = 'obj'."
        )

    package = output / asset_name
    meshes = package / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    visual_mesh = export_geometry_obj(cave_geometry, meshes / f"{asset_name}.obj")

    model_config = package / "model.config"
    model_config.write_text(
        "\n".join(
            (
                '<?xml version="1.0"?>',
                "<model>",
                f"  <name>{escape(asset_name)}</name>",
                "  <version>1.0</version>",
                "  <sdf version=\"1.10\">model.sdf</sdf>",
                "  <description>Procedural PLUME lava tube</description>",
                "</model>",
                "",
            )
        ),
        encoding="utf-8",
    )
    model_sdf = package / "model.sdf"
    uri = f"model://{asset_name}/meshes/{visual_mesh.name}"
    model_sdf.write_text(
        "\n".join(
            (
                '<?xml version="1.0"?>',
                '<sdf version="1.10">',
                f'  <model name="{escape(asset_name)}">',
                "    <static>true</static>",
                '    <link name="cave">',
                '      <visual name="visual">',
                "        <geometry><mesh>",
                f"          <uri>{escape(uri)}</uri>",
                "        </mesh></geometry>",
                "      </visual>",
                '      <collision name="collision">',
                "        <geometry><mesh>",
                f"          <uri>{escape(uri)}</uri>",
                "        </mesh></geometry>",
                "      </collision>",
                "    </link>",
                "  </model>",
                "</sdf>",
                "",
            )
        ),
        encoding="utf-8",
    )
    descriptor = package / "plume_export.json"
    warning = (
        "The first Gazebo adapter reuses the visual mesh for collision. "
        "A simplified tiled collision mesh is scheduled in Phase 5."
    )
    warnings = (warning,) + tuple(
        item
        for item in _pending_feature_warnings(export_config)
        if "collision" not in item.lower()
    )
    descriptor.write_text(
        json.dumps(
            {
                "schema": "plume.target_export.v1",
                "target": "gazebo",
                "coordinates": "right-handed Z-up metres",
                "requirements": asdict(export_config),
                "warnings": list(warnings),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files = [visual_mesh, model_config, model_sdf, descriptor]
    material = visual_mesh.with_suffix(".mtl")
    if material.exists():
        files.append(material)
    return ExportResult(
        target="gazebo",
        primary_asset=model_sdf,
        files=tuple(files),
        warnings=warnings,
    )


def _export_omniverse(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
) -> ExportResult:
    if export_config.file_format not in {"usd", "usdc"}:
        raise ValueError("Omniverse export.format must be 'usd' or 'usdc'")
    if export_config.file_format == "usdc":
        raise ValueError(
            "Binary USDC requires the optional OpenUSD Python package; "
            "select export.format = 'usd' for dependency-free USDA output."
        )

    asset = output / f"{asset_name}.usd"
    _write_usda(cave_geometry, asset, asset_name)
    descriptor = output / f"{asset_name}.omniverse.json"
    warnings = _pending_feature_warnings(export_config)
    descriptor.write_text(
        json.dumps(
            {
                "schema": "plume.target_export.v1",
                "target": "omniverse",
                "primary_asset": asset.name,
                "coordinates": "right-handed Z-up metres",
                "usd": {"upAxis": "Z", "metersPerUnit": 1.0},
                "requirements": asdict(export_config),
                "warnings": list(warnings),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ExportResult(
        target="omniverse",
        primary_asset=asset,
        files=(asset, descriptor),
        warnings=warnings,
    )


def _write_usda(
    cave_geometry: CaveGeometry,
    output: Path,
    asset_name: str,
) -> None:
    vertices, faces = _canonical_cave_mesh(cave_geometry)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "PLUME_Cave"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "PLUME_Cave" (',
        '    kind = "component"',
        ")",
        "{",
    ]
    lines.extend(_usda_mesh_lines("CaveWall", vertices, faces, indent="    "))
    for event_mesh in cave_geometry.event_meshes:
        event_name = f"Event_{event_mesh.event_id:04d}_{_safe_asset_name(event_mesh.kind)}"
        lines.extend(
            _usda_mesh_lines(
                event_name,
                np.asarray(event_mesh.vertices, dtype=np.float64),
                np.asarray(event_mesh.faces, dtype=np.int64),
                indent="    ",
            )
        )
    lines.extend(("}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")


def _canonical_cave_mesh(cave_geometry: CaveGeometry) -> tuple[np.ndarray, np.ndarray]:
    if cave_geometry.assembled_vertices and cave_geometry.assembled_faces:
        return (
            np.asarray(cave_geometry.assembled_vertices, dtype=np.float64),
            np.asarray(cave_geometry.assembled_faces, dtype=np.int64),
        )

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    offset = 0
    for chunk in cave_geometry.chunk_meshes:
        vertices.extend(chunk.vertices)
        faces.extend(
            tuple(int(index) + offset for index in face)
            for face in chunk.faces
        )
        offset += len(chunk.vertices)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _usda_mesh_lines(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    indent: str,
) -> list[str]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    point_text = ", ".join(_tuple_text(point) for point in vertices)
    normal_text = ", ".join(_tuple_text(normal) for normal in normals)
    count_text = ", ".join("3" for _ in faces)
    index_text = ", ".join(str(int(index)) for index in np.asarray(faces).reshape(-1))
    return [
        f'{indent}def Mesh "{name}"',
        f"{indent}{{",
        f"{indent}    uniform bool doubleSided = 1",
        f"{indent}    int[] faceVertexCounts = [{count_text}]",
        f"{indent}    int[] faceVertexIndices = [{index_text}]",
        f'{indent}    uniform token orientation = "rightHanded"',
        f"{indent}    point3f[] points = [{point_text}]",
        f"{indent}    normal3f[] normals = [{normal_text}] (",
        f'{indent}        interpolation = "vertex"',
        f"{indent}    )",
        f"{indent}    uniform token subdivisionScheme = \"none\"",
        f"{indent}}}",
    ]


def _tuple_text(values: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):.8g}" for value in values) + ")"


def _safe_asset_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return safe.strip("_") or "plume_cave"


def _pending_feature_warnings(export_config: ExportConfig) -> tuple[str, ...]:
    warnings: list[str] = []
    if export_config.generate_collision:
        warnings.append(
            "A separate simplified collision asset is not implemented yet for "
            f"the {export_config.target} adapter."
        )
    if export_config.generate_lods:
        warnings.append(
            f"Generated LOD meshes are not implemented yet for the {export_config.target} adapter."
        )
    if export_config.generate_wall_shell:
        warnings.append(
            "Finite wall-shell generation is configured but remains scheduled "
            "for the sparse-SDF geometry phase."
        )
    return tuple(warnings)
