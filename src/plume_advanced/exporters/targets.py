"""Target packages built from the canonical PLUME cave geometry.

The adapters intentionally use open interchange formats and Python code.  They
do not invoke Blender or require a GUI application.
"""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np
import trimesh

from plume_advanced.stages.geometry_export import (
    build_cave_visual_surface,
    export_cave_texture_files,
    export_geometry_glb,
    export_geometry_obj,
)
from plume_advanced.stages.geometry_types import CaveGeometry
from plume_advanced.world import APPLICATION_EXPORT_FORMATS, ExportConfig

from .atomic import atomic_output_directory
from .scene import PreparedExportScene, canonical_cave_mesh, prepare_export_scene


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
    """Prepare once, stage a complete package, and publish it atomically."""

    output = Path(output_root)
    safe_name = _safe_asset_name(asset_name)
    _validate_export_request(export_config)
    scene = prepare_export_scene(cave_geometry)
    with atomic_output_directory(output) as staging:
        staged = _export_target_asset_in_place(
            scene,
            export_config,
            staging,
            safe_name,
        )
    return ExportResult(
        target=staged.target,
        primary_asset=output / staged.primary_asset.relative_to(staging),
        files=tuple(output / path.relative_to(staging) for path in staged.files),
        warnings=staged.warnings,
    )


def _validate_export_request(export_config: ExportConfig) -> None:
    """Reject impossible direct API requests before preparing expensive geometry."""

    target = export_config.target
    file_format = export_config.file_format
    if target not in {"all", "neutral", *APPLICATION_EXPORT_FORMATS}:
        raise ValueError(f"Unsupported export target {target!r}")
    if target == "all" and file_format != "auto":
        raise ValueError("All-target export.format must be 'auto'")
    if target in {"neutral", "blender", "ue5", "unity"} and file_format not in {
        "glb",
        "obj",
    }:
        raise ValueError(
            f"Target {target!r} currently supports glb or obj; got {file_format!r}. "
            "USD is supported by the Omniverse adapter."
        )
    if target == "gazebo" and file_format != "obj":
        raise ValueError("Gazebo export.format must be 'obj'")
    if target == "omniverse" and file_format != "usd":
        raise ValueError("Omniverse export.format must be 'usd'")


def _export_target_asset_in_place(
    scene: PreparedExportScene,
    export_config: ExportConfig,
    output: Path,
    safe_name: str,
) -> ExportResult:
    """Serialize one already prepared scene inside an isolated directory."""

    cave_geometry = scene.geometry
    target = export_config.target

    if target == "all":
        return _export_all_targets(scene, export_config, output, safe_name)

    if target == "blender":
        result = _export_blender(
            cave_geometry,
            export_config,
            output,
            safe_name,
            scene,
        )
        descriptor = _write_target_descriptor(result, export_config, output, safe_name)
        return ExportResult(
            target=target,
            primary_asset=result.primary_asset,
            files=result.files + (descriptor,),
            warnings=result.warnings,
        )
    if target in {"neutral", "ue5", "unity"}:
        result = _export_glb_or_obj(
            cave_geometry,
            export_config,
            output,
            safe_name,
            scene,
        )
        files = list(result.files)
        if target == "neutral" and export_config.file_format == "glb":
            fallback_obj = export_geometry_obj(
                cave_geometry,
                output / f"{safe_name}_fallback.obj",
                visual_surface=scene.canonical_visual,
            )
            files.append(fallback_obj)
            fallback_mtl = fallback_obj.with_suffix(".mtl")
            if fallback_mtl.exists():
                files.append(fallback_mtl)
            result = ExportResult(
                target=result.target,
                primary_asset=result.primary_asset,
                files=tuple(files),
                warnings=result.warnings,
            )
        descriptor = _write_target_descriptor(result, export_config, output, safe_name)
        guide = _write_engine_import_guide(
            export_config.target,
            output,
            result.primary_asset,
            collision_asset=next(
                (path for path in result.files if path.name.endswith("_collision.obj")),
                None,
            ),
        )
        return ExportResult(
            target=target,
            primary_asset=result.primary_asset,
            files=result.files + (descriptor,) + ((guide,) if guide else ()),
            warnings=result.warnings,
        )
    if target == "gazebo":
        return _export_gazebo(cave_geometry, export_config, output, safe_name, scene)
    if target == "omniverse":
        return _export_omniverse(cave_geometry, export_config, output, safe_name, scene)
    raise ValueError(f"Unsupported export target {target!r}")


def _export_all_targets(
    scene: PreparedExportScene,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
) -> ExportResult:
    """Build every application package from one canonical cave geometry."""

    results: list[ExportResult] = []
    for target, file_format in APPLICATION_EXPORT_FORMATS.items():
        target_config = replace(
            export_config,
            target=target,
            file_format=file_format,
        )
        results.append(
            _export_target_asset_in_place(
                scene,
                target_config,
                output / target,
                asset_name,
            )
        )

    manifest = output / f"{asset_name}.all_exports.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "plume.all_exports.v1",
                "source_coordinates": "right-handed Z-up metres",
                "targets": {
                    result.target: {
                        "primary_asset": result.primary_asset.relative_to(output).as_posix(),
                        "files": [
                            path.relative_to(output).as_posix() for path in result.files
                        ],
                        "warnings": list(result.warnings),
                    }
                    for result in results
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files = tuple(path for result in results for path in result.files) + (manifest,)
    warnings = tuple(warning for result in results for warning in result.warnings)
    return ExportResult(
        target="all",
        primary_asset=manifest,
        files=files,
        warnings=warnings,
    )


def _export_blender(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
    scene: PreparedExportScene,
) -> ExportResult:
    """Write a Blender-oriented package without invoking Blender."""

    result = _export_glb_or_obj(
        cave_geometry,
        export_config,
        output,
        asset_name,
        scene,
    )
    files = list(result.files)
    fallback_obj: Path | None = None
    if export_config.file_format == "glb":
        fallback_obj = export_geometry_obj(
            cave_geometry,
            output / f"{asset_name}_fallback.obj",
            visual_surface=scene.canonical_visual,
        )
        files.append(fallback_obj)
        fallback_mtl = fallback_obj.with_suffix(".mtl")
        if fallback_mtl.exists():
            files.append(fallback_mtl)

    import_script = _write_blender_import_script(
        output=output,
        asset_name=asset_name,
        primary_asset=result.primary_asset,
        fallback_obj=fallback_obj,
    )
    instructions = _write_blender_import_instructions(
        output=output,
        primary_asset=result.primary_asset,
        fallback_obj=fallback_obj,
        import_script=import_script,
    )
    validation = _write_blender_validation_report(
        result.primary_asset,
        output / f"{asset_name}.blender_validation.json",
    )
    files.extend((import_script, instructions, validation))
    return ExportResult(
        target="blender",
        primary_asset=result.primary_asset,
        files=tuple(files),
        warnings=result.warnings,
    )


def _write_blender_import_script(
    *,
    output: Path,
    asset_name: str,
    primary_asset: Path,
    fallback_obj: Path | None,
) -> Path:
    script = output / f"{asset_name}_import_blender.py"
    fallback_name = (
        fallback_obj.name
        if fallback_obj is not None
        else primary_asset.name
        if primary_asset.suffix.lower() == ".obj"
        else ""
    )
    script.write_text(
        f'''"""Import the generated PLUME cave into Blender.

Open this file in Blender's Scripting workspace and choose Run Script.
"""

from pathlib import Path
import bpy

PACKAGE_DIR = Path(__file__).resolve().parent
PRIMARY_ASSET = PACKAGE_DIR / {primary_asset.name!r}
FALLBACK_OBJ = PACKAGE_DIR / {fallback_name!r}


def import_plume_cave() -> None:
    if PRIMARY_ASSET.suffix.lower() == ".glb":
        try:
            bpy.ops.import_scene.gltf(filepath=str(PRIMARY_ASSET))
            print(f"Imported PLUME cave from {{PRIMARY_ASSET}}")
            return
        except Exception as error:
            print(f"glTF import failed: {{error}}")
            if not FALLBACK_OBJ.name:
                raise

    if not FALLBACK_OBJ.is_file():
        raise FileNotFoundError(f"No Blender fallback asset found: {{FALLBACK_OBJ}}")
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(FALLBACK_OBJ))
    else:
        bpy.ops.import_scene.obj(filepath=str(FALLBACK_OBJ))
    print(f"Imported PLUME OBJ fallback from {{FALLBACK_OBJ}}")


if __name__ == "__main__":
    import_plume_cave()
''',
        encoding="utf-8",
    )
    return script


def _write_blender_import_instructions(
    *,
    output: Path,
    primary_asset: Path,
    fallback_obj: Path | None,
    import_script: Path,
) -> Path:
    instructions = output / "README_IMPORT_BLENDER.txt"
    fallback_line = (
        f"Fallback: File > Import > Wavefront (.obj), then select {fallback_obj.name}."
        if fallback_obj is not None
        else "No OBJ fallback was requested for this package."
    )
    instructions.write_text(
        "\n".join(
            (
                "PLUME-Advanced Blender import",
                "==============================",
                "",
                "Do not use File > Open. Blender uses that command for .blend projects.",
                "",
                "Recommended:",
                "1. In Blender choose File > Import > glTF 2.0 (.glb/.gltf).",
                f"2. Select {primary_asset.name}.",
                "",
                fallback_line,
                "",
                "Automated alternative:",
                "1. Open Blender's Scripting workspace.",
                f"2. Open {import_script.name}.",
                "3. Choose Run Script.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return instructions


def _write_blender_validation_report(asset: Path, output_path: Path) -> Path:
    """Record independent parse results for the generated interchange asset."""

    report: dict[str, object] = {
        "schema": "plume.blender_validation.v1",
        "asset": asset.name,
        "valid": False,
    }
    try:
        if asset.suffix.lower() == ".glb":
            data = asset.read_bytes()
            if len(data) < 20:
                raise ValueError("GLB is shorter than its required header")
            magic = data[:4]
            version = int.from_bytes(data[4:8], "little")
            declared_length = int.from_bytes(data[8:12], "little")
            if magic != b"glTF" or version != 2 or declared_length != len(data):
                raise ValueError("GLB header or declared byte length is invalid")
        loaded_scene: object = trimesh.load(asset, force="scene", process=False)
        if not isinstance(loaded_scene, trimesh.Scene):
            raise ValueError("Imported asset did not produce a scene")
        bounds = np.asarray(loaded_scene.bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("Imported scene has invalid bounds")
        report.update(
            {
                "valid": True,
                "parser": "trimesh",
                "geometry_count": len(loaded_scene.geometry),
                "scene_node_count": len(loaded_scene.graph.nodes_geometry),
                "bounds": bounds.tolist(),
            }
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["valid"]:
        raise ValueError(
            f"Generated Blender asset failed validation; see {output_path}"
        )
    return output_path


def _export_glb_or_obj(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
    scene: PreparedExportScene,
) -> ExportResult:
    file_format = export_config.file_format
    files: list[Path]
    if file_format == "glb":
        asset = export_geometry_glb(
            cave_geometry,
            output / f"{asset_name}.glb",
            visual_surface=scene.gltf_visual,
        )
        files = [asset, asset.with_suffix(".manifest.json")]
    elif file_format == "obj":
        asset = export_geometry_obj(
            cave_geometry,
            output / f"{asset_name}.obj",
            visual_surface=scene.canonical_visual,
        )
        files = [
            path
            for path in (asset, asset.with_suffix(".mtl"))
            if path.exists()
        ]
    else:
        raise ValueError(
            f"Target {export_config.target!r} currently supports glb or obj; "
            f"got {file_format!r}. USD is supported by the Omniverse adapter."
        )
    if export_config.generate_collision:
        files.append(
            _write_simplified_collision_obj(
                cave_geometry,
                output / f"{asset_name}_collision.obj",
                prepared_scene=scene,
            )
        )
    return ExportResult(
        target=export_config.target,
        primary_asset=asset,
        files=tuple(files),
        warnings=(),
    )


def _write_target_descriptor(
    result: ExportResult,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
) -> Path:
    conventions = {
        "neutral": {
            "application_coordinates": "glTF right-handed Y-up",
            "application_length_unit": "metre",
            "application_units_per_asset_metre": 1.0,
            "recommended_import_uniform_scale": 1.0,
            "note": (
                "Self-contained GLB with embedded PBR textures and visual "
                "displacement baked into vertex positions. Use the collision "
                "OBJ sidecar when the simulator requires a separate physics mesh."
            ),
        },
        "blender": {
            "application_coordinates": "right-handed Z-up",
            "application_length_unit": "metre",
            "application_units_per_asset_metre": 1.0,
            "recommended_import_uniform_scale": 1.0,
            "note": (
                "Use File > Import > glTF 2.0, not File > Open. "
                "The package includes an OBJ fallback and Blender import script."
            ),
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


def _write_engine_import_guide(
    target: str,
    output: Path,
    primary_asset: Path,
    *,
    collision_asset: Path | None,
) -> Path | None:
    """Write concise, version-resilient import settings beside engine assets."""

    lines: tuple[str, ...]
    if target == "ue5":
        name = "README_IMPORT_UE5.txt"
        lines = (
            "PLUME-Advanced Unreal Engine 5 import",
            "======================================",
            "",
            "1. Enable the Interchange Editor and Interchange Framework plugins.",
            f"2. Drag {primary_asset.name} into the Content Browser, or use Import Into Level.",
            "3. Keep import scale at 1.0; Interchange converts glTF metres to UE centimetres.",
            "4. Preserve imported normals/tangents and enable full-precision UVs for large caves.",
            "5. Disable automatically generated collision for the visual static meshes.",
            (
                f"6. Import {collision_asset.name} as the dedicated complex collision mesh."
                if collision_asset is not None
                else "6. Generate collision in UE if physics is required."
            ),
        )
    elif target == "unity":
        name = "README_IMPORT_UNITY.txt"
        lines = (
            "PLUME-Advanced Unity import",
            "============================",
            "",
            "Unity does not provide a universal built-in GLB importer across supported releases.",
            "Install a glTF 2.0 importer compatible with your Unity version (for example glTFast).",
            f"Import {primary_asset.name} with scale 1.0 and preserve normals/tangents.",
            "The GLB is Y-up, metre-based, self-contained, and uses metallic/roughness PBR.",
            "Keep the importer's generated materials and use shaders for your active render pipeline.",
            "PBR maps, when configured, are embedded: sRGB base color; linear OpenGL normal; "
            "linear roughness in green and metallic in blue. UVs repeat at the configured metre scale.",
            "Do not assign the packed glTF roughness image directly to a metallic-smoothness slot.",
            (
                f"Use {collision_asset.name} for a MeshCollider; disable rendering on that object."
                if collision_asset is not None
                else "Add collision in Unity if physics is required."
            ),
            "For large cave meshes, use a non-convex static MeshCollider and "
            "keep the object static.",
        )
    else:
        return None
    guide = output / name
    guide.write_text("\n".join((*lines, "")), encoding="utf-8")
    return guide


def _export_gazebo(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
    scene: PreparedExportScene,
) -> ExportResult:
    if export_config.file_format != "obj":
        raise ValueError("Gazebo export.format must be 'obj'")

    package = output / asset_name
    meshes = package / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    visual_mesh = export_geometry_obj(
        cave_geometry,
        meshes / f"{asset_name}.obj",
        visual_surface=scene.canonical_visual,
    )
    material = visual_mesh.with_suffix(".mtl")
    relocated_textures = _relocate_obj_material_textures(
        material,
        package / "materials" / "textures",
    )
    collision_mesh = (
        _write_simplified_collision_obj(
            cave_geometry,
            meshes / f"{asset_name}_collision.obj",
            prepared_scene=scene,
        )
        if export_config.generate_collision
        else visual_mesh
    )

    model_config = package / "model.config"
    model_config.write_text(
        "\n".join(
            (
                '<?xml version="1.0"?>',
                "<model>",
                f"  <name>{escape(asset_name)}</name>",
                "  <version>1.0</version>",
                "  <sdf version=\"1.12\">model.sdf</sdf>",
                "  <description>Procedural PLUME lava tube</description>",
                "</model>",
                "",
            )
        ),
        encoding="utf-8",
    )
    model_sdf = package / "model.sdf"
    uri = f"model://{asset_name}/meshes/{visual_mesh.name}"
    collision_uri = f"model://{asset_name}/meshes/{collision_mesh.name}"
    model_sdf.write_text(
        "\n".join(
            (
                '<?xml version="1.0"?>',
                '<sdf version="1.12">',
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
                f"          <uri>{escape(collision_uri)}</uri>",
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
    warnings: tuple[str, ...] = ()
    descriptor.write_text(
        json.dumps(
            {
                "schema": "plume.target_export.v1",
                "target": "gazebo",
                "gazebo_release": "Jetty",
                "gazebo_sim_major": 10,
                "sdformat_major": 16,
                "sdf_specification": "1.12",
                "coordinates": "right-handed Z-up metres",
                "requirements": asdict(export_config),
                "warnings": list(warnings),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    world = output / f"{asset_name}.world.sdf"
    world.write_text(
        "\n".join(
            (
                '<?xml version="1.0"?>',
                '<sdf version="1.12">',
                f'  <world name="{escape(asset_name)}_world">',
                '    <include>',
                f'      <uri>model://{escape(asset_name)}</uri>',
                '    </include>',
                '  </world>',
                '</sdf>',
                '',
            )
        ),
        encoding="utf-8",
    )
    guide = output / "README_RUN_GAZEBO.txt"
    guide.write_text(
        "\n".join(
            (
                "PLUME-Advanced Gazebo Jetty package",
                "====================================",
                "",
                "This package targets Gazebo Jetty (gz-sim 10, sdformat 16, SDF 1.12).",
                "From this directory run:",
                '  GZ_SIM_RESOURCE_PATH="$PWD${GZ_SIM_RESOURCE_PATH:+:'
                '$GZ_SIM_RESOURCE_PATH}" gz sim '
                f"{world.name}",
                "",
                "The model is static, metre-based, Z-up, and uses a separate collision mesh.",
                "",
            )
        ),
        encoding="utf-8",
    )
    files = [
        visual_mesh,
        collision_mesh,
        model_config,
        model_sdf,
        descriptor,
        world,
        guide,
        *relocated_textures,
    ]
    if material.exists():
        files.append(material)
    return ExportResult(
        target="gazebo",
        primary_asset=model_sdf,
        files=tuple(files),
        warnings=warnings,
    )


def _relocate_obj_material_textures(
    material_path: Path,
    texture_directory: Path,
) -> tuple[Path, ...]:
    """Copy MTL texture dependencies into a relocatable target package."""

    if not material_path.is_file():
        return ()
    rewritten: list[str] = []
    copied: dict[Path, Path] = {}
    map_directives = {"map_Kd", "map_Pr", "map_Pm", "map_Bump", "bump"}
    for line in material_path.read_text(encoding="utf-8").splitlines():
        try:
            tokens = shlex.split(line, comments=False, posix=True)
        except ValueError:
            tokens = []
        if tokens and tokens[0] in map_directives and len(tokens) >= 2:
            source = (material_path.parent / tokens[-1]).resolve()
            if source.is_file():
                texture_directory.mkdir(parents=True, exist_ok=True)
                destination = texture_directory / source.name
                if destination.exists() and source not in copied:
                    destination = texture_directory / (
                        f"{source.stem}_{len(copied):02d}{source.suffix}"
                    )
                if source not in copied:
                    shutil.copy2(source, destination)
                    copied[source] = destination
                relative = copied[source].relative_to(material_path.parent.parent).as_posix()
                tokens[-1] = f"../{relative}"
                line = " ".join(tokens)
        rewritten.append(line)
    material_path.write_text("\n".join((*rewritten, "")), encoding="utf-8")
    return tuple(copied.values())


def _export_omniverse(
    cave_geometry: CaveGeometry,
    export_config: ExportConfig,
    output: Path,
    asset_name: str,
    scene: PreparedExportScene,
) -> ExportResult:
    if export_config.file_format != "usd":
        raise ValueError("Omniverse export.format must be 'usd'")

    asset = output / f"{asset_name}.usd"
    texture_files = _write_usda(
        cave_geometry,
        asset,
        asset_name,
        generate_collision=export_config.generate_collision,
        prepared_scene=scene,
    )
    descriptor = output / f"{asset_name}.omniverse.json"
    warnings: tuple[str, ...] = ()
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
    guide = output / "README_IMPORT_OMNIVERSE.txt"
    guide.write_text(
        "\n".join(
            (
                "PLUME-Advanced NVIDIA Omniverse import",
                "=======================================",
                "",
                f"Open or drag {asset.name} into USD Composer's Content Browser.",
                f"Keep the adjacent {asset_name}_textures directory with the USD file.",
                "The stage declares Z-up and metersPerUnit=1 and uses UsdPreviewSurface.",
                "CaveCollision has PhysicsCollisionAPI and is hidden with guide purpose.",
                "Run usdchecker on the USD when an OpenUSD toolchain is installed.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return ExportResult(
        target="omniverse",
        primary_asset=asset,
        files=(asset, descriptor, guide, *texture_files),
        warnings=warnings,
    )


def _write_usda(
    cave_geometry: CaveGeometry,
    output: Path,
    asset_name: str,
    *,
    generate_collision: bool = False,
    prepared_scene: PreparedExportScene | None = None,
) -> tuple[Path, ...]:
    visual = (
        prepared_scene.canonical_visual
        if prepared_scene is not None
        else build_cave_visual_surface(cave_geometry, convert_to_gltf=False)
    )
    vertices = np.asarray(visual["positions"], dtype=np.float64)
    faces = np.asarray(visual["faces"], dtype=np.int64)
    normals = np.asarray(visual["normals"], dtype=np.float64)
    texcoords = np.asarray(visual["texcoords"], dtype=np.float64)
    texture_directory = output.parent / f"{asset_name}_textures"
    texture_files = export_cave_texture_files(
        cave_geometry,
        texture_directory,
    )
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "PLUME_Cave"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "PLUME_Cave" (',
        '    kind = "component"',
        ")",
        "{",
    ]
    lines.extend(
        _usda_mesh_lines(
            "CaveWall",
            vertices,
            faces,
            indent="    ",
            normals=normals,
            texcoords=texcoords,
            material_path="/PLUME_Cave/Looks/CaveMaterial",
            double_sided=False,
        )
    )
    if generate_collision:
        if prepared_scene is None:
            prepared_scene = prepare_export_scene(cave_geometry)
        collision_vertices = prepared_scene.collision_vertices
        collision_faces = prepared_scene.collision_faces
        lines.extend(
            _usda_mesh_lines(
                "CaveCollision",
                collision_vertices,
                collision_faces,
                indent="    ",
                api_schemas=("PhysicsCollisionAPI",),
                purpose="guide",
                visibility="invisible",
                collision_enabled=True,
            )
        )
    for event_mesh in cave_geometry.event_meshes:
        event_name = f"Event_{event_mesh.event_id:04d}_{_safe_asset_name(event_mesh.kind)}"
        event_texcoords = (
            np.asarray(event_mesh.face_uvs, dtype=np.float64).reshape((-1, 2))
            if len(event_mesh.face_uvs) == len(event_mesh.faces)
            else None
        )
        lines.extend(
            _usda_mesh_lines(
                event_name,
                np.asarray(event_mesh.vertices, dtype=np.float64),
                np.asarray(event_mesh.faces, dtype=np.int64),
                indent="    ",
                texcoords=event_texcoords,
                texcoord_interpolation="faceVarying",
                material_path=(
                    "/PLUME_Cave/Looks/CaveMaterial"
                    if _event_uses_cave_material(event_mesh, cave_geometry)
                    else None
                ),
                custom_strings={
                    "plume:kind": event_mesh.kind,
                    "plume:sourceGenerator": event_mesh.source_generator,
                    "plume:sourceShapeType": event_mesh.source_shape_type,
                    "plume:debrisFamilyId": str(event_mesh.debris_family_id),
                    "plume:familyAnchorEventId": str(
                        event_mesh.family_anchor_event_id
                    ),
                    "plume:debrisRole": event_mesh.debris_role,
                },
            )
        )
    lines.extend(
        _usda_cave_material_lines(
            texture_files,
            output=output,
            normal_scale=cave_geometry.config.cave_normal_scale,
            indent="    ",
        )
    )
    lines.extend(("}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return tuple(texture_files.values())



def _usda_mesh_lines(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    indent: str,
    normals: np.ndarray | None = None,
    texcoords: np.ndarray | None = None,
    texcoord_interpolation: str = "vertex",
    material_path: str | None = None,
    double_sided: bool = True,
    api_schemas: tuple[str, ...] = (),
    purpose: str | None = None,
    visibility: str | None = None,
    collision_enabled: bool = False,
    custom_strings: dict[str, str] | None = None,
) -> list[str]:
    if normals is None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        normal_values = np.asarray(mesh.vertex_normals, dtype=np.float64)
    else:
        normal_values = np.asarray(normals, dtype=np.float64)
    point_text = ", ".join(_tuple_text(point) for point in vertices)
    normal_text = ", ".join(_tuple_text(normal) for normal in normal_values)
    count_text = ", ".join("3" for _ in faces)
    index_text = ", ".join(str(int(index)) for index in np.asarray(faces).reshape(-1))
    applied_schemas = list(api_schemas)
    if material_path:
        applied_schemas.insert(0, "MaterialBindingAPI")
    if applied_schemas:
        schema_text = ", ".join(f'"{schema}"' for schema in applied_schemas)
        lines = [
            f'{indent}def Mesh "{name}" (',
            f"{indent}    prepend apiSchemas = [{schema_text}]",
            f"{indent})",
            f"{indent}{{",
        ]
    else:
        lines = [
            f'{indent}def Mesh "{name}"',
            f"{indent}{{",
        ]
    lines.extend(
        [
            f"{indent}    uniform bool doubleSided = {str(double_sided).lower()}",
            f"{indent}    int[] faceVertexCounts = [{count_text}]",
            f"{indent}    int[] faceVertexIndices = [{index_text}]",
            f'{indent}    uniform token orientation = "rightHanded"',
            f"{indent}    point3f[] points = [{point_text}]",
            f"{indent}    normal3f[] normals = [{normal_text}] (",
            f'{indent}        interpolation = "vertex"',
            f"{indent}    )",
        ]
    )
    if purpose is not None:
        lines.append(f'{indent}    uniform token purpose = "{purpose}"')
    if visibility is not None:
        lines.append(f'{indent}    token visibility = "{visibility}"')
    if collision_enabled:
        lines.append(f"{indent}    bool physics:collisionEnabled = true")
    for key, value in sorted((custom_strings or {}).items()):
        escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{indent}    custom string {key} = "{escaped_value}"')
    if texcoords is not None:
        uv_text = ", ".join(_tuple_text(uv) for uv in texcoords)
        lines.extend(
            (
                f"{indent}    texCoord2f[] primvars:st = [{uv_text}] (",
                f'{indent}        interpolation = "{texcoord_interpolation}"',
                f"{indent}    )",
            )
        )
    if material_path:
        lines.extend(
            (
                f"{indent}    rel material:binding = <{material_path}>",
            )
        )
    lines.extend(
        (
            f'{indent}    uniform token subdivisionScheme = "none"',
            f"{indent}}}",
        )
    )
    return lines


def _event_uses_cave_material(event_mesh, cave_geometry: CaveGeometry) -> bool:
    """Bind shared cave maps only when Rocky references the same source set."""

    event_maps = dict(event_mesh.material_maps)
    if not event_maps:
        return False
    cave_maps = {
        "diffuse": cave_geometry.config.cave_diffuse_texture,
        "normal": cave_geometry.config.cave_normal_texture,
        "roughness": cave_geometry.config.cave_roughness_texture,
    }
    compared = False
    for role, cave_path in cave_maps.items():
        event_path = event_maps.get(role)
        if not event_path:
            continue
        compared = True
        if not cave_path or Path(event_path).resolve() != Path(cave_path).resolve():
            return False
    return compared


def _usda_cave_material_lines(
    texture_files: dict[str, Path],
    *,
    output: Path,
    normal_scale: float,
    indent: str,
) -> list[str]:
    """Build a portable UsdPreviewSurface material driven by primvars:st."""

    material_path = "/PLUME_Cave/Looks/CaveMaterial"
    shader_path = f"{material_path}/PreviewSurface"
    reader_path = f"{material_path}/PrimvarReader"
    lines = [
        f'{indent}def Scope "Looks"',
        f"{indent}{{",
        f'{indent}    def Material "CaveMaterial"',
        f"{indent}    {{",
        f"{indent}        token outputs:surface.connect = <{shader_path}.outputs:surface>",
        f'{indent}        def Shader "PreviewSurface"',
        f"{indent}        {{",
        f'{indent}            uniform token info:id = "UsdPreviewSurface"',
        f"{indent}            color3f inputs:diffuseColor = (0.36, 0.35, 0.31)",
        f"{indent}            float inputs:metallic = 0",
        f"{indent}            float inputs:roughness = 0.92",
    ]
    if "diffuse" in texture_files:
        lines.append(
            f"{indent}            color3f inputs:diffuseColor.connect = "
            f"<{material_path}/BaseColor.outputs:rgb>"
        )
    if "metallic_roughness" in texture_files:
        lines.append(
            f"{indent}            float inputs:roughness.connect = "
            f"<{material_path}/MetallicRoughness.outputs:g>"
        )
    if "normal" in texture_files:
        lines.append(
            f"{indent}            normal3f inputs:normal.connect = "
            f"<{material_path}/Normal.outputs:rgb>"
        )
    lines.extend(
        (
            f"{indent}            token outputs:surface",
            f"{indent}        }}",
            f'{indent}        def Shader "PrimvarReader"',
            f"{indent}        {{",
            f'{indent}            uniform token info:id = "UsdPrimvarReader_float2"',
            f'{indent}            token inputs:varname = "st"',
            f"{indent}            float2 outputs:result",
            f"{indent}        }}",
        )
    )
    texture_specs = (
        ("diffuse", "BaseColor", "sRGB", None),
        ("metallic_roughness", "MetallicRoughness", "raw", None),
        (
            "normal",
            "Normal",
            "raw",
            (
                (2.0 * normal_scale, 2.0 * normal_scale, 2.0, 1.0),
                (-normal_scale, -normal_scale, -1.0, 0.0),
            ),
        ),
    )
    for role, shader_name, color_space, normal_transform in texture_specs:
        texture_path = texture_files.get(role)
        if texture_path is None:
            continue
        relative_path = texture_path.relative_to(output.parent).as_posix()
        lines.extend(
            (
                f'{indent}        def Shader "{shader_name}"',
                f"{indent}        {{",
                f'{indent}            uniform token info:id = "UsdUVTexture"',
                f"{indent}            asset inputs:file = @{relative_path}@",
                f'{indent}            token inputs:sourceColorSpace = "{color_space}"',
                f"{indent}            float2 inputs:st.connect = <{reader_path}.outputs:result>",
            )
        )
        if normal_transform is not None:
            scale, bias = normal_transform
            lines.extend(
                (
                    f"{indent}            float4 inputs:scale = {_tuple_text(scale)}",
                    f"{indent}            float4 inputs:bias = {_tuple_text(bias)}",
                )
            )
        lines.extend(
            (
                f"{indent}            float outputs:r",
                f"{indent}            float outputs:g",
                f"{indent}            float3 outputs:rgb",
                f"{indent}        }}",
            )
        )
    lines.extend(
        (
            f"{indent}    }}",
            f"{indent}}}",
        )
    )
    return lines


def _tuple_text(values: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):.8g}" for value in values) + ")"


def _safe_asset_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in value
    )
    return safe.strip("_") or "plume_cave"


def _write_simplified_collision_obj(
    cave_geometry: CaveGeometry,
    output_path: Path,
    *,
    prepared_scene: PreparedExportScene | None = None,
) -> Path:
    """Write a deterministic vertex-clustered collision approximation."""

    scene = prepared_scene or prepare_export_scene(cave_geometry)
    clustered = scene.collision_vertices
    remapped = scene.collision_faces
    collision = trimesh.Trimesh(
        vertices=clustered,
        faces=remapped,
        process=True,
    )
    collision.remove_unreferenced_vertices()
    if not collision.is_winding_consistent:
        collision.fix_normals(multibody=True)
    if (
        len(collision.faces) == 0
        or not np.isfinite(collision.vertices).all()
        or not collision.is_winding_consistent
    ):
        vertices, faces = canonical_cave_mesh(cave_geometry)
        collision = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=True,
        )
        collision.remove_unreferenced_vertices()
        collision.fix_normals(multibody=True)
    if len(collision.faces) == 0 or not np.isfinite(collision.vertices).all():
        raise ValueError("Collision simplification produced invalid geometry")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collision.export(output_path)
    return output_path
