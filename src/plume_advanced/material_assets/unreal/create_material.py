"""Run from UE5 with Python Editor Script Plugin enabled: creates a NEW asset folder.

Use Tools > Execute Python Script. Geometry/actors and existing materials are not
modified. Assign the resulting material to cave_wall after checking shader errors.
"""

from __future__ import annotations

import json
from pathlib import Path


def create_material(bundle_root=None, destination="/Game/PLUME_Continuous"):
    import unreal as ue

    root = Path(bundle_root) if bundle_root else Path(__file__).resolve().parents[1]
    settings = json.loads((root / "settings.json").read_text())
    library = ue.MaterialEditingLibrary
    if ue.EditorAssetLibrary.does_directory_exist(destination):
        raise FileExistsError(f"Choose a new Unreal destination folder: {destination}")
    files = {
        "ColorMap": "cave_base_color.png",
        "NormalMap": "cave_normal.png",
        "RoughnessMap": "cave_metallic_roughness.png",
    }
    for filename in files.values():
        if not (root / "textures" / filename).is_file():
            raise FileNotFoundError(filename)
    textures = {}
    tools = ue.AssetToolsHelpers.get_asset_tools()
    for name, filename in files.items():
        task = ue.AssetImportTask()
        task.set_editor_property("filename", str(root / "textures" / filename))
        task.set_editor_property("destination_path", destination)
        task.set_editor_property("destination_name", name)
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", False)
        task.set_editor_property("save", False)
        tools.import_asset_tasks([task])
        paths = task.get_editor_property("imported_object_paths")
        if len(paths) != 1:
            raise RuntimeError(f"Import failed for {name}: {paths}")
        texture = ue.load_asset(paths[0])
        texture.set_editor_property("srgb", name == "ColorMap")
        # A raw RGB map is decoded by the shared shader, not Unreal's normal sampler.
        compression = (
            ue.TextureCompressionSettings.TC_DEFAULT
            if name == "ColorMap"
            else ue.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP
        )
        texture.set_editor_property("compression_settings", compression)
        texture.set_editor_property("address_x", ue.TextureAddress.TA_WRAP)
        texture.set_editor_property("address_y", ue.TextureAddress.TA_WRAP)
        ue.EditorAssetLibrary.save_loaded_asset(texture)
        textures[name] = texture
    material = tools.create_asset(
        "M_PLUME_Continuous", destination, ue.Material, ue.MaterialFactoryNew()
    )
    if not material:
        raise RuntimeError("Could not create material")
    # Output normal is transformed to world space below; do not interpret it as tangent space.
    material.set_editor_property("tangent_space_normal", False)
    material.set_editor_property("two_sided", False)

    def expression(cls, x=-800, y=0):
        return library.create_material_expression(material, cls, x, y)

    def connect(source, output, target, pin):
        if not library.connect_material_expressions(source, output, target, pin):
            raise RuntimeError(f"Could not connect {source.get_name()}:{output} to {pin}")

    custom = expression(ue.MaterialExpressionCustom, 0)
    custom.set_editor_property("description", "PLUME continuous PBR / object metres")
    custom.set_editor_property("output_type", ue.CustomMaterialOutputType.CMOT_FLOAT3)
    names = [
        "ColorMap",
        "NormalMap",
        "RoughnessMap",
        "PositionCM",
        "BaseNormal",
        "TileSize",
        "NormalStrength",
        "BlendExponent",
    ]
    inputs = []
    for name in names:
        value = ue.CustomInput()
        value.set_editor_property("input_name", name)
        inputs.append(value)
    custom.set_editor_property("inputs", inputs)
    outputs = []
    for name, kind in [
        ("Roughness", ue.CustomMaterialOutputType.CMOT_FLOAT1),
        ("NormalObject", ue.CustomMaterialOutputType.CMOT_FLOAT3),
    ]:
        value = ue.CustomOutput()
        value.set_editor_property("output_name", name)
        value.set_editor_property("output_type", kind)
        outputs.append(value)
    custom.set_editor_property("additional_outputs", outputs)
    body = (root / "plume_triplanar_body.hlsl").read_text()
    custom.set_editor_property(
        "code",
        "float3 BaseColor;\nfloat3 PositionM = PositionCM * 0.01;\n"
        "float UVOriginTop = 1.0;\n" + body + "\nreturn BaseColor;\n",
    )
    for i, (name, texture) in enumerate(textures.items()):
        tex = expression(ue.MaterialExpressionTextureObjectParameter, y=i * 180)
        tex.set_editor_property("parameter_name", name)
        tex.set_editor_property("texture", texture)
        tex.set_editor_property(
            "sampler_type",
            ue.MaterialSamplerType.SAMPLERTYPE_COLOR
            if name == "ColorMap"
            else ue.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR,
        )
        connect(tex, "", custom, name)
    world_position = expression(ue.MaterialExpressionWorldPosition, x=-1300, y=600)
    local_position = expression(ue.MaterialExpressionTransformPosition, y=600)
    local_position.set_editor_property(
        "transform_source_type", ue.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_WORLD
    )
    local_position.set_editor_property(
        "transform_type", ue.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_LOCAL
    )
    # These transform expressions expose an unnamed first input in Unreal's
    # scripting API; the C++ member name "Input" is not a connectable pin name.
    connect(world_position, "", local_position, "")
    connect(local_position, "", custom, "PositionCM")
    vertex_normal = expression(ue.MaterialExpressionVertexNormalWS, x=-1300, y=800)
    interpolated_normal = expression(ue.MaterialExpressionVertexInterpolator, x=-1050, y=800)
    connect(vertex_normal, "", interpolated_normal, "")
    local_normal = expression(ue.MaterialExpressionTransform, y=800)
    local_normal.set_editor_property(
        "transform_source_type", ue.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_WORLD
    )
    local_normal.set_editor_property(
        "transform_type", ue.MaterialVectorCoordTransform.TRANSFORM_LOCAL
    )
    connect(interpolated_normal, "", local_normal, "")
    connect(local_normal, "", custom, "BaseNormal")
    for i, (name, value) in enumerate(
        [
            ("TileSize", settings["tile_size_m"]),
            ("NormalStrength", settings["normal_strength"]),
            ("BlendExponent", 4.0),
        ]
    ):
        scalar = expression(ue.MaterialExpressionScalarParameter, y=1000 + i * 150)
        scalar.set_editor_property("parameter_name", name)
        scalar.set_editor_property("default_value", value)
        connect(scalar, "", custom, name)
    world_normal = expression(ue.MaterialExpressionTransform, x=300, y=400)
    world_normal.set_editor_property(
        "transform_source_type", ue.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_LOCAL
    )
    world_normal.set_editor_property(
        "transform_type", ue.MaterialVectorCoordTransform.TRANSFORM_WORLD
    )
    connect(custom, "NormalObject", world_normal, "")
    for node, pin, prop in [
        (custom, "", ue.MaterialProperty.MP_BASE_COLOR),
        (custom, "Roughness", ue.MaterialProperty.MP_ROUGHNESS),
        (world_normal, "", ue.MaterialProperty.MP_NORMAL),
    ]:
        if not library.connect_material_property(node, pin, prop):
            raise RuntimeError(f"Could not connect material property {prop}")
    library.layout_material_expressions(material)
    errors = library.recompile_material(material)
    if errors:
        raise RuntimeError(f"PLUME material compilation failed: {errors}")
    ue.EditorAssetLibrary.save_loaded_asset(material)
    ue.log(
        f"Created {material.get_path_name()}. Check compiler errors and assign to cave_wall. "
        "Keep mesh/actor scale uniform; tile size is measured in object metres."
    )
    return material


if __name__ == "__main__":
    create_material()
