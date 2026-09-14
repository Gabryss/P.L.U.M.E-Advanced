"""Run in an isolated UE editor project with rendering enabled, never a user scene."""

import json
import runpy
import traceback
from pathlib import Path

import unreal as ue


def main(directory, attempt="01"):
    root = Path(directory)
    output = root / ("unreal_run_" + attempt)
    output.mkdir(exist_ok=False)
    report = {"passed": False, "version": ue.SystemLibrary.get_engine_version()}

    def progress(phase):
        report["phase"] = phase
        (output / "native_result.json").write_text(json.dumps(report, indent=2))
        ue.log("PLUME native check: " + phase)

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    try:
        expected = json.loads((root / "unreal_expected.json").read_text())
        source = json.loads((root / "native_input_receipt.json").read_text())["source_glb"]
        destination = "/Game/PLUME_Run" + attempt
        progress("native GLB import")
        task = ue.AssetImportTask()
        task.filename = source
        task.destination_path = destination + "/Cave"
        task.automated = True
        task.replace_existing = False
        task.save = True
        ue.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        paths = list(task.imported_object_paths)
        report["imported_assets"] = paths
        meshes = [
            ue.load_asset(path) for path in paths if isinstance(ue.load_asset(path), ue.StaticMesh)
        ]
        require(len(meshes) == 1, f"Expected one imported cave mesh, found {len(meshes)}")
        mesh = meshes[0]
        # UE 5.8 imports with Nanite enabled by default. Its reduced fallback is
        # also used for complex collision; validate the full source first.
        subsystem = ue.get_editor_subsystem(ue.StaticMeshEditorSubsystem)
        nanite = subsystem.get_nanite_settings(mesh)
        report["imported_nanite_enabled"] = nanite.enabled
        report["imported_fallback_triangles"] = mesh.get_num_triangles(0)
        nanite.enabled = False
        subsystem.set_nanite_settings(mesh, nanite, True)
        report["triangles"] = mesh.get_num_triangles(0)
        report["vertices"] = mesh.get_num_vertices(0)
        require(report["triangles"] == expected["triangles"], "Native triangle count differs")
        bounds = mesh.get_bounding_box()
        report["bounds_min_cm"] = [bounds.min.x, bounds.min.y, bounds.min.z]
        report["bounds_max_cm"] = [bounds.max.x, bounds.max.y, bounds.max.z]
        report["maximum_bounds_error_cm"] = max(
            abs(actual - wanted)
            for kind in ("min", "max")
            for actual, wanted in zip(
                report[f"bounds_{kind}_cm"], expected[f"bounds_{kind}"], strict=True
            )
        )
        require(report["maximum_bounds_error_cm"] < 0.2, "Native bounds differ by more than 2 mm")
        progress("native material construction and shader compilation")
        builder = runpy.run_path(str(root / "native_material/unreal/create_material.py"))
        material = builder["create_material"](root / "native_material", destination + "/Material")
        stats = ue.MaterialEditingLibrary.get_statistics(material)
        report["shader_statistics"] = {
            name: getattr(stats, name)
            for name in (
                "num_vertex_shader_instructions",
                "num_pixel_shader_instructions",
                "num_samplers",
                "num_vertex_texture_samples",
                "num_pixel_texture_samples",
            )
        }
        report["shader_count"] = len(ue.MaterialEditingLibrary.list_shaders(material))
        require(report["shader_count"] > 0, "Material has no compiled shaders")
        textures = []
        for name in ("ColorMap", "NormalMap", "RoughnessMap"):
            texture = ue.load_asset(destination + "/Material/" + name)
            item = dict(
                name=name,
                width=texture.blueprint_get_size_x(),
                height=texture.blueprint_get_size_y(),
                srgb=texture.get_editor_property("srgb"),
                compression=str(texture.get_editor_property("compression_settings")),
            )
            require(item["width"] == item["height"] == 4096, "Expected retained 4K maps")
            require(item["srgb"] == (name == "ColorMap"), "Incorrect native texture color space")
            textures.append(item)
        report["textures"] = textures
        progress("native scene and renders")
        world = ue.EditorLoadingAndSavingUtils.new_blank_map(False)
        actors = ue.get_editor_subsystem(ue.EditorActorSubsystem)
        body = mesh.get_editor_property("body_setup")
        body.set_editor_property(
            "collision_trace_flag", ue.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
        )
        body.set_editor_property("double_sided_geometry", True)
        actor = actors.spawn_actor_from_object(mesh, ue.Vector(0, 0, 0))
        actor.set_actor_label("PLUME inspected cave")
        component = actor.get_component_by_class(ue.StaticMeshComponent)
        component.set_material(0, material)
        component.set_collision_profile_name("BlockAll")
        report["passage_samples"] = len(expected["samples"])
        report["passage_passed"] = 0
        report["maximum_clearance_error_cm"] = 0.0
        report["clearance_failures"] = []
        for index, sample in enumerate(expected["samples"]):
            start = ue.Vector(*sample["point"])
            errors = []
            for direction, side in ((-1, "floor"), (1, "roof")):
                hit = ue.SystemLibrary.line_trace_single(
                    world,
                    start,
                    start + ue.Vector(0, 0, direction * 2000),
                    ue.TraceTypeQuery.TRACE_TYPE_QUERY1,
                    True,
                    [],
                    ue.DrawDebugTrace.NONE,
                )
                if hit is None:
                    errors.append(2000.0)
                else:
                    # HasNativeBreak exposes BreakHitResult through to_tuple; Distance
                    # is the fourth output after blocking, initial overlap and time.
                    errors.append(abs(hit.to_tuple()[3] - sample[side]))
            error = max(errors)
            report["maximum_clearance_error_cm"] = max(report["maximum_clearance_error_cm"], error)
            if error < 1.0:
                report["passage_passed"] += 1
            else:
                report["clearance_failures"].append(dict(sample=index, error_cm=error))
        require(
            report["passage_passed"] == report["passage_samples"],
            "Native passage ray checks failed",
        )
        progress("native collision passed; capturing materials")
        capture = actors.spawn_actor_from_class(ue.SceneCapture2D, ue.Vector(0, 0, 0))
        camera = capture.get_component_by_class(ue.SceneCaptureComponent2D)
        camera.capture_every_frame = False
        camera.capture_on_movement = False
        camera.fov_angle = 75.0
        camera.capture_source = ue.SceneCaptureSource.SCS_FINAL_COLOR_LDR
        target = ue.RenderingLibrary.create_render_target2d(
            world, 960, 640, ue.TextureRenderTargetFormat.RTF_RGBA8
        )
        camera.texture_target = target
        post = camera.get_editor_property("post_process_settings")
        post.override_auto_exposure_method = True
        post.auto_exposure_method = ue.AutoExposureMethod.AEM_MANUAL
        post.override_auto_exposure_bias = True
        post.auto_exposure_bias = 1.0
        camera.set_editor_property("post_process_settings", post)
        light_actor = actors.spawn_actor_from_class(ue.PointLight, ue.Vector(0, 0, 0))
        light = light_actor.get_component_by_class(ue.PointLightComponent)
        light.set_intensity(1500.0)
        light.set_attenuation_radius(2000.0)
        light.set_source_radius(10.0)

        def render(name):
            for _ in range(3):
                camera.capture_scene()
                pixels = ue.RenderingLibrary.read_render_target(world, target)
            ue.RenderingLibrary.export_render_target(world, target, str(output), name)
            return [(p.r, p.g, p.b) for p in pixels]

        for view, index in enumerate((15, 44), 1):
            location = ue.Vector(*expected["samples"][index]["point"])
            look = ue.Vector(*expected["samples"][index + 5]["point"])
            look.z -= 15
            capture.set_actor_location(location, False, False)
            capture.set_actor_rotation(ue.MathLibrary.find_look_at_rotation(location, look), False)
            light_actor.set_actor_location(location + ue.Vector(30, 0, 20), False, False)
            reference = render(f"interior_{view}.png")
        instance = component.create_dynamic_material_instance(0, material)
        instance.set_scalar_parameter_value("NormalStrength", 0.0)
        control = render("normal_off_control.png")
        component.set_material(0, material)
        report["normal_render_difference"] = sum(
            abs(a - b)
            for original, changed in zip(reference, control, strict=True)
            for a, b in zip(original, changed, strict=True)
        ) / (len(reference) * 3 * 255)
        require(
            report["normal_render_difference"] > 0.0001, "Normal map has no visible contribution"
        )
        ue.get_editor_subsystem(ue.UnrealEditorSubsystem).set_level_viewport_camera_info(
            capture.get_actor_location(), capture.get_actor_rotation()
        )
        ue.EditorLoadingAndSavingUtils.save_map(world, destination + "/PLUME_Inspection")
        ue.EditorAssetLibrary.save_loaded_asset(mesh)
        report["material"] = material.get_path_name()
        report["mesh"] = mesh.get_path_name()
        report["map"] = destination + "/PLUME_Inspection"
        report["passed"] = True
        progress("complete")
    except Exception:
        report["failure"] = traceback.format_exc()
        progress("failed")
        raise
