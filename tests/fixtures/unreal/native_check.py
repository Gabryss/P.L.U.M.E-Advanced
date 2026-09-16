"""Run in an isolated UE editor project with rendering enabled, never a user scene."""

import json
import math
import resource
import runpy
import time
import traceback
from pathlib import Path

import unreal as ue


def body_hits(world, a, b, radius, half_axis):
    """Multi traces include stationary initial overlaps as well as sweeps."""
    return bool(ue.SystemLibrary.capsule_trace_multi(
        world, ue.Vector(*a), ue.Vector(*b), radius, radius+half_axis,
        ue.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [], ue.DrawDebugTrace.NONE))


def box_hits(world, a, b, half, forward, up):
    rotation = ue.MathLibrary.make_rot_from_xz(ue.Vector(*forward), ue.Vector(*up))
    return bool(ue.SystemLibrary.box_trace_multi(world, ue.Vector(*a), ue.Vector(*b),
        ue.Vector(*half), rotation, ue.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [], ue.DrawDebugTrace.NONE))


def floor_hit(world, point, distance):
    return ue.SystemLibrary.line_trace_single(world, ue.Vector(*point),
        ue.Vector(point[0], point[1], point[2]-distance), ue.TraceTypeQuery.TRACE_TYPE_QUERY1,
        True, [], ue.DrawDebugTrace.NONE)


def ground_controls(world, actors, origin):
    cube = ue.load_asset('/Engine/BasicShapes/Cube.Cube')
    obstacle = actors.spawn_actor_from_object(cube, origin)
    obstacle.get_component_by_class(ue.StaticMeshComponent).set_collision_profile_name('BlockAll')
    obstacle.set_actor_scale3d(ue.Vector(.02, 4, 4))
    rows, second = [], None
    def point(x=0, y=0, z=0):
        return [origin.x+x, origin.y+y, origin.z+z]
    def record(name, expected, observed):
        rows.append(dict(name=name, expected=expected, observed=observed, passed=expected == observed))
    def hits(a, b, forward=(1, 0, 0)):
        return box_hits(world, a, b, [37, 27, 27], forward, [0, 0, 1])
    try:
        record('box_initial_overlap', True, hits(point(), point()))
        record('box_length_overlap', True, hits(point(34), point(34)))
        record('box_rotated_clear', False, hits(point(34), point(34), (0, 1, 0)))
        record('box_clear_start', False, hits(point(100), point(100)))
        record('box_thin_wall_forward', True, hits(point(-100), point(100)))
        record('box_thin_wall_reverse', True, hits(point(100), point(-100)))
        record('box_clear_sweep', False, hits(point(100), point(200)))
        obstacle.set_actor_scale3d(ue.Vector(4, 4, .02))
        record('floor_present', True, floor_hit(world, point(z=100), 200) is not None)
        record('floor_missing', False, floor_hit(world, point(x=500, z=100), 200) is not None)
        for angle in (15, 25):
            obstacle.set_actor_rotation(ue.Rotator(pitch=angle, yaw=0, roll=0), False)
            a, b = (floor_hit(world, point(x=x, z=100), 200) for x in (-50, 50))
            allowed = (a is not None and b is not None
                       and math.degrees(math.atan(abs(a.to_tuple()[3]-b.to_tuple()[3])/100)) <= 20)
            record('slope_below_limit' if angle == 15 else 'slope_above_limit', angle == 15, allowed)
        obstacle.set_actor_rotation(ue.Rotator(), False)
        obstacle.set_actor_location(ue.Vector(*point(-50)), False, False)
        obstacle.set_actor_scale3d(ue.Vector(1, 1, .02))
        second = actors.spawn_actor_from_object(cube, ue.Vector(*point(50)))
        second.get_component_by_class(ue.StaticMeshComponent).set_collision_profile_name('BlockAll')
        second.set_actor_scale3d(ue.Vector(1, 1, .02))
        for step in (6, 15):
            second.set_actor_location(ue.Vector(*point(50, z=step)), False, False)
            a, b = (floor_hit(world, point(x=x, z=100), 200) for x in (-50, 50))
            allowed = a is not None and b is not None and abs(a.to_tuple()[3]-b.to_tuple()[3]) <= 10
            record('step_below_limit' if step == 6 else 'step_above_limit', step == 6, allowed)
    finally:
        if second is not None:
            actors.destroy_actor(second)
        actors.destroy_actor(obstacle)
    return rows


def check_ground(world, actors, plan, origin):
    if not plan or not plan['enabled']:
        return dict(enabled=False)
    report = dict(enabled=True, passed=False, stations=0, sweeps=0, floor_samples=0, failures=0,
                  maximumFloorErrorM=0., maximumSlopeDeg=0., maximumStepM=0.,
                  **{key: plan[key] for key in ('length_m', 'width_m', 'height_m', 'margin_m',
                      'max_slope_deg', 'max_step_m', 'support_spacing_m')})
    for pose in plan['poses']:
        report['stations'] += 1
        report['failures'] += int(box_hits(world, pose['point'], pose['point'], plan['half_extents'],
                                          pose['forward'], pose['up']))
        heights, coefficients = [], [0., 0., 0.]
        for probe, support in zip(pose['probes'], plan['support']):
            report['floor_samples'] += 1
            hit = floor_hit(world, probe['point'], probe['floor']+10)
            if hit is None:
                report['failures'] += 1
                height = 0.
            else:
                distance = hit.to_tuple()[3]
                report['maximumFloorErrorM'] = max(report['maximumFloorErrorM'], abs(distance-probe['floor'])/100)
                height = (probe['point'][2]-distance)/100
            heights.append(height)
            for i, key in enumerate(('x', 'y', 'z')):
                coefficients[i] += support['weight'][key]*height
        a, b, c = coefficients
        residual = [height-(a*s['offset']['x']+b*s['offset']['y']+c)
                    for height, s in zip(heights, plan['support'])]
        report['maximumSlopeDeg'] = max(report['maximumSlopeDeg'], math.degrees(math.atan(math.hypot(a, b))))
        report['maximumStepM'] = max(report['maximumStepM'], max(residual)-min(residual))
    for motion in plan['motions']:
        report['sweeps'] += 1
        a, b = motion['start'], motion['end']
        if any(box_hits(world, p, q, motion['half_extents'], motion['forward'], motion['up'])
               for p, q in ((a, a), (b, b), (a, b), (b, a))):
            report['failures'] += 1
    report['controls'] = ground_controls(world, actors, origin)
    report['passed'] = (report['failures'] == 0
        and all(report[key] == plan[key] for key in ('stations', 'sweeps', 'floor_samples'))
        and report['maximumFloorErrorM'] <= .002 and report['maximumSlopeDeg'] <= plan['max_slope_deg']+1e-5
        and report['maximumStepM'] <= plan['max_step_m']+1e-5 and all(r['passed'] for r in report['controls']))
    return report


def physics_controls(world, actors, origin):
    """Exercise the same query on a temporary 2 cm wall outside cave bounds."""
    cube = ue.load_asset('/Engine/BasicShapes/Cube.Cube')
    if cube is None:
        raise RuntimeError('Missing native cube for collision controls')
    obstacle = actors.spawn_actor_from_object(cube, origin)
    component = obstacle.get_component_by_class(ue.StaticMeshComponent)
    component.set_collision_profile_name('BlockAll')
    obstacle.set_actor_scale3d(ue.Vector(.02, 4, 4))  # The engine cube is 100 cm.
    rows = []

    def point(x=0., z=0.):
        return [origin.x+x, origin.y, origin.z+z]

    def record(name, expected, observed):
        rows.append(dict(name=name, expected=expected, observed=observed, passed=expected == observed))

    try:
        for tall in (False, True):
            half_axis = 50. if tall else 0.
            prefix = 'capsule_' if tall else 'sphere_'
            record(prefix+'initial_overlap', True, body_hits(world, point(), point(), 27., half_axis))
            record(prefix+'grazing_overlap', True, body_hits(world, point(14), point(14), 27., half_axis))
            record(prefix+'clear_start', False, body_hits(world, point(100), point(100), 27., half_axis))
            record(prefix+'thin_wall_forward', True, body_hits(world, point(-100), point(100), 27., half_axis))
            record(prefix+'thin_wall_reverse', True, body_hits(world, point(100), point(-100), 27., half_axis))
            record(prefix+'clear_sweep', False, body_hits(world, point(100), point(200), 27., half_axis))
        obstacle.set_actor_scale3d(ue.Vector(4, 4, .02))
        record('capsule_axis_overlap', True, body_hits(world, point(z=55), point(z=55), 27., 50.))
        record('sphere_axis_clear', False, body_hits(world, point(z=55), point(z=55), 27., 0.))
    finally:
        actors.destroy_actor(obstacle)
    return rows


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
        import_start = time.perf_counter()
        task = ue.AssetImportTask()
        task.filename = source
        task.destination_path = destination + "/Cave"
        task.automated = True
        task.replace_existing = False
        task.save = True
        ue.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        report["visual_import_seconds"] = time.perf_counter()-import_start
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
        cooking_start = time.perf_counter()
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
        report["visual_setup_seconds"] = time.perf_counter()-cooking_start
        query_start = time.perf_counter()
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
        report["visual_query_seconds"] = time.perf_counter()-query_start
        component.set_collision_enabled(ue.CollisionEnabled.NO_COLLISION)
        progress("dedicated collider import and physics")
        import_start = time.perf_counter()
        collision_task = ue.AssetImportTask()
        collision_task.filename = str(root / "plume_collision.glb")
        collision_task.destination_path = destination + "/Collision"
        collision_task.automated = True
        collision_task.save = True
        ue.AssetToolsHelpers.get_asset_tools().import_asset_tasks([collision_task])
        collision_meshes = [ue.load_asset(p) for p in collision_task.imported_object_paths
                           if isinstance(ue.load_asset(p), ue.StaticMesh)]
        require(len(collision_meshes) == 1, "Expected one dedicated collider mesh")
        collision_mesh = collision_meshes[0]
        report["collision_import_seconds"] = time.perf_counter()-import_start
        cooking_start = time.perf_counter()
        nanite = subsystem.get_nanite_settings(collision_mesh)
        nanite.enabled = False
        subsystem.set_nanite_settings(collision_mesh, nanite, True)
        report["collision_triangles"] = collision_mesh.get_num_triangles(0)
        require(report["collision_triangles"] == expected["collision_triangles"], "Dedicated collider count changed")
        collision_body = collision_mesh.get_editor_property("body_setup")
        collision_body.set_editor_property("collision_trace_flag", ue.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        collision_body.set_editor_property("double_sided_geometry", True)
        collision_actor = actors.spawn_actor_from_object(collision_mesh, ue.Vector(0, 0, 0))
        collision_actor.set_actor_label("PLUME dedicated collision")
        collision_component = collision_actor.get_component_by_class(ue.StaticMeshComponent)
        collision_component.set_visibility(False)
        collision_component.set_hidden_in_game(True)
        collision_component.set_collision_profile_name("BlockAll")
        report["collision_setup_seconds"] = time.perf_counter()-cooking_start
        query_start = time.perf_counter()
        report["collision_samples"] = len(expected["collision_samples"])
        report["collision_passed"] = 0
        report["maximum_collision_error_cm"] = 0.
        for sample in expected["collision_samples"]:
            start = ue.Vector(*sample["point"])
            errors = []
            for sign, side in ((-1, "floor"), (1, "roof")):
                hit = ue.SystemLibrary.line_trace_single(world, start, start+ue.Vector(0, 0, sign*2000),
                    ue.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [], ue.DrawDebugTrace.NONE)
                errors.append(2000. if hit is None else abs(hit.to_tuple()[3]-sample[side]))
            error = max(errors)
            report["maximum_collision_error_cm"] = max(error, report["maximum_collision_error_cm"])
            report["collision_passed"] += int(error < 1.)
        report["collision_query_seconds"] = time.perf_counter()-query_start
        require(report["collision_passed"] == report["collision_samples"], "Dedicated collider ray checks failed")
        progress("dedicated collider finite-body overlaps and sweeps")
        body = expected["body"]
        native_body = dict(passed=False, stations=0, edges=0, station_failures=[], edge_failures=[],
                           height_m=body["height_m"], width_m=body["width_m"], margin_m=body["margin_m"])
        report["body"] = native_body

        def hits(a, b):
            return body_hits(world, a, b, body["radius"], body["half_axis"])

        for path in body["paths"]:
            points = path["points"]
            for i, centre in enumerate(points):
                native_body["stations"] += 1
                # Multi traces explicitly include initial overlaps, including zero-length casts.
                if hits(centre, centre):
                    native_body["station_failures"].append([path["segment_id"], i])
            for i, (a, b) in enumerate(zip(points, points[1:])):
                native_body["edges"] += 1
                if hits(a, b) or hits(b, a):
                    native_body["edge_failures"].append([path["segment_id"], i])
        start = body["paths"][0]["points"][0]
        floor_hit = ue.SystemLibrary.line_trace_single(
            world, ue.Vector(*start), ue.Vector(start[0], start[1], start[2]-2000),
            ue.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [], ue.DrawDebugTrace.NONE)
        require(floor_hit is not None, "No floor for body obstruction controls")
        floor = [start[0], start[1], start[2]-floor_hit.to_tuple()[3]]
        native_body["overlap_control"] = hits(floor, floor)
        native_body["sweep_control"] = hits(start, [floor[0], floor[1], floor[2]-body["radius"]])
        native_body["passed"] = (native_body["stations"] == body["stations"]
            and native_body["edges"] == body["edges"] and not native_body["station_failures"]
            and not native_body["edge_failures"] and native_body["overlap_control"]
            and native_body["sweep_control"])
        require(native_body["passed"], "Imported finite-body checks or floor obstruction controls failed")
        report['physics_controls'] = physics_controls(
            world, actors, collision_mesh.get_bounding_box().max + ue.Vector(10000, 10000, 10000))
        require(all(row['passed'] for row in report['physics_controls']), 'Native collision negative controls failed')
        progress('dedicated collider ground-robot poses, floor support and sweeps')
        report['ground'] = check_ground(world, actors, expected.get('ground'),
            collision_mesh.get_bounding_box().max + ue.Vector(11000, 11000, 11000))
        require(not report['ground']['enabled'] or report['ground']['passed'], 'Native ground robot checks failed')
        ue.EditorAssetLibrary.save_loaded_asset(collision_mesh)
        progress("dedicated collision passed; capturing materials")
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

        report["view_exposure"] = []
        require(len(expected["views"]) >= 2, "Missing branch inspection view plan")
        for view, specification in enumerate(expected["views"], 1):
            progress(f"material view {view}/{len(expected['views'])}")
            location = ue.Vector(*specification["point"])
            look = ue.Vector(*specification["look"])
            capture.set_actor_location(location, False, False)
            capture.set_actor_rotation(ue.MathLibrary.find_look_at_rotation(location, look), False)
            # The verified centre stays in air even in low passages; an arbitrary
            # light offset can cross the roof. Retain every exposure trial.
            light_actor.set_actor_location(location, False, False)
            intensity = 1500.0
            for attempt in range(8):
                light.set_intensity(intensity)
                trial = f"exposure_{view}_{attempt}.png"
                reference = render(trial)
                clipped = sum(all(c >= 254 for c in pixel) for pixel in reference) / len(reference)
                luminance = sorted((.2126*r+.7152*g+.0722*b)/255
                                   for r, g, b in reference[::32])
                p95 = luminance[int(.95*(len(luminance)-1))]
                if clipped <= 0.005 and p95 <= .7:
                    (output / f"interior_{view}.png").write_bytes((output / trial).read_bytes())
                    report["view_exposure"].append(dict(
                        view=view, intensity=intensity, clipped_fraction=clipped,
                        luminance_p95=p95, attempts=attempt+1))
                    break
                intensity *= 0.5
            else:
                raise RuntimeError("Inspection exposure exhausted its eight attempts")
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
        report["editor_peak_memory_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report["passed"] = True
        progress("complete")
    except Exception:
        report["failure"] = traceback.format_exc()
        progress("failed")
        raise
