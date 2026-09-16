using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public static class PlumeNativeCheck
{
    [Serializable] class Sample { public Vector3 point; public float floor, roof; }
    [Serializable] class View { public Vector3 point, look; }
    [Serializable] class Route { public int segment_id; public Vector3[] points; }
    [Serializable] class Body { public float radius, half_axis, height_m, width_m, margin_m; public int stations, edges; public Route[] paths; }
    [Serializable] class GroundSupport { public Vector2 offset; public Vector3 weight; }
    [Serializable] class GroundPose { public Vector3 point, forward, up; public Sample[] probes; }
    [Serializable] class GroundMotion { public Vector3 start, end, forward, up, half_extents; }
    [Serializable] class Ground {
        public bool enabled; public float length_m, width_m, height_m, margin_m, max_slope_deg, max_step_m, support_spacing_m;
        public Vector3 half_extents; public int stations, sweeps, floor_samples;
        public GroundPose[] poses; public GroundMotion[] motions; public GroundSupport[] support;
    }
    [Serializable] class GroundReport {
        public bool enabled, passed; public int stations, sweeps, floor_samples, failures;
        public float length_m, width_m, height_m, margin_m, max_slope_deg, max_step_m, support_spacing_m;
        public float maximumFloorErrorM, maximumSlopeDeg, maximumStepM;
        public PhysicsControl[] controls;
    }
    [Serializable] class Expected { public int triangles, vertices, collision_triangles; public Sample[] samples, collision_samples; public View[] views; public Body body; public Ground ground; }
    [Serializable] class TextureCheck {
        public string property, type, compression, wrap; public bool srgb; public int width, height;
    }
    [Serializable] class PhysicsControl {
        public string name; public bool expected, observed, passed;
    }
    [Serializable] class NativeReport {
        public bool passed; public string unity, api, graphics, failure;
        public int triangles, vertices, passageSamples, passagePassed, collisionTriangles, collisionSamples, collisionPassed;
        public float maximumCollisionErrorM;
        public double reimportSeconds, visualCookingSeconds, collisionCookingSeconds, visualQuerySeconds, collisionQuerySeconds;
        public long editorPeakMemoryBytes;
        public float maximumVertexErrorM, maximumClearanceErrorM, uvRenderDifference, normalRenderDifference;
        public string[] shaderErrors; public TextureCheck[] textures;
        public float[] viewLightIntensity, viewClippedFraction, viewLuminanceP95;
        public bool bodyPassed, bodyOverlapControl, bodySweepControl;
        public int bodyStations, bodyEdges, bodyStationFailures, bodyEdgeFailures;
        public float bodyHeightM, bodyWidthM, bodyMarginM;
        public PhysicsControl[] physicsControls;
        public GroundReport ground;
    }
    static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }
    static string Output(string name) { return Path.Combine(Application.dataPath, "../" + name); }

    public static void EvaluateGroundFixture() {
        var report=new NativeReport {unity=Application.unityVersion};
        try {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
            AssetDatabase.ImportAsset("Assets/Cave/ground_fixture.glb",ImportAssetOptions.ForceUpdate);
            var prefab=AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Cave/ground_fixture.glb");
            Require(prefab != null,"Ground fixture failed import");
            var room=UnityEngine.Object.Instantiate(prefab);
            var filter=room.GetComponentInChildren<MeshFilter>();
            var collider=filter.gameObject.AddComponent<MeshCollider>();
            collider.sharedMesh=filter.sharedMesh; collider.convex=false; collider.gameObject.layer=8;
            Physics.queriesHitBackfaces=true; Physics.SyncTransforms();
            var plan=JsonUtility.FromJson<Ground>(File.ReadAllText(Output("ground_expected.json")));
            report.ground=CheckGround(plan,collider); report.passed=report.ground.passed;
        } catch (Exception error) { report.failure=error.ToString(); }
        File.WriteAllText(Output("ground_result.json"),JsonUtility.ToJson(report,true));
        EditorApplication.Exit(report.passed ? 0 : 1);
    }

    public static void Evaluate()
    {
        var report = new NativeReport { unity = Application.unityVersion,
            api = SystemInfo.graphicsDeviceType.ToString(), graphics = SystemInfo.graphicsDeviceName };
        try
        {
            ShaderUtil.allowAsyncCompilation = false;
            var expected = JsonUtility.FromJson<Expected>(File.ReadAllText(Output("expected.json")));
            var timer = System.Diagnostics.Stopwatch.StartNew();
            AssetDatabase.ImportAsset("Assets/Cave/plume_cave.glb", ImportAssetOptions.ForceUpdate);
            AssetDatabase.ImportAsset("Assets/Cave/plume_collision.glb", ImportAssetOptions.ForceUpdate);
            report.reimportSeconds = timer.Elapsed.TotalSeconds;
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Cave/plume_cave.glb");
            Require(prefab != null, "glTFast did not import a prefab");
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var cave = UnityEngine.Object.Instantiate(prefab);
            cave.name = "PLUME inspected cave";
            var filters = cave.GetComponentsInChildren<MeshFilter>();
            Require(filters.Length == 1, "Expected one cave primitive");
            var filter = filters[0];
            var mesh = UnityEngine.Object.Instantiate(filter.sharedMesh);
            filter.sharedMesh = mesh;
            report.vertices = mesh.vertexCount;
            report.triangles = (int)(mesh.GetIndexCount(0) / 3);
            Require(report.vertices == expected.vertices && report.triangles == expected.triangles,
                "Imported geometry counts changed");
            var vertices = mesh.vertices.Select(p => filter.transform.TransformPoint(p))
                .OrderBy(p => p.x).ThenBy(p => p.y).ThenBy(p => p.z).ToArray();
            using (var reader = new BinaryReader(File.OpenRead(Output("positions_unity.f32"))))
                foreach (var point in vertices) {
                    var original = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                    report.maximumVertexErrorM = Mathf.Max(report.maximumVertexErrorM,
                        (point-original).magnitude);
                }
            Require(report.maximumVertexErrorM < 0.002f, "Native vertex positions changed by more than 2 mm");
            timer.Restart();
            var collider = filter.gameObject.AddComponent<MeshCollider>();
            collider.sharedMesh = mesh; collider.convex = false;
            Physics.queriesHitBackfaces = true;
            Physics.SyncTransforms();
            report.visualCookingSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            report.passageSamples = expected.samples.Length;
            foreach (var sample in expected.samples) {
                bool floor = collider.Raycast(new Ray(sample.point, Vector3.down), out var lower, 20);
                bool roof = collider.Raycast(new Ray(sample.point, Vector3.up), out var upper, 20);
                if (floor && roof) {
                    float error = Mathf.Max(Mathf.Abs(lower.distance-sample.floor), Mathf.Abs(upper.distance-sample.roof));
                    report.maximumClearanceErrorM = Mathf.Max(error, report.maximumClearanceErrorM);
                    if (error < 0.01f) report.passagePassed++;
                }
            }
            Require(report.passagePassed == report.passageSamples, "Imported passage ray checks failed");
            report.visualQuerySeconds = timer.Elapsed.TotalSeconds;
            collider.enabled = false;
            var collisionPrefab = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Cave/plume_collision.glb");
            Require(collisionPrefab != null, "Dedicated collider failed import");
            var collisionObject = UnityEngine.Object.Instantiate(collisionPrefab);
            collisionObject.name = "PLUME dedicated collision";
            foreach (var renderer in collisionObject.GetComponentsInChildren<Renderer>()) renderer.enabled = false;
            var collisionFilters = collisionObject.GetComponentsInChildren<MeshFilter>();
            Require(collisionFilters.Length == 1, "Expected one dedicated collider primitive");
            var collisionFilter = collisionFilters[0];
            report.collisionTriangles = (int)(collisionFilter.sharedMesh.GetIndexCount(0)/3);
            Require(report.collisionTriangles == expected.collision_triangles, "Dedicated collision triangle count changed");
            timer.Restart();
            var dedicated = collisionFilter.gameObject.AddComponent<MeshCollider>();
            dedicated.sharedMesh = collisionFilter.sharedMesh;
            dedicated.convex = false;
            Physics.SyncTransforms();
            report.collisionCookingSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            report.collisionSamples = expected.collision_samples.Length;
            foreach (var sample in expected.collision_samples) {
                bool floor = dedicated.Raycast(new Ray(sample.point, Vector3.down), out var lower, 20);
                bool roof = dedicated.Raycast(new Ray(sample.point, Vector3.up), out var upper, 20);
                if (floor && roof) {
                    float error = Mathf.Max(Mathf.Abs(lower.distance-sample.floor), Mathf.Abs(upper.distance-sample.roof));
                    report.maximumCollisionErrorM = Mathf.Max(error, report.maximumCollisionErrorM);
                    if (error < 0.01f) report.collisionPassed++;
                }
            }
            report.collisionQuerySeconds = timer.Elapsed.TotalSeconds;
            Require(report.collisionPassed == report.collisionSamples, "Dedicated collider ray checks failed");
            CheckBody(expected.body, dedicated, report);
            report.physicsControls = CheckPhysicsControls(dedicated.bounds.max + Vector3.one*100);
            Require(report.physicsControls.All(row => row.passed), "Native collision negative controls failed");
            report.ground = CheckGround(expected.ground, dedicated);
            Require(!report.ground.enabled || report.ground.passed, "Ground robot import checks failed");
            var material = PlumeMaterialInstaller.CreateFromSettings(Path.Combine(Application.dataPath, "PlumeMaterial/settings.json"));
            foreach (var renderer in cave.GetComponentsInChildren<Renderer>()) renderer.sharedMaterial = material;
            report.textures = new[] { "_ColorMap", "_NormalMap", "_RoughnessMap" }.Select(name => {
                var texture = material.GetTexture(name);
                var importer = (TextureImporter)AssetImporter.GetAtPath(AssetDatabase.GetAssetPath(texture));
                Require(texture.width == 4096 && texture.height == 4096, "Expected retained 4K maps");
                Require(importer.sRGBTexture == (name == "_ColorMap"), "Incorrect texture color space");
                Require(importer.textureType == TextureImporterType.Default && importer.wrapMode == TextureWrapMode.Repeat,
                    "Raw RGB/repeat texture settings changed");
                return new TextureCheck { property=name, width=texture.width, height=texture.height,
                    srgb=importer.sRGBTexture, type=importer.textureType.ToString(),
                    compression=importer.textureCompression.ToString(), wrap=importer.wrapMode.ToString() };
            }).ToArray();
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.08f, 0.08f, 0.08f);
            var camera = new GameObject("Inspection camera").AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = Color.black;
            camera.nearClipPlane = 0.025f; camera.farClipPlane = 150; camera.fieldOfView = 75;
            camera.allowHDR = false; camera.allowMSAA = false;
            camera.gameObject.AddComponent<UniversalAdditionalCameraData>().renderPostProcessing = false;
            var light = new GameObject("Inspection point light").AddComponent<Light>();
            light.type = LightType.Point; light.range = 20; light.intensity = 12; light.shadows = LightShadows.Soft;
            Require(expected.views != null && expected.views.Length >= 2, "Missing branch inspection view plan");
            report.viewLightIntensity = new float[expected.views.Length];
            report.viewClippedFraction = new float[expected.views.Length];
            report.viewLuminanceP95 = new float[expected.views.Length];
            Color32[] reference = null;
            for (int view=0; view<expected.views.Length; view++) {
                camera.transform.position = expected.views[view].point;
                camera.transform.LookAt(expected.views[view].look);
                Debug.Log("PLUME material view " + (view+1) + "/" + expected.views.Length);
                // Keep the inspection light at the verified air sample. Fixed offsets can
                // put it inside the rock when passage height is only a few decimetres.
                light.transform.position = camera.transform.position;
                light.intensity = 12;
                bool exposed = false;
                for (int attempt=0; attempt<8; attempt++) {
                    string trial = "exposure_"+(view+1)+"_"+attempt+".png";
                    reference = Render(camera, trial);
                    float clipped = reference.Count(p => p.r>=254 && p.g>=254 && p.b>=254)/(float)reference.Length;
                    // Meter the bright surfaces too: almost-white pixels can hide
                    // material detail while staying just below a clipping cutoff.
                    var luminance = reference.Where((p,i) => i % 32 == 0)
                        .Select(p => (0.2126f*p.r + 0.7152f*p.g + 0.0722f*p.b)/255f).OrderBy(v => v).ToArray();
                    float p95 = luminance[(int)(0.95f*(luminance.Length-1))];
                    if (clipped <= 0.005f && p95 <= 0.7f) {
                        File.Copy(Output(trial), Output("interior_"+(view+1)+".png"), true);
                        report.viewLightIntensity[view] = light.intensity;
                        report.viewClippedFraction[view] = clipped;
                        report.viewLuminanceP95[view] = p95;
                        exposed = true;
                        break;
                    }
                    light.intensity *= 0.5f;
                }
                Require(exposed, "Inspection exposure exhausted its eight attempts");
            }
            var originalUV = mesh.uv;
            mesh.uv = Enumerable.Repeat(new Vector2(0.12f,0.91f), mesh.vertexCount).ToArray();
            var changedUV = Render(camera, "uv_control.png");
            report.uvRenderDifference = Difference(reference, changedUV);
            Require(report.uvRenderDifference < 0.001f, "Continuous material depends on UV charts");
            mesh.uv = originalUV;
            float strength = material.GetFloat("_NormalStrength");
            material.SetFloat("_NormalStrength", 0);
            report.normalRenderDifference = Difference(reference, Render(camera, "normal_off_control.png"));
            material.SetFloat("_NormalStrength", strength);
            Require(report.normalRenderDifference > 0.0001f, "Normal map has no visible contribution");
            report.shaderErrors = ShaderUtil.GetShaderMessages(material.shader)
                .Where(m => m.severity.ToString() == "Error").Select(m => m.message).ToArray();
            Require(report.shaderErrors.Length == 0 && material.shader.isSupported, "Rendered shader has errors");
            AssetDatabase.CreateAsset(mesh, "Assets/PLUME_Inspection_Mesh.asset");
            AssetDatabase.SaveAssets();
            EditorSceneManager.SaveScene(camera.gameObject.scene, "Assets/PLUME_Inspection.unity");
            report.passed = true;
        }
        catch (Exception error) { report.failure=error.ToString(); Debug.LogException(error); }
        report.editorPeakMemoryBytes = System.Diagnostics.Process.GetCurrentProcess().PeakWorkingSet64;
        File.WriteAllText(Output("native_result.json"), JsonUtility.ToJson(report,true));
        EditorApplication.Exit(report.passed ? 0 : 1);
    }

    static bool BodyOverlap(Vector3 centre, Body body) {
        var offset = Vector3.up * body.half_axis;
        return body.half_axis == 0
            ? Physics.CheckSphere(centre, body.radius, 1 << 8, QueryTriggerInteraction.Ignore)
            : Physics.CheckCapsule(centre-offset, centre+offset, body.radius, 1 << 8, QueryTriggerInteraction.Ignore);
    }

    static bool BoxOverlap(Vector3 point, Vector3 half, Quaternion rotation) {
        return Physics.CheckBox(point, half, rotation, 1 << 8, QueryTriggerInteraction.Ignore);
    }
    static bool BoxSweep(Vector3 a, Vector3 b, Vector3 half, Quaternion rotation) {
        var delta = b-a;
        return delta.sqrMagnitude > 1e-16f && Physics.BoxCast(a, half, delta.normalized,
            out _, rotation, delta.magnitude, 1 << 8, QueryTriggerInteraction.Ignore);
    }
    static GroundReport CheckGround(Ground plan, MeshCollider collider) {
        var r = new GroundReport { enabled=plan != null && plan.enabled };
        if (!r.enabled) return r;
        r.length_m=plan.length_m; r.width_m=plan.width_m; r.height_m=plan.height_m;
        r.margin_m=plan.margin_m; r.max_slope_deg=plan.max_slope_deg;
        r.max_step_m=plan.max_step_m; r.support_spacing_m=plan.support_spacing_m;
        foreach (var pose in plan.poses) {
            r.stations++;
            if (BoxOverlap(pose.point, plan.half_extents, Quaternion.LookRotation(pose.forward, pose.up))) r.failures++;
            var heights = new double[pose.probes.Length];
            double a=0, b=0, c=0;
            for (int i=0; i<heights.Length; i++) {
                var probe=pose.probes[i]; r.floor_samples++;
                if (!collider.Raycast(new Ray(probe.point, Vector3.down), out var hit, probe.floor+0.1f)) {
                    r.failures++; continue;
                }
                r.maximumFloorErrorM=Mathf.Max(r.maximumFloorErrorM, Mathf.Abs(hit.distance-probe.floor));
                heights[i]=probe.point.y-hit.distance;
                var w=plan.support[i].weight;
                a+=w.x*heights[i]; b+=w.y*heights[i]; c+=w.z*heights[i];
            }
            double low=double.PositiveInfinity, high=double.NegativeInfinity;
            for (int i=0; i<heights.Length; i++) {
                var xy=plan.support[i].offset;
                double residual=heights[i]-(a*xy.x+b*xy.y+c);
                low=Math.Min(low,residual); high=Math.Max(high,residual);
            }
            r.maximumSlopeDeg=Mathf.Max(r.maximumSlopeDeg,(float)(Math.Atan(Math.Sqrt(a*a+b*b))*180/Math.PI));
            r.maximumStepM=Mathf.Max(r.maximumStepM,(float)(high-low));
        }
        foreach (var motion in plan.motions) {
            r.sweeps++;
            var rotation=Quaternion.LookRotation(motion.forward,motion.up);
            if (BoxOverlap(motion.start,motion.half_extents,rotation)
                || BoxOverlap(motion.end,motion.half_extents,rotation)
                || BoxSweep(motion.start,motion.end,motion.half_extents,rotation)
                || BoxSweep(motion.end,motion.start,motion.half_extents,rotation)) r.failures++;
        }
        r.controls=CheckGroundControls(collider.bounds.max+Vector3.one*110);
        r.passed=r.failures == 0 && r.stations == plan.stations && r.sweeps == plan.sweeps
            && r.floor_samples == plan.floor_samples && r.maximumFloorErrorM <= .002f
            && r.maximumSlopeDeg <= plan.max_slope_deg+1e-5f && r.maximumStepM <= plan.max_step_m+1e-5f
            && r.controls.All(row=>row.passed);
        return r;
    }

    static PhysicsControl[] CheckGroundControls(Vector3 origin) {
        var rows=new System.Collections.Generic.List<PhysicsControl>();
        Action<string,bool,bool> record=(name,expected,observed)=>rows.Add(
            new PhysicsControl {name=name,expected=expected,observed=observed,passed=expected==observed});
        var obstacle=new GameObject("PLUME temporary ground controls");
        obstacle.layer=8; obstacle.transform.position=origin;
        var wall=obstacle.AddComponent<BoxCollider>(); wall.size=new Vector3(.02f,4,4);
        var half=new Vector3(.27f,.27f,.37f);
        var rotation=Quaternion.LookRotation(Vector3.right,Vector3.up);
        GameObject second=null;
        try {
            Physics.SyncTransforms();
            record("box_initial_overlap",true,BoxOverlap(origin,half,rotation));
            record("box_length_overlap",true,BoxOverlap(origin+Vector3.right*.34f,half,rotation));
            record("box_rotated_clear",false,BoxOverlap(origin+Vector3.right*.34f,half,Quaternion.identity));
            record("box_clear_start",false,BoxOverlap(origin+Vector3.right,half,rotation));
            record("box_thin_wall_forward",true,BoxSweep(origin-Vector3.right,origin+Vector3.right,half,rotation));
            record("box_thin_wall_reverse",true,BoxSweep(origin+Vector3.right,origin-Vector3.right,half,rotation));
            record("box_clear_sweep",false,BoxSweep(origin+Vector3.right,origin+Vector3.right*2,half,rotation));
            wall.size=new Vector3(4,.02f,4); Physics.SyncTransforms();
            record("floor_present",true,wall.Raycast(new Ray(origin+Vector3.up,Vector3.down),out _,2));
            record("floor_missing",false,wall.Raycast(new Ray(origin+Vector3.right*5+Vector3.up,Vector3.down),out _,2));
            foreach (float angle in new[] {15f,25f}) {
                obstacle.transform.rotation=Quaternion.Euler(0,0,angle); Physics.SyncTransforms();
                bool hit=wall.Raycast(new Ray(origin+Vector3.up,Vector3.down),out var floor,2);
                record(angle == 15 ? "slope_below_limit" : "slope_above_limit", angle == 15,
                    hit && Vector3.Angle(floor.normal,Vector3.up) <= 20);
            }
            obstacle.transform.rotation=Quaternion.identity;
            obstacle.transform.position=origin-Vector3.right*.5f; wall.size=new Vector3(1,.02f,1);
            second=new GameObject("PLUME temporary step"); second.layer=8;
            var top=second.AddComponent<BoxCollider>(); top.size=wall.size;
            foreach (float step in new[] {.06f,.15f}) {
                second.transform.position=origin+Vector3.right*.5f+Vector3.up*step; Physics.SyncTransforms();
                bool low=wall.Raycast(new Ray(origin-Vector3.right*.5f+Vector3.up,Vector3.down),out var a,2);
                bool high=top.Raycast(new Ray(origin+Vector3.right*.5f+Vector3.up,Vector3.down),out var b,2);
                record(step < .1 ? "step_below_limit" : "step_above_limit",step < .1,
                    low && high && Mathf.Abs(a.distance-b.distance) <= .1f);
            }
        } finally {
            if (second != null) UnityEngine.Object.DestroyImmediate(second);
            UnityEngine.Object.DestroyImmediate(obstacle); Physics.SyncTransforms();
        }
        return rows.ToArray();
    }
    static bool BodySweep(Vector3 a, Vector3 b, Body body) {
        var delta = b-a;
        if (delta.sqrMagnitude < 1e-16f) return false;
        var offset = Vector3.up * body.half_axis;
        return body.half_axis == 0
            ? Physics.SphereCast(a, body.radius, delta.normalized, out _, delta.magnitude, 1 << 8, QueryTriggerInteraction.Ignore)
            : Physics.CapsuleCast(a-offset, a+offset, body.radius, delta.normalized, out _, delta.magnitude, 1 << 8, QueryTriggerInteraction.Ignore);
    }
    static void CheckBody(Body body, MeshCollider dedicated, NativeReport report) {
        Require(body != null && body.paths != null && body.paths.Length > 0, "Missing finite-body routes");
        dedicated.gameObject.layer = 8;
        Physics.SyncTransforms();
        report.bodyHeightM=body.height_m; report.bodyWidthM=body.width_m; report.bodyMarginM=body.margin_m;
        foreach (var path in body.paths) {
            foreach (var centre in path.points) {
                report.bodyStations++;
                if (BodyOverlap(centre, body)) report.bodyStationFailures++;
            }
            for (int i=1; i<path.points.Length; i++) {
                report.bodyEdges++;
                // Query both orientations: mesh backface handling differs across native engines.
                if (BodySweep(path.points[i-1], path.points[i], body)
                    || BodySweep(path.points[i], path.points[i-1], body)) report.bodyEdgeFailures++;
            }
        }
        var start=body.paths[0].points[0];
        Require(dedicated.Raycast(new Ray(start, Vector3.down), out var floor, 20), "No floor for body controls");
        report.bodyOverlapControl=BodyOverlap(floor.point, body);
        report.bodySweepControl=BodySweep(start, floor.point-Vector3.up*body.radius, body);
        report.bodyPassed=report.bodyStations == body.stations && report.bodyEdges == body.edges
            && report.bodyStationFailures == 0 && report.bodyEdgeFailures == 0
            && report.bodyOverlapControl && report.bodySweepControl;
        Require(report.bodyPassed, "Imported finite-body overlap/sweep checks or floor obstruction controls failed");
    }

    static PhysicsControl[] CheckPhysicsControls(Vector3 origin) {
        // A temporary obstacle away from the cave exercises the same query
        // methods used above. No control object is retained in the saved scene.
        var controls = new System.Collections.Generic.List<PhysicsControl>();
        var obstacle = new GameObject("PLUME temporary collision control");
        obstacle.layer = 8; obstacle.transform.position = origin;
        var wall = obstacle.AddComponent<BoxCollider>();
        wall.size = new Vector3(0.02f, 4, 4);
        Action<string,bool,bool> record = (name, expected, observed) => controls.Add(
            new PhysicsControl { name=name, expected=expected, observed=observed, passed=expected==observed });
        try {
            Physics.SyncTransforms();
            foreach (bool tall in new[] {false, true}) {
                var body = new Body {radius=0.27f, half_axis=tall ? 0.5f : 0};
                string prefix = tall ? "capsule_" : "sphere_";
                record(prefix+"initial_overlap", true, BodyOverlap(origin, body));
                record(prefix+"grazing_overlap", true, BodyOverlap(origin+Vector3.right*0.14f, body));
                record(prefix+"clear_start", false, BodyOverlap(origin+Vector3.right, body));
                record(prefix+"thin_wall_forward", true, BodySweep(origin-Vector3.right, origin+Vector3.right, body));
                record(prefix+"thin_wall_reverse", true, BodySweep(origin+Vector3.right, origin-Vector3.right, body));
                record(prefix+"clear_sweep", false, BodySweep(origin+Vector3.right, origin+Vector3.right*2, body));
            }
            wall.size = new Vector3(4, 0.02f, 4);
            Physics.SyncTransforms();
            record("capsule_axis_overlap", true, BodyOverlap(origin+Vector3.up*0.55f,
                new Body {radius=0.27f, half_axis=0.5f}));
            record("sphere_axis_clear", false, BodyOverlap(origin+Vector3.up*0.55f,
                new Body {radius=0.27f, half_axis=0}));
        }
        finally { UnityEngine.Object.DestroyImmediate(obstacle); Physics.SyncTransforms(); }
        return controls.ToArray();
    }

    static Color32[] Render(Camera camera, string name)
    {
        var target = new RenderTexture(960, 640, 24, RenderTextureFormat.ARGB32);
        target.Create();
        // A freshly constructed render pipeline needs its first camera submission
        // before a screenshot can be treated as inspection evidence.
        for (int frame=0; frame<3; frame++)
            RenderPipeline.SubmitRenderRequest(camera, new UniversalRenderPipeline.SingleCameraRequest { destination=target });
        var previous = RenderTexture.active; RenderTexture.active=target;
        var image = new Texture2D(960, 640, TextureFormat.RGBA32, false);
        image.ReadPixels(new Rect(0,0,960,640),0,0); image.Apply();
        var pixels=image.GetPixels32();
        File.WriteAllBytes(Output(name), image.EncodeToPNG());
        RenderTexture.active=previous; target.Release();
        UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(image);
        return pixels;
    }
    static float Difference(Color32[] a, Color32[] b) {
        double sum=0;
        for(int i=0;i<a.Length;i++) sum+=Math.Abs(a[i].r-b[i].r)+Math.Abs(a[i].g-b[i].g)+Math.Abs(a[i].b-b[i].b);
        return (float)(sum/(a.Length*3*255.0));
    }
    [Serializable] class BootstrapReport
    {
        public string unity, graphics, api;
        public bool shaderFound, shaderSupported;
        public string[] shaderErrors;
    }

    public static void Bootstrap()
    {
        try
        {
            var renderer = ScriptableObject.CreateInstance<UniversalRendererData>();
            AssetDatabase.CreateAsset(renderer, "Assets/PLUME_Renderer.asset");
            var pipeline = UniversalRenderPipelineAsset.Create(renderer);
            pipeline.supportsHDR = false;
            pipeline.msaaSampleCount = 1;
            AssetDatabase.CreateAsset(pipeline, "Assets/PLUME_URP.asset");
            GraphicsSettings.defaultRenderPipeline = pipeline;
            QualitySettings.renderPipeline = pipeline;
            AssetDatabase.SaveAssets();
            var shader = Shader.Find("PLUME/Continuous Rock URP");
            var report = new BootstrapReport {
                unity = Application.unityVersion, graphics = SystemInfo.graphicsDeviceName,
                api = SystemInfo.graphicsDeviceType.ToString(), shaderFound = shader != null,
                shaderSupported = shader != null && shader.isSupported,
                shaderErrors = shader == null ? new[] { "Shader missing" } :
                    ShaderUtil.GetShaderMessages(shader).Where(m => m.severity.ToString() == "Error")
                        .Select(m => m.message).ToArray()
            };
            File.WriteAllText(Path.Combine(Application.dataPath, "../bootstrap.json"), JsonUtility.ToJson(report, true));
            if (!report.shaderFound || report.shaderErrors.Length != 0)
                throw new Exception("Shader bootstrap failed");
            Debug.Log("PLUME native bootstrap completed");
        }
        catch (Exception error) { Debug.LogException(error); EditorApplication.Exit(1); }
    }
}
