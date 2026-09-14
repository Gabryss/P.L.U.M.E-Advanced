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
    [Serializable] class Expected { public int triangles, vertices, collision_triangles; public Sample[] samples, collision_samples; public View[] views; }
    [Serializable] class TextureCheck {
        public string property, type, compression, wrap; public bool srgb; public int width, height;
    }
    [Serializable] class NativeReport {
        public bool passed; public string unity, api, graphics, failure;
        public int triangles, vertices, passageSamples, passagePassed, collisionTriangles, collisionSamples, collisionPassed;
        public float maximumCollisionErrorM;
        public double reimportSeconds, visualCookingSeconds, collisionCookingSeconds, visualQuerySeconds, collisionQuerySeconds;
        public long editorPeakMemoryBytes;
        public float maximumVertexErrorM, maximumClearanceErrorM, uvRenderDifference, normalRenderDifference;
        public string[] shaderErrors; public TextureCheck[] textures;
        public float[] viewLightIntensity, viewClippedFraction;
    }
    static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }
    static string Output(string name) { return Path.Combine(Application.dataPath, "../" + name); }

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
                    if (clipped <= 0.005f) {
                        File.Copy(Output(trial), Output("interior_"+(view+1)+".png"), true);
                        report.viewLightIntensity[view] = light.intensity;
                        report.viewClippedFraction[view] = clipped;
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
