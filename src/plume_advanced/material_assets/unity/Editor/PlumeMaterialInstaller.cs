using System;
using System.IO;
using UnityEditor;
using UnityEngine;

// Creates a new material, without assigning it or changing existing scene assets.
public static class PlumeMaterialInstaller
{
    [Serializable] private class Settings { public float tile_size_m = 4; public float normal_strength = 1; }

    [MenuItem("Tools/PLUME/Create continuous rock material (URP)")]
    public static void Create()
    {
        string selected = EditorUtility.OpenFilePanel("Choose this bundle's settings.json", Application.dataPath, "json");
        if (string.IsNullOrEmpty(selected)) return;
        CreateFromSettings(selected);
    }

    // The same installation path is callable from automated editor validation.
    public static Material CreateFromSettings(string selected)
    {
        selected = Path.GetFullPath(selected);
        string root = Path.GetDirectoryName(selected).Replace('\\', '/');
        string assets = Application.dataPath.Replace('\\', '/');
        if (!root.StartsWith(assets + "/", StringComparison.Ordinal))
            throw new InvalidOperationException("Copy the material bundle inside Assets first.");
        string assetRoot = "Assets" + root.Substring(assets.Length);
        Shader shader = Shader.Find("PLUME/Continuous Rock URP");
        if (shader == null || ShaderUtil.ShaderHasError(shader))
            throw new InvalidOperationException("Continuous Rock URP shader is missing or has compiler errors. Check Console and URP installation.");
        Settings settings = JsonUtility.FromJson<Settings>(File.ReadAllText(selected));
        Material material = new Material(shader);
        material.SetFloat("_TileSize", settings.tile_size_m);
        material.SetFloat("_NormalStrength", settings.normal_strength);
        material.SetFloat("_BlendExponent", 4);
        foreach (var pair in new[] { ("_ColorMap", "cave_base_color.png", true),
                                    ("_NormalMap", "cave_normal.png", false),
                                    ("_RoughnessMap", "cave_metallic_roughness.png", false) })
        {
            string path = assetRoot + "/textures/" + pair.Item2;
            var importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer == null) throw new FileNotFoundException("Missing texture", path);
            importer.textureType = TextureImporterType.Default;
            importer.sRGBTexture = pair.Item3;
            importer.wrapMode = TextureWrapMode.Repeat;
            importer.mipmapEnabled = true;
            importer.maxTextureSize = 4096;
            importer.filterMode = FilterMode.Trilinear;
            importer.anisoLevel = 4;
            // RGB normals must remain RGB; platform normal-map swizzling is not used.
            importer.textureCompression = TextureImporterCompression.Uncompressed;
            importer.SaveAndReimport();
            material.SetTexture(pair.Item1, AssetDatabase.LoadAssetAtPath<Texture2D>(path));
        }
        string output = AssetDatabase.GenerateUniqueAssetPath(assetRoot + "/PLUME_Continuous_Rock.mat");
        AssetDatabase.CreateAsset(material, output);
        AssetDatabase.SaveAssets();
        Selection.activeObject = material;
        Debug.Log("Created " + output + ". Assign to the imported cave_wall renderer. Validate in your active URP renderer.");
        return material;
    }
}
