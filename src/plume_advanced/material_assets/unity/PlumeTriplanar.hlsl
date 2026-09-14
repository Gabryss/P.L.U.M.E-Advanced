#ifndef PLUME_TRIPLANAR_INCLUDED
#define PLUME_TRIPLANAR_INCLUDED
// Import normal PNG as Default / sRGB OFF, NOT Unity Normal Map compression.
// Shader Graph Custom Function uses the _float entry point and Float precision.
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Texture.hlsl"
#define Texture2DSample(T, S, UV) SAMPLE_TEXTURE2D(T, S, UV)
void PlumeSample(Texture2D ColorMap, SamplerState ColorMapSampler,
                 Texture2D NormalMap, SamplerState NormalMapSampler,
                 Texture2D RoughnessMap, SamplerState RoughnessMapSampler,
                 float3 PositionM, float3 BaseNormal, float TileSize,
                 float NormalStrength, float BlendExponent, float UVOriginTop,
                 out float3 BaseColor, out float Roughness, out float3 NormalObject)
{
    #include "../plume_triplanar_body.hlsl"
}
#undef Texture2DSample

// Object-space position and normal in Unity. The cyclic frame is remapped to
// right-handed Z-up before sampling, matching Blender for a glTFast X-flipped mesh.
void PlumeTriplanar_float(UnityTexture2D ColorMap, UnityTexture2D NormalMap,
    UnityTexture2D RoughnessMap, float3 PositionObject, float3 NormalObjectInput,
    float TileSize, float NormalStrength, float BlendExponent,
    out float3 BaseColor, out float Smoothness, out float3 NormalObject)
{
    float3 p = float3(-PositionObject.x, -PositionObject.z, PositionObject.y);
    float3 n = float3(-NormalObjectInput.x, -NormalObjectInput.z, NormalObjectInput.y);
    float3 projected;
    float roughness;
    PlumeSample(ColorMap.tex, ColorMap.samplerstate, NormalMap.tex, NormalMap.samplerstate,
        RoughnessMap.tex, RoughnessMap.samplerstate, p, n, TileSize, NormalStrength,
        BlendExponent, 0.0, BaseColor, roughness, projected);
    NormalObject = float3(-projected.x, projected.z, -projected.y);
    Smoothness = 1.0 - saturate(roughness);
}
#endif
