// Shared by Unity and Unreal. Inputs are in one object-space frame, metres.
// ColorMap: sRGB; NormalMap: raw linear OpenGL RGB; RoughnessMap: linear glTF MR.
// Outputs: BaseColor (linear), Roughness, NormalObject. No UV/tangent inputs.
float3 n = normalize(BaseNormal);
float3 p = PositionM / max(TileSize, 0.000001);
float3 signN = float3(n.x > 0 ? 1 : -1, n.y > 0 ? 1 : -1, n.z > 0 ? 1 : -1);
float3 w = pow(abs(n), clamp(BlendExponent, 1.0, 16.0));
w /= max(w.x + w.y + w.z, 0.000001);
float2 uvX = float2(p.y, p.z * signN.x);
float2 uvY = float2(p.z, p.x * signN.y);
float2 uvZ = float2(p.x, p.y * signN.z);
// Raw PNG row conventions differ between APIs. The normal's +Y convention does not.
uvX.y = lerp(uvX.y, 1.0 - uvX.y, UVOriginTop);
uvY.y = lerp(uvY.y, 1.0 - uvY.y, UVOriginTop);
uvZ.y = lerp(uvZ.y, 1.0 - uvZ.y, UVOriginTop);
BaseColor = Texture2DSample(ColorMap, ColorMapSampler, uvX).rgb * w.x
          + Texture2DSample(ColorMap, ColorMapSampler, uvY).rgb * w.y
          + Texture2DSample(ColorMap, ColorMapSampler, uvZ).rgb * w.z;
Roughness = Texture2DSample(RoughnessMap, RoughnessMapSampler, uvX).g * w.x
          + Texture2DSample(RoughnessMap, RoughnessMapSampler, uvY).g * w.y
          + Texture2DSample(RoughnessMap, RoughnessMapSampler, uvZ).g * w.z;
float3 nx = Texture2DSample(NormalMap, NormalMapSampler, uvX).rgb * 2.0 - 1.0;
float3 ny = Texture2DSample(NormalMap, NormalMapSampler, uvY).rgb * 2.0 - 1.0;
float3 nz = Texture2DSample(NormalMap, NormalMapSampler, uvZ).rgb * 2.0 - 1.0;
float2 sx = nx.xy / max(nx.z, 0.1);
float2 sy = ny.xy / max(ny.z, 0.1);
float2 sz = nz.xy / max(nz.z, 0.1);
float3 gradient = float3(0, sx.x, sx.y * signN.x) * w.x
                + float3(sy.y * signN.y, 0, sy.x) * w.y
                + float3(sz.x, sz.y * signN.z, 0) * w.z;
gradient -= n * dot(n, gradient);
NormalObject = normalize(n + clamp(NormalStrength, 0.0, 10.0) * gradient);
