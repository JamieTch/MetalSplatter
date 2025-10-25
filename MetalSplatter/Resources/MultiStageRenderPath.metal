#include "SplatProcessing.h"

typedef struct
{
    half4 albedoMetallic [[raster_order_group(0)]];
    half4 normalRoughness [[raster_order_group(0)]];
    half4 viewAlpha [[raster_order_group(0)]];
    half2 ambientOcclusion [[raster_order_group(0)]];
    float depth [[raster_order_group(0)]];
} FragmentValues;

typedef struct
{
    FragmentValues values [[imageblock_data]];
} FragmentStore;

typedef struct
{
    half4 color [[color(0)]];
    float depth [[depth(any)]];
} FragmentOut;

kernel void initializeFragmentStore(imageblock<FragmentValues, imageblock_layout_explicit> blockData,
                                    ushort2 localThreadID [[thread_position_in_threadgroup]]) {
    threadgroup_imageblock FragmentValues *values = blockData.data(localThreadID);
    values->albedoMetallic = half4(0);
    values->normalRoughness = half4(0);
    values->viewAlpha = half4(0);
    values->ambientOcclusion = half2(0);
    values->depth = 0;
}

vertex FragmentIn multiStageSplatVertexShader(uint vertexID [[vertex_id]],
                                              uint instanceID [[instance_id]],
                                              ushort amplificationID [[amplification_id]],
                                              constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                              constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];

    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.albedo = half3(0);
        out.metallic = half(0);
        out.roughness = half(0);
        out.normal = half3(0);
        out.worldPosition = float3(0);
        out.viewDirection = float3(0);
        return out;
    }

    Splat splat = splatArray[splatID];

    return splatVertex(splat, uniforms, vertexID % 4);
}

fragment FragmentStore multiStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                                     FragmentValues previousFragmentValues [[imageblock_data]]) {
    FragmentStore out;

    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    if (alpha <= 0) {
        out.values = previousFragmentValues;
        return out;
    }

    half oneMinusAlpha = 1 - alpha;
    half ao = computeAmbientOcclusion(in.color.a);

    half4 albedoMetallic = half4(in.albedo * alpha, in.metallic * alpha);
    half4 normalRoughness = half4(in.normal * alpha, in.roughness * alpha);
    half4 viewAlpha = half4(half3(in.viewDirection) * alpha, alpha);
    half2 ambientOcclusion = half2(ao * alpha, 0);

    out.values.albedoMetallic = previousFragmentValues.albedoMetallic * oneMinusAlpha + albedoMetallic;
    out.values.normalRoughness = previousFragmentValues.normalRoughness * oneMinusAlpha + normalRoughness;
    out.values.viewAlpha = previousFragmentValues.viewAlpha * oneMinusAlpha + viewAlpha;
    out.values.ambientOcclusion = previousFragmentValues.ambientOcclusion * oneMinusAlpha + ambientOcclusion;

    float depth = in.position.z;
    out.values.depth = previousFragmentValues.depth * oneMinusAlpha + depth * alpha;

    return out;
}

/// Generate a single triangle covering the entire screen
vertex FragmentIn postprocessVertexShader(uint vertexID [[vertex_id]]) {
    FragmentIn out;

    float4 position;
    position.x = (vertexID == 2) ? 3.0 : -1.0;
    position.y = (vertexID == 0) ? -3.0 : 1.0;
    position.zw = 1.0;

    out.position = position;
    out.relativePosition = half2(0);
    out.color = half4(0);
    out.albedo = half3(0);
    out.metallic = half(0);
    out.roughness = half(0);
    out.normal = half3(0);
    out.worldPosition = float3(0);
    out.viewDirection = float3(0);
    return out;
}

inline half4 resolveFragmentValues(FragmentValues fragmentValues,
                                   texturecube<half> environmentMap,
                                   texture2d<half> brdfLUT,
                                   sampler environmentSampler,
                                   sampler brdfSampler) {
    half accumulatedAlpha = fragmentValues.viewAlpha.w;
    if (accumulatedAlpha <= 0) {
        return half4(0);
    }

    half invAlpha = half(1.0) / accumulatedAlpha;
    half3 albedo = half3(fragmentValues.albedoMetallic.xyz * invAlpha);
    half metallic = fragmentValues.albedoMetallic.w * invAlpha;
    half3 normal = half3(fragmentValues.normalRoughness.xyz * invAlpha);
    half roughness = fragmentValues.normalRoughness.w * invAlpha;
    half3 viewDirection = half3(fragmentValues.viewAlpha.xyz * invAlpha);
    half ao = fragmentValues.ambientOcclusion.x * invAlpha;

    half3 shaded = shadeGaussian(albedo,
                                 metallic,
                                 roughness,
                                 normal,
                                 viewDirection,
                                 ao,
                                 environmentMap,
                                 brdfLUT,
                                 environmentSampler,
                                 brdfSampler);

    return half4(shaded * accumulatedAlpha, accumulatedAlpha);
}

fragment FragmentOut postprocessFragmentShader(FragmentValues fragmentValues [[imageblock_data]],
                                               texturecube<half> environmentMap [[texture(TextureIndexEnvironment)]],
                                               texture2d<half> brdfLUT [[texture(TextureIndexBRDF)]],
                                               sampler environmentSampler [[sampler(SamplerIndexEnvironment)]],
                                               sampler brdfSampler [[sampler(SamplerIndexBRDF)]]) {
    FragmentOut out;
    half accumulatedAlpha = fragmentValues.viewAlpha.w;
    out.depth = (accumulatedAlpha == 0) ? 0 : fragmentValues.depth / accumulatedAlpha;
    out.color = resolveFragmentValues(fragmentValues,
                                      environmentMap,
                                      brdfLUT,
                                      environmentSampler,
                                      brdfSampler);
    return out;
}

fragment half4 postprocessFragmentShaderNoDepth(FragmentValues fragmentValues [[imageblock_data]],
                                               texturecube<half> environmentMap [[texture(TextureIndexEnvironment)]],
                                               texture2d<half> brdfLUT [[texture(TextureIndexBRDF)]],
                                               sampler environmentSampler [[sampler(SamplerIndexEnvironment)]],
                                               sampler brdfSampler [[sampler(SamplerIndexBRDF)]]) {
    return resolveFragmentValues(fragmentValues,
                                 environmentMap,
                                 brdfLUT,
                                 environmentSampler,
                                 brdfSampler);
}
