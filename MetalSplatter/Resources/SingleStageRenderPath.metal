#include "SplatProcessing.h"

vertex FragmentIn singleStageSplatVertexShader(uint vertexID [[vertex_id]],
                                               uint instanceID [[instance_id]],
                                               ushort amplificationID [[amplification_id]],
                                               constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                               constant SplatSHCoefficients* splatSHArray [[ buffer(BufferIndexSphericalHarmonics) ]],
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
        out.splatIndex = 0;
        return out;
    }

    Splat splat = splatArray[splatID];

    (void)splatSHArray;
    return splatVertex(splat, uniforms, vertexID % 4, splatID);
}

fragment half4 singleStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                             texturecube<half> environmentMap [[texture(TextureIndexEnvironment)]],
                                             texture2d<half> brdfLUT [[texture(TextureIndexBRDF)]],
                                             sampler environmentSampler [[sampler(SamplerIndexEnvironment)]],
                                             sampler brdfSampler [[sampler(SamplerIndexBRDF)]],
                                             constant SplatSHCoefficients* splatSHArray [[ buffer(BufferIndexSphericalHarmonics) ]]) {
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    if (alpha <= 0) {
        return half4(0);
    }

    (void)splatSHArray;

    half ao = computeAmbientOcclusion(in.color.a);
    half3 shaded = shadeGaussian(in.albedo,
                                 in.metallic,
                                 in.roughness,
                                 in.normal,
                                 half3(in.viewDirection),
                                 ao,
                                 environmentMap,
                                 brdfLUT,
                                 environmentSampler,
                                 brdfSampler);

    return half4(shaded * alpha, alpha);
}
