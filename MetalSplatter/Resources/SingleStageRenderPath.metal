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
        out.diffuseSH = float3(0);
        out.specularSH = float3(0);
        return out;
    }

    Splat splat = splatArray[splatID];

    FragmentIn out = splatVertex(splat, uniforms, vertexID % 4, splatID);

    ushort configuredCoefficientCount = ushort(min(uniforms.shCoefficientCount, 16u));
    SplatSHCoefficients shCoefficients = splatSHArray[splatID];
    float3 normal = safeNormalize(float3(out.normal), float3(0, 0, 1));
    float3 viewDirection = safeNormalize(float3(out.viewDirection), float3(0, 0, 1));
    float3 reflectionDirection = reflect(-viewDirection, normal);
    out.diffuseSH = evaluateSplatSHForDiffuse(shCoefficients, configuredCoefficientCount, normal);
    out.specularSH = evaluateSplatSHForSpecular(shCoefficients, configuredCoefficientCount, reflectionDirection);

    return out;
}

fragment half4 singleStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                             ushort viewIndex [[render_target_array_index]],
                                             texturecube<half> environmentMap [[texture(TextureIndexEnvironment)]],
                                             texture2d<half> brdfLUT [[texture(TextureIndexBRDF)]],
                                             sampler environmentSampler [[sampler(SamplerIndexEnvironment)]],
                                             sampler brdfSampler [[sampler(SamplerIndexBRDF)]],
                                             constant SplatSHCoefficients* splatSHArray [[ buffer(BufferIndexSphericalHarmonics) ]],
                                             constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    (void)splatSHArray;
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    if (alpha <= 0) {
        return half4(0);
    }

    Uniforms uniforms = uniformsArray.uniforms[min(int(viewIndex), kMaxViewCount)];
    float3 normal = safeNormalize(float3(in.normal), float3(0, 0, 1));
    float3 viewDirection = safeNormalize(float3(in.viewDirection), float3(0, 0, 1));
    float3 reflectionDirection = reflect(-viewDirection, normal);

    float3 diffuseSH = in.diffuseSH;
    float3 specularSH = in.specularSH;

    uint shMask = uniforms.useSHMask;
    if ((shMask & SphericalHarmonicsUsageDiffuse) == 0) {
        diffuseSH = float3(0);
    }
    if ((shMask & SphericalHarmonicsUsageSpecular) == 0) {
        specularSH = float3(0);
    }

    half ao = computeAmbientOcclusion(in.color.a);
    half3 shaded = shadeGaussian(in.albedo,
                                 in.metallic,
                                 in.roughness,
                                 normal,
                                 viewDirection,
                                 reflectionDirection,
                                 ao,
                                 diffuseSH,
                                 specularSH,
                                 environmentMap,
                                 brdfLUT,
                                 environmentSampler,
                                 brdfSampler);

    return half4(shaded * alpha, alpha);
}
