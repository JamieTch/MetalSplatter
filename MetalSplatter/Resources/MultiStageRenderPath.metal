#include "SplatProcessing.h"
#include <metal_math>

// Debug visualization selector:
// 0: coverage (alpha), 1: albedo, 2: normal, 3: roughness, 4: metallic, 5: AO, 6: depth, 7: shaded (default)
// Debug view selection driven by UI (function constant).
// 28: diffuse SH only, 29: specular SH only
constant uint DEBUG_VIEW [[function_constant(0)]];
constant bool _DEBUG_VIEW_IS_SET = is_function_constant_defined(DEBUG_VIEW);
constant uint DEBUG_VIEW_VALUE = _DEBUG_VIEW_IS_SET ? DEBUG_VIEW : 1; // default fallback

// Optional alias so existing code can still read DEBUG_VIEW:
#define DEBUG_VIEW DEBUG_VIEW_VALUE

// --- Legacy macro-based path (disabled) ---
// #ifndef DEBUG_VIEW
// #define DEBUG_VIEW 1
// #endif

// ---- Normal decoding controls (tweak and rebuild) ----
#ifndef NORMAL_STORAGE_01
#define NORMAL_STORAGE_01 0   // 1: incoming normal stored in [0,1] -> remap to [-1,1]; 0: already in [-1,1]
#endif
#ifndef NORMAL_FLIP_X
#define NORMAL_FLIP_X 0
#endif
#ifndef NORMAL_FLIP_Y
#define NORMAL_FLIP_Y 0
#endif
#ifndef NORMAL_FLIP_Z
#define NORMAL_FLIP_Z 0
#endif

typedef struct
{
    half4 albedoMetallic [[raster_order_group(0)]];
    half4 normalRoughness [[raster_order_group(0)]];
    half4 viewAlpha [[raster_order_group(0)]];
    half2 ambientOcclusion [[raster_order_group(0)]];
    half4 diffuseIrradiance [[raster_order_group(0)]];
    half4 specularRadiance [[raster_order_group(0)]];
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
    values->diffuseIrradiance = half4(0);
    values->specularRadiance = half4(0);
    values->depth = 0;
}

vertex FragmentIn multiStageSplatVertexShader(uint vertexID [[vertex_id]],
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

fragment FragmentStore multiStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                                     ushort viewIndex [[render_target_array_index]],
                                                     FragmentValues previousFragmentValues [[imageblock_data]],
                                                     constant SplatSHCoefficients* splatSHArray [[ buffer(BufferIndexSphericalHarmonics) ]],
                                                     constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                                     constant SphericalHarmonicsDebugUniforms & shDebug [[ buffer(BufferIndexSphericalHarmonicsDebug) ]]) {
    FragmentStore out;

    (void)splatSHArray;

    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a); // restored: use real coverage
    if (alpha <= 0) {
        out.values = previousFragmentValues;
        return out;
    }

    half oneMinusAlpha = 1 - alpha;
    half ao = computeAmbientOcclusion(in.color.a);

    Uniforms uniforms = uniformsArray.uniforms[min(int(viewIndex), kMaxViewCount)];
    float3 normal = safeNormalize(float3(in.normal), float3(0, 0, 1));
    float3 viewDirection = safeNormalize(float3(in.viewDirection), float3(0, 0, 1));
    float3 reflectionDirection = reflect(-viewDirection, normal);

    float3 diffuseSH = in.diffuseSH;
    float3 specularSH = in.specularSH;
    float diffuseMagnitude = length(diffuseSH);
    float specularMagnitude = length(specularSH);

    uint shMask = uniforms.useSHMask;
    bool maskDebugRequested = shDebug.enableMaskDebug != 0 && (DEBUG_VIEW_VALUE == 28 || DEBUG_VIEW_VALUE == 29);
    if (maskDebugRequested) {
        float packedMask = clamp(float(shMask & 0x3u) / 3.0f, 0.0f, 1.0f);
        float3 debugVector = float3(diffuseMagnitude, specularMagnitude, packedMask);
        diffuseSH = debugVector;
        specularSH = debugVector;
    } else {
        if ((shMask & SphericalHarmonicsUsageDiffuse) == 0) {
            diffuseSH = float3(0);
        }
        if ((shMask & SphericalHarmonicsUsageSpecular) == 0) {
            specularSH = float3(0);
        }
    }

    half4 albedoMetallic = half4(in.albedo * alpha, in.metallic * alpha);
    half4 normalRoughness = half4(half3(normal) * alpha, in.roughness * alpha);
    half4 viewAlpha = half4(half3(viewDirection) * alpha, alpha);
    half2 ambientOcclusion = half2(ao * alpha, 0);
    half4 diffuseIrradiance = half4(half3(diffuseSH) * alpha, half(0));
    half4 specularRadiance = half4(half3(specularSH) * alpha, half(0));

    out.values.albedoMetallic = previousFragmentValues.albedoMetallic * oneMinusAlpha + albedoMetallic;
    out.values.normalRoughness = previousFragmentValues.normalRoughness * oneMinusAlpha + normalRoughness;
    out.values.viewAlpha = previousFragmentValues.viewAlpha * oneMinusAlpha + viewAlpha;
    out.values.ambientOcclusion = previousFragmentValues.ambientOcclusion * oneMinusAlpha + ambientOcclusion;
    out.values.diffuseIrradiance = previousFragmentValues.diffuseIrradiance * oneMinusAlpha + diffuseIrradiance;
    out.values.specularRadiance = previousFragmentValues.specularRadiance * oneMinusAlpha + specularRadiance;

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
    out.splatIndex = 0;
    out.diffuseSH = float3(0);
    out.specularSH = float3(0);
    return out;
}

// sRGB to Linear helper (for debug/diagnostic use)
inline half3 srgbToLinear(half3 c) {
    half3 lo = c / 12.92h;
    half3 hi = pow(max(half3(0), (c + 0.055h) / 1.055h), half3(2.4h));
    return mix(lo, hi, step(0.04045h, c));
}

inline half3 remap01ToN11(half3 v) { return v * 2.0h - 1.0h; }

inline half3 applyNormalFlips(half3 n) {
#if NORMAL_FLIP_X
    n.x = -n.x;
#endif
#if NORMAL_FLIP_Y
    n.y = -n.y;
#endif
#if NORMAL_FLIP_Z
    n.z = -n.z;
#endif
    return n;
}

inline half3 decodeNormal(half3 src) {
#if NORMAL_STORAGE_01
    half3 n = remap01ToN11(src);
#else
    half3 n = src;
#endif
    n = applyNormalFlips(n);
    return normalize(n);
}

inline half4 resolveFragmentValues(FragmentValues fragmentValues,
                                   texturecube<half> environmentMap [[texture(0)]],
                                   texture2d<half> brdfLUT [[texture(1)]],
                                   sampler environmentSampler [[sampler(0)]],
                                   sampler brdfSampler [[sampler(1)]]) {
    half accumulatedAlpha = fragmentValues.viewAlpha.w;
    if (accumulatedAlpha <= 0) {
        return half4(0);
    }

    half invAlpha = half(1.0) / accumulatedAlpha;
    float invAlphaF = float(invAlpha);
    half3 albedo     = half3(fragmentValues.albedoMetallic.xyz * invAlpha);
    half  metallic   =        fragmentValues.albedoMetallic.w   * invAlpha;
    half3 normal     = half3(fragmentValues.normalRoughness.xyz * invAlpha);
    half  roughness  =        fragmentValues.normalRoughness.w   * invAlpha;
    half3 viewDir    = half3(fragmentValues.viewAlpha.xyz        * invAlpha);
    half  ao         =        fragmentValues.ambientOcclusion.x  * invAlpha;
    float3 diffuseSH = float3(fragmentValues.diffuseIrradiance.xyz) * invAlphaF;
    float3 specularSH = float3(fragmentValues.specularRadiance.xyz) * invAlphaF;

    // Clamp and normalize core material inputs for stability during debugging
    albedo    = clamp(albedo,   0.0h, 1.0h);
    metallic  = clamp(metallic, 0.0h, 1.0h);
    roughness = clamp(roughness,0.0h, 1.0h);

    // Compute raw-as-is and decoded normals
    half3 normal_raw = normalize(normal);
    half3 normal_dec = decodeNormal(normal);

// assumes you already defined:
// constant uint DEBUG_VIEW [[function_constant(0)]];
// constant bool _DEBUG_VIEW_IS_SET = is_function_constant_defined(DEBUG_VIEW);
// constant uint DEBUG_VIEW_VALUE = _DEBUG_VIEW_IS_SET ? DEBUG_VIEW : 1;

float3 shadingNormal     = safeNormalize(float3(normal_dec), float3(0, 0, 1));
float3 shadingView       = safeNormalize(float3(viewDir),    float3(0, 0, 1));
float3 shadingReflection = reflect(-shadingView, shadingNormal);

// Debug outputs (early returns). If none matches, continue with regular shading below.
if (DEBUG_VIEW_VALUE == 1) {
    // Albedo
    return half4(albedo, 1);
} else if (DEBUG_VIEW_VALUE == 2) {
    // Normal (visualized as 0..1)
    return half4(normalize(normal) * 0.5h + 0.5h, 1);
} else if (DEBUG_VIEW_VALUE == 3) {
    // Roughness
    return half4(roughness, roughness, roughness, 1);
} else if (DEBUG_VIEW_VALUE == 4) {
    // Metallic
    return half4(metallic, metallic, metallic, 1);
} else if (DEBUG_VIEW_VALUE == 5) {
    // Ambient occlusion
    return half4(ao, ao, ao, 1);
} else if (DEBUG_VIEW_VALUE == 21) {
    // Albedo after sRGB->linear (diagnostic)
    return half4(srgbToLinear(albedo), 1);
} else if (DEBUG_VIEW_VALUE == 22) {
    // Roughness (duplicate of 3 for sweep)
    return half4(roughness, roughness, roughness, 1);
} else if (DEBUG_VIEW_VALUE == 23) {
    // Metallic (duplicate of 4 for sweep)
    return half4(metallic, metallic, metallic, 1);
} else if (DEBUG_VIEW_VALUE == 24) {
    // Normal (duplicate of 2 for sweep)
    return half4(normal_dec * 0.5h + 0.5h, 1);
} else if (DEBUG_VIEW_VALUE == 25) {
    // Raw (pre-decode) normal visualization
    return half4(normal_raw * 0.5h + 0.5h, 1);
} else if (DEBUG_VIEW_VALUE == 26) {
    // Difference heatmap between decoded and raw normals
    half3 diff = abs(normal_dec - normal_raw);
    return half4(diff, 1);
} else if (DEBUG_VIEW_VALUE == 27) {
    // N·V comparison: R=using decoded normal, G=using raw normal
    half3 V = normalize(viewDir);
    half ndv_dec = saturate(dot(normal_dec, V));
    half ndv_raw = saturate(dot(normal_raw, V));
    return half4(ndv_dec, ndv_raw, 0.0h, 1);
} else if (DEBUG_VIEW_VALUE == 8) {
    // Shaded result with AO forced to 1 (tests if AO is zeroing energy)
    half3 shaded = shadeGaussian(albedo,
                                 metallic,
                                 roughness,
                                 shadingNormal,
                                 shadingView,
                                 shadingReflection,
                                 half(1.0),   // force AO = 1
                                 diffuseSH,
                                 specularSH,
                                 environmentMap,
                                 brdfLUT,
                                 environmentSampler,
                                 brdfSampler);
    return half4(shaded * accumulatedAlpha, accumulatedAlpha);
} else if (DEBUG_VIEW_VALUE == 9) {
    // Direct environment reflection sample (tests env binding/indices)
    half3 N = normalize(normal);
    half3 V = normalize(viewDir);
    half3 R = reflect(-V, N);
    half3 env = environmentMap.sample(environmentSampler, float3(R)).rgb;
    return half4(env, 1);
} else if (DEBUG_VIEW_VALUE == 10) {
    // N·V visualization (tests geometry/normal-view relationship)
    half3 N = normalize(normal);
    half3 V = normalize(viewDir);
    half ndv = saturate(dot(N, V));
    return half4(ndv, ndv, ndv, 1);
} else if (DEBUG_VIEW_VALUE == 12) {
    // Simple Lambert with a fixed key light (no env/BRDF) to prove shading path works
    half3 N = normalize(normal);
    half3 L = normalize(half3(0.4h, 0.8h, 0.4h));
    half ndl = saturate(dot(N, L));
    half3 lit = albedo * ndl;
    return half4(lit * accumulatedAlpha, accumulatedAlpha);
} else if (DEBUG_VIEW_VALUE == 13) {
    // BRDF LUT debug: show LUT sample at center to validate BRDF binding
    half2 l = brdfLUT.sample(brdfSampler, float2(0.5, 0.5)).rg;
    return half4(l.x, l.y, 0, 1);
} else if (DEBUG_VIEW_VALUE == 7) {
    // Shaded result (uses environment)
    half3 shaded = shadeGaussian(albedo,
                                 metallic,
                                 roughness,
                                 shadingNormal,
                                 shadingView,
                                 shadingReflection,
                                 ao,
                                 diffuseSH,
                                 specularSH,
                                 environmentMap,
                                 brdfLUT,
                                 environmentSampler,
                                 brdfSampler);
    return half4(shaded * accumulatedAlpha, accumulatedAlpha);
} else {
    // Coverage and depth handled in postprocess
    return half4(0);
}
// UI-driven debug modes (function constant)
// 28: diffuse SH only, 29: specular SH only, 7: full shaded, else: coverage/depth handled in post

if (DEBUG_VIEW_VALUE == 28) {
    // Diffuse spherical harmonics contribution only (accumulated)
    half3 sh = half3(diffuseSH) * accumulatedAlpha;
    return half4(sh, accumulatedAlpha);
} else if (DEBUG_VIEW_VALUE == 29) {
    // Specular spherical harmonics contribution only (accumulated)
    half3 sh = half3(specularSH) * accumulatedAlpha;
    return half4(sh, accumulatedAlpha);
} else if (DEBUG_VIEW_VALUE == 7) {
    // Shaded result (uses environment)
    half3 shaded = shadeGaussian(
        albedo,
        metallic,
        roughness,
        shadingNormal,
        shadingView,
        shadingReflection,
        ao,
        diffuseSH,
        specularSH,
        environmentMap,
        brdfLUT,
        environmentSampler,
        brdfSampler
    );
    return half4(shaded * accumulatedAlpha, accumulatedAlpha);
} else {
    // Coverage and depth handled in postprocess
    return half4(0);
}
}

fragment FragmentOut postprocessFragmentShader(FragmentValues fragmentValues [[imageblock_data]],
                                               texturecube<half> environmentMap [[texture(0)]],
                                               texture2d<half> brdfLUT [[texture(1)]],
                                               sampler environmentSampler [[sampler(0)]],
                                               sampler brdfSampler [[sampler(1)]]) {
    {
        FragmentOut out;
        half accumulatedAlpha = fragmentValues.viewAlpha.w;
        out.depth = (accumulatedAlpha == 0) ? 0 : fragmentValues.depth / accumulatedAlpha;

        if (DEBUG_VIEW_VALUE == 0) {
            // Coverage view (grayscale alpha)
            half a = clamp(accumulatedAlpha, 0.0h, 1.0h);
            out.color = half4(a, a, a, 1);
            return out;
        } else if (DEBUG_VIEW_VALUE == 6) {
            // Depth visualization mapped to [0,1]
            half d = half(out.depth);
            out.color = half4(d, d, d, 1);
            return out;
        } else if (DEBUG_VIEW_VALUE == 11) {
            // Direct environment sample (panorama from screen UV -> direction), ignores accumulation
            float2 uv = float2((fragmentValues.viewAlpha.x + 1.0f) * 0.5f, (fragmentValues.viewAlpha.y + 1.0f) * 0.5f);
            // Equirectangular-like mapping to direction (yaw=pitch)
            float phi = (uv.x * 2.0f - 1.0f) * M_PI_F;        // -pi..pi
            float theta = (uv.y * 1.0f - 0.5f) * M_PI_F;      // -pi/2..pi/2
            float3 dir = float3(cos(theta) * sin(phi), sin(theta), cos(theta) * cos(phi));
            half3 env = environmentMap.sample(environmentSampler, dir).rgb;
            out.color = half4(env, 1);
            return out;
        } else if (DEBUG_VIEW_VALUE == 14) {
            // Env sample at fixed direction, forced LOD 0
            float3 dir = float3(0.0, 1.0, 0.0);
            half3 env = environmentMap.sample(environmentSampler, dir, level(0.0)).rgb;
            out.color = half4(env, 1);
            return out;
        } else if (DEBUG_VIEW_VALUE == 15) {
            // Env sample at fixed direction, largest mip (very blurred)
            float3 dir = float3(0.0, 1.0, 0.0);
            float maxLevel = log2((float)environmentMap.get_width());
            half3 env = environmentMap.sample(environmentSampler, dir, level(maxLevel)).rgb;
            out.color = half4(env, 1);
            return out;
        } else {
            // Use resolveFragmentValues for attribute views or shaded output
            out.color = resolveFragmentValues(fragmentValues,
                                              environmentMap,
                                              brdfLUT,
                                              environmentSampler,
                                              brdfSampler);
            return out;
        }
    }
}

fragment half4 postprocessFragmentShaderNoDepth(FragmentValues fragmentValues [[imageblock_data]],
                                               texturecube<half> environmentMap [[texture(0)]],
                                               texture2d<half> brdfLUT [[texture(1)]],
                                               sampler environmentSampler [[sampler(0)]],
                                               sampler brdfSampler [[sampler(1)]]) {
    {
        // No depth resolve path — choose debug view or shaded like the main postprocess
        if (DEBUG_VIEW_VALUE == 0) {
            // Coverage is not available without depth resolve; show AO as a proxy
            half a = fragmentValues.ambientOcclusion.x;
            return half4(a, a, a, 1);
        } else if (DEBUG_VIEW_VALUE == 6) {
            // No depth resolve here; return black
            return half4(0,0,0,1);
        } else {
            return resolveFragmentValues(fragmentValues,
                                         environmentMap,
                                         brdfLUT,
                                         environmentSampler,
                                         brdfSampler);
        }
    }
}
