#include <metal_stdlib>
using namespace metal;

// Temporary diagnostic: set to 1 to force a solid color write into specular mip 0 to verify write path
#ifndef PREFILTER_DIAG
#define PREFILTER_DIAG 0
#endif

struct PrefilterUniforms {
    uint mipLevel;
    uint dimension;
    float roughness;
    uint sampleCount;
};

struct DiffuseUniforms {
    uint mipLevel;
    uint dimension;
    uint sampleCount;
};

struct BRDFUniforms {
    uint dimension;
    uint sampleCount;
};

static inline float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f; // / 0x100000000
}

static inline float2 hammersley(uint i, uint N) {
    return float2(float(i) / float(N), radicalInverse_VdC(i));
}

static inline float3 hemisphereDirection(uint faceIndex, float2 uv) {
    float2 xy = 2.0f * uv - 1.0f;
    switch (faceIndex) {
        case 0: return normalize(float3(1.0f, -xy.y, -xy.x));
        case 1: return normalize(float3(-1.0f, -xy.y, xy.x));
        case 2: return normalize(float3(xy.x, 1.0f, xy.y));
        case 3: return normalize(float3(xy.x, -1.0f, -xy.y));
        case 4: return normalize(float3(xy.x, -xy.y, 1.0f));
        default: return normalize(float3(-xy.x, -xy.y, -1.0f));
    }
}

static inline float3 importanceSampleGGX(float2 Xi, float roughness, float3 N) {
    float a = roughness * roughness;
    float phi = 2.0f * M_PI_F * Xi.x;
    float cosTheta = sqrt(max(0.0f, (1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y)));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    float3 up = fabs(N.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangentX = normalize(cross(up, N));
    float3 tangentY = cross(N, tangentX);

    return normalize(tangentX * H.x + tangentY * H.y + N * H.z);
}

static inline float distributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(1e-4f, M_PI_F * denom * denom);
}

static inline float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / max(1e-4f, NdotV * (1.0f - k) + k);
}

static inline float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx1 = geometrySchlickGGX(NdotV, roughness);
    float ggx2 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

static inline float3 sampleEnvironment(texturecube<float, access::sample> source, sampler cubeSampler, float3 dir) {
    return source.sample(cubeSampler, dir).xyz;
}

kernel void prefilterEnvironmentSpecular(texturecube<float, access::sample> source [[texture(0)]],
                                         texturecube<half, access::write> destination [[texture(1)]],
                                         constant PrefilterUniforms &uniforms [[buffer(0)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.dimension || gid.y >= uniforms.dimension || gid.z >= 6) {
        return;
    }

#if PREFILTER_DIAG == 1
    if (uniforms.mipLevel == 0) {
        // Solid red only for mip 0
        destination.write(half4(1.0h, 0.0h, 0.0h, 1.0h), uint2(gid.xy), gid.z, uniforms.mipLevel);
        return;
    }
#elif PREFILTER_DIAG == 2
    {
        // Distinct pattern for ALL mips and faces so we can verify full coverage.
        // R encodes mip (brighter for lower mips), G encodes face index, B encodes u coordinate.
        float2 uv_dbg = (float2(gid.xy) + 0.5f) / float(max(1u, uniforms.dimension));
        half r = half(1.0f / (1.0f + float(uniforms.mipLevel)));
        half g = half(float(gid.z) / 5.0f);
        half b = half(uv_dbg.x);
        destination.write(half4(r, g, b, 1.0h), uint2(gid.xy), gid.z, uniforms.mipLevel);
        return;
    }
#endif

    constexpr sampler cubeSampler(filter::linear, address::clamp_to_edge);

    float2 uv = (float2(gid.xy) + 0.5f) / float(uniforms.dimension);
    float3 N = hemisphereDirection(gid.z, uv);
    float3 V = N;
    float roughness = max(uniforms.roughness, 0.045f);

    uint sampleCount = max(uniforms.sampleCount, 1u);
    float3 prefiltered = float3(0.0f);
    float totalWeight = 0.0f;

    for (uint i = 0; i < sampleCount; ++i) {
        float2 Xi = hammersley(i, sampleCount);
        float3 H = importanceSampleGGX(Xi, roughness, N);
        float3 L = normalize(2.0f * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0f);
        if (NdotL > 0.0f) {
            prefiltered += sampleEnvironment(source, cubeSampler, L) * NdotL;
            totalWeight += NdotL;
        }
    }

    if (totalWeight > 0.0f) {
        prefiltered /= totalWeight;
    }

    destination.write(half4(half3(prefiltered), half(1.0f)), uint2(gid.xy), gid.z, uniforms.mipLevel);
}

kernel void prefilterEnvironmentDiffuse(texturecube<float, access::sample> source [[texture(0)]],
                                        texturecube<half, access::write> destination [[texture(1)]],
                                        constant DiffuseUniforms &uniforms [[buffer(0)]],
                                        uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.dimension || gid.y >= uniforms.dimension || gid.z >= 6) {
        return;
    }

    constexpr sampler cubeSampler(filter::linear, address::clamp_to_edge);

    float2 uv = (float2(gid.xy) + 0.5f) / float(uniforms.dimension);
    float3 N = hemisphereDirection(gid.z, uv);

    uint sampleCount = max(uniforms.sampleCount, 1u);
    float3 diffuse = float3(0.0f);
    float totalWeight = 0.0f;

    for (uint i = 0; i < sampleCount; ++i) {
        float2 Xi = hammersley(i, sampleCount);
        float phi = 2.0f * M_PI_F * Xi.x;
        float cosTheta = Xi.y;
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

        float3 H = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
        float3 up = fabs(N.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
        float3 tangentX = normalize(cross(up, N));
        float3 tangentY = cross(N, tangentX);
        float3 L = normalize(tangentX * H.x + tangentY * H.y + N * H.z);

        float NdotL = max(dot(N, L), 0.0f);
        if (NdotL > 0.0f) {
            diffuse += sampleEnvironment(source, cubeSampler, L) * NdotL;
            totalWeight += NdotL;
        }
    }

    if (totalWeight > 0.0f) {
        diffuse /= totalWeight;
    }

    destination.write(half4(half3(diffuse), half(1.0f)), uint2(gid.xy), gid.z, uniforms.mipLevel);
}

static inline float2 integrateBRDF(float NdotV, float roughness, uint sampleCount) {
    float3 V = float3(sqrt(max(0.0f, 1.0f - NdotV * NdotV)), 0.0f, NdotV);
    float A = 0.0f;
    float B = 0.0f;
    float3 N = float3(0.0f, 0.0f, 1.0f);

    for (uint i = 0; i < sampleCount; ++i) {
        float2 Xi = hammersley(i, sampleCount);
        float3 H = importanceSampleGGX(Xi, roughness, N);
        float3 L = normalize(2.0f * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0f);
        float NdotH = max(H.z, 0.0f);
        float VdotH = max(dot(V, H), 0.0f);

        if (NdotL > 0.0f) {
            float G = geometrySmith(N, V, L, roughness);
            float GVis = (G * VdotH) / max(1e-4f, NdotH * NdotV);
            float Fc = pow(1.0f - VdotH, 5.0f);
            A += (1.0f - Fc) * GVis;
            B += Fc * GVis;
        }
    }

    return float2(A, B) / float(sampleCount);
}

kernel void integrateBRDFLUT(texture2d<half, access::write> lut [[texture(0)]],
                              constant BRDFUniforms &uniforms [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.dimension || gid.y >= uniforms.dimension) {
        return;
    }

    uint dimension = uniforms.dimension;
    float NdotV = (float(gid.x) + 0.5f) / float(dimension);
    float roughness = (float(gid.y) + 0.5f) / float(dimension);

    uint sampleCount = max(uniforms.sampleCount, 1u);
    float2 brdf = integrateBRDF(NdotV, roughness, sampleCount);

    lut.write(half4(half(brdf.x), half(brdf.y), half(0.0f), half(1.0f)), gid);
}
