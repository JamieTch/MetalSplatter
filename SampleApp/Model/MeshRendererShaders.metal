#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3x3 normalMatrix;
    float roughness;
    float metallic;
    float2 pad;
};

struct UniformArray {
    Uniforms u0;
    Uniforms u1;
};

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 texcoord [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 texcoord;
    uint viewportIndex [[viewport_array_index]];
};

vertex VertexOut meshVertex(VertexIn in                 [[stage_in]],
                            constant UniformArray &uArr [[buffer(3)]],
                            uint vID                    [[vertex_id]],
                            uint viewID                 [[view_id]]) {
    VertexOut out;
    const Uniforms u = (viewID == 0) ? uArr.u0 : uArr.u1;

    float4 worldPos = u.modelMatrix * float4(in.position, 1.0);
    float4 viewPos = u.viewMatrix * worldPos;
    out.position = u.projectionMatrix * viewPos;
    out.worldPos = worldPos.xyz;
    out.normal = normalize(u.normalMatrix * in.normal);
    out.texcoord = in.texcoord;
    out.viewportIndex = viewID;
    return out;
}

float3 sampleEnvironment(texturecube<float> environment, sampler s, float3 dir, float roughness) {
    if (!environment.get_type()) { return float3(0.0); }
    constexpr float lodBias = 0.0;
    float lod = roughness * (environment.get_num_mip_levels() - 1);
    return environment.sample(s, dir, level(lod + lodBias)).rgb;
}

fragment float4 meshFragment(VertexOut in [[stage_in]],
                             texture2d<float> baseColor [[texture(0)]],
                             texturecube<float> environment [[texture(1)]],
                             texture2d<float> brdfLUT [[texture(2)]],
                             sampler textureSampler [[sampler(0)]]) {
    float3 albedo = float3(0.8);
    if (baseColor.get_type()) {
        albedo = baseColor.sample(textureSampler, in.texcoord).rgb;
    }

    float3 N = normalize(in.normal);
    float3 V = normalize(-in.worldPos);
    float NdotV = max(dot(N, V), 0.001);

    float3 diffuse = albedo * 0.5;
    float3 specular = float3(0.0);

    if (environment.get_type()) {
        float3 R = reflect(-V, N);
        float3 envSample = sampleEnvironment(environment, textureSampler, R, 0.35);
        float2 brdf = brdfLUT.get_type() ? brdfLUT.sample(textureSampler, float2(NdotV, 0.35)).rg : float2(1.0, 0.0);
        specular = envSample * (brdf.x + brdf.y);
    }

    float3 color = diffuse + specular;
    return float4(color, 1.0);
}
