#include <metal_stdlib>
using namespace metal;

struct CalibrationVertexIn {
    float3 position [[attribute(0)]];
    float3 color [[attribute(1)]];
};

struct CalibrationUniforms {
    float4x4 modelViewProjection;
};

struct CalibrationVertexOut {
    float4 position [[position]];
    float3 color;
};

vertex CalibrationVertexOut calibrationAnchorVertex(CalibrationVertexIn in [[stage_in]],
                                                    constant CalibrationUniforms &uniforms [[buffer(1)]]) {
    CalibrationVertexOut out;
    out.position = uniforms.modelViewProjection * float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

fragment float4 calibrationAnchorFragment(CalibrationVertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
