#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant const int kMaxViewCount = 2;
constant static const half kBoundsRadius = 3;
constant static const half kBoundsRadiusSquared = kBoundsRadius*kBoundsRadius;

enum BufferIndex: int32_t
{
    BufferIndexUniforms = 0,
    BufferIndexSplat    = 1,
    BufferIndexSphericalHarmonics = 2,
};

enum TextureIndex: int32_t
{
    TextureIndexEnvironment = 0,
    TextureIndexBRDF        = 1,
};

enum SamplerIndex: int32_t
{
    SamplerIndexEnvironment = 0,
    SamplerIndexBRDF        = 1,
};

enum SphericalHarmonicsUsageMask : uint
{
    SphericalHarmonicsUsageDiffuse  = 1u << 0,
    SphericalHarmonicsUsageSpecular = 1u << 1,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    uint2 screenSize;
    uint2 _paddingScreen;
    float4 cameraPosition;

    /*
     The first N splats are represented as as 2N primitives and 4N vertex indices. The remained are represented
     as instanced of these first N. This allows us to limit the size of the indexed array (and associated memory),
     but also avoid the performance penalty of a very large number of instances.
     */
    uint splatCount;
    uint indexedSplatCount;
    uint shCoefficientCount;
    uint useSHMask;
} Uniforms;

typedef struct
{
    Uniforms uniforms[kMaxViewCount];
} UniformsArray;

typedef struct
{
    packed_float3 position;
    packed_half4 color;
    packed_half3 covA;
    packed_half3 covB;
    packed_half3 albedo;
    half          metallic;
    half          roughness;
    packed_half3 normal;
    packed_half4 rotation;
} Splat;

typedef struct
{
    ushort count;
    ushort padding;
    packed_half3 coefficient0;
    packed_half3 coefficient1;
    packed_half3 coefficient2;
    packed_half3 coefficient3;
    packed_half3 coefficient4;
    packed_half3 coefficient5;
    packed_half3 coefficient6;
    packed_half3 coefficient7;
    packed_half3 coefficient8;
    packed_half3 coefficient9;
    packed_half3 coefficient10;
    packed_half3 coefficient11;
    packed_half3 coefficient12;
    packed_half3 coefficient13;
    packed_half3 coefficient14;
    packed_half3 coefficient15;
} SplatSHCoefficients;

typedef struct
{
    float4 position [[position]];
    half2 relativePosition; // Ranges from -kBoundsRadius to +kBoundsRadius
    half4 color;
    half3 albedo;
    half metallic;
    half roughness;
    half3 normal;
    float3 worldPosition;
    float3 viewDirection;
    uint  splatIndex;
} FragmentIn;
