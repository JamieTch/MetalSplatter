#import "ShaderCommon.h"

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize);

void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2);

float3 safeNormalize(float3 value, float3 fallback);

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       uint splatIndex);

half splatFragmentAlpha(half2 relativePosition, half splatAlpha);

half computeAmbientOcclusion(half opacity);

float3 evaluateSplatSHForDiffuse(SplatSHCoefficients coefficients,
                                 ushort coefficientCount,
                                 float3 normal);
float3 evaluateSplatSHForSpecular(SplatSHCoefficients coefficients,
                                  ushort coefficientCount,
                                  float3 reflectionDirection);

half3 shadeGaussian(half3 albedo,
                    half metallic,
                    half roughness,
                    float3 normal,
                    float3 viewDirection,
                    float3 reflectionDirection,
                    half ambientOcclusion,
                    float3 diffuseSHIrradiance,
                    float3 specularSHRadiance,
                    texturecube<half> environmentMap,
                    texture2d<half> brdfLUT,
                    sampler environmentSampler,
                    sampler brdfSampler);
