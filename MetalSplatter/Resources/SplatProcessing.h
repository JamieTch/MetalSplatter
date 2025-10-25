#import "ShaderCommon.h"

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize);

void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2);

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex);

half splatFragmentAlpha(half2 relativePosition, half splatAlpha);

half computeAmbientOcclusion(half opacity);

half3 shadeGaussian(half3 albedo,
                    half metallic,
                    half roughness,
                    half3 normal,
                    half3 viewDirection,
                    half ambientOcclusion,
                    texturecube<half> environmentMap,
                    texture2d<half> brdfLUT,
                    sampler environmentSampler,
                    sampler brdfSampler);
