#ifndef NORMAL_ROTATE_BY_QUAT
#define NORMAL_ROTATE_BY_QUAT 1 // set to 0 to bypass per-splat quaternion rotation for normals
#endif

#import "SplatProcessing.h"

float4 normalizeQuaternion(float4 quaternion) {
    float lengthSquared = dot(quaternion, quaternion);
    if (!isfinite(lengthSquared) || lengthSquared <= 0) {
        return float4(0, 0, 0, 1);
    }
    float length = sqrt(lengthSquared);
    return quaternion / length;
}

float3 rotateVectorByQuaternion(float4 quaternion, float3 vector) {
    float4 normalizedQuaternion = normalizeQuaternion(quaternion);
    float3 qVec = normalizedQuaternion.xyz;
    float qW = normalizedQuaternion.w;
    float3 t = 2.0 * cross(qVec, vector);
    return vector + qW * t + cross(qVec, t);
}

float3 safeNormalize(float3 value, float3 fallback) {
    float lengthSquared = dot(value, value);
    if (!isfinite(lengthSquared) || lengthSquared <= 0) {
        return fallback;
    }
    return normalize(value);
}

half computeAmbientOcclusion(half opacity) {
    half clampedOpacity = clamp(opacity, half(0), half(1));
    return half(1) - half(fast::exp(-float(clampedOpacity)));
}

float3 sampleDiffuseIrradiance(texturecube<half> environmentMap,
                               sampler environmentSampler,
                               float3 normal) {
    uint mipCount = environmentMap.get_num_mip_levels();
    uint diffuseMip = mipCount == 0 ? 0 : mipCount - 1;
    return float3(environmentMap.sample(environmentSampler, normal, level(float(diffuseMip))).rgb);
}

float3 evaluateSplatSHCoefficients(SplatSHCoefficients coefficients,
                                   ushort coefficientCount,
                                   float3 direction) {
    ushort count = min(coefficients.count, coefficientCount);
    if (count == 0) {
        return float3(0);
    }

    const float shC0 = 0.28209479177387814f;
    const float shC1 = 0.4886025119029199f;
    const float shC2_0 = 1.0925484305920792f;
    const float shC2_1 = 0.31539156525252005f;
    const float shC2_2 = 0.5462742152960396f;
    const float shC3_0 = 0.5900435899266435f;
    const float shC3_1 = 2.890611442640554f;
    const float shC3_2 = 0.4570457994644658f;
    const float shC3_3 = 0.3731763325901154f;
    const float shC3_4 = 1.445305721320277f;

    float x = direction.x;
    float y = direction.y;
    float z = direction.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    float3 result = float3(coefficients.coefficient0) * shC0;
    if (count == 1) {
        return result;
    }

    result += float3(coefficients.coefficient1) * (-shC1 * y);
    if (count == 2) {
        return result;
    }

    result += float3(coefficients.coefficient2) * (shC1 * z);
    if (count == 3) {
        return result;
    }

    result += float3(coefficients.coefficient3) * (-shC1 * x);
    if (count == 4) {
        return result;
    }

    result += float3(coefficients.coefficient4) * (shC2_0 * xy);
    if (count == 5) {
        return result;
    }

    result += float3(coefficients.coefficient5) * (-shC2_0 * yz);
    if (count == 6) {
        return result;
    }

    result += float3(coefficients.coefficient6) * (shC2_1 * (3.0f * zz - 1.0f));
    if (count == 7) {
        return result;
    }

    result += float3(coefficients.coefficient7) * (-shC2_0 * xz);
    if (count == 8) {
        return result;
    }

    result += float3(coefficients.coefficient8) * (shC2_2 * (xx - yy));
    if (count == 9) {
        return result;
    }

    result += float3(coefficients.coefficient9) * (-shC3_0 * y * (3.0f * xx - yy));
    if (count == 10) {
        return result;
    }

    result += float3(coefficients.coefficient10) * (shC3_1 * xy * z);
    if (count == 11) {
        return result;
    }

    result += float3(coefficients.coefficient11) * (-shC3_2 * y * (5.0f * zz - 1.0f));
    if (count == 12) {
        return result;
    }

    result += float3(coefficients.coefficient12) * (shC3_3 * (5.0f * zz * z - 3.0f * z));
    if (count == 13) {
        return result;
    }

    result += float3(coefficients.coefficient13) * (-shC3_2 * x * (5.0f * zz - 1.0f));
    if (count == 14) {
        return result;
    }

    result += float3(coefficients.coefficient14) * (shC3_4 * z * (xx - yy));
    if (count == 15) {
        return result;
    }

    result += float3(coefficients.coefficient15) * (-shC3_0 * x * (xx - 3.0f * yy));
    return result;
}

float3 evaluateSplatSHForDiffuse(SplatSHCoefficients coefficients,
                                 ushort coefficientCount,
                                 float3 normal) {
    return (coefficients.count == 0 || coefficientCount == 0)
        ? float3(0)
        : evaluateSplatSHCoefficients(coefficients, coefficientCount, normal);
}

float3 evaluateSplatSHForSpecular(SplatSHCoefficients coefficients,
                                  ushort coefficientCount,
                                  float3 reflectionDirection) {
    return (coefficients.count == 0 || coefficientCount == 0)
        ? float3(0)
        : evaluateSplatSHCoefficients(coefficients, coefficientCount, reflectionDirection);
}

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
                    sampler brdfSampler) {
    float3 N = safeNormalize(normal, float3(0, 0, 1));
    float3 V = safeNormalize(viewDirection, float3(0, 0, 1));
    float3 R = safeNormalize(reflectionDirection, float3(0, 0, 1));
    float perceptualRoughness = clamp(float(roughness), 0.045, 1.0);
    uint mipCount = environmentMap.get_num_mip_levels();
    float lod = perceptualRoughness * float(max(int(mipCount) - 1, 0));
    float3 prefilteredColor = float3(environmentMap.sample(environmentSampler, R, level(lod)).rgb);
    float3 irradiance = sampleDiffuseIrradiance(environmentMap, environmentSampler, N);

    float3 combinedDiffuseIrradiance = irradiance + diffuseSHIrradiance;
    float3 combinedSpecularRadiance = prefilteredColor + specularSHRadiance;

    float NdotV = clamp(dot(N, V), 1e-4, 1.0);
    float3 baseAlbedo = float3(albedo);
    float metallicValue = clamp(float(metallic), 0.0, 1.0);
    float3 F0 = mix(float3(0.04), baseAlbedo, metallicValue);
    float Fc = pow(1.0 - NdotV, 5.0);
    float3 F = F0 + (1.0 - F0) * Fc;

    float3 kS = F;
    float3 kD = (float3(1.0) - kS) * (1.0 - metallicValue);

    float2 brdfSample = float2(brdfLUT.sample(brdfSampler, float2(NdotV, perceptualRoughness)).rg);
    float3 specular = combinedSpecularRadiance * (F0 * brdfSample.x + brdfSample.y);

    float3 diffuse = combinedDiffuseIrradiance * baseAlbedo * kD;
    float3 color = (diffuse + specular) * float(clamp(ambientOcclusion, half(0), half(1)));
    return half3(color);
}

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize) {
    float invViewPosZ = 1 / viewPos.z;
    float invViewPosZSquared = invViewPosZ * invViewPosZ;

    float tanHalfFovX = 1 / projectionMatrix[0][0];
    float tanHalfFovY = 1 / projectionMatrix[1][1];
    float limX = 1.3 * tanHalfFovX;
    float limY = 1.3 * tanHalfFovY;
    viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;

    float focalX = screenSize.x * projectionMatrix[0][0] / 2;
    float focalY = screenSize.y * projectionMatrix[1][1] / 2;

    float3x3 J = float3x3(
        focalX * invViewPosZ, 0, 0,
        0, focalY * invViewPosZ, 0,
        -(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0
    );
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 Vrk = float3x3(
        cov3Da.x, cov3Da.y, cov3Da.z,
        cov3Da.y, cov3Db.x, cov3Db.y,
        cov3Da.z, cov3Db.y, cov3Db.z
    );
    float3x3 cov = T * Vrk * transpose(T);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

// cov2D is a flattened 2d covariance matrix. Given
// covariance = | a b |
//              | c d |
// (where b == c because the Gaussian covariance matrix is symmetric),
// cov2D = ( a, b, d )
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b; // matrix is symmetric, so "c" is same as "b"
    float trace = a + d;

    float mean = 0.5 * trace;
    float dist = max(0.1, sqrt(mean * mean - det)); // based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu

    // Eigenvalues
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;

    float2 eigenvector1;
    if (b == 0) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        eigenvector1 = normalize(float2(b, d - lambda2));
    }

    // Gaussian axes are orthogonal
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);

    v1 = eigenvector1 * sqrt(lambda1);
    v2 = eigenvector2 * sqrt(lambda2);
}

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       uint splatIndex) {
    FragmentIn out;

    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position, 1);
    float3 viewPosition3 = viewPosition4.xyz;

    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
                                    uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);

    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;

    float bounds = 1.2 * projectedCenter.w;
    if (projectedCenter.z < 0.0 ||
        projectedCenter.z > projectedCenter.w ||
        projectedCenter.x < -bounds ||
        projectedCenter.x > bounds ||
        projectedCenter.y < -bounds ||
        projectedCenter.y > bounds) {
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.albedo = half3(0);
        out.metallic = half(0);
        out.roughness = half(0);
        out.normal = half3(0);
        out.worldPosition = float3(0);
        out.viewDirection = float3(0);
        out.splatIndex = splatIndex;
        out.diffuseSH = float3(0);
        out.specularSH = float3(0);
        return out;
    }

    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[relativeVertexIndex];
    half2 screenSizeFloat = half2(uniforms.screenSize.x, uniforms.screenSize.y);
    half2 projectedScreenDelta =
        (relativeCoordinates.x * half2(axis1) + relativeCoordinates.y * half2(axis2))
        * 2
        * kBoundsRadius
        / screenSizeFloat;

    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
                          projectedCenter.z,
                          projectedCenter.w);
    out.relativePosition = kBoundsRadius * relativeCoordinates;
    out.color = splat.color;
    out.albedo = half3(splat.albedo);
    out.metallic = splat.metallic;
    out.roughness = splat.roughness;

    float3 worldPosition = float3(splat.position);

    // Base normal straight from the splat (sanitized)
    float3 baseNormal = safeNormalize(float3(splat.normal), float3(0, 0, 1));

    // Rotate normal by the splat's quaternion (default behavior)
    float4 rotation = float4(splat.rotation);
    float3 n = safeNormalize(rotateVectorByQuaternion(rotation, baseNormal), float3(0, 0, 1));

    out.normal = half3(n);
    out.worldPosition = worldPosition;

    float3 cameraPosition = uniforms.cameraPosition.xyz;
    float3 viewDirection = safeNormalize(cameraPosition - worldPosition, float3(0, 0, 1));
    out.viewDirection = viewDirection;
    out.splatIndex = splatIndex;
    out.diffuseSH = float3(0);
    out.specularSH = float3(0);
    return out;
}

half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? 0 : exp(0.5 * negativeMagnitudeSquared) * splatAlpha;
}
