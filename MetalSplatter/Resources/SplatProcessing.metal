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

half3 shadeGaussian(half3 albedo,
                    half metallic,
                    half roughness,
                    half3 normal,
                    half3 viewDirection,
                    half ambientOcclusion,
                    texturecube<half> environmentMap,
                    texture2d<half> brdfLUT,
                    sampler environmentSampler,
                    sampler brdfSampler) {
    float3 N = safeNormalize(float3(normal), float3(0, 0, 1));
    float3 V = safeNormalize(float3(viewDirection), float3(0, 0, 1));
    float perceptualRoughness = clamp(float(roughness), 0.045, 1.0);
    float3 R = reflect(-V, N);
    uint mipCount = environmentMap.get_num_mip_levels();
    float lod = perceptualRoughness * float(max(int(mipCount) - 1, 0));
    float3 prefilteredColor = float3(environmentMap.sample(environmentSampler, R, level(lod)).rgb);
    float3 irradiance = sampleDiffuseIrradiance(environmentMap, environmentSampler, N);

    float NdotV = clamp(dot(N, V), 1e-4, 1.0);
    float3 baseAlbedo = float3(albedo);
    float metallicValue = clamp(float(metallic), 0.0, 1.0);
    float3 F0 = mix(float3(0.04), baseAlbedo, metallicValue);
    float Fc = pow(1.0 - NdotV, 5.0);
    float3 F = F0 + (1.0 - F0) * Fc;

    float3 kS = F;
    float3 kD = (float3(1.0) - kS) * (1.0 - metallicValue);

    float2 brdfSample = float2(brdfLUT.sample(brdfSampler, float2(NdotV, perceptualRoughness)).rg);
    float3 specular = prefilteredColor * (F0 * brdfSample.x + brdfSample.y);

    float3 diffuse = irradiance * baseAlbedo * kD;
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
                       uint relativeVertexIndex) {
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
    float3 normal = safeNormalize(float3(splat.normal), float3(0, 0, 1));
    float4 rotation = float4(splat.rotation);
    float3 rotatedNormal = safeNormalize(rotateVectorByQuaternion(rotation, normal), float3(0, 0, 1));
    out.normal = half3(rotatedNormal);
    out.worldPosition = worldPosition;

    float3 cameraPosition = uniforms.cameraPosition.xyz;
    float3 viewDirection = safeNormalize(cameraPosition - worldPosition, float3(0, 0, 1));
    out.viewDirection = viewDirection;
    return out;
}

half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? 0 : exp(0.5 * negativeMagnitudeSquared) * splatAlpha;
}
