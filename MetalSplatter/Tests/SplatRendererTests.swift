#if canImport(Metal)
import Metal
import MetalKit
import simd
import XCTest
@testable import MetalSplatter
import SplatIO

final class SplatRendererPackingTests: XCTestCase {
    private func expectedGIRNormal(rotation: simd_quatf, scale: SIMD3<Float>) -> SIMD3<Float> {
        let rotationMatrix = simd_float3x3(rotation)
        let scaleComponents = [scale.x, scale.y, scale.z]
        var smallestIndex = 0
        var smallestValue = scaleComponents[0]
        for index in 1..<scaleComponents.count {
            if scaleComponents[index] < smallestValue {
                smallestIndex = index
                smallestValue = scaleComponents[index]
            }
        }

        var column = rotationMatrix[smallestIndex]
        let length = simd_length(column)
        if length > .leastNonzeroMagnitude, length.isFinite {
            column /= length
        }
        return column
    }

    func testSplatPacksMaterialProperties() {
        let rotation = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1))
        let scale = SIMD3<Float>(0.25, 1.0, 0.5)
        let point = SplatScenePoint(position: SIMD3<Float>(1, 2, 3),
                                    color: .linearFloat(SIMD3<Float>(0.25, 0.5, 0.75)),
                                    opacity: .linearFloat(0.8),
                                    scale: .linearFloat(scale),
                                    rotation: rotation,
                                    albedo: SIMD3<Float>(0.2, 0.4, 0.6),
                                    metallic: 0.3,
                                    roughness: 0.7,
                                    normal: SIMD3<Float>(0, 0, 1))

        let splat = SplatRenderer.Splat(point, index: 0)

        let expectedNormal = expectedGIRNormal(rotation: rotation, scale: scale)

        XCTAssertEqual(Float(splat.albedo.x), 0.2, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.albedo.y), 0.4, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.albedo.z), 0.6, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.metallic), 0.3, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.roughness), 0.7, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.x), expectedNormal.x, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.y), expectedNormal.y, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.z), expectedNormal.z, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.color.a), 0.8, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.rotation.x), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.rotation.y), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.rotation.z), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.rotation.w), 1, accuracy: 1e-3)
    }

    func testSerializedNormalOverridesReconstruction() {
        let rotation = simd_quatf(angle: .pi / 3, axis: SIMD3<Float>(0, 1, 0))
        let scale = SIMD3<Float>(0.1, 0.2, 0.3)
        let providedNormal = simd_normalize(SIMD3<Float>(1, 1, 0))
        let point = SplatScenePoint(position: SIMD3<Float>(1, 2, 3),
                                    color: .linearFloat(SIMD3<Float>(repeating: 0.25)),
                                    opacity: .linearFloat(0.6),
                                    scale: .linearFloat(scale),
                                    rotation: rotation,
                                    albedo: SIMD3<Float>(repeating: 0.5),
                                    metallic: 0.1,
                                    roughness: 0.2,
                                    normal: providedNormal,
                                    normalWasProvided: true)

        let splat = SplatRenderer.Splat(point, index: 1)

        XCTAssertEqual(Float(splat.normal.x), providedNormal.x, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.y), providedNormal.y, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.z), providedNormal.z, accuracy: 1e-3)
    }

    func testInvalidMaterialValuesFallBackToDefaults() {
        var point = SplatScenePoint(position: .zero,
                                    color: .linearFloat(SIMD3<Float>(Float.nan, Float.nan, Float.nan)),
                                    opacity: .linearFloat(Float.nan),
                                    scale: .linearFloat(SIMD3<Float>(repeating: 1)),
                                    rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)))
        point.albedo = SIMD3<Float>(repeating: .nan)
        point.metallic = .nan
        point.roughness = .nan
        point.normal = SIMD3<Float>(repeating: .nan)

        let splat = SplatRenderer.Splat(point, index: 5)

        let expectedNormal = expectedGIRNormal(rotation: point.rotation, scale: point.scale.asLinearFloat)

        XCTAssertEqual(Float(splat.albedo.x), SplatScenePoint.defaultAlbedo.x, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.albedo.y), SplatScenePoint.defaultAlbedo.y, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.albedo.z), SplatScenePoint.defaultAlbedo.z, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.metallic), SplatScenePoint.defaultMetallic, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.roughness), SplatScenePoint.defaultRoughness, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.x), expectedNormal.x, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.y), expectedNormal.y, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.normal.z), expectedNormal.z, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.color.x), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.color.y), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.color.z), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.color.a), 0, accuracy: 1e-3)
    }

    func testInvalidRotationFallsBackToIdentity() {
        var point = SplatScenePoint(position: .zero,
                                    color: .linearFloat(SIMD3<Float>(repeating: 0.5)),
                                    opacity: .linearFloat(0.5),
                                    scale: .linearFloat(SIMD3<Float>(repeating: 1)),
                                    rotation: simd_quatf())
        point.rotation = simd_quatf(vector: SIMD4<Float>(repeating: .nan))

        let splat = SplatRenderer.Splat(point, index: 3)

        XCTAssertEqual(Float(splat.rotation.x), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.rotation.y), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.rotation.z), 0, accuracy: 1e-3)
        XCTAssertEqual(Float(splat.rotation.w), 1, accuracy: 1e-3)
    }

    func testSplatStrideMatchesSize() throws {
        #if arch(x86_64)
        throw XCTSkip("Float16 layout assertions are skipped on x86_64")
        #else
        XCTAssertEqual(MemoryLayout<SplatRenderer.Splat>.stride, MemoryLayout<SplatRenderer.Splat>.size)
        #endif
    }

    func testRejectsInvalidEnvironmentTextureType() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable in test environment")
        }

        let renderer = try SplatRenderer(device: device,
                                         colorFormat: .bgra8Unorm,
                                         depthFormat: .invalid,
                                         sampleCount: 1,
                                         maxViewCount: 1,
                                         maxSimultaneousRenders: 1)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                   width: 1,
                                                                   height: 1,
                                                                   mipmapped: false)
        descriptor.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            XCTFail("Failed to allocate test texture")
            return
        }

        XCTAssertThrowsError(try renderer.setEnvironmentMap(texture)) { error in
            guard case SplatRenderer.Error.invalidMaterialResource(let resource, _) = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
            XCTAssertEqual(resource, "environmentMap")
        }
    }

    func testRejectsInvalidBRDFTextureType() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable in test environment")
        }

        let renderer = try SplatRenderer(device: device,
                                         colorFormat: .bgra8Unorm,
                                         depthFormat: .invalid,
                                         sampleCount: 1,
                                         maxViewCount: 1,
                                         maxSimultaneousRenders: 1)

        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rg16Float,
                                                                    size: 1,
                                                                    mipmapped: false)
        descriptor.usage = [.shaderRead]

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            XCTFail("Failed to allocate test texture")
            return
        }

        XCTAssertThrowsError(try renderer.setBRDFLookupTexture(texture)) { error in
            guard case SplatRenderer.Error.invalidMaterialResource(let resource, _) = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
            XCTAssertEqual(resource, "brdfLUT")
        }
    }
}
#endif
