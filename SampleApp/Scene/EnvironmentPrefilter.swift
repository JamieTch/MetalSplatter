#if os(visionOS)

import Foundation
import Metal
import os

struct EnvironmentPrefilterResult {
    let environmentMap: MTLTexture
    let brdfLookup: MTLTexture
    let revision: UInt64
    let timestamp: Date
    let sphericalHarmonics: [Float]
}

private struct PrefilterUniforms {
    var mipLevel: UInt32
    var dimension: UInt32
    var roughness: Float
    var sampleCount: UInt32
}

private struct DiffuseUniforms {
    var mipLevel: UInt32
    var dimension: UInt32
    var sampleCount: UInt32
}

private struct BRDFUniforms {
    var dimension: UInt32
    var sampleCount: UInt32
}

final class EnvironmentPrefilter {
    enum Error: Swift.Error {
        case missingCommandQueue
        case missingPipeline(function: String)
        case unsupportedTextureType(MTLTextureType)
        case unsupportedPixelFormat(MTLPixelFormat)
        case commandBufferFailed
    }

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "EnvironmentPrefilter",
                                     category: "EnvironmentPrefilter")

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let specularPipeline: MTLComputePipelineState
    private let diffusePipeline: MTLComputePipelineState
    private let brdfPipeline: MTLComputePipelineState

    private let targetMapSize: Int
    private let diffuseSampleCount: UInt32
    private let specularSampleCount: UInt32
    private let brdfSampleCount: UInt32
    private let brdfResolution: Int

    private var cachedBRDFLUT: MTLTexture?

    init(device: MTLDevice,
         mapSize: Int = 256,
         specularSamples: UInt32 = 256,
         diffuseSamples: UInt32 = 128,
         brdfResolution: Int = 256,
         brdfSamples: UInt32 = 512) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw Error.missingCommandQueue
        }
        self.commandQueue = queue
        self.targetMapSize = mapSize
        self.specularSampleCount = specularSamples
        self.diffuseSampleCount = diffuseSamples
        self.brdfResolution = brdfResolution
        self.brdfSampleCount = brdfSamples

        let library = try device.makeDefaultLibrary(bundle: Bundle.main)
        guard let specularFunction = library.makeFunction(name: "prefilterEnvironmentSpecular") else {
            throw Error.missingPipeline(function: "prefilterEnvironmentSpecular")
        }
        guard let diffuseFunction = library.makeFunction(name: "prefilterEnvironmentDiffuse") else {
            throw Error.missingPipeline(function: "prefilterEnvironmentDiffuse")
        }
        guard let brdfFunction = library.makeFunction(name: "integrateBRDFLUT") else {
            throw Error.missingPipeline(function: "integrateBRDFLUT")
        }

        specularPipeline = try device.makeComputePipelineState(function: specularFunction)
        diffusePipeline = try device.makeComputePipelineState(function: diffuseFunction)
        brdfPipeline = try device.makeComputePipelineState(function: brdfFunction)
    }

    func prefilter(snapshot: EnvironmentProbeManager.Snapshot) throws -> EnvironmentPrefilterResult {
        guard snapshot.texture.textureType == .typeCube else {
            throw Error.unsupportedTextureType(snapshot.texture.textureType)
        }

        guard let sourceTexture = sanitizedSourceTexture(from: snapshot.texture) else {
            throw Error.unsupportedPixelFormat(snapshot.texture.pixelFormat)
        }

        let environmentTexture = try makeEnvironmentTexture()
        try encodePrefilter(from: sourceTexture, to: environmentTexture)
        let brdf = try makeBRDFLookupTexture()

        return EnvironmentPrefilterResult(environmentMap: environmentTexture,
                                          brdfLookup: brdf,
                                          revision: snapshot.revision,
                                          timestamp: snapshot.timestamp,
                                          sphericalHarmonics: snapshot.sphericalHarmonics)
    }

    private func sanitizedSourceTexture(from texture: MTLTexture) -> MTLTexture? {
        switch texture.pixelFormat {
        case .rgba16Float, .rgba32Float,
             .rgba8Unorm, .rgba8Unorm_sRGB,
             .bgra8Unorm, .bgra8Unorm_sRGB,
             .bgr10_xr, .bgr10_xr_sRGB:
            return texture
        default:
            if let view = texture.makeTextureView(pixelFormat: .rgba16Float) {
                Self.log.debug("Created float16 view for environment probe texture (format: \(texture.pixelFormat.rawValue))")
                return view
            }
            Self.log.error("Unsupported environment probe pixel format: \(texture.pixelFormat.rawValue)")
            return nil
        }
    }

    private func makeEnvironmentTexture() throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float,
                                                                     size: targetMapSize,
                                                                     mipmapped: true)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Error.unsupportedPixelFormat(.rgba16Float)
        }
        texture.label = "PrefilteredEnvironmentCube"
        return texture
    }

    private func makeBRDFLookupTexture() throws -> MTLTexture {
        if let cachedBRDFLUT {
            return cachedBRDFLUT
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float,
                                                                   width: brdfResolution,
                                                                   height: brdfResolution,
                                                                   mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Error.unsupportedPixelFormat(.rg16Float)
        }
        texture.label = "PrefilteredBRDFLUT"

        try encodeBRDF(into: texture)
        cachedBRDFLUT = texture
        return texture
    }

    private func encodePrefilter(from source: MTLTexture, to destination: MTLTexture) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Error.missingCommandQueue
        }
        commandBuffer.label = "EnvironmentPrefilter"

        let mipCount = destination.mipmapLevelCount
        var dimension = destination.width
        let uniformBuffer = device.makeBuffer(length: MemoryLayout<PrefilterUniforms>.stride,
                                              options: .storageModeShared)
        let diffuseBuffer = device.makeBuffer(length: MemoryLayout<DiffuseUniforms>.stride,
                                              options: .storageModeShared)

        for mipLevel in 0..<mipCount {
            let encoder = commandBuffer.makeComputeCommandEncoder()
            encoder?.label = "SpecularPrefilterMip\(mipLevel)"
            encoder?.setComputePipelineState(specularPipeline)
            encoder?.setTexture(source, index: 0)
            encoder?.setTexture(destination, index: 1)

            var uniforms = PrefilterUniforms(mipLevel: UInt32(mipLevel),
                                             dimension: UInt32(dimension),
                                             roughness: Float(mipCount <= 1 ? 0 : Float(mipLevel) / Float(mipCount - 1)),
                                             sampleCount: specularSampleCount)
            if let buffer = uniformBuffer {
                memcpy(buffer.contents(), &uniforms, MemoryLayout<PrefilterUniforms>.stride)
                encoder?.setBuffer(buffer, offset: 0, index: 0)
            }

            let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroups = MTLSize(width: max(1, (dimension + 7) / 8),
                                       height: max(1, (dimension + 7) / 8),
                                       depth: 6)
            encoder?.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
            encoder?.endEncoding()

            dimension = max(1, dimension >> 1)
        }

        if let diffuseBuffer, mipCount > 0 {
            let diffuseEncoder = commandBuffer.makeComputeCommandEncoder()
            diffuseEncoder?.label = "DiffusePrefilter"
            diffuseEncoder?.setComputePipelineState(diffusePipeline)
            diffuseEncoder?.setTexture(source, index: 0)
            diffuseEncoder?.setTexture(destination, index: 1)
            let diffuseMipLevel = mipCount - 1
            let diffuseDimension = max(1, destination.width >> diffuseMipLevel)
            var uniforms = DiffuseUniforms(mipLevel: UInt32(diffuseMipLevel),
                                           dimension: UInt32(diffuseDimension),
                                           sampleCount: diffuseSampleCount)
            memcpy(diffuseBuffer.contents(), &uniforms, MemoryLayout<DiffuseUniforms>.stride)
            diffuseEncoder?.setBuffer(diffuseBuffer, offset: 0, index: 0)
            let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
            let threadgroups = MTLSize(width: diffuseDimension,
                                       height: diffuseDimension,
                                       depth: 6)
            diffuseEncoder?.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
            diffuseEncoder?.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            let description = commandBuffer.error?.localizedDescription ?? "unknown"
            Self.log.error("Environment prefilter command buffer failed: \(description)")
            throw Error.commandBufferFailed
        }
    }

    private func encodeBRDF(into texture: MTLTexture) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Error.missingCommandQueue
        }
        commandBuffer.label = "BRDFIntegration"

        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder?.label = "IntegrateBRDF"
        encoder?.setComputePipelineState(brdfPipeline)
        encoder?.setTexture(texture, index: 0)

        var uniforms = BRDFUniforms(dimension: UInt32(brdfResolution),
                                     sampleCount: brdfSampleCount)
        let uniformBuffer = device.makeBuffer(bytes: &uniforms,
                                              length: MemoryLayout<BRDFUniforms>.stride,
                                              options: .storageModeShared)
        encoder?.setBuffer(uniformBuffer, offset: 0, index: 0)

        let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(width: (brdfResolution + 7) / 8,
                                   height: (brdfResolution + 7) / 8,
                                   depth: 1)
        encoder?.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder?.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            Self.log.error("BRDF integration command buffer failed: \(commandBuffer.error?.localizedDescription ?? "unknown")")
            throw Error.commandBufferFailed
        }
    }
}

#endif // os(visionOS)
