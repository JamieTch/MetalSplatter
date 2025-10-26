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
    enum Error: Swift.Error, LocalizedError {
        case missingCommandQueue
        case missingPipeline(function: String)
        case unsupportedTextureType(MTLTextureType)
        case unsupportedPixelFormat(MTLPixelFormat)
        case commandBufferFailed
        
        var errorDescription: String? {
            switch self {
            case .missingCommandQueue:
                return "Environment prefilter could not create a Metal command queue."
            case .missingPipeline(let function):
                return "Environment prefilter pipeline function '\(function)' could not be created."
            case .unsupportedTextureType(let type):
                return "Environment prefilter received unsupported texture type: \(String(describing: type))."
            case .unsupportedPixelFormat(let format):
                return "Environment prefilter received unsupported pixel format: \(String(describing: format))."
            case .commandBufferFailed:
                return "Environment prefilter command buffer failed to complete successfully."
            }
        }
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
    private var luminanceScratchBuffer: MTLBuffer?
    private var lastLoggedRevision: UInt64?

    private struct LuminanceStatistics {
        var minimum: Float
        var maximum: Float
        var average: Float
    }

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

        if lastLoggedRevision != snapshot.revision {
            logLuminanceDiagnostics(for: environmentTexture, revision: snapshot.revision)
        }

        let brdf = try makeBRDFLookupTexture()

        return EnvironmentPrefilterResult(environmentMap: environmentTexture,
                                          brdfLookup: brdf,
                                          revision: snapshot.revision,
                                          timestamp: snapshot.timestamp,
                                          sphericalHarmonics: snapshot.sphericalHarmonics)
    }

    private func sanitizedSourceTexture(from texture: MTLTexture) -> MTLTexture? {
        if texture.pixelFormat == .rgba16Float || texture.pixelFormat == .rgba32Float {
            return texture
        }
        return texture.makeTextureView(pixelFormat: .rgba16Float)
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

    private func logLuminanceDiagnostics(for texture: MTLTexture, revision: UInt64) {
        let faceSize = texture.width
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<UInt16>.stride
        let faceByteCount = faceSize * faceSize * bytesPerPixel
        let totalByteCount = faceByteCount * 6

        if luminanceScratchBuffer == nil || luminanceScratchBuffer?.length ?? 0 < totalByteCount {
            luminanceScratchBuffer = device.makeBuffer(length: totalByteCount, options: .storageModeShared)
            luminanceScratchBuffer?.label = "EnvironmentPrefilterLuminanceScratch"
        }

        guard let scratchBuffer = luminanceScratchBuffer else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        commandBuffer.label = "EnvironmentPrefilterLuminanceCopy"

        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            let bytesPerRow = faceSize * bytesPerPixel
            let bytesPerImage = faceByteCount
            let origin = MTLOrigin(x: 0, y: 0, z: 0)
            let size = MTLSize(width: faceSize, height: faceSize, depth: 1)

            for face in 0..<6 {
                blitEncoder.copy(from: texture,
                                 sourceSlice: face,
                                 sourceLevel: 0,
                                 sourceOrigin: origin,
                                 sourceSize: size,
                                 to: scratchBuffer,
                                 destinationOffset: faceByteCount * face,
                                 destinationBytesPerRow: bytesPerRow,
                                 destinationBytesPerImage: bytesPerImage)
            }

            blitEncoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            return
        }

        let pixelCountPerFace = faceSize * faceSize
        let totalComponentCount = pixelCountPerFace * componentsPerPixel * 6
        let componentPointer = scratchBuffer.contents().bindMemory(to: UInt16.self, capacity: totalComponentCount)

        var statistics: [LuminanceStatistics] = []
        statistics.reserveCapacity(6)

        for face in 0..<6 {
            let faceOffset = face * pixelCountPerFace * componentsPerPixel
            var minLum = Float.greatestFiniteMagnitude
            var maxLum: Float = -Float.greatestFiniteMagnitude
            var sumLum: Float = 0

            for pixel in 0..<pixelCountPerFace {
                let baseIndex = faceOffset + pixel * componentsPerPixel
                let r = Float(Float16(bitPattern: componentPointer[baseIndex]))
                let g = Float(Float16(bitPattern: componentPointer[baseIndex + 1]))
                let b = Float(Float16(bitPattern: componentPointer[baseIndex + 2]))

                let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                minLum = min(minLum, luminance)
                maxLum = max(maxLum, luminance)
                sumLum += luminance
            }

            let averageLum = sumLum / Float(pixelCountPerFace)
            statistics.append(LuminanceStatistics(minimum: minLum, maximum: maxLum, average: averageLum))
        }

        var message = "Environment revision \(revision) luminance"
        for (index, stat) in statistics.enumerated() {
            let faceSummary = String(format: " F%u[min:%.3f max:%.3f avg:%.3f]", UInt32(index), stat.minimum, stat.maximum, stat.average)
            message.append(faceSummary)
        }
        Self.log.debug("\(message)")

        lastLoggedRevision = revision
    }
}

#endif // os(visionOS)
