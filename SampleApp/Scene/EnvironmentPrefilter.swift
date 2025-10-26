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
    private let linearClampSampler: MTLSamplerState

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

        let sdesc = MTLSamplerDescriptor()
        sdesc.minFilter = .linear
        sdesc.magFilter = .linear
        sdesc.mipFilter = .linear
        sdesc.sAddressMode = .clampToEdge
        sdesc.tAddressMode = .clampToEdge
        sdesc.rAddressMode = .clampToEdge
        sdesc.lodMinClamp = 0
        sdesc.lodMaxClamp = Float(mapSize)
        guard let sampler = device.makeSamplerState(descriptor: sdesc) else {
            throw Error.missingPipeline(function: "sampler")
        }
        self.linearClampSampler = sampler
    }

    func prefilter(snapshot: EnvironmentProbeManager.Snapshot) throws -> EnvironmentPrefilterResult {
        Self.log.debug("[Prefilter] begin (snapshot rev: \(snapshot.revision), ts: \(snapshot.timestamp)) src type=\(snapshot.texture.textureType.rawValue) fmt=\(snapshot.texture.pixelFormat.rawValue) size=\(snapshot.texture.width)x\(snapshot.texture.height)x\(snapshot.texture.depth) mips=\(snapshot.texture.mipmapLevelCount)")

        guard snapshot.texture.textureType == .typeCube else {
            throw Error.unsupportedTextureType(snapshot.texture.textureType)
        }

        guard let sourceTexture = sanitizedSourceTexture(from: snapshot.texture) else {
            throw Error.unsupportedPixelFormat(snapshot.texture.pixelFormat)
        }
        Self.log.debug("[Prefilter] using source texture view: type=\(sourceTexture.textureType.rawValue) fmt=\(sourceTexture.pixelFormat.rawValue) size=\(sourceTexture.width)x\(sourceTexture.height)x\(sourceTexture.depth) mips=\(sourceTexture.mipmapLevelCount)")

        let environmentTexture = try makeEnvironmentTexture()
        Self.log.debug("[Prefilter] made destination cube: label=\(environmentTexture.label ?? "<none>") type=\(environmentTexture.textureType.rawValue) fmt=\(environmentTexture.pixelFormat.rawValue) size=\(environmentTexture.width)x\(environmentTexture.height)x\(environmentTexture.depth) mips=\(environmentTexture.mipmapLevelCount)")

        try encodePrefilter(from: sourceTexture, to: environmentTexture)

        // DEBUG: Probe the prefiltered destination cube for non-zero energy at LOD 0 and max LOD
        if let cb = commandQueue.makeCommandBuffer(),
           let blit = cb.makeBlitCommandEncoder() {
            let dstMipCount = environmentTexture.mipmapLevelCount
            let dstLOD0W = max(1, min(4, environmentTexture.width))
            let dstLOD0H = max(1, min(4, environmentTexture.height))
            let dstMaxLOD = max(0, dstMipCount - 1)
            let dstMaxW = max(1, environmentTexture.width >> dstMaxLOD)
            let dstMaxH = max(1, environmentTexture.height >> dstMaxLOD)

            func makeStaging(_ w: Int, _ h: Int) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: environmentTexture.pixelFormat, width: w, height: h, mipmapped: false)
                d.storageMode = .shared
                return device.makeTexture(descriptor: d)
            }

            guard let stagingLOD0 = makeStaging(dstLOD0W, dstLOD0H),
                  let stagingMax = makeStaging(dstMaxW, dstMaxH) else {
                Self.log.error("[Prefilter] Failed to create staging textures for dst probe")
                throw Error.commandBufferFailed
            }

            // Copy face +X, LOD 0
            let origin = MTLOrigin(x: 0, y: 0, z: 0)
            let sizeLOD0 = MTLSize(width: dstLOD0W, height: dstLOD0H, depth: 1)
            blit.copy(from: environmentTexture,
                      sourceSlice: 0,
                      sourceLevel: 0,
                      sourceOrigin: origin,
                      sourceSize: sizeLOD0,
                      to: stagingLOD0,
                      destinationSlice: 0,
                      destinationLevel: 0,
                      destinationOrigin: origin)

            // Copy face +X, max LOD
            let sizeMax = MTLSize(width: dstMaxW, height: dstMaxH, depth: 1)
            blit.copy(from: environmentTexture,
                      sourceSlice: 0,
                      sourceLevel: dstMaxLOD,
                      sourceOrigin: origin,
                      sourceSize: sizeMax,
                      to: stagingMax,
                      destinationSlice: 0,
                      destinationLevel: 0,
                      destinationOrigin: origin)

            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            func anyNonZero(_ tex: MTLTexture) -> Bool {
                let bpp: Int
                switch tex.pixelFormat {
                case .rgba16Float: bpp = 8
                case .rgba32Float: bpp = 16
                case .rg16Float:   bpp = 4
                case .rg32Float:   bpp = 8
                default:           bpp = 8
                }
                let row = tex.width * bpp
                let count = row * tex.height
                var buf = [UInt8](repeating: 0, count: count)
                buf.withUnsafeMutableBytes { p in
                    tex.getBytes(p.baseAddress!, bytesPerRow: row, from: MTLRegionMake2D(0, 0, tex.width, tex.height), mipmapLevel: 0)
                }
                return buf.contains { $0 != 0 }
            }

            let nonZeroLOD0 = anyNonZero(stagingLOD0)
            let nonZeroMax  = anyNonZero(stagingMax)
            if nonZeroLOD0 { Self.log.debug("[Prefilter] DST probe LOD0 is NON-zero") } else { Self.log.warning("[Prefilter] DST probe LOD0 is all zeros") }
            if nonZeroMax  { Self.log.debug("[Prefilter] DST probe max LOD is NON-zero") } else { Self.log.warning("[Prefilter] DST probe max LOD is all zeros") }

            // TEMP: If both are zero, copy a 1x1 from the source smallest mip into destination smallest mip to validate writes
            if !nonZeroLOD0 && !nonZeroMax {
                if let cb2 = commandQueue.makeCommandBuffer(), let blit2 = cb2.makeBlitCommandEncoder() {
                    let srcMinLevel = max(0, sourceTexture.mipmapLevelCount - 1)
                    let dstMinLevel = max(0, environmentTexture.mipmapLevelCount - 1)
                    let sz = MTLSize(width: 1, height: 1, depth: 1)
                    blit2.copy(from: sourceTexture,
                               sourceSlice: 0,
                               sourceLevel: srcMinLevel,
                               sourceOrigin: origin,
                               sourceSize: sz,
                               to: environmentTexture,
                               destinationSlice: 0,
                               destinationLevel: dstMinLevel,
                               destinationOrigin: origin)
                    blit2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
                    Self.log.warning("[Prefilter] Wrote a 1x1 texel from source(min mip) to destination(min mip) for validation")
                }
            }
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

        Self.log.debug("[BRDF] integrating LUT: size=\(self.brdfResolution)x\(self.brdfResolution) fmt=\(texture.pixelFormat.rawValue)")
        try encodeBRDF(into: texture)
        cachedBRDFLUT = texture
        return texture
    }

    private func encodePrefilter(from source: MTLTexture, to destination: MTLTexture) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Error.missingCommandQueue
        }
        commandBuffer.label = "EnvironmentPrefilter"
        Self.log.debug("[Prefilter] encode start: src type=\(source.textureType.rawValue) fmt=\(source.pixelFormat.rawValue) dst type=\(destination.textureType.rawValue) fmt=\(destination.pixelFormat.rawValue) mips=\(destination.mipmapLevelCount)")

        commandBuffer.addCompletedHandler { cb in
            if cb.status == .error {
                if let e = cb.error as NSError? {
                    Self.log.error("[Prefilter] command buffer error: domain=\(e.domain) code=\(e.code) userInfo=\(e.userInfo)")
                } else {
                    Self.log.error("[Prefilter] command buffer error with no NSError")
                }
            }
        }

        let mipCount = destination.mipmapLevelCount
        var dimension = destination.width

        for mipLevel in 0..<mipCount {
            let encoder = commandBuffer.makeComputeCommandEncoder()
            encoder?.label = "SpecularPrefilterMip\(mipLevel)"
            encoder?.setComputePipelineState(specularPipeline)

            // Primary expectation: source@0, dest@1, sampler@0
            encoder?.setTexture(source, index: 0)
            encoder?.setTexture(destination, index: 1)
            encoder?.setSamplerState(linearClampSampler, index: 0)
            // Also bind duplicates at alt indices in case the compiled kernel expects them swapped or shifted
            encoder?.setTexture(source, index: 2)
            encoder?.setTexture(destination, index: 3)
            encoder?.setSamplerState(linearClampSampler, index: 1)

            Self.log.debug("[Prefilter] specular mip=\(mipLevel) dim=\(dimension) dispatch=\(max(1, (dimension + 7) / 8))x\(max(1, (dimension + 7) / 8)) faces=6")

            // Bind per-dispatch uniforms for this mip via setBytes (avoids stale buffer contents)
            var u = PrefilterUniforms(
                mipLevel: UInt32(mipLevel),
                dimension: UInt32(dimension),
                roughness: Float(mipCount <= 1 ? 0 : Float(mipLevel) / Float(mipCount - 1)),
                sampleCount: specularSampleCount
            )
            encoder?.setBytes(&u, length: MemoryLayout<PrefilterUniforms>.stride, index: 0)
            // Duplicate at index 1 to tolerate alternate kernel signatures
            encoder?.setBytes(&u, length: MemoryLayout<PrefilterUniforms>.stride, index: 1)
            Self.log.debug("[Prefilter] specular uniforms mip=\(u.mipLevel) dim=\(u.dimension) rough=\(u.roughness) samples=\(u.sampleCount)")

            let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroups = MTLSize(width: max(1, (dimension + 7) / 8),
                                       height: max(1, (dimension + 7) / 8),
                                       depth: 6)
            encoder?.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
            encoder?.endEncoding()

            dimension = max(1, dimension >> 1)
        }

        if mipCount > 0 {
            let diffuseEncoder = commandBuffer.makeComputeCommandEncoder()
            diffuseEncoder?.label = "DiffusePrefilter"
            diffuseEncoder?.setComputePipelineState(diffusePipeline)
            diffuseEncoder?.setTexture(source, index: 0)
            diffuseEncoder?.setTexture(destination, index: 1)
            diffuseEncoder?.setSamplerState(linearClampSampler, index: 0)
            let diffuseMipLevel = mipCount - 1
            let diffuseDimension = max(1, destination.width >> diffuseMipLevel)
            var du = DiffuseUniforms(mipLevel: UInt32(diffuseMipLevel),
                                     dimension: UInt32(diffuseDimension),
                                     sampleCount: diffuseSampleCount)
            diffuseEncoder?.setBytes(&du, length: MemoryLayout<DiffuseUniforms>.stride, index: 0)
            diffuseEncoder?.setBytes(&du, length: MemoryLayout<DiffuseUniforms>.stride, index: 1)
            Self.log.debug("[Prefilter] diffuse bind: src@0 dst@1 smp@0 uniforms@0&1 mip=\(diffuseMipLevel) dim=\(diffuseDimension)")
            let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
            let threadgroups = MTLSize(width: diffuseDimension,
                                       height: diffuseDimension,
                                       depth: 6)
            diffuseEncoder?.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
            diffuseEncoder?.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let status = commandBuffer.status
        Self.log.info("[Prefilter] completed (mips: \(mipCount)) status=\(status.rawValue)")

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

        Self.log.debug("[BRDF] dispatch: groups=\((self.brdfResolution + 7) / 8)x\((self.brdfResolution + 7) / 8) threads=8x8")

        let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(width: (brdfResolution + 7) / 8,
                                   height: (brdfResolution + 7) / 8,
                                   depth: 1)
        encoder?.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder?.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        Self.log.info("[BRDF] completed status=\(commandBuffer.status.rawValue)")

        if commandBuffer.status == .error {
            Self.log.error("BRDF integration command buffer failed: \(commandBuffer.error?.localizedDescription ?? "unknown")")
            throw Error.commandBufferFailed
        }
    }
}

#endif // os(visionOS)
