import Foundation
import Metal
import MetalKit
import ModelIO
import simd

/// A simple mesh renderer that lights Model I/O assets with a PBR-inspired shader.
final class MeshModelRenderer: EnvironmentBindableRenderer {
    private enum Error: Swift.Error {
        case missingPipeline
        case missingDepthState
        case missingMesh
    }

    private struct Uniforms {
        var modelMatrix: simd_float4x4
        var viewMatrix: simd_float4x4
        var projectionMatrix: simd_float4x4
        var normalMatrix: simd_float3x3
        var roughness: Float
        var metallic: Float
        var pad: SIMD2<Float> = .zero
    }

    private struct UniformArray {
        var u0: Uniforms
        var u1: Uniforms

        static var alignedSize: Int { (MemoryLayout<UniformArray>.size + 0xFF) & -0x100 }

        mutating func set(_ uniforms: Uniforms, at index: Int) {
            switch index {
            case 0: u0 = uniforms
            case 1: u1 = uniforms
            default: break
            }
        }
    }

    private enum BufferIndex: Int {
        case positions = 0
        case normals = 1
        case texcoords = 2
        case uniforms = 3
    }

    private enum TextureIndex: Int {
        case baseColor = 0
        case environment = 1
        case brdf = 2
    }

    private let device: MTLDevice
    private let mesh: MTKMesh
    private let baseColor: MTLTexture?
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let sampler: MTLSamplerState
    private let maxViewCount: Int
    private let maxSimultaneousRenders: Int
    private let uniformBuffer: MTLBuffer
    private var uniformBufferOffset = 0
    private var uniformBufferIndex = 0
    private var uniforms: UnsafeMutablePointer<UniformArray>

    private var environmentMap: MTLTexture?
    private var brdfLUT: MTLTexture?

    init(device: MTLDevice,
         colorFormat: MTLPixelFormat,
         depthFormat: MTLPixelFormat,
         sampleCount: Int,
         maxViewCount: Int,
         maxSimultaneousRenders: Int,
         url: URL,
         defaultRoughness: Float = 0.4,
         defaultMetallic: Float = 0.0) throws {
        self.device = device
        self.maxViewCount = maxViewCount
        self.maxSimultaneousRenders = maxSimultaneousRenders

        let allocator = MTKMeshBufferAllocator(device: device)
        let asset = MDLAsset(url: url, vertexDescriptor: Self.modelIODescriptor(), bufferAllocator: allocator)
        asset.loadTextures()
        guard let mdlMesh = asset.childObjects(of: MDLMesh.self).first as? MDLMesh else {
            throw Error.missingMesh
        }
        mdlMesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.01)
        mdlMesh.addOrthTanBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                normalAttributeNamed: MDLVertexAttributeNormal,
                                tangentAttributeNamed: MDLVertexAttributeTangent)

        let mtkMesh = try MTKMesh(mesh: mdlMesh, device: device)
        self.mesh = mtkMesh
        self.baseColor = Self.findBaseColor(in: mdlMesh, device: device)

        let library = try device.makeDefaultLibrary(bundle: Bundle.main)
        guard let vertex = library.makeFunction(name: "meshVertex"),
              let fragment = library.makeFunction(name: "meshFragment") else {
            throw Error.missingPipeline
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Mesh Renderer"
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.vertexDescriptor = Self.metalVertexDescriptor()
        descriptor.rasterSampleCount = sampleCount
        descriptor.colorAttachments[0].pixelFormat = colorFormat
        descriptor.depthAttachmentPixelFormat = depthFormat
        descriptor.maxVertexAmplificationCount = maxViewCount

        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.sAddressMode = .repeat
        samplerDescriptor.tAddressMode = .repeat
        samplerDescriptor.rAddressMode = .repeat
        samplerDescriptor.lodMinClamp = 0
        samplerDescriptor.lodMaxClamp = Float.greatestFiniteMagnitude
        sampler = device.makeSamplerState(descriptor: samplerDescriptor)!

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .greater
        depthDescriptor.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            throw Error.missingDepthState
        }
        self.depthState = depthState

        let uniformBufferLength = UniformArray.alignedSize * maxSimultaneousRenders
        guard let buffer = device.makeBuffer(length: uniformBufferLength, options: .storageModeShared) else {
            throw Error.missingPipeline
        }
        uniformBuffer = buffer
        uniforms = buffer.contents().bindMemory(to: UniformArray.self, capacity: 1)

        var defaults = Uniforms(modelMatrix: matrix_identity_float4x4,
                                viewMatrix: matrix_identity_float4x4,
                                projectionMatrix: matrix_identity_float4x4,
                                normalMatrix: matrix_identity_float3x3,
                                roughness: defaultRoughness,
                                metallic: defaultMetallic)
        uniforms.pointee = UniformArray(u0: defaults, u1: defaults)
    }

    func render(viewports: [ModelRendererViewportDescriptor],
                colorTexture: MTLTexture,
                colorStoreAction: MTLStoreAction,
                depthTexture: MTLTexture?,
                rasterizationRateMap: MTLRasterizationRateMap?,
                renderTargetArrayLength: Int,
                to commandBuffer: MTLCommandBuffer) throws {
        updateDynamicUniforms(for: viewports)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = colorTexture
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = colorStoreAction
        if let depthTexture {
            pass.depthAttachment.texture = depthTexture
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
        }
        pass.rasterizationRateMap = rasterizationRateMap
        pass.renderTargetArrayLength = renderTargetArrayLength

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Error.missingPipeline
        }

        encoder.label = "Mesh Renderer"
        encoder.setViewports(viewports.map(\.viewport))

        if viewports.count > 1 {
            var mappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            encoder.setVertexAmplificationCount(viewports.count, viewMappings: &mappings)
        }

        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)

        encoder.setVertexBuffer(uniformBuffer, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        for (index, buffer) in mesh.vertexBuffers.enumerated() {
            encoder.setVertexBuffer(buffer.buffer, offset: buffer.offset, index: index)
        }

        encoder.setFragmentTexture(baseColor, index: TextureIndex.baseColor.rawValue)
        encoder.setFragmentTexture(environmentMap, index: TextureIndex.environment.rawValue)
        encoder.setFragmentTexture(brdfLUT, index: TextureIndex.brdf.rawValue)
        encoder.setFragmentSamplerState(sampler, index: 0)

        for submesh in mesh.submeshes {
            encoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                          indexCount: submesh.indexCount,
                                          indexType: submesh.indexType,
                                          indexBuffer: submesh.indexBuffer.buffer,
                                          indexBufferOffset: submesh.indexBuffer.offset)
        }

        encoder.endEncoding()
    }

    func setEnvironmentMap(_ texture: MTLTexture?) {
        environmentMap = texture
    }

    func setBRDFLookupTexture(_ texture: MTLTexture?) {
        brdfLUT = texture
    }

    func applyEnvironment(environmentMap: MTLTexture?, brdfLookup: MTLTexture?) throws {
        setEnvironmentMap(environmentMap)
        setBRDFLookupTexture(brdfLookup)
    }

    private func updateDynamicUniforms(for viewports: [ModelRendererViewportDescriptor]) {
        uniformBufferIndex = (uniformBufferIndex + 1) % maxSimultaneousRenders
        uniformBufferOffset = UniformArray.alignedSize * uniformBufferIndex
        uniforms = uniformBuffer.contents().advanced(by: uniformBufferOffset).bindMemory(to: UniformArray.self, capacity: 1)

        for (idx, viewport) in viewports.enumerated() where idx < 2 {
            let viewModel = viewport.viewMatrix * viewport.modelMatrix
            let normalMatrix = simd_float3x3(normalFrom: viewModel)
            let uniformsEntry = Uniforms(modelMatrix: viewport.modelMatrix,
                                         viewMatrix: viewport.viewMatrix,
                                         projectionMatrix: viewport.projectionMatrix,
                                         normalMatrix: normalMatrix,
                                         roughness: 0.4,
                                         metallic: 0.0)
            uniforms.pointee.set(uniformsEntry, at: idx)
        }
    }

    private static func modelIODescriptor() -> MDLVertexDescriptor {
        let metalDescriptor = metalVertexDescriptor()
        let mdlDescriptor = MTKModelIOVertexDescriptorFromMetal(metalDescriptor)
        guard let attributes = mdlDescriptor.attributes as? [MDLVertexAttribute] else {
            return MDLVertexDescriptor()
        }
        attributes[0].name = MDLVertexAttributePosition
        attributes[1].name = MDLVertexAttributeNormal
        attributes[2].name = MDLVertexAttributeTextureCoordinate
        return mdlDescriptor
    }

    private static func metalVertexDescriptor() -> MTLVertexDescriptor {
        let descriptor = MTLVertexDescriptor()
        descriptor.attributes[0].format = .float3
        descriptor.attributes[0].offset = 0
        descriptor.attributes[0].bufferIndex = BufferIndex.positions.rawValue
        descriptor.layouts[BufferIndex.positions.rawValue].stride = 12

        descriptor.attributes[1].format = .float3
        descriptor.attributes[1].offset = 0
        descriptor.attributes[1].bufferIndex = BufferIndex.normals.rawValue
        descriptor.layouts[BufferIndex.normals.rawValue].stride = 12

        descriptor.attributes[2].format = .float2
        descriptor.attributes[2].offset = 0
        descriptor.attributes[2].bufferIndex = BufferIndex.texcoords.rawValue
        descriptor.layouts[BufferIndex.texcoords.rawValue].stride = 8
        return descriptor
    }

    private static func findBaseColor(in mesh: MDLMesh, device: MTLDevice) -> MTLTexture? {
        guard let material = mesh.submeshes.first?.material else { return nil }
        guard let property = material.property(with: .baseColor) else { return nil }
        if property.type == .string, let filename = property.stringValue {
            let url = URL(fileURLWithPath: filename)
            return try? MTKTextureLoader(device: device).newTexture(URL: url, options: nil)
        }
        if property.type == .URL, let url = property.urlValue {
            return try? MTKTextureLoader(device: device).newTexture(URL: url, options: nil)
        }
        return nil
    }
}

private extension simd_float3x3 {
    init(normalFrom matrix: simd_float4x4) {
        self.init(simd_normalize(matrix.columns.0.xyz),
                  simd_normalize(matrix.columns.1.xyz),
                  simd_normalize(matrix.columns.2.xyz))
    }
}
