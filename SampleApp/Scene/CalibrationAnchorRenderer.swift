import Foundation
import Metal
import simd

#if os(visionOS)

final class CalibrationAnchorRenderer {
    private struct Vertex {
        var position: SIMD3<Float>
        var color: SIMD3<Float>
    }

    private struct Uniforms {
        var modelViewProjection: simd_float4x4
    }

    private let pipelineState: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let planeVertexBuffer: MTLBuffer
    private let axisVertexBuffer: MTLBuffer
    private let planeVertexCount: Int
    private let axisVertexCount: Int

    init(device: MTLDevice, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) throws {
        let library = try device.makeDefaultLibrary(bundle: Bundle.main)
        guard let vertexFunction = library.makeFunction(name: "calibrationAnchorVertex"),
              let fragmentFunction = library.makeFunction(name: "calibrationAnchorFragment") else {
            throw RendererError.missingShader
        }

        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float3
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
        vertexDescriptor.layouts[0].stepFunction = .perVertex

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "Calibration Anchor Pipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .lessEqual
        depthDescriptor.isDepthWriteEnabled = false
        guard let depthState = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            throw RendererError.missingDepthState
        }
        self.depthState = depthState

        let planeColor = SIMD3<Float>(repeating: 0.4)
        let planeSize: Float = 0.4
        let planeVertices: [Vertex] = [
            Vertex(position: SIMD3<Float>(-planeSize, 0, -planeSize), color: planeColor),
            Vertex(position: SIMD3<Float>(planeSize, 0, -planeSize), color: planeColor),
            Vertex(position: SIMD3<Float>(-planeSize, 0, planeSize), color: planeColor),
            Vertex(position: SIMD3<Float>(planeSize, 0, -planeSize), color: planeColor),
            Vertex(position: SIMD3<Float>(planeSize, 0, planeSize), color: planeColor),
            Vertex(position: SIMD3<Float>(-planeSize, 0, planeSize), color: planeColor)
        ]
        planeVertexCount = planeVertices.count

        let axisLength: Float = 0.5
        let axisThicknessColor: Float = 0.85
        let axisVertices: [Vertex] = [
            Vertex(position: .zero, color: SIMD3<Float>(axisThicknessColor, 0, 0)),
            Vertex(position: SIMD3<Float>(axisLength, 0, 0), color: SIMD3<Float>(axisThicknessColor, 0, 0)),
            Vertex(position: .zero, color: SIMD3<Float>(0, axisThicknessColor, 0)),
            Vertex(position: SIMD3<Float>(0, axisLength, 0), color: SIMD3<Float>(0, axisThicknessColor, 0)),
            Vertex(position: .zero, color: SIMD3<Float>(0, 0, axisThicknessColor)),
            Vertex(position: SIMD3<Float>(0, 0, axisLength), color: SIMD3<Float>(0, 0, axisThicknessColor))
        ]
        axisVertexCount = axisVertices.count

        guard let planeBuffer = device.makeBuffer(bytes: planeVertices,
                                                  length: planeVertices.count * MemoryLayout<Vertex>.stride,
                                                  options: .storageModeShared) else {
            throw RendererError.missingBuffer
        }
        guard let axisBuffer = device.makeBuffer(bytes: axisVertices,
                                                 length: axisVertices.count * MemoryLayout<Vertex>.stride,
                                                 options: .storageModeShared) else {
            throw RendererError.missingBuffer
        }

        planeVertexBuffer = planeBuffer
        axisVertexBuffer = axisBuffer
    }

    func render(viewports: [ModelRendererViewportDescriptor],
                colorTexture: MTLTexture,
                depthTexture: MTLTexture?,
                commandBuffer: MTLCommandBuffer) {
        guard !viewports.isEmpty else { return }

        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = colorTexture
        descriptor.colorAttachments[0].loadAction = .load
        descriptor.colorAttachments[0].storeAction = .store
        if let depthTexture {
            descriptor.depthAttachment.texture = depthTexture
            descriptor.depthAttachment.loadAction = .load
            descriptor.depthAttachment.storeAction = .store
        }

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        encoder.label = "Calibration Anchor Encoder"
        encoder.setRenderPipelineState(pipelineState)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none)

        for viewport in viewports {
            var uniforms = Uniforms(modelViewProjection: viewport.projectionMatrix * viewport.viewMatrix)
            encoder.setViewport(viewport.viewport)
            encoder.setVertexBytes(&uniforms,
                                   length: MemoryLayout<Uniforms>.stride,
                                   index: 1)

            encoder.setVertexBuffer(planeVertexBuffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: planeVertexCount)

            encoder.setVertexBuffer(axisVertexBuffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: axisVertexCount)
        }

        encoder.endEncoding()
    }

    private enum RendererError: Error {
        case missingShader
        case missingBuffer
        case missingDepthState
    }
}

#else

final class CalibrationAnchorRenderer {
    init(device: MTLDevice, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) throws {}

    func render(viewports: [ModelRendererViewportDescriptor],
                colorTexture: MTLTexture,
                depthTexture: MTLTexture?,
                commandBuffer: MTLCommandBuffer) {}
}

#endif
