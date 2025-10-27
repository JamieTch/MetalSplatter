import Foundation
import Metal
import MetalKit
import os
#if canImport(os.signpost)
import os.signpost
#endif
import simd
import SplatIO

#if arch(x86_64)
typealias Float16 = Float
#warning("x86_64 targets are unsupported by MetalSplatter and will fail at runtime. MetalSplatter builds on x86_64 only because Xcode builds Swift Packages as universal binaries and provides no way to override this. When Swift supports Float16 on x86_64, this may be revisited.")
#endif

public class SplatRenderer {
    enum Constants {
        // Keep in sync with Shaders.metal : maxViewCount
        static let maxViewCount = 2
        // Sort by euclidian distance squared from camera position (true), or along the "forward" vector (false)
        // TODO: compare the behaviour and performance of sortByDistance
        // notes: sortByDistance introduces unstable artifacts when you get close to an object; whereas !sortByDistance introduces artifacts are you turn -- but they're a little subtler maybe?
        static let sortByDistance = true
        // Only store indices for 1024 splats; for the remainder, use instancing of these existing indices.
        // Setting to 1 uses only instancing (with a significant performance penalty); setting to a number higher than the splat count
        // uses only indexing (with a significant memory penalty for th elarge index array, and a small performance penalty
        // because that can't be cached as easiliy). Anywhere within an order of magnitude (or more?) of 1k seems to be the sweet spot,
        // with effectively no memory penalty compated to instancing, and slightly better performance than even using all indexing.
        static let maxIndexedSplatCount = 1024

        static let tileSize = MTLSize(width: 16, height: 16, depth: 1)
    }

    private static let log =
        Logger(subsystem: Bundle.module.bundleIdentifier!,
               category: "SplatRenderer")
#if canImport(os.signpost)
    private static let signposter = OSSignposter(logger: log)
#endif

    public enum DebugViewMode: UInt, CaseIterable {
        case coverage = 0
        case albedo = 1
        case normal = 2
        case roughness = 3
        case metallic = 4
        case ambientOcclusion = 5
        case depth = 6
        case shaded = 7
        case shadedAmbientOcclusionUnity = 8
        case environmentReflection = 9
        case normalViewRelationship = 10
        case environmentPanorama = 11
        case lambert = 12
        case brdfLookup = 13
        case environmentFixedLod0 = 14
        case environmentFixedMaxLod = 15
        case albedoLinear = 21
        case roughnessSweep = 22
        case metallicSweep = 23
        case normalSweep = 24
        case normalRaw = 25
        case normalDifference = 26
        case normalDotComparison = 27
    }

    public enum Error: Swift.Error, LocalizedError {
        case invalidMaterialResource(resource: String, reason: String)

        public var errorDescription: String? {
            switch self {
            case .invalidMaterialResource(let resource, let reason):
                return "Invalid material resource for \(resource): \(reason)"
            }
        }
    }

    private static func sanitizeScalar(_ value: Float,
                                       defaultValue: Float,
                                       field: String,
                                       pointIndex: Int) -> Float {
        guard value.isFinite else {
            log.error("Non-finite \(field, privacy: .public) for splat index \(pointIndex, privacy: .public); using default \(defaultValue, privacy: .public)")
            return defaultValue
        }
        return value
    }

    private static func sanitizeUnitScalar(_ value: Float,
                                           defaultValue: Float,
                                           field: String,
                                           pointIndex: Int) -> Float {
        let sanitized = sanitizeScalar(value, defaultValue: defaultValue, field: field, pointIndex: pointIndex)
        return simd_clamp(sanitized, 0, 1)
    }

    private static func sanitizeVector(_ vector: SIMD3<Float>,
                                       defaultValue: SIMD3<Float>,
                                       field: String,
                                       pointIndex: Int) -> SIMD3<Float> {
        SIMD3(
            sanitizeScalar(vector.x, defaultValue: defaultValue.x, field: "\(field).x", pointIndex: pointIndex),
            sanitizeScalar(vector.y, defaultValue: defaultValue.y, field: "\(field).y", pointIndex: pointIndex),
            sanitizeScalar(vector.z, defaultValue: defaultValue.z, field: "\(field).z", pointIndex: pointIndex)
        )
    }

    private static func sanitizeVector(_ vector: SIMD4<Float>,
                                       defaultValue: SIMD4<Float>,
                                       field: String,
                                       pointIndex: Int) -> SIMD4<Float> {
        SIMD4(
            sanitizeScalar(vector.x, defaultValue: defaultValue.x, field: "\(field).x", pointIndex: pointIndex),
            sanitizeScalar(vector.y, defaultValue: defaultValue.y, field: "\(field).y", pointIndex: pointIndex),
            sanitizeScalar(vector.z, defaultValue: defaultValue.z, field: "\(field).z", pointIndex: Int(pointIndex)),
            sanitizeScalar(vector.w, defaultValue: defaultValue.w, field: "\(field).w", pointIndex: pointIndex)
        )
    }

    private static func sanitizeColor(_ color: SIMD4<Float>, pointIndex: Int) -> SIMD4<Float> {
        let sanitized = sanitizeVector(color, defaultValue: SIMD4<Float>(repeating: 0), field: "color", pointIndex: pointIndex)
        return simd_clamp(sanitized, SIMD4<Float>(repeating: 0), SIMD4<Float>(repeating: 1))
    }

    private static func sanitizeAlbedo(_ albedo: SIMD3<Float>, pointIndex: Int) -> SIMD3<Float> {
        let sanitized = sanitizeVector(albedo, defaultValue: SplatScenePoint.defaultAlbedo, field: "albedo", pointIndex: pointIndex)
        return simd_clamp(sanitized, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
    }

    private static func sanitizeNormal(_ normal: SIMD3<Float>, pointIndex: Int) -> SIMD3<Float> {
        var sanitized = sanitizeVector(normal, defaultValue: SplatScenePoint.defaultNormal, field: "normal", pointIndex: pointIndex)
        let length = simd_length(sanitized)
        if length > .leastNonzeroMagnitude && length.isFinite {
            sanitized /= length
        } else {
            log.error("Invalid normal for splat index \(pointIndex, privacy: .public); falling back to default")
            sanitized = SplatScenePoint.defaultNormal
        }
        return sanitized
    }

    private static func sanitizeQuaternion(_ quaternion: simd_quatf, pointIndex: Int) -> simd_quatf {
        let vector = quaternion.vector
        guard vector.x.isFinite, vector.y.isFinite, vector.z.isFinite, vector.w.isFinite else {
            log.error("Non-finite rotation for splat index \(pointIndex, privacy: .public); using identity quaternion")
            return simd_quatf()
        }

        let lengthSquared = simd_length_squared(vector)
        guard lengthSquared.isFinite, lengthSquared > .leastNonzeroMagnitude else {
            log.error("Invalid rotation magnitude for splat index \(pointIndex, privacy: .public); using identity quaternion")
            return simd_quatf()
        }

        return simd_normalize(quaternion)
    }

    private static func packHalf3(_ vector: SIMD3<Float>) -> PackedHalf3 {
        PackedHalf3(x: Float16(vector.x), y: Float16(vector.y), z: Float16(vector.z))
    }

    private static func packHalf4(_ vector: SIMD4<Float>) -> PackedHalf4 {
        PackedHalf4(x: Float16(vector.x),
                    y: Float16(vector.y),
                    z: Float16(vector.z),
                    w: Float16(vector.w))
    }

    private static func makeFallbackEnvironmentMap(device: MTLDevice) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba8Unorm,
                                                                    size: 1,
                                                                    mipmapped: false)
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            fatalError("Unable to create fallback environment map texture")
        }
        texture.label = "SplatRendererFallbackEnvironment"

        var pixel: [UInt8] = [0, 0, 0, 0]
        let region = MTLRegionMake2D(0, 0, 1, 1)
        pixel.withUnsafeBytes { bytes in
            for slice in 0..<6 {
                texture.replace(region: region,
                                mipmapLevel: 0,
                                slice: slice,
                                withBytes: bytes.baseAddress!,
                                bytesPerRow: bytes.count,
                                bytesPerImage: bytes.count)
            }
        }

        return texture
    }

    private static func makeFallbackBRDFLUT(device: MTLDevice) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg8Unorm,
                                                                  width: 1,
                                                                  height: 1,
                                                                  mipmapped: false)
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            fatalError("Unable to create fallback BRDF LUT texture")
        }
        texture.label = "SplatRendererFallbackBRDFLUT"

        var pixel: [UInt8] = [0, 0]
        let region = MTLRegionMake2D(0, 0, 1, 1)
        pixel.withUnsafeBytes { bytes in
            texture.replace(region: region,
                            mipmapLevel: 0,
                            withBytes: bytes.baseAddress!,
                            bytesPerRow: bytes.count)
        }

        return texture
    }

    private static func makeMaterialSampler(device: MTLDevice, label: String) -> MTLSamplerState {
        let descriptor = MTLSamplerDescriptor()
        descriptor.label = label
        descriptor.minFilter = .linear
        descriptor.magFilter = .linear
        descriptor.mipFilter = .notMipmapped
        descriptor.sAddressMode = .clampToEdge
        descriptor.tAddressMode = .clampToEdge
        descriptor.rAddressMode = .clampToEdge
        descriptor.normalizedCoordinates = true
        guard let sampler = device.makeSamplerState(descriptor: descriptor) else {
            fatalError("Unable to create sampler state: \(label)")
        }
        return sampler
    }

    public struct ViewportDescriptor {
        public var viewport: MTLViewport
        public var projectionMatrix: simd_float4x4
        public var viewMatrix: simd_float4x4
        public var screenSize: SIMD2<Int>

        public init(viewport: MTLViewport, projectionMatrix: simd_float4x4, viewMatrix: simd_float4x4, screenSize: SIMD2<Int>) {
            self.viewport = viewport
            self.projectionMatrix = projectionMatrix
            self.viewMatrix = viewMatrix
            self.screenSize = screenSize
        }
    }

    // Keep in sync with Shaders.metal : BufferIndex
    enum BufferIndex: NSInteger {
        case uniforms = 0
        case splat    = 1
    }

    // Keep in sync with Shaders.metal : TextureIndex
    enum TextureIndex: NSInteger {
        case environment = 0
        case brdf        = 1
    }

    // Keep in sync with Shaders.metal : SamplerIndex
    enum SamplerIndex: NSInteger {
        case environment = 0
        case brdf        = 1
    }

    // Keep in sync with Shaders.metal : Uniforms
    struct Uniforms {
        var projectionMatrix: matrix_float4x4
        var viewMatrix: matrix_float4x4
        var screenSize: SIMD2<UInt32> // Size of screen in pixels
        var screenPadding: SIMD2<UInt32> = .zero
        var cameraPosition: SIMD4<Float>

        var splatCount: UInt32
        var indexedSplatCount: UInt32
        var paddingCounts: SIMD2<UInt32> = .zero
    }

    // Keep in sync with Shaders.metal : UniformsArray
    struct UniformsArray {
        // maxViewCount = 2, so we have 2 entries
        var uniforms0: Uniforms
        var uniforms1: Uniforms

        // The 256 byte aligned size of our uniform structure
        static var alignedSize: Int { (MemoryLayout<UniformsArray>.size + 0xFF) & -0x100 }

        mutating func setUniforms(index: Int, _ uniforms: Uniforms) {
            switch index {
            case 0: uniforms0 = uniforms
            case 1: uniforms1 = uniforms
            default: break
            }
        }
    }

    struct PackedHalf3 {
        var x: Float16
        var y: Float16
        var z: Float16
    }

    struct PackedHalf4 {
        var x: Float16
        var y: Float16
        var z: Float16
        var w: Float16
    }

    struct PackedRGBHalf4 {
        var r: Float16
        var g: Float16
        var b: Float16
        var a: Float16
    }

    // Keep in sync with Shaders.metal : Splat
    struct Splat {
        var position: MTLPackedFloat3
        var color: PackedRGBHalf4
        var covA: PackedHalf3
        var covB: PackedHalf3
        var albedo: PackedHalf3
        var metallic: Float16
        var roughness: Float16
        var normal: PackedHalf3
        var rotation: PackedHalf4
    }

    struct SplatIndexAndDepth {
        var index: UInt32
        var depth: Float
    }

    public let device: MTLDevice
    public let colorFormat: MTLPixelFormat
    public let depthFormat: MTLPixelFormat
    public let sampleCount: Int
    public let maxViewCount: Int
    public let maxSimultaneousRenders: Int

    /**
     High-quality depth takes longer, but results in a continuous, more-representative depth buffer result, which is useful for reducing artifacts during Vision Pro's frame reprojection.
     */
    public var highQualityDepth: Bool = true

    public var debugViewMode: DebugViewMode = .albedo {
        didSet {
            guard oldValue != debugViewMode else { return }
            resetPipelineStates()
        }
    }

    private var writeDepth: Bool {
        depthFormat != .invalid
    }

    /**
     The SplatRenderer has two shader pipelines.
     - The single stage has a vertex shader, and a fragment shader. It can produce depth (or not), but the depth it produces is the depth of the nearest splat, whether it's visible or now.
     - The multi-stage pipeline uses a set of shaders which communicate using imageblock tile memory: initialization (which clears the tile memory), draw splats (similar to the single-stage
     pipeline but the end result is tile memory, not color+depth), and a post-process stage which merely copies the tile memory (color and optionally depth) to the frame's buffers.
     This is neccessary so that the primary stage can do its own blending -- of both color and depth -- by reading the previous values and writing new ones, which isn't possible without tile
     memory. Color blending works the same as the hardcoded path, but depth blending uses color alpha and results in mostly-transparent splats contributing only slightly to the depth,
     resulting in a much more continuous and representative depth value, which is important for reprojection on Vision Pro.
     */
    private var useMultiStagePipeline: Bool {
#if targetEnvironment(simulator)
        false
#else
        writeDepth && highQualityDepth
#endif
    }

    public var clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)

    public var onSortStart: (() -> Void)?
    public var onSortComplete: ((TimeInterval) -> Void)?

    private let library: MTLLibrary
    // Single-stage pipeline
    private var singleStagePipelineState: MTLRenderPipelineState?
    private var singleStageDepthState: MTLDepthStencilState?
    // Multi-stage pipeline
    private var initializePipelineState: MTLRenderPipelineState?
    private var drawSplatPipelineState: MTLRenderPipelineState?
    private var drawSplatDepthState: MTLDepthStencilState?
    private var postprocessPipelineState: MTLRenderPipelineState?
    private var postprocessDepthState: MTLDepthStencilState?

    private let fallbackEnvironmentMap: MTLTexture
    private let fallbackBRDFLUT: MTLTexture
    private let environmentSamplerState: MTLSamplerState
    private let brdfSamplerState: MTLSamplerState
    private var environmentMapTexture: MTLTexture?
    private var brdfLookupTexture: MTLTexture?

    // dynamicUniformBuffers contains maxSimultaneousRenders uniforms buffers,
    // which we round-robin through, one per render; this is managed by switchToNextDynamicBuffer.
    // uniforms = the i'th buffer (where i = uniformBufferIndex, which varies from 0 to maxSimultaneousRenders-1)
    var dynamicUniformBuffers: MTLBuffer
    var uniformBufferOffset = 0
    var uniformBufferIndex = 0
    var uniforms: UnsafeMutablePointer<UniformsArray>

    // cameraWorldPosition and Forward vectors are the latest mean camera position across all viewports
    var cameraWorldPosition: SIMD3<Float> = .zero
    var cameraWorldForward: SIMD3<Float> = .init(x: 0, y: 0, z: -1)

    typealias IndexType = UInt32
    // splatBuffer contains one entry for each gaussian splat
    var splatBuffer: MetalBuffer<Splat>
    // splatBufferPrime is a copy of splatBuffer, which is not currenly in use for rendering.
    // We use this for sorting, and when we're done, swap it with splatBuffer.
    // There's a good chance that we'll sometimes end up sorting a splatBuffer still in use for
    // rendering.
    // TODO: Replace this with a more robust multiple-buffer scheme to guarantee we're never actively sorting a buffer still in use for rendering
    var splatBufferPrime: MetalBuffer<Splat>

    var indexBuffer: MetalBuffer<UInt32>

    public var splatCount: Int { splatBuffer.count }

    var sorting = false
    var orderAndDepthTempSort: [SplatIndexAndDepth] = []

    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int,
                maxSimultaneousRenders: Int) throws {
#if arch(x86_64)
        fatalError("MetalSplatter is unsupported on Intel architecture (x86_64)")
#endif

        self.device = device

        self.colorFormat = colorFormat
        self.depthFormat = depthFormat
        self.sampleCount = sampleCount
        self.maxViewCount = min(maxViewCount, Constants.maxViewCount)
        self.maxSimultaneousRenders = maxSimultaneousRenders

        self.fallbackEnvironmentMap = Self.makeFallbackEnvironmentMap(device: device)
        self.fallbackBRDFLUT = Self.makeFallbackBRDFLUT(device: device)
        self.environmentSamplerState = Self.makeMaterialSampler(device: device, label: "SplatRendererEnvironmentSampler")
        self.brdfSamplerState = Self.makeMaterialSampler(device: device, label: "SplatRendererBRDFSampler")
        self.environmentMapTexture = nil
        self.brdfLookupTexture = nil

        let dynamicUniformBuffersSize = UniformsArray.alignedSize * maxSimultaneousRenders
        self.dynamicUniformBuffers = device.makeBuffer(length: dynamicUniformBuffersSize,
                                                       options: .storageModeShared)!
        self.dynamicUniformBuffers.label = "Uniform Buffers"
        self.uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents()).bindMemory(to: UniformsArray.self, capacity: 1)

        self.splatBuffer = try MetalBuffer(device: device)
        self.splatBufferPrime = try MetalBuffer(device: device)
        self.indexBuffer = try MetalBuffer(device: device)

#if !arch(x86_64)
        precondition(MemoryLayout<Splat>.stride == MemoryLayout<Splat>.size,
                     "SplatRenderer.Splat contains unexpected padding; verify ShaderCommon.Splat matches Swift layout")
#endif

        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            fatalError("Unable to initialize SplatRenderer: \(error)")
        }
    }

    public func reset() {
        splatBuffer.count = 0
        try? splatBuffer.setCapacity(0)
    }

    public func setEnvironmentMap(_ texture: MTLTexture?) throws {
        guard let texture else {
            environmentMapTexture = nil
            return
        }

        guard texture.device === device else {
            throw Error.invalidMaterialResource(resource: "environmentMap",
                                                reason: "Texture must be created with the renderer device")
        }

        guard texture.textureType == .typeCube else {
            throw Error.invalidMaterialResource(resource: "environmentMap",
                                                reason: "Expected cube texture, received \(texture.textureType)")
        }

        guard texture.usage.contains(.shaderRead) else {
            throw Error.invalidMaterialResource(resource: "environmentMap",
                                                reason: "Texture must include shaderRead usage")
        }

        guard texture.width > 0 else {
            throw Error.invalidMaterialResource(resource: "environmentMap",
                                                reason: "Texture must have non-zero dimensions")
        }

        environmentMapTexture = texture
    }

    public func setBRDFLookupTexture(_ texture: MTLTexture?) throws {
        guard let texture else {
            brdfLookupTexture = nil
            return
        }

        guard texture.device === device else {
            throw Error.invalidMaterialResource(resource: "brdfLUT",
                                                reason: "Texture must be created with the renderer device")
        }

        guard texture.textureType == .type2D else {
            throw Error.invalidMaterialResource(resource: "brdfLUT",
                                                reason: "Expected 2D texture, received \(texture.textureType)")
        }

        guard texture.usage.contains(.shaderRead) else {
            throw Error.invalidMaterialResource(resource: "brdfLUT",
                                                reason: "Texture must include shaderRead usage")
        }

        guard texture.width > 0 && texture.height > 0 else {
            throw Error.invalidMaterialResource(resource: "brdfLUT",
                                                reason: "Texture must have non-zero dimensions")
        }

        brdfLookupTexture = texture
    }

    public func read(from url: URL) async throws {
        var newPoints = SplatMemoryBuffer()
        try await newPoints.read(from: try AutodetectSceneReader(url))
        try add(newPoints.points)
    }

    private func resetPipelineStates() {
        singleStagePipelineState = nil
        initializePipelineState = nil
        drawSplatPipelineState = nil
        drawSplatDepthState = nil
        postprocessPipelineState = nil
        postprocessDepthState = nil
    }

    private func makeDebugViewFunction(named name: String) -> MTLFunction {
        var value = UInt32(debugViewMode.rawValue)
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&value, type: .uint, index: 0)
        return library.makeRequiredFunction(name: name, constantValues: constants)
    }

    private func buildSingleStagePipelineStatesIfNeeded() throws {
        guard singleStagePipelineState == nil else { return }

        singleStagePipelineState = try buildSingleStagePipelineState()
        singleStageDepthState = try buildSingleStageDepthState()
    }

    private func buildMultiStagePipelineStatesIfNeeded() throws {
        guard initializePipelineState == nil else { return }

        initializePipelineState = try buildInitializePipelineState()
        drawSplatPipelineState = try buildDrawSplatPipelineState()
        drawSplatDepthState = try buildDrawSplatDepthState()
        postprocessPipelineState = try buildPostprocessPipelineState()
        postprocessDepthState = try buildPostprocessDepthState()
    }

    private func buildSingleStagePipelineState() throws -> MTLRenderPipelineState {
        assert(!useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "SingleStagePipeline"
        pipelineDescriptor.vertexFunction = library.makeRequiredFunction(name: "singleStageSplatVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeRequiredFunction(name: "singleStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.pixelFormat = colorFormat
        colorAttachment.isBlendingEnabled = true
        colorAttachment.rgbBlendOperation = .add
        colorAttachment.alphaBlendOperation = .add
        colorAttachment.sourceRGBBlendFactor = .one
        colorAttachment.sourceAlphaBlendFactor = .one
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0] = colorAttachment

        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildSingleStageDepthState() throws -> MTLDepthStencilState {
        assert(!useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    private func buildInitializePipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLTileRenderPipelineDescriptor()

        pipelineDescriptor.label = "InitializePipeline"
        pipelineDescriptor.tileFunction = library.makeRequiredFunction(name: "initializeFragmentStore")
        pipelineDescriptor.threadgroupSizeMatchesTileSize = true;
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat

        return try device.makeRenderPipelineState(tileDescriptor: pipelineDescriptor, options: [], reflection: nil)
    }

    private func buildDrawSplatPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "DrawSplatPipeline"
        pipelineDescriptor.vertexFunction = library.makeRequiredFunction(name: "multiStageSplatVertexShader")
        pipelineDescriptor.fragmentFunction = makeDebugViewFunction(named: "multiStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildDrawSplatDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    private func buildPostprocessPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "PostprocessPipeline"
        pipelineDescriptor.vertexFunction =
            library.makeRequiredFunction(name: "postprocessVertexShader")
        pipelineDescriptor.fragmentFunction =
            writeDepth
            ? makeDebugViewFunction(named: "postprocessFragmentShader")
            : makeDebugViewFunction(named: "postprocessFragmentShaderNoDepth")

        pipelineDescriptor.colorAttachments[0]!.pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildPostprocessDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    public func ensureAdditionalCapacity(_ pointCount: Int) throws {
        try splatBuffer.ensureCapacity(splatBuffer.count + pointCount)
    }

    public func add(_ points: [SplatScenePoint]) throws {
        do {
            try ensureAdditionalCapacity(points.count)
        } catch {
            Self.log.error("Failed to grow buffers for \(points.count, privacy: .public) points: \(String(describing: error), privacy: .public)")
            throw error
        }

        let startIndex = splatBuffer.count
        for (offset, point) in points.enumerated() {
            let index = startIndex + offset
            splatBuffer.append(Splat(point, index: index))
        }
    }

    public func add(_ point: SplatScenePoint) throws {
        try add([ point ])
    }

    private enum TelemetryConstants {
        static let fallbackEscalationFrameThreshold: UInt32 = 120
    }

    public struct MaterialFallbackTelemetry {
        public var environmentFallbackBindings: UInt32
        public var brdfFallbackBindings: UInt32

        public init(environmentFallbackBindings: UInt32 = 0,
                    brdfFallbackBindings: UInt32 = 0) {
            self.environmentFallbackBindings = environmentFallbackBindings
            self.brdfFallbackBindings = brdfFallbackBindings
        }
    }

    public private(set) var materialFallbackTelemetry = MaterialFallbackTelemetry()

    public var automaticallyLogsFallbackTelemetry = true

    private var didLogEnvironmentFallbackThisFrame = false
    private var didLogBRDFFallbackThisFrame = false
    private var fallbackFrameStreak: UInt32 = 0
    private var didEscalateFallback = false

    private func beginTelemetryFrame() {
        materialFallbackTelemetry = MaterialFallbackTelemetry()
        didLogEnvironmentFallbackThisFrame = false
        didLogBRDFFallbackThisFrame = false
    }

    private func switchToNextDynamicBuffer() {
        uniformBufferIndex = (uniformBufferIndex + 1) % maxSimultaneousRenders
        uniformBufferOffset = UniformsArray.alignedSize * uniformBufferIndex
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents() + uniformBufferOffset).bindMemory(to: UniformsArray.self, capacity: 1)
    }

    private func bindMaterialResources(to renderEncoder: MTLRenderCommandEncoder) {
        let environmentTexture: MTLTexture
        if let environmentMapTexture {
            environmentTexture = environmentMapTexture
        } else {
            environmentTexture = fallbackEnvironmentMap
            materialFallbackTelemetry.environmentFallbackBindings &+= 1
            if !didLogEnvironmentFallbackThisFrame {
                didLogEnvironmentFallbackThisFrame = true
                Self.log.notice("Binding fallback environment map texture")
#if canImport(os.signpost)
                Self.signposter.emitEvent("FallbackEnvironmentMapBound")
#endif
            }
        }

        let brdfTexture: MTLTexture
        if let brdfLookupTexture {
            brdfTexture = brdfLookupTexture
        } else {
            brdfTexture = fallbackBRDFLUT
            materialFallbackTelemetry.brdfFallbackBindings &+= 1
            if !didLogBRDFFallbackThisFrame {
                didLogBRDFFallbackThisFrame = true
                Self.log.notice("Binding fallback BRDF lookup texture")
#if canImport(os.signpost)
                Self.signposter.emitEvent("FallbackBRDFLookupBound")
#endif
            }
        }

        // Unconditional detailed logging of the textures we are about to bind (helps verify content/type)
        let envIdx = TextureIndex.environment.rawValue
        let brdfIdx = TextureIndex.brdf.rawValue
        func describe(_ tex: MTLTexture?) -> String {
            guard let t = tex else { return "nil" }
            return "label=\(t.label ?? "<none>") type=\(t.textureType.rawValue) fmt=\(t.pixelFormat.rawValue) size=\(t.width)x\(t.height)x\(t.depth) array=\(t.arrayLength) mips=\(t.mipmapLevelCount) storage=\(t.storageMode.rawValue) usage=\(t.usage.rawValue)"
        }
        Self.log.debug("[bindMaterialResources] env slot=\(envIdx) tex { \(describe(environmentTexture)) }")
        Self.log.debug("[bindMaterialResources] brdf slot=\(brdfIdx) tex { \(describe(brdfTexture)) }")

        // Log fragment binding indices to verify shader <-> Swift agreement
        Self.log.debug("Binding env cube at frag slot \(TextureIndex.environment.rawValue), BRDF LUT at slot \(TextureIndex.brdf.rawValue)")
        renderEncoder.setFragmentTexture(environmentTexture, index: TextureIndex.environment.rawValue)
        Self.log.debug("[bindMaterialResources] setFragmentTexture env at slot \(envIdx)")
        renderEncoder.setFragmentTexture(brdfTexture, index: TextureIndex.brdf.rawValue)
        Self.log.debug("[bindMaterialResources] setFragmentTexture brdf at slot \(brdfIdx)")
        renderEncoder.setFragmentSamplerState(environmentSamplerState, index: SamplerIndex.environment.rawValue)
        renderEncoder.setFragmentSamplerState(brdfSamplerState, index: SamplerIndex.brdf.rawValue)
    }

    // Moved out to class scope: probe a single texel from the environment cube to verify non-zero content.
    // Copies face 0, mip 0, texel (0,0) into a 1x1 shared staging texture and logs whether it's non-zero.
    private func debugProbeEnvironmentTexel(commandBuffer: MTLCommandBuffer) {
        guard let env = environmentMapTexture else {
            Self.log.warning("[EnvProbe] environmentMapTexture is nil")
            return
        }
        // Only support common float formats here
        switch env.pixelFormat {
        case .rgba16Float, .rgba32Float, .rg16Float, .rg32Float:
            break
        default:
            Self.log.debug("[EnvProbe] Skipping probe for unsupported pixel format \(env.pixelFormat.rawValue)")
            return
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: env.pixelFormat, width: 1, height: 1, mipmapped: false)
        desc.storageMode = .shared
        // No .blit usage; blit encoders can still copy to/from textures.
        desc.usage = [.shaderRead, .shaderWrite]
        guard let staging = device.makeTexture(descriptor: desc) else {
            Self.log.error("[EnvProbe] Failed to create staging texture")
            return
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            Self.log.error("[EnvProbe] Failed to create blit encoder")
            return
        }
        let origin = MTLOrigin(x: 0, y: 0, z: 0)
        let size = MTLSize(width: 1, height: 1, depth: 1)
        // Probe face 0 (positive X) level 0
        blit.copy(from: env,
                  sourceSlice: 0,
                  sourceLevel: 0,
                  sourceOrigin: origin,
                  sourceSize: size,
                  to: staging,
                  destinationSlice: 0,
                  destinationLevel: 0,
                  destinationOrigin: origin)
        blit.endEncoding()
        commandBuffer.addCompletedHandler { _ in
            let bytesPerPixel: Int
            switch staging.pixelFormat {
            case .rgba16Float: bytesPerPixel = 8
            case .rg16Float:   bytesPerPixel = 4
            case .rgba32Float: bytesPerPixel = 16
            case .rg32Float:   bytesPerPixel = 8
            default:           bytesPerPixel = 8
            }
            var storage = [UInt8](repeating: 0, count: bytesPerPixel)
            storage.withUnsafeMutableBytes { ptr in
                staging.getBytes(ptr.baseAddress!, bytesPerRow: bytesPerPixel,
                                 from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0)
            }
            let nonZero = storage.contains { $0 != 0 }
            if nonZero {
                Self.log.debug("[EnvProbe] Face0 LOD0 texel (0,0) appears NON-zero: \(storage)")
            } else {
                Self.log.warning("[EnvProbe] Face0 LOD0 texel (0,0) is all zeros")
            }
        }
    }

    private func updateUniforms(forViewports viewports: [ViewportDescriptor],
                                splatCount: UInt32,
                                indexedSplatCount: UInt32) {
        for (i, viewport) in viewports.enumerated() where i <= maxViewCount {
            let cameraPosition = Self.cameraWorldPosition(forViewMatrix: viewport.viewMatrix)
            let uniforms = Uniforms(projectionMatrix: viewport.projectionMatrix,
                                    viewMatrix: viewport.viewMatrix,
                                    screenSize: SIMD2(x: UInt32(viewport.screenSize.x), y: UInt32(viewport.screenSize.y)),
                                    cameraPosition: SIMD4<Float>(cameraPosition, 1),
                                    splatCount: splatCount,
                                    indexedSplatCount: indexedSplatCount)
            self.uniforms.pointee.setUniforms(index: i, uniforms)
        }

        cameraWorldPosition = viewports.map { Self.cameraWorldPosition(forViewMatrix: $0.viewMatrix) }.mean ?? .zero
        cameraWorldForward = viewports.map { Self.cameraWorldForward(forViewMatrix: $0.viewMatrix) }.mean?.normalized ?? .init(x: 0, y: 0, z: -1)

        if !sorting {
            resort()
        }
    }

    private static func cameraWorldForward(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: -1, w: 0)).xyz
    }

    private static func cameraWorldPosition(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: 0, w: 1)).xyz
    }

    func renderEncoder(multiStage: Bool,
                       viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       for commandBuffer: MTLCommandBuffer) -> MTLRenderCommandEncoder {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = clearColor
        if let depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = 0.0
        }
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength

        renderPassDescriptor.tileWidth  = Constants.tileSize.width
        renderPassDescriptor.tileHeight = Constants.tileSize.height

        if multiStage {
            if let initializePipelineState {
                renderPassDescriptor.imageblockSampleLength = initializePipelineState.imageblockSampleLength
            } else {
                Self.log.error("initializePipeline == nil in renderEncoder()")
            }
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.setViewports(viewports.map(\.viewport))

        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        return renderEncoder
    }

    public func render(viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        beginTelemetryFrame()

        let splatCount = splatBuffer.count
        guard splatBuffer.count != 0 else { return }
        let indexedSplatCount = min(splatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (splatCount + indexedSplatCount - 1) / indexedSplatCount

        switchToNextDynamicBuffer()
        updateUniforms(forViewports: viewports, splatCount: UInt32(splatCount), indexedSplatCount: UInt32(indexedSplatCount))

        let multiStage = useMultiStagePipeline
        if multiStage {
            try buildMultiStagePipelineStatesIfNeeded()
        } else {
            try buildSingleStagePipelineStatesIfNeeded()
        }

        let renderEncoder = renderEncoder(multiStage: multiStage,
                                          viewports: viewports,
                                          colorTexture: colorTexture,
                                          colorStoreAction: colorStoreAction,
                                          depthTexture: depthTexture,
                                          rasterizationRateMap: rasterizationRateMap,
                                          renderTargetArrayLength: renderTargetArrayLength,
                                          for: commandBuffer)

        let indexCount = indexedSplatCount * 6
        if indexBuffer.count < indexCount {
            do {
                try indexBuffer.ensureCapacity(indexCount)
            } catch {
                return
            }
            indexBuffer.count = indexCount
            for i in 0..<indexedSplatCount {
                indexBuffer.values[i * 6 + 0] = UInt32(i * 4 + 0)
                indexBuffer.values[i * 6 + 1] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 2] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 3] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 4] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 5] = UInt32(i * 4 + 3)
            }
        }

        if multiStage {
            guard let initializePipelineState,
                  let drawSplatPipelineState
            else { return }

            renderEncoder.pushDebugGroup("Initialize")
            renderEncoder.setRenderPipelineState(initializePipelineState)
            renderEncoder.dispatchThreadsPerTile(Constants.tileSize)
            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(drawSplatPipelineState)
            renderEncoder.setDepthStencilState(drawSplatDepthState)
            bindMaterialResources(to: renderEncoder)
        } else {
            guard let singleStagePipelineState
            else { return }

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(singleStagePipelineState)
            renderEncoder.setDepthStencilState(singleStageDepthState)
            bindMaterialResources(to: renderEncoder)
        }

        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)

        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                            indexCount: indexCount,
                                            indexType: .uint32,
                                            indexBuffer: indexBuffer.buffer,
                                            indexBufferOffset: 0,
                                            instanceCount: instanceCount)

        if multiStage {
            guard let postprocessPipelineState
            else { return }

            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Postprocess")
            renderEncoder.setRenderPipelineState(postprocessPipelineState)
            renderEncoder.setDepthStencilState(postprocessDepthState)
            renderEncoder.setCullMode(.none)
            Self.log.debug("Postprocess: binding material resources before draw")
            bindMaterialResources(to: renderEncoder)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renderEncoder.popDebugGroup()
        } else {
            renderEncoder.popDebugGroup()
        }

        renderEncoder.endEncoding()

        // Probe one env texel to verify content (must occur outside any active encoder)
        debugProbeEnvironmentTexel(commandBuffer: commandBuffer)

        emitFallbackTelemetryIfNeeded()
    }

    private func emitFallbackTelemetryIfNeeded() {
        guard automaticallyLogsFallbackTelemetry else { return }

        let telemetry = materialFallbackTelemetry
        let usedEnvironmentFallback = telemetry.environmentFallbackBindings > 0
        let usedBRDFFallback = telemetry.brdfFallbackBindings > 0

        guard usedEnvironmentFallback || usedBRDFFallback else {
            guard fallbackFrameStreak != 0 else { return }
            Self.log.info("Fallback material bindings resolved after \(self.self.fallbackFrameStreak) frame(s)")
            fallbackFrameStreak = 0
            didEscalateFallback = false
            return
        }

        fallbackFrameStreak &+= 1

        var reasons: [String] = []
        if usedEnvironmentFallback {
            if environmentMapTexture == nil {
                reasons.append("environmentMapTexture was nil")
            } else {
                reasons.append("environment map texture was invalid")
            }
        }
        if usedBRDFFallback {
            if brdfLookupTexture == nil {
                reasons.append("brdfLookupTexture was nil")
            } else {
                reasons.append("BRDF lookup texture was invalid")
            }
        }

        let reasonSummary: String
        if reasons.isEmpty {
            reasonSummary = "material resources were missing or invalid"
        } else {
            reasonSummary = reasons.joined(separator: "; ")
        }
        Self.log.warning("Renderer bound fallback material resources this frame (environment: \(telemetry.environmentFallbackBindings), brdf: \(telemetry.brdfFallbackBindings)). Reason: \(reasonSummary). Provide valid textures using setEnvironmentMap(_:) and setBRDFLookupTexture(_:).")

        if fallbackFrameStreak >= TelemetryConstants.fallbackEscalationFrameThreshold && !didEscalateFallback {
            didEscalateFallback = true
            Self.log.error("Fallback material resources persisted for \(self.self.fallbackFrameStreak) consecutive frames")
#if DEBUG
            assertionFailure("Fallback material resources persisted for \(fallbackFrameStreak) consecutive frames")
#endif
        }
    }

    // Sort splatBuffer (read-only), storing the results in splatBuffer (write-only) then swap splatBuffer and splatBufferPrime
    public func resort() {
        guard !sorting else { return }
        sorting = true
        onSortStart?()
        let sortStartTime = Date()

        let splatCount = splatBuffer.count

        let cameraWorldForward = cameraWorldForward
        let cameraWorldPosition = cameraWorldPosition

        Task(priority: .high) {
            defer {
                sorting = false
                onSortComplete?(-sortStartTime.timeIntervalSinceNow)
            }

            if orderAndDepthTempSort.count != splatCount {
                orderAndDepthTempSort = Array(repeating: SplatIndexAndDepth(index: .max, depth: 0), count: splatCount)
            }

            if Constants.sortByDistance {
                for i in 0..<splatCount {
                    orderAndDepthTempSort[i].index = UInt32(i)
                    let splatPosition = splatBuffer.values[i].position.simd
                    orderAndDepthTempSort[i].depth = (splatPosition - cameraWorldPosition).lengthSquared
                }
            } else {
                for i in 0..<splatCount {
                    orderAndDepthTempSort[i].index = UInt32(i)
                    let splatPosition = splatBuffer.values[i].position.simd
                    orderAndDepthTempSort[i].depth = dot(splatPosition, cameraWorldForward)
                }
            }

            orderAndDepthTempSort.sort { $0.depth > $1.depth }

            do {
                try splatBufferPrime.setCapacity(splatCount)
                splatBufferPrime.count = 0
                for newIndex in 0..<orderAndDepthTempSort.count {
                    let oldIndex = Int(orderAndDepthTempSort[newIndex].index)
                    splatBufferPrime.append(splatBuffer, fromIndex: oldIndex)
                }

                swap(&splatBuffer, &splatBufferPrime)
            } catch {
                Self.log.error("Failed to resort splats: \(String(describing: error), privacy: .public)")
            }
        }
    }
}

extension SplatRenderer.Splat {
    init(_ splat: SplatScenePoint, index: Int) {
        let normalized = splat.linearNormalized
        let color = SIMD4<Float>(normalized.color.asLinearFloat.sRGBToLinear, normalized.opacity.asLinearFloat)
        self.init(position: normalized.position,
                  color: color,
                  scale: normalized.scale.asLinearFloat,
                  rotation: normalized.rotation.normalized,
                  albedo: normalized.albedo,
                  metallic: normalized.metallic,
                  roughness: normalized.roughness,
                  normal: normalized.normal,
                  pointIndex: index)
    }

    init(position: SIMD3<Float>,
         color: SIMD4<Float>,
         scale: SIMD3<Float>,
         rotation: simd_quatf,
         albedo: SIMD3<Float>,
         metallic: Float,
         roughness: Float,
         normal: SIMD3<Float>,
         pointIndex: Int) {
        let sanitizedRotation = SplatRenderer.sanitizeQuaternion(rotation, pointIndex: pointIndex)
        let transform = simd_float3x3(sanitizedRotation) * simd_float3x3(diagonal: scale)
        let cov3D = transform * transform.transpose

        let sanitizedColor = SplatRenderer.sanitizeColor(color, pointIndex: pointIndex)
        let sanitizedAlbedo = SplatRenderer.sanitizeAlbedo(albedo, pointIndex: pointIndex)
        let sanitizedMetallic = SplatRenderer.sanitizeUnitScalar(metallic,
                                                                 defaultValue: SplatScenePoint.defaultMetallic,
                                                                 field: "metallic",
                                                                 pointIndex: pointIndex)
        let sanitizedRoughness = SplatRenderer.sanitizeUnitScalar(roughness,
                                                                  defaultValue: SplatScenePoint.defaultRoughness,
                                                                  field: "roughness",
                                                                  pointIndex: pointIndex)
        let sanitizedNormal = SplatRenderer.sanitizeNormal(normal, pointIndex: pointIndex)

        let covA = SIMD3<Float>(cov3D[0, 0], cov3D[0, 1], cov3D[0, 2])
        let covB = SIMD3<Float>(cov3D[1, 1], cov3D[1, 2], cov3D[2, 2])
        let sanitizedCovA = SplatRenderer.sanitizeVector(covA,
                                                         defaultValue: SIMD3<Float>(repeating: 0),
                                                         field: "covA",
                                                         pointIndex: pointIndex)
        let sanitizedCovB = SplatRenderer.sanitizeVector(covB,
                                                         defaultValue: SIMD3<Float>(repeating: 0),
                                                         field: "covB",
                                                         pointIndex: pointIndex)

        self.position = MTLPackedFloat3Make(position.x, position.y, position.z)
        self.color = SplatRenderer.PackedRGBHalf4(r: Float16(sanitizedColor.x),
                                                  g: Float16(sanitizedColor.y),
                                                  b: Float16(sanitizedColor.z),
                                                  a: Float16(sanitizedColor.w))
        self.covA = SplatRenderer.packHalf3(sanitizedCovA)
        self.covB = SplatRenderer.packHalf3(sanitizedCovB)
        self.albedo = SplatRenderer.packHalf3(sanitizedAlbedo)
        self.metallic = Float16(sanitizedMetallic)
        self.roughness = Float16(sanitizedRoughness)
        self.normal = SplatRenderer.packHalf3(sanitizedNormal)
        self.rotation = SplatRenderer.packHalf4(sanitizedRotation.vector)
    }
}

protocol MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { get }
}

extension UInt32: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint32 }
}
extension UInt16: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint16 }
}

extension Array where Element == SIMD3<Float> {
    var mean: SIMD3<Float>? {
        guard !isEmpty else { return nil }
        return reduce(.zero, +) / Float(count)
    }
}

private extension MTLPackedFloat3 {
    var simd: SIMD3<Float> {
        SIMD3(x: x, y: y, z: z)
    }
}

private extension SIMD3 where Scalar: BinaryFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
    var normalized: SIMD3<Scalar> {
        self / Scalar(sqrt(lengthSquared))
    }

    var lengthSquared: Scalar {
        x*x + y*y + z*z
    }

    func vector4(w: Scalar) -> SIMD4<Scalar> {
        SIMD4<Scalar>(x: x, y: y, z: z, w: w)
    }

    static func random(in range: Range<Scalar>) -> SIMD3<Scalar> {
        Self(x: Scalar.random(in: range), y: .random(in: range), z: .random(in: range))
    }
}

private extension SIMD3<Float> {
    var sRGBToLinear: SIMD3<Float> {
        SIMD3(x: pow(x, 2.2), y: pow(y, 2.2), z: pow(z, 2.2))
    }
}

private extension SIMD4 where Scalar: BinaryFloatingPoint {
    var xyz: SIMD3<Scalar> {
        .init(x: x, y: y, z: z)
    }
}

private extension MTLLibrary {
    func makeRequiredFunction(name: String) -> MTLFunction {
        guard let result = makeFunction(name: name) else {
            fatalError("Unable to load required shader function: \"\(name)\"")
        }
        return result
    }

    func makeRequiredFunction(name: String, constantValues: MTLFunctionConstantValues) -> MTLFunction {
        do {
            return try makeFunction(name: name, constantValues: constantValues)
        } catch {
            fatalError("Unable to load required shader function: \"\(name)\" with constants: \(error)")
        }
    }
}
