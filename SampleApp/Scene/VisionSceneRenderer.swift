#if os(visionOS)

import ARKit
import CompositorServices
import Foundation
import Metal
import MetalSplatter
import os
import SampleBoxRenderer
import simd
import Spatial
import SwiftUI

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

class VisionSceneRenderer {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "VisionSceneRenderer")
    private static let iso8601Formatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private enum ProbeGuardReason: Equatable {
        case noSnapshot
        case staleRevision(UInt64)
        case prefilterUnavailable(UInt64)
    }

    private enum EnvironmentBindingSkipReason: Equatable {
        case rendererUnavailable
        case resultMissing
        case notDirty
    }

    private enum TelemetryConstants {
        static let fallbackEscalationFrameThreshold = 120
    }

    let layerRenderer: LayerRenderer
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    private struct CachedHandState {
        let chirality: HandAnchor.Chirality
        let pinchPosition: SIMD3<Float>?
        let palmPosition: SIMD3<Float>?
        let isPinching: Bool
    }

    private struct ModelInteractionState {
        enum GestureMode {
            case idle
            case singleHandGrab(chirality: HandAnchor.Chirality)
            case twoHandManipulate
        }

        var translation = SIMD3<Float>(0, 0, Constants.modelCenterZ)
        var rotation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        var scale: Float = 1.0
        var gestureMode: GestureMode = .idle

        var singleHandOffset: SIMD3<Float>?
        var twoHandInitialDistance: Float?
        var twoHandInitialVector: SIMD3<Float>?
        var twoHandInitialRotation: simd_quatf?
        var twoHandInitialScale: Float = 1.0
        var twoHandMidpointOffset: SIMD3<Float>?
    }

    private enum InteractionConstants {
        static let pinchThreshold: Float = 0.03
    }

    private var lastRotationUpdateTimestamp: Date? = nil
    private var interactionState = ModelInteractionState()

    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider
    let environmentLightEstimation: EnvironmentLightEstimationProvider
    let handTracking: HandTrackingProvider

    private let handStateLock = NSLock()
    private var cachedHandStates: [HandAnchor.Chirality: CachedHandState] = [:]
    private var handUpdateTask: Task<Void, Never>?

    private let environmentProbeManager: EnvironmentProbeManager
    private let environmentPrefilter: EnvironmentPrefilter?
    private var environmentPrefilterResult: EnvironmentPrefilterResult?
    private var environmentResourcesDirty = false
    private var latestPrefilterRevision: UInt64 = 0
    private var lastAppliedEnvironmentRevision: UInt64 = 0
    private var lastPrefilterDuration: TimeInterval = 0
    private var lastProbeGuardReason: ProbeGuardReason?
    private var lastBindingSkipReason: EnvironmentBindingSkipReason?
    private var fallbackFrameStreak: Int = 0
    private var didEscalateFallback: Bool = false
    // Debug/telemetry extensions
    private var frameCounter: UInt64 = 0
    private var didLogTargetFormats: Bool = false
    private var didAutoCapture: Bool = false
    private var captureNextFrame: Bool = false
    private var lastBrightnessProbeFrame: UInt64 = 0

    init(_ layerRenderer: LayerRenderer) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!

        worldTracking = WorldTrackingProvider()
        environmentLightEstimation = EnvironmentLightEstimationProvider()
        handTracking = HandTrackingProvider()
        arSession = ARKitSession()
        environmentProbeManager = EnvironmentProbeManager(session: arSession,
                                                          worldTracking: worldTracking,
                                                          environmentLightEstimation: environmentLightEstimation,
                                                          device: device)
        do {
            environmentPrefilter = try EnvironmentPrefilter(device: device)
        } catch {
            Self.log.error("Failed to initialize environment prefilter: \(error.localizedDescription)")
            environmentPrefilter = nil
        }

        handUpdateTask = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            for await update in self.handTracking.anchorUpdates {
                let anchor = update.anchor
                self.cacheHandAnchors([anchor])
            }
        }
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        interactionState = ModelInteractionState()

        if let splat = modelRenderer as? SplatRenderer {
            try? splat.setEnvironmentMap(nil)
            try? splat.setBRDFLookupTexture(nil)
        }
        modelRenderer = nil
        switch model {
        case .gaussianSplat(let url):
            let splat = try SplatRenderer(device: device,
                                          colorFormat: layerRenderer.configuration.colorFormat,
                                          depthFormat: layerRenderer.configuration.depthFormat,
                                          sampleCount: 1,
                                          maxViewCount: layerRenderer.properties.viewCount,
                                          maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            try await splat.read(from: url)
            modelRenderer = splat
            if environmentPrefilterResult != nil {
                environmentResourcesDirty = true
                lastAppliedEnvironmentRevision = 0
            }
        case .sampleBox:
            modelRenderer = try! SampleBoxRenderer(device: device,
                                                   colorFormat: layerRenderer.configuration.colorFormat,
                                                   depthFormat: layerRenderer.configuration.depthFormat,
                                                   sampleCount: 1,
                                                   maxViewCount: layerRenderer.properties.viewCount,
                                                   maxSimultaneousRenders: Constants.maxSimultaneousRenders)
        case .none:
            break
        }
    }

    func startRenderLoop() {
        Task {
            do {
                try await arSession.run([worldTracking, environmentLightEstimation, handTracking])
            } catch {
                fatalError("Failed to initialize ARSession")
            }

            let renderThread = Thread {
                self.renderLoop()
            }
            renderThread.name = "Render Thread"
            renderThread.start()
        }
    }

    private func viewports(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor?) -> [ModelRendererViewportDescriptor] {
        let translationMatrix = matrix4x4_translation(interactionState.translation.x,
                                                     interactionState.translation.y,
                                                     interactionState.translation.z)
        let rotationMatrix = matrix_float4x4(interactionState.rotation)
        let scaleMatrix = matrix4x4_scale(interactionState.scale)
        // Turn common 3D GS PLY files rightside-up. This isn't generally meaningful, it just
        // happens to be a useful default for the most common datasets at the moment.
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        let simdDeviceAnchor = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        return drawable.views.enumerated().map { (i, view) in
            let userViewpointMatrix = (simdDeviceAnchor * view.transform).inverse
            // Compute per-view projection:
            // On visionOS 2.0+, use the compositor-provided projection helper for mixed reality.
            // On older SDKs, fall back to constructing from tangents.
            let projMatrixSIMD: simd_float4x4
            if #available(visionOS 2.0, *) {
                // Swift wrapper for the C API cp_drawable_compute_projection.
                // If your SDK uses a different symbol (e.g., on drawable or view), adjust the call name.
                projMatrixSIMD = drawable.computeProjection(viewIndex: i)
            } else {
                // Legacy path for visionOS 1.x
                let legacyProj = ProjectiveTransform3D(
                    leftTangent:  Double(view.tangents[0]),
                    rightTangent: Double(view.tangents[1]),
                    topTangent:   Double(view.tangents[2]),
                    bottomTangent:Double(view.tangents[3]),
                    nearZ:  Double(drawable.depthRange.y),
                    farZ:   Double(drawable.depthRange.x),
                    reverseZ: true
                )
                // Convert ProjectiveTransform3D to simd_float4x4; adjust accessor if your type differs.
                projMatrixSIMD = simd_float4x4(legacyProj)
            }
            let screenSize = SIMD2(x: Int(view.textureMap.viewport.width),
                                   y: Int(view.textureMap.viewport.height))
            // Matrix sanity checks (log only when invalid)
            if !Self.isFinite(projMatrixSIMD) || !Self.isFinite(userViewpointMatrix) {
                Self.log.error("Non-finite matrix detected (proj finite: \(Self.isFinite(projMatrixSIMD)), view finite: \(Self.isFinite(userViewpointMatrix)))")
            }
            return ModelRendererViewportDescriptor(viewport: view.textureMap.viewport,
                                                   projectionMatrix: projMatrixSIMD,
                                                   viewMatrix: userViewpointMatrix * translationMatrix * rotationMatrix * scaleMatrix * commonUpCalibration,
                                                   screenSize: screenSize)
        }
    }

    private func logRenderTargetFormatsOnce(for drawable: LayerRenderer.Drawable) {
        guard !didLogTargetFormats else { return }
        didLogTargetFormats = true
        let cfg = layerRenderer.configuration
        let colorFormat = cfg.colorFormat
        let depthFormat = cfg.depthFormat
        let viewCount = drawable.views.count
        let colorTex = drawable.colorTextures.first
        let depthTex = drawable.depthTextures.first
        // Convert MTLPixelFormat and other enums to String to satisfy os.Logger interpolation
        let colorFormatStr = String(describing: colorFormat)
        let depthFormatStr = String(describing: depthFormat)
        let colorTexPFStr = String(describing: colorTex?.pixelFormat)
        let depthTexPFStr = String(describing: depthTex?.pixelFormat)
        Self.log.info("Render target formats: cfg.color=\(colorFormatStr) cfg.depth=\(depthFormatStr) drawable.color[0]=\(colorTexPFStr) drawable.depth[0]=\(depthTexPFStr) views=\(viewCount) layout=\(cfg.layout.rawValue)")
    }

    private func probeBrightnessIfNeeded(drawable: LayerRenderer.Drawable, commandBuffer: MTLCommandBuffer) {
        // Run at most once every ~60 frames
        guard frameCounter - lastBrightnessProbeFrame >= 60 else { return }
        lastBrightnessProbeFrame = frameCounter
        guard let colorTex = drawable.colorTextures.first else { return }

        // Only probe formats we can trivially inspect
        switch colorTex.pixelFormat {
        case .rgba16Float, .bgra8Unorm, .rgba8Unorm, .bgra8Unorm_srgb, .rgba8Unorm_srgb:
            break
        default:
            let pf = String(describing: colorTex.pixelFormat)
            Self.log.debug("Brightness probe skipped for unsupported pixel format \(pf)")
            return
        }

        let w = min(8, colorTex.width)
        let h = min(8, colorTex.height)

        // Create a small SHARED staging texture to allow CPU readback
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: colorTex.pixelFormat,
                                                            width: w, height: h, mipmapped: false)
        desc.storageMode = .shared
        // No .blit usage exists; blit operations don't require a usage flag.
        // Keep shader usages only if you plan to bind this texture in shaders; otherwise, [] is fine.
        desc.usage = [.shaderRead, .shaderWrite]
        guard let staging = device.makeTexture(descriptor: desc) else {
            Self.log.error("Brightness probe: failed to create staging texture")
            return
        }

        // Copy a top-left region from the render target into the staging texture
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            Self.log.error("Brightness probe: failed to create blit encoder")
            return
        }
        let origin = MTLOrigin(x: 0, y: 0, z: 0)
        let size   = MTLSize(width: w, height: h, depth: 1)
        blit.copy(from: colorTex,
                  sourceSlice: 0,
                  sourceLevel: 0,
                  sourceOrigin: origin,
                  sourceSize: size,
                  to: staging,
                  destinationSlice: 0,
                  destinationLevel: 0,
                  destinationOrigin: origin)
        blit.endEncoding()

        // Read back on completion (after GPU finishes blit)
        let frameAtEnqueue = frameCounter
        commandBuffer.addCompletedHandler { _ in
            let w = staging.width
            let h = staging.height
            let bytesPerPixel: Int = (staging.pixelFormat == .rgba16Float) ? 8 : 4 // RGBA16F=8 bytes/px, 8-bit=4
            let row = w * bytesPerPixel
            let count = row * h
            let ptr = UnsafeMutableRawPointer.allocate(byteCount: count, alignment: 64)
            defer { ptr.deallocate() }
            staging.getBytes(ptr,
                             bytesPerRow: row,
                             from: MTLRegionMake2D(0, 0, w, h),
                             mipmapLevel: 0)

            // Cheap non-zero check
            var anyNonZero = false
            let buf = ptr.bindMemory(to: UInt8.self, capacity: count)
            for i in 0..<count where buf[i] != 0 {
                anyNonZero = true
                break
            }
            if anyNonZero {
                Self.log.debug("Brightness probe: 8x8 region NON-zero (frame \(frameAtEnqueue))")
            } else {
                Self.log.warning("Brightness probe: 8x8 region all zeros (frame \(frameAtEnqueue))")
            }
        }
    }

    private func scheduleAutoMetalCaptureIfNeeded() {
#if DEBUG
        guard !didAutoCapture else { return }
        captureNextFrame = true
        Self.log.info("Auto Metal capture scheduled for next frame (fallback persisted)")
#endif
    }

    private static func isFinite(_ m: simd_float4x4) -> Bool {
        for r in 0..<4 {
            for c in 0..<4 {
                if !m[r][c].isFinite { return false }
            }
        }
        return true
    }

    private func updateInteractionState() {
        let now = Date()
        defer { lastRotationUpdateTimestamp = now }

        guard let lastTimestamp = lastRotationUpdateTimestamp else { return }

        let deltaTime = now.timeIntervalSince(lastTimestamp)

        handStateLock.lock()
        let handStates = cachedHandStates
        handStateLock.unlock()

        let pinchedHands = handStates.filter { $0.value.isPinching }

        switch interactionState.gestureMode {
        case .idle:
            if pinchedHands.count == 1, let entry = pinchedHands.first?.value, let pinchPosition = entry.pinchPosition {
                interactionState.gestureMode = .singleHandGrab(chirality: entry.chirality)
                interactionState.singleHandOffset = interactionState.translation - pinchPosition
            } else if pinchedHands.count >= 2 {
                beginTwoHandGesture(with: handStates)
            } else {
                let deltaAngle = Constants.rotationPerSecond * deltaTime
                let deltaQuat = simd_quatf(angle: Float(deltaAngle.radians), axis: Constants.rotationAxis)
                interactionState.rotation = deltaQuat * interactionState.rotation
            }
        case .singleHandGrab(let chirality):
            let otherPinchedCount = pinchedHands.count
            if otherPinchedCount >= 2 {
                beginTwoHandGesture(with: handStates)
                return
            }

            guard
                let hand = handStates[chirality],
                hand.isPinching,
                let pinchPosition = hand.pinchPosition,
                let offset = interactionState.singleHandOffset
            else {
                resetToIdle()
                return
            }

            interactionState.translation = pinchPosition + offset
        case .twoHandManipulate:
            guard
                let left = handStates[.left], left.isPinching,
                let right = handStates[.right], right.isPinching,
                let leftPalm = left.palmPosition ?? left.pinchPosition,
                let rightPalm = right.palmPosition ?? right.pinchPosition,
                let midpointOffset = interactionState.twoHandMidpointOffset,
                let initialVector = interactionState.twoHandInitialVector,
                let initialRotation = interactionState.twoHandInitialRotation,
                let initialDistance = interactionState.twoHandInitialDistance,
                initialDistance > 0
            else {
                resetToIdle()
                return
            }

            let midpoint = (leftPalm + rightPalm) * 0.5
            interactionState.translation = midpoint + midpointOffset

            let currentVector = rightPalm - leftPalm
            let currentDistance = simd_length(currentVector)
            if currentDistance > 0 {
                interactionState.scale = interactionState.twoHandInitialScale * (currentDistance / initialDistance)
                let rotationDelta = simd_quatf(from: simd_normalize(initialVector), to: simd_normalize(currentVector))
                interactionState.rotation = rotationDelta * initialRotation
            }

            if pinchedHands.count < 2 {
                resetToIdle()
            }
        }
    }

    private func beginTwoHandGesture(with handStates: [HandAnchor.Chirality: CachedHandState]) {
        guard
            let left = handStates[.left], left.isPinching,
            let right = handStates[.right], right.isPinching,
            let leftPalm = left.palmPosition ?? left.pinchPosition,
            let rightPalm = right.palmPosition ?? right.pinchPosition
        else {
            resetToIdle()
            return
        }

        let midpoint = (leftPalm + rightPalm) * 0.5
        interactionState.twoHandMidpointOffset = interactionState.translation - midpoint
        interactionState.twoHandInitialDistance = simd_length(rightPalm - leftPalm)
        interactionState.twoHandInitialVector = rightPalm - leftPalm
        interactionState.twoHandInitialRotation = interactionState.rotation
        interactionState.twoHandInitialScale = interactionState.scale
        interactionState.gestureMode = .twoHandManipulate
        interactionState.singleHandOffset = nil
    }

    private func resetToIdle() {
        interactionState.gestureMode = .idle
        interactionState.singleHandOffset = nil
        interactionState.twoHandInitialDistance = nil
        interactionState.twoHandInitialVector = nil
        interactionState.twoHandInitialRotation = nil
        interactionState.twoHandMidpointOffset = nil
        interactionState.twoHandInitialScale = interactionState.scale
    }

    private func cacheHandAnchors<S: Sequence>(_ anchors: S) where S.Element == HandAnchor {
        var updates: [HandAnchor.Chirality: CachedHandState] = [:]

        for anchor in anchors {
            let anchorTransform = anchor.originFromAnchorTransform
            let skeleton = anchor.handSkeleton

            let thumbPosition = jointPosition(.thumbTip, skeleton: skeleton, anchorTransform: anchorTransform)
            let indexPosition = jointPosition(.indexFingerTip, skeleton: skeleton, anchorTransform: anchorTransform)
            let palmPosition = jointPosition(.wrist, skeleton: skeleton, anchorTransform: anchorTransform)

            let isPinching: Bool
            let pinchPosition: SIMD3<Float>?
            if let thumbPosition, let indexPosition {
                let distance = simd_distance(thumbPosition, indexPosition)
                isPinching = distance < InteractionConstants.pinchThreshold
                pinchPosition = (thumbPosition + indexPosition) * 0.5
            } else {
                isPinching = false
                pinchPosition = nil
            }

            let cachedState = CachedHandState(chirality: anchor.chirality,
                                              pinchPosition: pinchPosition,
                                              palmPosition: palmPosition ?? pinchPosition,
                                              isPinching: isPinching)
            updates[anchor.chirality] = cachedState
        }

        handStateLock.lock()
        for (chirality, state) in updates {
            cachedHandStates[chirality] = state
        }
        let missing = Set(cachedHandStates.keys).subtracting(updates.keys)
        for chirality in missing {
            cachedHandStates.removeValue(forKey: chirality)
        }
        handStateLock.unlock()
    }

    private func jointPosition(_ name: HandSkeleton.JointName,
                               skeleton: HandSkeleton?,
                               anchorTransform: simd_float4x4) -> SIMD3<Float>? {
        guard
            let joint = skeleton?.joint(name)
        else { return nil }

        let jointWorld = anchorTransform * joint.anchorFromJointTransform
        return SIMD3<Float>(jointWorld.columns.3.x,
                            jointWorld.columns.3.y,
                            jointWorld.columns.3.z)
    }

    func renderFrame() {
        guard let frame = layerRenderer.queryNextFrame() else { return }
        frameCounter &+= 1

        frame.startUpdate()
        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

        guard let drawable = frame.queryDrawable() else { return }
        logRenderTargetFormatsOnce(for: drawable)

#if DEBUG
        if captureNextFrame, !didAutoCapture {
            captureNextFrame = false
            let mgr = MTLCaptureManager.shared()
            let desc = MTLCaptureDescriptor()
            desc.captureObject = commandQueue
            do {
                try mgr.startCapture(with: desc)
                Self.log.info("Metal capture started for this frame")
                commandBuffer.addCompletedHandler { _ in
                    MTLCaptureManager.shared().stopCapture()
                    Self.log.info("Metal capture stopped")
                }
                didAutoCapture = true
            } catch {
                Self.log.error("Failed to start Metal capture: \(error.localizedDescription)")
            }
        }
#endif

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        frame.startSubmission()

        let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        updateInteractionState()
        updateEnvironmentProbeIfNeeded()
        applyEnvironmentResourcesIfNeeded()

        let viewports = self.viewports(drawable: drawable, deviceAnchor: deviceAnchor)

        probeBrightnessIfNeeded(drawable: drawable, commandBuffer: commandBuffer)

        do {
            try modelRenderer?.render(viewports: viewports,
                                      colorTexture: drawable.colorTextures[0],
                                      colorStoreAction: .store,
                                      depthTexture: drawable.depthTextures[0],
                                      rasterizationRateMap: drawable.rasterizationRateMaps.first,
                                      renderTargetArrayLength: layerRenderer.configuration.layout == .layered ? drawable.views.count : 1,
                                      to: commandBuffer)
        } catch {
            let cfg = layerRenderer.configuration
            let colorFormatStr = String(describing: cfg.colorFormat)
            let depthFormatStr = String(describing: cfg.depthFormat)
            Self.log.error("Unable to render scene: \(error.localizedDescription). colorFormat=\(colorFormatStr) depthFormat=\(depthFormatStr) views=\(drawable.views.count) frame=\(self.frameCounter)")
        }

        if let splatRenderer = modelRenderer as? SplatRenderer {
            inspectMaterialFallbackTelemetry(from: splatRenderer)
        } else {
            resetFallbackTrackingIfNeeded(didResolve: fallbackFrameStreak > 0)
        }

        drawable.encodePresent(commandBuffer: commandBuffer)

        commandBuffer.commit()

        frame.endSubmission()
    }

    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                Self.log.warning("Layer is invalidated")
                return
            } else if layerRenderer.state == .paused {
                layerRenderer.waitUntilRunning()
                continue
            } else {
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }

    private func updateEnvironmentProbeIfNeeded() {
        guard let snapshot = environmentProbeManager.consumeLatestSnapshot() else {
            logProbeGuardChange(.noSnapshot)
            return
        }

        guard snapshot.revision != latestPrefilterRevision else {
            logProbeGuardChange(.staleRevision(snapshot.revision))
            return
        }

        guard let prefilter = environmentPrefilter else {
            logProbeGuardChange(.prefilterUnavailable(snapshot.revision))
            return
        }

        do {
            let start = Date()
            let result = try prefilter.prefilter(snapshot: snapshot)
            latestPrefilterRevision = snapshot.revision
            environmentPrefilterResult = result
            environmentResourcesDirty = true
            lastPrefilterDuration = Date().timeIntervalSince(start)
            clearProbeGuardIfNeeded(resumedRevision: snapshot.revision)
            let durationMS = lastPrefilterDuration * 1000.0
            Self.log.debug("Prefiltered environment revision \(snapshot.revision) in \(durationMS) ms using texture \(Self.describeTexture(snapshot.texture))")
        } catch {
            Self.log.error("Failed to prefilter environment map for revision \(snapshot.revision) (texture: \(Self.describeTexture(snapshot.texture))): \(error.localizedDescription)")
        }
    }

    private func applyEnvironmentResourcesIfNeeded() {
        guard environmentResourcesDirty else {
            logBindingSkip(.notDirty)
            return
        }

        guard let result = environmentPrefilterResult else {
            logBindingSkip(.resultMissing)
            return
        }

        guard let splatRenderer = modelRenderer as? SplatRenderer else {
            logBindingSkip(.rendererUnavailable)
            return
        }

        clearBindingSkipIfNeeded()

        Self.log.debug("Applying environment resources revision \(result.revision) (environment: \(Self.describeTexture(result.environmentMap)), brdf: \(Self.describeTexture(result.brdfLookup)))")

        do {
            try splatRenderer.setEnvironmentMap(result.environmentMap)
            try splatRenderer.setBRDFLookupTexture(result.brdfLookup)
            environmentResourcesDirty = false
            lastAppliedEnvironmentRevision = result.revision
        } catch {
            Self.log.error("Unable to bind environment resources revision \(result.revision): \(error.localizedDescription)")
        }
    }

    private func inspectMaterialFallbackTelemetry(from renderer: SplatRenderer) {
        let telemetry = renderer.materialFallbackTelemetry
        guard telemetry.environmentFallbackBindings > 0 || telemetry.brdfFallbackBindings > 0 else {
            resetFallbackTrackingIfNeeded(didResolve: fallbackFrameStreak > 0)
            return
        }

        fallbackFrameStreak &+= 1

        let diagnostics = environmentProbeManager.diagnostics()
        let latestResultRevision = environmentPrefilterResult?.revision
        // environmentPrefilterResult?.timestamp is a non-optional Date inside an optional container.
        let timestamp = environmentPrefilterResult
            .map { Self.iso8601Formatter.string(from: $0.timestamp) } ?? "nil"
        // diagnostics.latestSnapshotTimestamp is Optional<Date>, so map it directly.
        let snapshotTimestamp = diagnostics.latestSnapshotTimestamp
            .map { Self.iso8601Formatter.string(from: $0) } ?? "nil"
        let prefilterDurationMS = lastPrefilterDuration * 1000.0
        let formattedDuration = String(format: "%.2f", prefilterDurationMS)
        Self.log.warning("Renderer bound fallback materials (environment: \(telemetry.environmentFallbackBindings), brdf: \(telemetry.brdfFallbackBindings)). Prefilter latest revision: \(self.latestPrefilterRevision), applied revision: \(self.lastAppliedEnvironmentRevision), current result revision: \(latestResultRevision.map(String.init) ?? "nil"), result timestamp: \(timestamp), resourcesDirty: \(self.environmentResourcesDirty), pending snapshot revision: \(diagnostics.pendingSnapshotRevision.map(String.init) ?? "nil"), delivered revision: \(diagnostics.deliveredRevision), latest snapshot timestamp: \(snapshotTimestamp), probe running: \(diagnostics.isRunning), last prefilter duration: \(formattedDuration) ms, frame: \(self.frameCounter)")

        if fallbackFrameStreak >= TelemetryConstants.fallbackEscalationFrameThreshold && !didEscalateFallback {
            didEscalateFallback = true
            scheduleAutoMetalCaptureIfNeeded()
            Self.log.error("Fallback environment resources persisted for \(self.fallbackFrameStreak) consecutive frames")
#if DEBUG
            assertionFailure("SplatRenderer is still using fallback environment resources after \(fallbackFrameStreak) frames")
#endif
        }
    }

    private func resetFallbackTrackingIfNeeded(didResolve: Bool) {
        guard fallbackFrameStreak != 0 || didEscalateFallback else { return }
        if didResolve {
            Self.log.info("Fallback environment bindings resolved after \(self.fallbackFrameStreak) frames")
        }
        fallbackFrameStreak = 0
        didEscalateFallback = false
    }

    private func logProbeGuardChange(_ reason: ProbeGuardReason) {
        guard reason != lastProbeGuardReason else { return }
        lastProbeGuardReason = reason
        switch reason {
        case .noSnapshot:
            Self.log.debug("No environment probe snapshot available yet; waiting for updates")
        case .staleRevision(let revision):
            Self.log.debug("Latest environment probe revision \(revision) already prefiltered; skipping reprocessing")
        case .prefilterUnavailable(let revision):
            Self.log.error("Environment prefilter unavailable when snapshot revision \(revision) arrived")
        }
    }

    private func clearProbeGuardIfNeeded(resumedRevision: UInt64) {
        guard let reason = lastProbeGuardReason else { return }
        let description: String
        switch reason {
        case .noSnapshot:
            description = "waiting for initial snapshot"
        case .staleRevision(let revision):
            description = "receiving already-processed revision \(revision)"
        case .prefilterUnavailable(let revision):
            description = "environment prefilter unavailable for revision \(revision)"
        }
        Self.log.debug("Environment probe updates resumed with revision \(resumedRevision) after \(description)")
        lastProbeGuardReason = nil
    }

    private func logBindingSkip(_ reason: EnvironmentBindingSkipReason) {
        guard reason != lastBindingSkipReason else { return }
        lastBindingSkipReason = reason
        switch reason {
        case .rendererUnavailable:
            Self.log.debug("Skipping environment binding: current renderer is not a SplatRenderer")
        case .resultMissing:
            Self.log.debug("Skipping environment binding: prefilter result unavailable")
        case .notDirty:
            Self.log.debug("Skipping environment binding: environment resources are up to date (applied revision: \(self.lastAppliedEnvironmentRevision))")
        }
    }

    private func clearBindingSkipIfNeeded() {
        guard lastBindingSkipReason != nil else { return }
        lastBindingSkipReason = nil
        Self.log.debug("Environment resource binding proceeding with new prefilter result")
    }

    private static func describeTexture(_ texture: MTLTexture) -> String {
        let label = texture.label ?? "<unlabeled>"
        let dimensions = "\(texture.width)x\(texture.height)x\(max(texture.depth, 1))"
        let arrayInfo = texture.arrayLength > 1 ? ", arrayLength: \(texture.arrayLength)" : ""
        let mipInfo = texture.mipmapLevelCount > 1 ? ", mips: \(texture.mipmapLevelCount)" : ""
        let usage = textureUsageDescription(texture.usage)
        let deviceID = ObjectIdentifier(texture.device)
        return "\(label) [type: \(texture.textureType), pixelFormat: \(texture.pixelFormat), dimensions: \(dimensions)\(arrayInfo)\(mipInfo), storage: \(texture.storageMode), usage: \(usage), device: \(deviceID)]"
    }

    private static func textureUsageDescription(_ usage: MTLTextureUsage) -> String {
        var components: [String] = []
        if usage.contains(.shaderRead) { components.append("shaderRead") }
        if usage.contains(.shaderWrite) { components.append("shaderWrite") }
        if usage.contains(.renderTarget) { components.append("renderTarget") }
        if usage.contains(.pixelFormatView) { components.append("pixelFormatView") }
        if components.isEmpty { components.append("none") }
        return components.joined(separator: "|")
    }

    deinit {
        environmentProbeManager.stop()
        handUpdateTask?.cancel()
    }
}

#endif // os(visionOS)
