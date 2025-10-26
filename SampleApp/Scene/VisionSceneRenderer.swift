#if os(visionOS)

import ARKit
import CompositorServices
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

    let layerRenderer: LayerRenderer
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    var lastRotationUpdateTimestamp: Date? = nil
    var rotation: Angle = .zero

    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider
    let environmentProbeSession: ARSession

    private let environmentProbeManager: EnvironmentProbeManager
    private let environmentPrefilter: EnvironmentPrefilter?
    private var environmentPrefilterResult: EnvironmentPrefilterResult?
    private var environmentResourcesDirty = false
    private var latestPrefilterRevision: UInt64 = 0
    private var lastAppliedEnvironmentRevision: UInt64 = 0
    private var lastPrefilterDuration: TimeInterval = 0

    init(_ layerRenderer: LayerRenderer) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!

        worldTracking = WorldTrackingProvider()
        arSession = ARKitSession()
        environmentProbeSession = ARSession()
        environmentProbeManager = EnvironmentProbeManager(session: environmentProbeSession,
                                                          device: device)
        do {
            environmentPrefilter = try EnvironmentPrefilter(device: device)
        } catch {
            Self.log.error("Failed to initialize environment prefilter: \(error.localizedDescription)")
            environmentPrefilter = nil
        }
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

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
                try await arSession.run([worldTracking])
            } catch {
                fatalError("Failed to initialize ARKitSession")
            }

            environmentProbeManager.start()

            let renderThread = Thread {
                self.renderLoop()
            }
            renderThread.name = "Render Thread"
            renderThread.start()
        }
    }

    private func viewports(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor?) -> [ModelRendererViewportDescriptor] {
        let rotationMatrix = matrix4x4_rotation(radians: Float(rotation.radians),
                                                axis: Constants.rotationAxis)
        let translationMatrix = matrix4x4_translation(0.0, 0.0, Constants.modelCenterZ)
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
            return ModelRendererViewportDescriptor(viewport: view.textureMap.viewport,
                                                   projectionMatrix: projMatrixSIMD,
                                                   viewMatrix: userViewpointMatrix * translationMatrix * rotationMatrix * commonUpCalibration,
                                                   screenSize: screenSize)
        }
    }

    private func updateRotation() {
        let now = Date()
        defer {
            lastRotationUpdateTimestamp = now
        }

        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func renderFrame() {
        guard let frame = layerRenderer.queryNextFrame() else { return }

        frame.startUpdate()
        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

        guard let drawable = frame.queryDrawable() else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        frame.startSubmission()

        let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        updateRotation()
        updateEnvironmentProbeIfNeeded()
        applyEnvironmentResourcesIfNeeded()

        let viewports = self.viewports(drawable: drawable, deviceAnchor: deviceAnchor)

        do {
            try modelRenderer?.render(viewports: viewports,
                                      colorTexture: drawable.colorTextures[0],
                                      colorStoreAction: .store,
                                      depthTexture: drawable.depthTextures[0],
                                      rasterizationRateMap: drawable.rasterizationRateMaps.first,
                                      renderTargetArrayLength: layerRenderer.configuration.layout == .layered ? drawable.views.count : 1,
                                      to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }

        drawable.encodePresent(commandBuffer: commandBuffer)

        commandBuffer.commit()

        frame.endSubmission()
    }

    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                Self.log.warning("Layer is invalidated")
                environmentProbeManager.stop()
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
        guard let snapshot = environmentProbeManager.consumeLatestSnapshot() else { return }
        guard snapshot.revision != latestPrefilterRevision else { return }
        guard let prefilter = environmentPrefilter else {
            Self.log.error("Environment prefilter unavailable when probe revision \(snapshot.revision) arrived")
            return
        }

        do {
            let start = Date()
            let result = try prefilter.prefilter(snapshot: snapshot)
            latestPrefilterRevision = snapshot.revision
            environmentPrefilterResult = result
            environmentResourcesDirty = true
            lastPrefilterDuration = Date().timeIntervalSince(start)
            let durationMS = lastPrefilterDuration * 1000.0
            Self.log.debug("Prefiltered environment revision \(snapshot.revision) in \(durationMS) ms")
        } catch {
            Self.log.error("Failed to prefilter environment map: \(error.localizedDescription)")
        }
    }

    private func applyEnvironmentResourcesIfNeeded() {
        guard environmentResourcesDirty,
              let result = environmentPrefilterResult,
              let splatRenderer = modelRenderer as? SplatRenderer else { return }

        do {
            try splatRenderer.setEnvironmentMap(result.environmentMap)
            try splatRenderer.setBRDFLookupTexture(result.brdfLookup)
            environmentResourcesDirty = false
            lastAppliedEnvironmentRevision = result.revision
        } catch {
            Self.log.error("Unable to bind environment resources: \(error.localizedDescription)")
        }
    }

    deinit {
        environmentProbeManager.stop()
    }
}

#endif // os(visionOS)

