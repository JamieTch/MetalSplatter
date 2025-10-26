#if os(visionOS)

import ARKit
import Foundation
import Metal
import os

final class EnvironmentProbeManager: NSObject {
    struct Snapshot {
        let texture: MTLTexture
        let revision: UInt64
        let timestamp: Date
        let sphericalHarmonics: [Float]
    }

    struct Diagnostics {
        let latestSnapshotRevision: UInt64?
        let deliveredRevision: UInt64
        let pendingSnapshotRevision: UInt64?
        let latestSnapshotTimestamp: Date?
        let isRunning: Bool
    }

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "EnvironmentProbeManager",
                                     category: "EnvironmentProbe")

    private let session: ARKitSession
    private let worldTracking: WorldTrackingProvider
    private let environmentLightEstimation: EnvironmentLightEstimationProvider
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let stateQueue = DispatchQueue(label: "com.metalsplatter.environmentProbe.state")

    private var latestSnapshot: Snapshot?
    private var deliveredRevision: UInt64 = 0
    private var environmentTask: Task<Void, Never>?
    private var isRunning = false

    init(session: ARKitSession,
         worldTracking: WorldTrackingProvider,
         environmentLightEstimation: EnvironmentLightEstimationProvider,
         device: MTLDevice) {
        self.session = session
        self.worldTracking = worldTracking
        self.environmentLightEstimation = environmentLightEstimation
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        super.init()
        start()
    }

    deinit {
        stop()
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true

        environmentTask = Task { [weak self] in
            await self?.listenForEnvironmentUpdates()
        }
        Self.log.debug("Subscribed to environment light estimation updates (session: \(String(describing: self.session)))")
    }

    func stop() {
        guard isRunning else { return }
        environmentTask?.cancel()
        environmentTask = nil
        isRunning = false
        Self.log.debug("Cancelled environment light estimation update subscription")
    }

    func consumeLatestSnapshot() -> Snapshot? {
        stateQueue.sync {
            guard let snapshot = latestSnapshot else { return nil }
            guard snapshot.revision != deliveredRevision else { return nil }
            deliveredRevision = snapshot.revision
            return snapshot
        }
    }

    func diagnostics() -> Diagnostics {
        stateQueue.sync {
            let latestRevision = latestSnapshot?.revision
            let pendingRevision = latestRevision != deliveredRevision ? latestRevision : nil
            return Diagnostics(latestSnapshotRevision: latestRevision,
                               deliveredRevision: deliveredRevision,
                               pendingSnapshotRevision: pendingRevision,
                               latestSnapshotTimestamp: latestSnapshot?.timestamp,
                               isRunning: isRunning)
        }
    }

    private func describe(_ tex: MTLTexture) -> String {
        return "label=\(tex.label ?? "<none>") type=\(tex.textureType.rawValue) fmt=\(tex.pixelFormat.rawValue) size=\(tex.width)x\(tex.height)x\(tex.depth) array=\(tex.arrayLength) mips=\(tex.mipmapLevelCount) storage=\(tex.storageMode.rawValue) usage=\(tex.usage.rawValue)"
    }

    /// Returns true if the cube appears to contain non-zero texels at face 0 / mip 0 (quick check), false if zeros or unsupported format.
    private func sourceCubeHasEnergy(_ cube: MTLTexture) -> Bool {
        // Only probe float/half formats commonly used for env maps
        switch cube.pixelFormat {
        case .rgba16Float, .rgba32Float, .rg16Float, .rg32Float: break
        default:
            Self.log.debug("[EnvSrcProbe] Skipping unsupported pixel format \(cube.pixelFormat.rawValue)")
            return false
        }
        // commandQueue is non-optional; only unwrap the optionals we make from it.
        guard let cb = commandQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            Self.log.error("[EnvSrcProbe] Failed to create command buffer/encoder")
            return false
        }
        let w = max(1, min(4, cube.width))
        let h = max(1, min(4, cube.height))
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: cube.pixelFormat, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared
        guard let staging = device.makeTexture(descriptor: desc) else {
            Self.log.error("[EnvSrcProbe] Failed to create staging texture")
            return false
        }
        let origin = MTLOrigin(x: 0, y: 0, z: 0)
        let size = MTLSize(width: w, height: h, depth: 1)
        // Copy face 0, level 0 into staging
        blit.copy(from: cube,
                  sourceSlice: 0,
                  sourceLevel: 0,
                  sourceOrigin: origin,
                  sourceSize: size,
                  to: staging,
                  destinationSlice: 0,
                  destinationLevel: 0,
                  destinationOrigin: origin)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let bytesPerPixel: Int
        switch staging.pixelFormat {
        case .rgba16Float: bytesPerPixel = 8
        case .rg16Float:   bytesPerPixel = 4
        case .rgba32Float: bytesPerPixel = 16
        case .rg32Float:   bytesPerPixel = 8
        default:           bytesPerPixel = 8
        }
        let row = w * bytesPerPixel
        let count = row * h
        var buf = [UInt8](repeating: 0, count: count)
        buf.withUnsafeMutableBytes { p in
            staging.getBytes(p.baseAddress!, bytesPerRow: row, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        }
        let anyNonZero = buf.contains { $0 != 0 }
        if anyNonZero {
            Self.log.debug("[EnvSrcProbe] Face0 LOD0 \(w)x\(h) sample appears NON-zero")
        } else {
            Self.log.warning("[EnvSrcProbe] Face0 LOD0 \(w)x\(h) sample is all zeros")
        }
        return anyNonZero
    }

    private func updateSnapshot(with texture: MTLTexture?, sphericalHarmonics: [Float], timestamp: Date) {
        guard let texture else {
            Self.log.error("Received environment probe without texture")
            return
        }

        guard texture.device === device else {
            Self.log.error("Environment texture device mismatch; expected \(String(describing: self.device)), received \(String(describing: texture.device))")
            return
        }

        assignSnapshot(texture: texture,
                       sphericalHarmonics: sphericalHarmonics,
                       timestamp: timestamp)
    }

    private func assignSnapshot(texture: MTLTexture, sphericalHarmonics: [Float], timestamp: Date) {
        stateQueue.async {
            let revision = (self.latestSnapshot?.revision ?? 0) &+ 1
            self.latestSnapshot = Snapshot(texture: texture,
                                           revision: revision,
                                           timestamp: timestamp,
                                           sphericalHarmonics: sphericalHarmonics)
            Self.log.debug("Updated environment probe snapshot (revision: \(revision))")
        }
    }

    private func listenForEnvironmentUpdates() async {
        await waitUntilProvidersRunning()

        for await update in environmentLightEstimation.anchorUpdates {
            if Task.isCancelled { return }
            handleAnchorUpdate(update)
        }
    }

    private func waitUntilProvidersRunning() async {
        while worldTracking.state != .running || environmentLightEstimation.state != .running {
            try? await Task.sleep(nanoseconds: 50_000_000)
            if Task.isCancelled { return }
        }
    }

    private func handleAnchorUpdate(_ update: AnchorUpdate<EnvironmentProbeAnchor>) {
        switch update.event {
        case .removed:
            return
        case .added, .updated:
            break
        @unknown default:
            return
        }

        let anchor = update.anchor

        guard let texture = anchor.environmentTexture else {
            Self.log.debug("Environment probe anchor missing cube map texture")
            return
        }

        // Log the raw environment texture from the anchor and probe its energy before prefiltering
        Self.log.debug("Received EnvironmentProbeAnchor texture: \(self.describe(texture))")
        let hasEnergy = sourceCubeHasEnergy(texture)
        if !hasEnergy {
            Self.log.warning("EnvironmentProbeAnchor environment cube appears zero-energy at source (before prefilter)")
        }

        // Assign snapshot without spherical harmonics
        updateSnapshot(with: texture,
                       sphericalHarmonics: [],
                       timestamp: Date())

    }
}

#endif // os(visionOS)
