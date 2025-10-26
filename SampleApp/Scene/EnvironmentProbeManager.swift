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

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "EnvironmentProbeManager",
                                     category: "EnvironmentProbe")

    private let session: ARKitSession
    private let worldTracking: WorldTrackingProvider
    private let environmentLightEstimation: EnvironmentLightEstimationProvider
    private let device: MTLDevice
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
        Self.log.debug("Subscribed to environment light estimation updates (session: \(String(describing: session)))")
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

    private func updateSnapshot(with texture: MTLTexture?, sphericalHarmonics: [Float], timestamp: Date) {
        guard let texture else {
            Self.log.error("Received environment probe without texture")
            return
        }

        guard texture.device === device else {
            Self.log.error("Environment texture device mismatch; expected \(String(describing: device)), received \(String(describing: texture.device))")
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

        let harmonics = sphericalHarmonicsCoefficients(from: anchor)
        if harmonics.isEmpty {
            Self.log.debug("Environment probe anchor missing spherical harmonics coefficients")
        }

        updateSnapshot(with: texture,
                       sphericalHarmonics: harmonics,
                       timestamp: Date())
    }

    private func sphericalHarmonicsCoefficients(from anchor: EnvironmentProbeAnchor) -> [Float] {
        guard let coefficients = anchor.sphericalHarmonicsCoefficients else { return [] }

        if let floats = coefficients as? [Float] {
            return floats
        }
        if let doubles = coefficients as? [Double] {
            return doubles.map { Float($0) }
        }
        if let numbers = coefficients as? [NSNumber] {
            return numbers.map { $0.floatValue }
        }
        return []
    }
}

#endif // os(visionOS)
