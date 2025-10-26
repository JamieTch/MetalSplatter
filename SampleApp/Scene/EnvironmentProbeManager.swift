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

    private let session: ARSession
    private let device: MTLDevice
    private let delegateQueue = DispatchQueue(label: "com.metalsplatter.environmentProbe.delegate")
    private let stateQueue = DispatchQueue(label: "com.metalsplatter.environmentProbe.state")

    private var latestSnapshot: Snapshot?
    private var deliveredRevision: UInt64 = 0
    private var isRunning = false
    private var latestSphericalHarmonics: [Float] = []

    init(session: ARSession, device: MTLDevice) {
        self.session = session
        self.device = device
        super.init()
    }

    deinit {
        stop()
    }

    func start() {
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in
                self?.start()
            }
            return
        }

        guard !isRunning else { return }
        isRunning = true

        session.delegateQueue = delegateQueue
        session.delegate = self

        let configuration = ARWorldTrackingConfiguration()
        configuration.environmentTexturing = .automatic

        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        Self.log.debug("Started ARSession for environment probes")
    }

    func stop() {
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in
                self?.stop()
            }
            return
        }

        guard isRunning else { return }
        session.pause()
        if session.delegate === self {
            session.delegate = nil
        }
        isRunning = false
        Self.log.debug("Paused ARSession for environment probes")
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

    private func storeSphericalHarmonics(_ harmonics: [Float]) {
        stateQueue.async {
            self.latestSphericalHarmonics = harmonics
        }
    }

    private func currentSphericalHarmonics() -> [Float] {
        stateQueue.sync {
            latestSphericalHarmonics
        }
    }
}

extension EnvironmentProbeManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if let coefficients = frame.lightEstimate?.sphericalHarmonicsCoefficients {
            let harmonics = coefficients.map { Float(truncating: $0) }
            storeSphericalHarmonics(harmonics)
        } else {
            storeSphericalHarmonics([])
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        handleAnchors(anchors)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        handleAnchors(anchors)
    }

    private func handleAnchors(_ anchors: [ARAnchor]) {
        let harmonics = currentSphericalHarmonics()
        for anchor in anchors {
            guard let probe = anchor as? AREnvironmentProbeAnchor else { continue }
            updateSnapshot(with: probe.environmentTexture,
                           sphericalHarmonics: harmonics,
                           timestamp: Date())
        }
    }
}

#endif // os(visionOS)
