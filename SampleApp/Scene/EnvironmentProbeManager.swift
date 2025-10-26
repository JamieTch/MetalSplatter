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
    private let stateQueue = DispatchQueue(label: "com.metalsplatter.environmentProbe.state")

    private var latestSnapshot: Snapshot?
    private var deliveredRevision: UInt64 = 0
    private var isRunning = false

    override init() {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("EnvironmentProbeManager requires a Metal device")
        }
        self.device = defaultDevice
        self.session = ARSession()
        super.init()
        self.session.delegate = self
    }

    init(device: MTLDevice) {
        self.device = device
        self.session = ARSession()
        super.init()
        self.session.delegate = self
    }

    func start() {
        guard !isRunning else { return }

        let configuration = ARWorldTrackingConfiguration()
        configuration.environmentTexturing = .automatic

        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        Self.log.debug("Started ARSession with automatic environment texturing")
    }

    func stop() {
        guard isRunning else { return }
        session.pause()
        isRunning = false
        Self.log.debug("Stopped ARSession environment probe updates")
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
            Self.log.error("Environment texture device mismatch; expected \(device), received \(String(describing: texture.device))")
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

    private func currentSphericalHarmonics(from frame: ARFrame?) -> [Float] {
        guard let coefficients = frame?.lightEstimate?.sphericalHarmonicsCoefficients else { return [] }
        return coefficients.map { Float(truncating: $0) }
    }
}

extension EnvironmentProbeManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRunning else { return }
        let harmonics = currentSphericalHarmonics(from: frame)
        if harmonics.isEmpty {
            Self.log.debug("No spherical harmonics coefficients available in current frame")
        }
        if let probeTexture = frame.environmentTexture {
            updateSnapshot(with: probeTexture,
                           sphericalHarmonics: harmonics,
                           timestamp: Date())
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        handleProbeAnchors(anchors)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        handleProbeAnchors(anchors)
    }

    private func handleProbeAnchors(_ anchors: [ARAnchor]) {
        guard isRunning else { return }
        var handled = false
        for anchor in anchors {
            guard let probe = anchor as? AREnvironmentProbeAnchor else { continue }
            handled = true
            let harmonics = currentSphericalHarmonics(from: session.currentFrame)
            updateSnapshot(with: probe.environmentTexture,
                           sphericalHarmonics: harmonics,
                           timestamp: Date())
        }
        if !handled {
            Self.log.debug("Received anchor update without environment probe")
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        Self.log.error("Environment probe session failed: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        Self.log.warning("Environment probe session interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        Self.log.info("Environment probe session interruption ended")
        if isRunning {
            start()
        }
    }
}

#endif // os(visionOS)
