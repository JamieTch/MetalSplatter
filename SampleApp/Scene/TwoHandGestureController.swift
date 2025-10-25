#if os(visionOS)

import ARKit
import os
import simd

final class TwoHandGestureController {
    enum Phase {
        case idle
        case began
        case changed
    }

    private struct State {
        var phase: Phase = .idle
        var initialMidpoint: SIMD3<Float> = .zero
        var initialSeparation: Float = 0
        var initialOrientation: simd_quatf = simd_quatf()
    }

    private let interactionState: SceneInteractionState
    private let log: Logger
    private let stateLock: OSAllocatedUnfairLock<State>

    init(interactionState: SceneInteractionState, log: Logger) {
        self.interactionState = interactionState
        self.log = log
        self.stateLock = OSAllocatedUnfairLock(initialState: State())
    }

    func handleTwoHandPinch(_ pinch: SpatialEventCollection.TwoHandPinch) {
        guard let sample = Sample(pinch: pinch) else {
            endGestureIfNeeded(reason: "invalidPinchSample")
            return
        }

        let pinchPhase = pinch.gesturePhase
        switch pinchPhase {
        case .began:
            begin(sample: sample)
        case .changed:
            update(sample: sample)
        case .ended, .cancelled:
            endGestureIfNeeded(reason: "phaseEnded")
        @unknown default:
            endGestureIfNeeded(reason: "unknownPhase")
        }
    }

    func updateFromAnchors(_ anchors: HandTrackingProvider.HandAnchors) {
        guard let left = anchors.left,
              let right = anchors.right else {
            endGestureIfNeeded(reason: "missingAnchors")
            return
        }

        guard left.isPinching && right.isPinching,
              let sample = Sample(left: left.originFromAnchorTransform,
                                   right: right.originFromAnchorTransform) else {
            endGestureIfNeeded(reason: "noPinchDetected")
            return
        }

        stateLock.withLock { state in
            switch state.phase {
            case .idle:
                state.phase = .began
                state.initialMidpoint = sample.midpoint
                state.initialSeparation = sample.separation
                state.initialOrientation = sample.orientation
                log.debug("Two-hand pinch began (fallback)")
                let resetUpdate = SceneInteractionState.Update(translation: .zero,
                                                               rotation: simd_quatf(),
                                                               scale: 1.0)
                interactionState.update(resetUpdate)
            case .began, .changed:
                state.phase = .changed
                let update = makeUpdate(for: sample, state: state)
                interactionState.update(update)
            }
        }
    }
}

private extension TwoHandGestureController {
    struct Sample {
        let midpoint: SIMD3<Float>
        let separation: Float
        let orientation: simd_quatf

        init?(pinch: SpatialEventCollection.TwoHandPinch) {
            guard let left = pinch.leftHand?.originFromAnchorTransform,
                  let right = pinch.rightHand?.originFromAnchorTransform else {
                return nil
            }
            self.init(left: left, right: right)
        }

        init?(left: simd_float4x4, right: simd_float4x4) {
            let leftPosition = left.translation
            let rightPosition = right.translation
            let vector = rightPosition - leftPosition
            let distance = simd_length(vector)
            if distance.isZero {
                return nil
            }

            midpoint = (leftPosition + rightPosition) * 0.5
            separation = distance
            orientation = TwoHandGestureController.orientation(for: leftPosition, right: rightPosition)
        }
    }

    func begin(sample: Sample) {
        stateLock.withLock { state in
            state.phase = .began
            state.initialMidpoint = sample.midpoint
            state.initialSeparation = sample.separation
            state.initialOrientation = sample.orientation
            log.debug("Two-hand pinch began")
            let resetUpdate = SceneInteractionState.Update(translation: .zero,
                                                           rotation: simd_quatf(),
                                                           scale: 1.0)
            interactionState.update(resetUpdate)
        }
    }

    func update(sample: Sample) {
        stateLock.withLock { state in
            guard state.phase != .idle else {
                state.phase = .began
                state.initialMidpoint = sample.midpoint
                state.initialSeparation = sample.separation
                state.initialOrientation = sample.orientation
                log.debug("Two-hand pinch began implicitly")
                let resetUpdate = SceneInteractionState.Update(translation: .zero,
                                                               rotation: simd_quatf(),
                                                               scale: 1.0)
                interactionState.update(resetUpdate)
                return
            }

            state.phase = .changed
            let update = makeUpdate(for: sample, state: state)
            interactionState.update(update)
        }
    }

    func endGestureIfNeeded(reason: StaticString) {
        stateLock.withLock { state in
            guard state.phase != .idle else { return }
            log.debug("Two-hand pinch ended: \(String(describing: reason))")
            state.phase = .idle
            state.initialMidpoint = .zero
            state.initialSeparation = 0
            state.initialOrientation = simd_quatf()
        }
    }

    func makeUpdate(for sample: Sample, state: State) -> SceneInteractionState.Update {
        let translation = sample.midpoint - state.initialMidpoint
        let scaleRatio = max(sample.separation / max(state.initialSeparation, 0.001), 0.01)
        let rotationDelta = sample.orientation * state.initialOrientation.inverse
        return SceneInteractionState.Update(translation: translation,
                                            rotation: rotationDelta.normalized,
                                            scale: scaleRatio)
    }
}

private extension simd_float4x4 {
    var translation: SIMD3<Float> {
        SIMD3<Float>(columns.3.x, columns.3.y, columns.3.z)
    }
}

private extension TwoHandGestureController {
    static func orientation(for left: SIMD3<Float>, right: SIMD3<Float>) -> simd_quatf {
        let forward = normalize(right - left)
        let reference = SIMD3<Float>(0, 0, -1)
        let axis = simd_cross(reference, forward)
        let axisLength = simd_length(axis)
        if axisLength < 1e-5 {
            if simd_dot(reference, forward) > 0.0 {
                return simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
            } else {
                return simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 1, 0))
            }
        }
        let angle = acos(simd_dot(reference, forward))
        return simd_quatf(angle: angle, axis: axis / axisLength)
    }
}

#endif // os(visionOS)
