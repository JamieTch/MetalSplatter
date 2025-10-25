#if os(visionOS)

import os
import simd

final class SceneInteractionState {
    struct Values {
        var translation: SIMD3<Float> = .zero
        var rotation: simd_quatf = simd_quatf()
        var scale: Float = 1.0
    }

    private let scaleRange: ClosedRange<Float>
    private let lock: OSAllocatedUnfairLock<Values>

    init(scaleRange: ClosedRange<Float> = 0.2...5.0) {
        self.scaleRange = scaleRange
        let initialState = Values()
        self.lock = OSAllocatedUnfairLock(initialState: initialState)
        assert(SceneInteractionState.isApproximatelyIdentity(SceneInteractionState.matrix(for: initialState)))
    }

    func currentMatrix() -> simd_float4x4 {
        return lock.withLock { values in
            SceneInteractionState.matrix(for: values)
        }
    }

    func update(_ body: (inout Values) -> Void) {
        lock.withLock { values in
            body(&values)
            values.rotation = values.rotation.normalized
            values.scale = SceneInteractionState.clamp(values.scale, to: scaleRange)
            assert(scaleRange.contains(values.scale))
        }
    }
}

private extension SceneInteractionState {
    static func clamp(_ value: Float, to range: ClosedRange<Float>) -> Float {
        return min(max(value, range.lowerBound), range.upperBound)
    }

    static func matrix(for values: Values) -> simd_float4x4 {
        let translationMatrix = matrix4x4_translation(values.translation.x,
                                                      values.translation.y,
                                                      values.translation.z)
        let rotationMatrix = simd_float4x4(values.rotation.normalized)
        let scaleVector = SIMD4<Float>(values.scale, values.scale, values.scale, 1)
        let scaleMatrix = simd_float4x4(diagonal: scaleVector)
        return translationMatrix * rotationMatrix * scaleMatrix
    }

    static func isApproximatelyIdentity(_ matrix: simd_float4x4, epsilon: Float = 1e-5) -> Bool {
        var isIdentity = true
        for row in 0..<4 {
            for column in 0..<4 {
                let expected: Float = row == column ? 1 : 0
                if abs(matrix[row][column] - expected) > epsilon {
                    isIdentity = false
                    break
                }
            }
            if !isIdentity {
                break
            }
        }
        return isIdentity
    }
}

#endif // os(visionOS)
