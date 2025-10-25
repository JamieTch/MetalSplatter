import Foundation
import SwiftUI

enum Constants {
    static let maxSimultaneousRenders = 3
#if !os(visionOS)
    static let rotationPerSecond = Angle(degrees: 7)
#endif
    static let rotationAxis = SIMD3<Float>(0, 1, 0)
#if !os(visionOS)
    static let fovy = Angle(degrees: 65)
#endif
    static let modelCenterZ: Float = -8
}

