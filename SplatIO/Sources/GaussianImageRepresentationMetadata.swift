import Foundation

public struct GaussianImageRepresentationSphericalHarmonicsUsage: OptionSet, Sendable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    public static let diffuse = GaussianImageRepresentationSphericalHarmonicsUsage(rawValue: 1 << 0)
    public static let specular = GaussianImageRepresentationSphericalHarmonicsUsage(rawValue: 1 << 1)
    public static let all: GaussianImageRepresentationSphericalHarmonicsUsage = [.diffuse, .specular]
}

public struct GaussianImageRepresentationMetadata: Sendable {
    public var sphericalHarmonicsCoefficientCount: UInt32?
    public var sphericalHarmonicsUsage: GaussianImageRepresentationSphericalHarmonicsUsage?

    public init(sphericalHarmonicsCoefficientCount: UInt32? = nil,
                sphericalHarmonicsUsage: GaussianImageRepresentationSphericalHarmonicsUsage? = nil) {
        self.sphericalHarmonicsCoefficientCount = sphericalHarmonicsCoefficientCount
        self.sphericalHarmonicsUsage = sphericalHarmonicsUsage
    }
}
