import Foundation
import simd

public struct SplatScenePoint {
    public static let defaultAlbedo = SIMD3<Float>(repeating: 1)
    public static let defaultMetallic: Float = 0
    public static let defaultRoughness: Float = 1
    public static let defaultNormal = SIMD3<Float>(x: 0, y: 0, z: 1)

    public enum Color {
        static let SH_C0: Float = 0.28209479177387814
        static let INV_SH_C0: Float = 1.0 / SH_C0

        // .sphericalHarmonic should have 1 values (for N=0 aka first-order spherical harmonics), 4 (for N=1), 9 (for N=2), or 16 (for N=3)
        case sphericalHarmonic([SIMD3<Float>])
        case linearFloat(SIMD3<Float>)
        case linearFloat256(SIMD3<Float>)
        case linearUInt8(SIMD3<UInt8>)

        private static func primarySphericalHarmonicToLinear(_ sh0: SIMD3<Float>) -> SIMD3<Float> {
            SIMD3(x: (0.5 + Self.SH_C0 * sh0.x).clamped(to: 0...1),
                  y: (0.5 + Self.SH_C0 * sh0.y).clamped(to: 0...1),
                  z: (0.5 + Self.SH_C0 * sh0.z).clamped(to: 0...1))
        }

        private static func linearToPrimarySphericalHarmonic(_ values: SIMD3<Float>) -> SIMD3<Float> {
            (values - 0.5) * Self.INV_SH_C0
        }

        public var asSphericalHarmonic: [SIMD3<Float>] {
            switch self {
            case let .sphericalHarmonic(values):
                values
            case let .linearFloat(values):
                [Self.linearToPrimarySphericalHarmonic(values)]
            case let .linearFloat256(values):
                [Self.linearToPrimarySphericalHarmonic(values / 256)]
            case let .linearUInt8(values):
                [Self.linearToPrimarySphericalHarmonic(values.asFloat / 255)]
            }
        }

        public var asLinearFloat: SIMD3<Float> {
            switch self {
            case let .sphericalHarmonic(sh):
                Self.primarySphericalHarmonicToLinear(sh[0])
            case let .linearFloat(value):
                value
            case let .linearFloat256(value):
                value / 256
            case let .linearUInt8(value):
                value.asFloat / 255
            }
        }

        public var asLinearFloat256: SIMD3<Float> {
            switch self {
            case let .sphericalHarmonic(sh):
                Self.primarySphericalHarmonicToLinear(sh[0]) * 256
            case let .linearFloat(value):
                value * 256
            case let .linearFloat256(value):
                value
            case let .linearUInt8(value):
                value.asFloat
            }
        }

        public var asLinearUInt8: SIMD3<UInt8> {
            switch self {
            case let .sphericalHarmonic(sh):
                (Self.primarySphericalHarmonicToLinear(sh[0]) * 256).asUInt8
            case let .linearFloat(value):
                (value * 256).asUInt8
            case let .linearFloat256(value):
                value.asUInt8
            case let .linearUInt8(value):
                value
            }
        }
    }

    public enum Opacity {
        case logitFloat(Float)
        case linearFloat(Float)
        case linearUInt8(UInt8)

        static func sigmoid(_ value: Float) -> Float {
            1 / (1 + exp(-value))
        }

        // Inverse sigmoid
        static func logit(_ value: Float) -> Float {
            log(value / (1 - value))
        }

        public var asLogitFloat: Float {
            switch self {
            case let .logitFloat(value):
                value
            case let .linearFloat(value):
                Self.logit(value)
            case let .linearUInt8(value):
                Self.logit((Float(value) + 0.5) / 256) // logit(0) and logit(1) are undefined; so map them to >0 and <1
            }
        }

        public var asLinearFloat: Float {
            switch self {
            case let .logitFloat(value):
                Self.sigmoid(value)
            case let .linearFloat(value):
                value
            case let .linearUInt8(value):
                Float(value) / 255.0
            }
        }

        public var asLinearUInt8: UInt8 {
            switch self {
            case let .logitFloat(value):
                (Self.sigmoid(value) * 256).asUInt8
            case let .linearFloat(value):
                (value * 256).asUInt8
            case let .linearUInt8(value):
                value
            }
        }
    }

    public enum Scale {
        case exponent(SIMD3<Float>)
        case linearFloat(SIMD3<Float>)

        public var asExponent: SIMD3<Float> {
            switch self {
            case let .exponent(value):
                value
            case let .linearFloat(value):
                log(value)
            }
        }

        public var asLinearFloat: SIMD3<Float> {
            switch self {
            case let .exponent(value):
                exp(value)
            case let .linearFloat(value):
                value
            }
        }
    }

    public enum AlbedoInput {
        case girRawLinearFloat(SIMD3<Float>)
        case linearFloat(SIMD3<Float>)
        case linearFloat256(SIMD3<Float>)
        case linearUInt8(SIMD3<UInt8>)
    }

    public enum UnitScalarInput {
        case float(Float)
        case float256(Float)
        case uint8(UInt8)
    }

    public var position: SIMD3<Float>
    public var color: Color
    public var opacity: Opacity
    public var scale: Scale
    public var rotation: simd_quatf
    public var albedo: SIMD3<Float>
    public var metallic: Float
    public var roughness: Float
    public var normal: SIMD3<Float>
    public var hasSerializedNormal: Bool

    public init(position: SIMD3<Float>,
                color: Color,
                opacity: Opacity,
                scale: Scale,
                rotation: simd_quatf,
                albedo: SIMD3<Float> = Self.defaultAlbedo,
                metallic: Float = Self.defaultMetallic,
                roughness: Float = Self.defaultRoughness,
                normal: SIMD3<Float> = Self.defaultNormal,
                normalWasProvided: Bool = false) {
        self.position = position
        self.color = color
        self.opacity = opacity
        self.scale = scale
        self.rotation = rotation
        self.albedo = albedo.clamped01()
        self.metallic = metallic.clamped(to: 0...1)
        self.roughness = roughness.clamped(to: 0...1)
        self.normal = normal.normalizedOrDefault(Self.defaultNormal)
        self.hasSerializedNormal = normalWasProvided
    }

    public var linearNormalized: SplatScenePoint {
        SplatScenePoint(position: position,
                        color: .linearFloat(color.asLinearFloat),
                        opacity: .linearFloat(opacity.asLinearFloat),
                        scale: .linearFloat(scale.asLinearFloat),
                        rotation: rotation.normalized,
                        albedo: albedo.clamped01(),
                        metallic: metallic.clamped(to: 0...1),
                        roughness: roughness.clamped(to: 0...1),
                        normal: normal.normalizedOrDefault(Self.defaultNormal),
                        normalWasProvided: hasSerializedNormal)
    }

    public mutating func setAlbedo(_ input: AlbedoInput) {
        switch input {
        case .girRawLinearFloat(let values):
            let converted = values + SIMD3<Float>(repeating: 0.5)
            albedo = converted.clamped01()
        case .linearFloat(let values):
            albedo = values.clamped01()
        case .linearFloat256(let values):
            albedo = (values / 256).clamped01()
        case .linearUInt8(let values):
            albedo = (values.asFloat / 255).clamped01()
        }
    }

    public mutating func setMetallic(_ input: UnitScalarInput) {
        switch input {
        case .float(let value):
            metallic = value.clamped(to: 0...1)
        case .float256(let value):
            metallic = (value / 256).clamped(to: 0...1)
        case .uint8(let value):
            metallic = (Float(value) / 255.0).clamped(to: 0...1)
        }
    }

    public mutating func setRoughness(_ input: UnitScalarInput) {
        switch input {
        case .float(let value):
            roughness = value.clamped(to: 0...1)
        case .float256(let value):
            roughness = (value / 256).clamped(to: 0...1)
        case .uint8(let value):
            roughness = (Float(value) / 255.0).clamped(to: 0...1)
        }
    }

    public mutating func setNormal(_ newNormal: SIMD3<Float>, isProvided: Bool = true) {
        normal = newNormal.normalizedOrDefault(Self.defaultNormal)
        hasSerializedNormal = isProvided
    }
}

fileprivate extension SIMD3 where Scalar == Float {
    var asUInt8: SIMD3<UInt8> {
        SIMD3<UInt8>(x: x.asUInt8, y: y.asUInt8, z: z.asUInt8)
    }

    func clamped01() -> SIMD3<Float> {
        SIMD3(x: x.clamped(to: 0...1),
              y: y.clamped(to: 0...1),
              z: z.clamped(to: 0...1))
    }

    func normalizedOrDefault(_ defaultValue: SIMD3<Float>) -> SIMD3<Float> {
        let length = simd_length(self)
        if length > .leastNonzeroMagnitude {
            return self / length
        }
        return defaultValue
    }
}

fileprivate extension SIMD3 where Scalar == UInt8 {
    var asFloat: SIMD3<Float> {
        SIMD3<Float>(x: Float(x), y: Float(y), z: Float(z))
    }
}

fileprivate extension Float {
    var asUInt8: UInt8 {
        UInt8(clamped(to: 0.0...255.0))
    }
}
