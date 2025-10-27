import Foundation
import simd

#if os(visionOS)

struct ModelCalibration: Codable {
    struct AnchorMetadata: Codable {
        var transform: simd_float4x4

        init(transform: simd_float4x4) {
            self.transform = transform
        }

        private enum CodingKeys: String, CodingKey {
            case transform
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(Self.flatten(transform), forKey: .transform)
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let values = try container.decode([Float].self, forKey: .transform)
            guard let matrix = Self.matrix(from: values) else {
                throw DecodingError.dataCorruptedError(forKey: .transform,
                                                      in: container,
                                                      debugDescription: "Expected 16-element array for simd_float4x4")
            }
            transform = matrix
        }

        private static func flatten(_ matrix: simd_float4x4) -> [Float] {
            let columns = matrix.columns
            return [
                columns.0.x, columns.0.y, columns.0.z, columns.0.w,
                columns.1.x, columns.1.y, columns.1.z, columns.1.w,
                columns.2.x, columns.2.y, columns.2.z, columns.2.w,
                columns.3.x, columns.3.y, columns.3.z, columns.3.w
            ]
        }

        private static func matrix(from values: [Float]) -> simd_float4x4? {
            guard values.count == 16 else { return nil }
            return simd_float4x4(columns: (
                SIMD4(values[0], values[1], values[2], values[3]),
                SIMD4(values[4], values[5], values[6], values[7]),
                SIMD4(values[8], values[9], values[10], values[11]),
                SIMD4(values[12], values[13], values[14], values[15])
            ))
        }
    }

    var translation: SIMD3<Float>
    var rotation: simd_quatf
    var scale: Float
    var anchor: AnchorMetadata?

    init(translation: SIMD3<Float>,
         rotation: simd_quatf,
         scale: Float,
         anchor: AnchorMetadata? = nil) {
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.anchor = anchor
    }

    private enum CodingKeys: String, CodingKey {
        case translation
        case rotation
        case scale
        case anchor
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        translation = try container.decode(SIMD3<Float>.self, forKey: .translation)
        let rotationVector = try container.decode(SIMD4<Float>.self, forKey: .rotation)
        rotation = simd_quatf(vector: rotationVector)
        scale = try container.decode(Float.self, forKey: .scale)
        anchor = try container.decodeIfPresent(AnchorMetadata.self, forKey: .anchor)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(translation, forKey: .translation)
        try container.encode(rotation.vector, forKey: .rotation)
        try container.encode(scale, forKey: .scale)
        try container.encodeIfPresent(anchor, forKey: .anchor)
    }
}

#else

struct ModelCalibration: Codable {
    struct AnchorMetadata: Codable {}

    var translation: SIMD3<Float>
    var rotation: simd_quatf
    var scale: Float
    var anchor: AnchorMetadata?

    init(translation: SIMD3<Float> = .zero,
         rotation: simd_quatf = simd_quatf(),
         scale: Float = 1.0,
         anchor: AnchorMetadata? = nil) {
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.anchor = anchor
    }
}

#endif
