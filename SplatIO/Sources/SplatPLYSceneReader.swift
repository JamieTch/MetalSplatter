import Foundation
import PLYIO
import simd

public class SplatPLYSceneReader: SplatSceneReader {
    enum Error: LocalizedError {
        case unsupportedFileContents(String?)
        case unexpectedPointCountDiscrepancy
        case internalConsistency(String?)

        public var errorDescription: String? {
            switch self {
            case .unsupportedFileContents(let description):
                if let description {
                    "Unexpected file contents for a splat PLY: \(description)"
                } else {
                    "Unexpected file contents for a splat PLY"
                }
            case .unexpectedPointCountDiscrepancy:
                "Unexpected point count discrepancy"
            case .internalConsistency(let description):
                "Internal error in SplatPLYSceneReader: \(description ?? "(unknown)")"
            }
        }
    }

    private let ply: PLYReader

    public convenience init(_ url: URL) throws {
        self.init(try PLYReader(url))
    }

    public convenience init(_ inputStream: InputStream) {
        self.init(PLYReader(inputStream))
    }

    public init(_ ply: PLYReader) {
        self.ply = ply
    }

    public func read(to delegate: SplatSceneReaderDelegate) {
        SplatPLYSceneReaderStream().read(ply, to: delegate)
    }
}

private class SplatPLYSceneReaderStream {
    private weak var delegate: SplatSceneReaderDelegate? = nil
    private var active = false
    private var elementMapping: ElementInputMapping?
    private var expectedPointCount: UInt32 = 0
    private var pointCount: UInt32 = 0
    private var reusablePoint = SplatScenePoint(position: .zero,
                                                color: .linearUInt8(.zero),
                                                opacity: .linearFloat(.zero),
                                                scale: .exponent(.zero),
                                                rotation: .init(vector: .zero))

    func read(_ ply: PLYReader, to delegate: SplatSceneReaderDelegate) {
        self.delegate = delegate
        active = true
        elementMapping = nil
        expectedPointCount = 0
        pointCount = 0

        ply.read(to: self)

        assert(!active)
    }
}

extension SplatPLYSceneReaderStream: PLYReaderDelegate {
    func didStartReading(withHeader header: PLYHeader) {
        guard active else { return }
        guard elementMapping == nil else {
            delegate?.didFailReading(withError: SplatPLYSceneReader.Error.internalConsistency("didStart called while elementMapping is non-nil"))
            active = false
            return
        }

        do {
            let elementMapping = try ElementInputMapping.elementMapping(for: header)
            self.elementMapping = elementMapping
            expectedPointCount = header.elements[elementMapping.elementTypeIndex].count
            delegate?.didStartReading(withPointCount: expectedPointCount)
        } catch {
            delegate?.didFailReading(withError: error)
            active = false
            return
        }
    }

    func didRead(element: PLYElement, typeIndex: Int, withHeader elementHeader: PLYHeader.Element) {
        guard active else { return }
        guard let elementMapping else {
            delegate?.didFailReading(withError: SplatPLYSceneReader.Error.internalConsistency("didRead(element:typeIndex:withHeader:) called but elementMapping is nil"))
            active = false
            return
        }

        guard typeIndex == elementMapping.elementTypeIndex else { return }
        do {
            try elementMapping.apply(from: element, to: &reusablePoint)
            pointCount += 1
            delegate?.didRead(points: [ reusablePoint ])
        } catch {
            delegate?.didFailReading(withError: error)
            active = false
            return
        }
    }

    func didFinishReading() {
        guard active else { return }
        guard expectedPointCount == pointCount else {
            delegate?.didFailReading(withError: SplatPLYSceneReader.Error.unexpectedPointCountDiscrepancy)
            active = false
            return
        }

        delegate?.didFinishReading()
        active = false
    }

    func didFailReading(withError error: Swift.Error?) {
        guard active else { return }
        delegate?.didFailReading(withError: error)
        active = false
    }
}

private struct ElementInputMapping {
    public enum Color {
        case sphericalHarmonic([SIMD3<Int>])
        case linearFloat256(SIMD3<Int>)
        case linearUInt8(SIMD3<Int>)
    }

    enum Albedo {
        case float32(SIMD3<Int>)
        case uint8(SIMD3<Int>)
    }

    enum UnitScalar {
        case float32(Int)
        case uint8(Int)
    }

    static let sphericalHarmonicsCount = 45
    static let float256Threshold: Float = 1.0 + 1e-4

    let elementTypeIndex: Int

    let positionXPropertyIndex: Int
    let positionYPropertyIndex: Int
    let positionZPropertyIndex: Int
    let colorPropertyIndices: Color
    let scaleXPropertyIndex: Int
    let scaleYPropertyIndex: Int
    let scaleZPropertyIndex: Int
    let opacityPropertyIndex: Int
    let rotation0PropertyIndex: Int
    let rotation1PropertyIndex: Int
    let rotation2PropertyIndex: Int
    let rotation3PropertyIndex: Int
    let normalPropertyIndices: SIMD3<Int>?
    let albedoPropertyIndices: Albedo?
    let metallicPropertyIndex: UnitScalar?
    let roughnessPropertyIndex: UnitScalar?

    static func elementMapping(for header: PLYHeader) throws -> ElementInputMapping {
        guard let elementTypeIndex = header.index(forElementNamed: SplatPLYConstants.ElementName.point.rawValue) else {
            throw SplatPLYSceneReader.Error.unsupportedFileContents("No element type \"\(SplatPLYConstants.ElementName.point.rawValue)\" found")
        }
        let headerElement = header.elements[elementTypeIndex]

        let positionXPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.positionX)
        let positionYPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.positionY)
        let positionZPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.positionZ)

        let color: Color
        if let sh0_rPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.sh0_r),
           let sh0_gPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.sh0_g),
            let sh0_bPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.sh0_b) {
            let primaryColorPropertyIndices = SIMD3<Int>(x: sh0_rPropertyIndex, y: sh0_gPropertyIndex, z: sh0_bPropertyIndex)
            if headerElement.hasProperty(forName: "\(SplatPLYConstants.PropertyName.sphericalHarmonicsPrefix)0") {
                let individualSphericalHarmonicsPropertyIndices: [Int] = try (0..<sphericalHarmonicsCount).map {
                    try headerElement.index(forFloat32PropertyNamed: [ "\(SplatPLYConstants.PropertyName.sphericalHarmonicsPrefix)\($0)" ])
                }
                let sphericalHarmonicsPropertyIndices: [SIMD3<Int>] = stride(from: 0, to: individualSphericalHarmonicsPropertyIndices.count, by: 3).map {
                    SIMD3<Int>(individualSphericalHarmonicsPropertyIndices[$0],
                               individualSphericalHarmonicsPropertyIndices[$0 + 1],
                               individualSphericalHarmonicsPropertyIndices[$0 + 2])
                }
                color = .sphericalHarmonic([primaryColorPropertyIndices] + sphericalHarmonicsPropertyIndices)
            } else {
                color = .sphericalHarmonic([primaryColorPropertyIndices])
            }
        } else if headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorR, type: .float32) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorG, type: .float32) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorB, type: .float32) {
            // Special case for NRRFStudio SH=0 files. This may be fixed now?
            let colorRPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorR, type: .float32)
            let colorGPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorG, type: .float32)
            let colorBPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorB, type: .float32)
            color = .linearFloat256(SIMD3(colorRPropertyIndex, colorGPropertyIndex, colorBPropertyIndex))
        } else if headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorR, type: .uint8) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorG, type: .uint8) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorB, type: .uint8) {
            let colorRPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorR, type: .uint8)
            let colorGPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorG, type: .uint8)
            let colorBPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorB, type: .uint8)
            color = .linearUInt8(SIMD3(colorRPropertyIndex, colorGPropertyIndex, colorBPropertyIndex))
        } else {
            throw SplatPLYSceneReader.Error.unsupportedFileContents("No color property elements found with the expected types")
        }

        let scaleXPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.scaleX)
        let scaleYPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.scaleY)
        let scaleZPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.scaleZ)
        let opacityPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.opacity)

        let rotation0PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation0)
        let rotation1PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation1)
        let rotation2PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation2)
        let rotation3PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation3)

        let normalPropertyIndices: SIMD3<Int>?
        if let normalXPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.normalX),
           let normalYPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.normalY),
           let normalZPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.normalZ) {
            normalPropertyIndices = SIMD3(normalXPropertyIndex, normalYPropertyIndex, normalZPropertyIndex)
        } else {
            normalPropertyIndices = nil
        }

        let albedoPropertyIndices: Albedo?
        if let albedoRFloatPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.albedoR, type: .float32),
           let albedoGFloatPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.albedoG, type: .float32),
           let albedoBFloatPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.albedoB, type: .float32) {
            albedoPropertyIndices = .float32(SIMD3(albedoRFloatPropertyIndex, albedoGFloatPropertyIndex, albedoBFloatPropertyIndex))
        } else if let albedoRUIntPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.albedoR, type: .uint8),
                    let albedoGUIntPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.albedoG, type: .uint8),
                    let albedoBUIntPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.albedoB, type: .uint8) {
            albedoPropertyIndices = .uint8(SIMD3(albedoRUIntPropertyIndex, albedoGUIntPropertyIndex, albedoBUIntPropertyIndex))
        } else {
            albedoPropertyIndices = nil
        }

        let metallicPropertyIndex: UnitScalar?
        if let metallicFloatPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.metallic, type: .float32) {
            metallicPropertyIndex = .float32(metallicFloatPropertyIndex)
        } else if let metallicUIntPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.metallic, type: .uint8) {
            metallicPropertyIndex = .uint8(metallicUIntPropertyIndex)
        } else {
            metallicPropertyIndex = nil
        }

        let roughnessPropertyIndex: UnitScalar?
        if let roughnessFloatPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.roughness, type: .float32) {
            roughnessPropertyIndex = .float32(roughnessFloatPropertyIndex)
        } else if let roughnessUIntPropertyIndex = try headerElement.index(forOptionalPropertyNamed: SplatPLYConstants.PropertyName.roughness, type: .uint8) {
            roughnessPropertyIndex = .uint8(roughnessUIntPropertyIndex)
        } else {
            roughnessPropertyIndex = nil
        }

        return ElementInputMapping(elementTypeIndex: elementTypeIndex,
                                   positionXPropertyIndex: positionXPropertyIndex,
                                   positionYPropertyIndex: positionYPropertyIndex,
                                   positionZPropertyIndex: positionZPropertyIndex,
                                   colorPropertyIndices: color,
                                   scaleXPropertyIndex: scaleXPropertyIndex,
                                   scaleYPropertyIndex: scaleYPropertyIndex,
                                   scaleZPropertyIndex: scaleZPropertyIndex,
                                   opacityPropertyIndex: opacityPropertyIndex,
                                   rotation0PropertyIndex: rotation0PropertyIndex,
                                   rotation1PropertyIndex: rotation1PropertyIndex,
                                   rotation2PropertyIndex: rotation2PropertyIndex,
                                   rotation3PropertyIndex: rotation3PropertyIndex,
                                   normalPropertyIndices: normalPropertyIndices,
                                   albedoPropertyIndices: albedoPropertyIndices,
                                   metallicPropertyIndex: metallicPropertyIndex,
                                   roughnessPropertyIndex: roughnessPropertyIndex)
    }

    func apply(from element: PLYElement, to result: inout SplatScenePoint) throws {
        result.position = SIMD3(x: try element.float32Value(forPropertyIndex: positionXPropertyIndex),
                                y: try element.float32Value(forPropertyIndex: positionYPropertyIndex),
                                z: try element.float32Value(forPropertyIndex: positionZPropertyIndex))

        switch colorPropertyIndices {
        case .sphericalHarmonic(let sphericalHarmonicsPropertyIndices):
            result.color = .sphericalHarmonic(try sphericalHarmonicsPropertyIndices.map {
                try SIMD3<Float>(x: element.float32Value(forPropertyIndex: $0.x),
                                 y: element.float32Value(forPropertyIndex: $0.y),
                                 z: element.float32Value(forPropertyIndex: $0.z))
            })
        case .linearFloat256(let propertyIndices):
            result.color = .linearFloat256(SIMD3(try element.float32Value(forPropertyIndex: propertyIndices.x),
                                                 try element.float32Value(forPropertyIndex: propertyIndices.y),
                                                 try element.float32Value(forPropertyIndex: propertyIndices.z)))
        case .linearUInt8(let propertyIndices):
            result.color = .linearUInt8(SIMD3(try element.uint8Value(forPropertyIndex: propertyIndices.x),
                                              try element.uint8Value(forPropertyIndex: propertyIndices.y),
                                              try element.uint8Value(forPropertyIndex: propertyIndices.z)))
        }

        result.scale =
            .exponent(SIMD3(try element.float32Value(forPropertyIndex: scaleXPropertyIndex),
                            try element.float32Value(forPropertyIndex: scaleYPropertyIndex),
                            try element.float32Value(forPropertyIndex: scaleZPropertyIndex)))
        result.opacity = .logitFloat(try element.float32Value(forPropertyIndex: opacityPropertyIndex))
        result.rotation.real   = try element.float32Value(forPropertyIndex: rotation0PropertyIndex)
        result.rotation.imag.x = try element.float32Value(forPropertyIndex: rotation1PropertyIndex)
        result.rotation.imag.y = try element.float32Value(forPropertyIndex: rotation2PropertyIndex)
        result.rotation.imag.z = try element.float32Value(forPropertyIndex: rotation3PropertyIndex)

        if let normalPropertyIndices {
            result.setNormal(try element.normalizedVector(for: normalPropertyIndices))
        } else {
            result.setNormal(SplatScenePoint.defaultNormal)
        }

        if let albedoPropertyIndices {
            switch albedoPropertyIndices {
            case .float32(let propertyIndices):
                let values = try element.float32Vector(forPropertyIndices: propertyIndices)
                let maxComponent = max(values.x, max(values.y, values.z))
                if maxComponent > ElementInputMapping.float256Threshold {
                    result.setAlbedo(.linearFloat256(values))
                } else {
                    result.setAlbedo(.linearFloat(values))
                }
            case .uint8(let propertyIndices):
                let values = try element.uint8Vector(forPropertyIndices: propertyIndices)
                result.setAlbedo(.linearUInt8(values))
            }
        } else {
            result.albedo = SplatScenePoint.defaultAlbedo
        }

        if let metallicPropertyIndex {
            switch metallicPropertyIndex {
            case .float32(let propertyIndex):
                let value = try element.float32Value(forPropertyIndex: propertyIndex)
                if value > ElementInputMapping.float256Threshold {
                    result.setMetallic(.float256(value))
                } else {
                    result.setMetallic(.float(value))
                }
            case .uint8(let propertyIndex):
                let value = try element.uint8Value(forPropertyIndex: propertyIndex)
                result.setMetallic(.uint8(value))
            }
        } else {
            result.metallic = SplatScenePoint.defaultMetallic
        }

        if let roughnessPropertyIndex {
            switch roughnessPropertyIndex {
            case .float32(let propertyIndex):
                let value = try element.float32Value(forPropertyIndex: propertyIndex)
                if value > ElementInputMapping.float256Threshold {
                    result.setRoughness(.float256(value))
                } else {
                    result.setRoughness(.float(value))
                }
            case .uint8(let propertyIndex):
                let value = try element.uint8Value(forPropertyIndex: propertyIndex)
                result.setRoughness(.uint8(value))
            }
        } else {
            result.roughness = SplatScenePoint.defaultRoughness
        }
    }
}

private extension PLYHeader.Element {
    func hasProperty(forName name: String, type: PLYHeader.PrimitivePropertyType? = nil) -> Bool {
        guard let index = index(forPropertyNamed: name) else {
            return false
        }

        if let type {
            guard case .primitive(type) = properties[index].type else {
                return false
            }
        }

        return true
    }

    func hasProperty(forName names: [String], type: PLYHeader.PrimitivePropertyType? = nil) -> Bool {
        for name in names {
            if hasProperty(forName: name, type: type) {
                return true
            }
        }
        return false
    }

    func index(forOptionalPropertyNamed names: [String], type: PLYHeader.PrimitivePropertyType) throws -> Int? {
        for name in names {
            if let index = index(forPropertyNamed: name) {
                guard case .primitive(type) = properties[index].type else { throw SplatPLYSceneReader.Error.unsupportedFileContents("Unexpected type for property \"\(name)\"") }
                return index
            }
        }
        return nil
    }

    func index(forPropertyNamed names: [String], type: PLYHeader.PrimitivePropertyType) throws -> Int {
        guard let result = try index(forOptionalPropertyNamed: names, type: type) else {
            throw SplatPLYSceneReader.Error.unsupportedFileContents("No property named \"\(names.first ?? "(none)")\" found")
        }
        return result
    }

    func index(forOptionalFloat32PropertyNamed names: [String]) throws -> Int? {
        try index(forOptionalPropertyNamed: names, type: .float32)
    }

    func index(forFloat32PropertyNamed names: [String]) throws -> Int {
        try index(forPropertyNamed: names, type: .float32)
    }
}

private extension PLYElement {
    func float32Value(forPropertyIndex propertyIndex: Int) throws -> Float {
        guard case .float32(let typedValue) = properties[propertyIndex] else { throw SplatPLYSceneReader.Error.internalConsistency("Unexpected type for property at index \(propertyIndex)") }
        return typedValue
    }

    func uint8Value(forPropertyIndex propertyIndex: Int) throws -> UInt8 {
        guard case .uint8(let typedValue) = properties[propertyIndex] else { throw SplatPLYSceneReader.Error.internalConsistency("Unexpected type for property at index \(propertyIndex)") }
        return typedValue
    }

    func float32Vector(forPropertyIndices propertyIndices: SIMD3<Int>) throws -> SIMD3<Float> {
        SIMD3(x: try float32Value(forPropertyIndex: propertyIndices.x),
              y: try float32Value(forPropertyIndex: propertyIndices.y),
              z: try float32Value(forPropertyIndex: propertyIndices.z))
    }

    func uint8Vector(forPropertyIndices propertyIndices: SIMD3<Int>) throws -> SIMD3<UInt8> {
        SIMD3(x: try uint8Value(forPropertyIndex: propertyIndices.x),
              y: try uint8Value(forPropertyIndex: propertyIndices.y),
              z: try uint8Value(forPropertyIndex: propertyIndices.z))
    }

    func normalizedVector(for propertyIndices: SIMD3<Int>) throws -> SIMD3<Float> {
        let vector = try float32Vector(forPropertyIndices: propertyIndices)
        let length = simd_length(vector)
        if length > .leastNonzeroMagnitude {
            return vector / length
        }
        return SplatScenePoint.defaultNormal
    }
}
