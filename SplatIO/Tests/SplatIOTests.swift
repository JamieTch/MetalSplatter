import XCTest
import Spatial
import SplatIO
import simd

final class SplatIOTests: XCTestCase {
    class ContentCounter: SplatSceneReaderDelegate {
        var expectedPointCount: UInt32?
        var pointCount: UInt32 = 0
        var didFinish = false
        var didFail = false

        func reset() {
            expectedPointCount = nil
            pointCount = 0
            didFinish = false
            didFail = false
        }

        func didStartReading(withPointCount pointCount: UInt32?) {
            XCTAssertNil(expectedPointCount)
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            expectedPointCount = pointCount
        }

        func didRead(points: [SplatIO.SplatScenePoint]) {
            pointCount += UInt32(points.count)
        }

        func didFinishReading() {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFinish = true
        }

        func didFailReading(withError error: Error?) {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFail = true
        }
    }

    class ContentStorage: SplatSceneReaderDelegate {
        var points: [SplatIO.SplatScenePoint] = []
        var didFinish = false
        var didFail = false

        func reset() {
            points = []
            didFinish = false
            didFail = false
        }

        func didStartReading(withPointCount pointCount: UInt32?) {
            XCTAssertTrue(points.isEmpty)
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
        }

        func didRead(points: [SplatScenePoint]) {
            self.points.append(contentsOf: points)
        }

        func didFinishReading() {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFinish = true
        }

        func didFailReading(withError error: Error?) {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFail = true
        }

        static func testApproximatelyEqual(lhs: ContentStorage, rhs: ContentStorage) {
            XCTAssertEqual(lhs.points.count, rhs.points.count, "Same number of points")
            for (lhsPoint, rhsPoint) in zip(lhs.points, rhs.points) {
                XCTAssertTrue(lhsPoint ~= rhsPoint)
            }
        }
    }

    let plyURL = Bundle.module.url(forResource: "test-splat.3-points-from-train", withExtension: "ply", subdirectory: "TestData")!
    let dotSplatURL = Bundle.module.url(forResource: "test-splat.3-points-from-train", withExtension: "splat", subdirectory: "TestData")!

    func testReadPLY() throws {
        try testRead(plyURL)
    }

    func textReadDotSplat() throws {
        try testRead(dotSplatURL)
    }

    func testFormatsEqual() throws {
        try testEqual(plyURL, dotSplatURL)
    }

    func testRewritePLY() throws {
        try testReadWriteRead(plyURL, writePLY: true)
        try testReadWriteRead(plyURL, writePLY: false)
    }

    func testRewriteDotSplat() throws {
        try testReadWriteRead(dotSplatURL, writePLY: true)
        try testReadWriteRead(dotSplatURL, writePLY: false)
    }

    func testEqual(_ urlA: URL, _ urlB: URL) throws {
        let readerA = try AutodetectSceneReader(urlA)
        let contentA = ContentStorage()
        readerA.read(to: contentA)

        let readerB = try AutodetectSceneReader(urlB)
        let contentB = ContentStorage()
        readerB.read(to: contentB)

        ContentStorage.testApproximatelyEqual(lhs: contentA, rhs: contentB)
    }

    func testReadWriteRead(_ url: URL, writePLY: Bool) throws {
        let readerA = try AutodetectSceneReader(url)
        let contentA = ContentStorage()
        readerA.read(to: contentA)

        let memoryOutput = DataOutputStream()
        memoryOutput.open()
        let writer: any SplatSceneWriter
        switch writePLY {
        case true:
            let plyWriter = SplatPLYSceneWriter(memoryOutput)
            try plyWriter.start(pointCount: contentA.points.count)
            writer = plyWriter
        case false:
            writer = DotSplatSceneWriter(memoryOutput)
        }
        try writer.write(contentA.points)

        let memoryInput = InputStream(data: memoryOutput.data)
        memoryInput.open()

        let readerB: any SplatSceneReader = writePLY ? SplatPLYSceneReader(memoryInput) : DotSplatSceneReader(memoryInput)
        let contentB = ContentStorage()
        readerB.read(to: contentB)

        ContentStorage.testApproximatelyEqual(lhs: contentA, rhs: contentB)
    }

    func testRead(_ url: URL) throws {
        let reader = try AutodetectSceneReader(url)

        let content = ContentCounter()
        reader.read(to: content)
        XCTAssertTrue(content.didFinish)
        XCTAssertFalse(content.didFail)
        if let expectedPointCount = content.expectedPointCount {
            XCTAssertEqual(expectedPointCount, content.pointCount)
        }
    }

    private func readPoints(fromASCII ascii: String) throws -> [SplatScenePoint] {
        let data = Data(ascii.utf8)
        let stream = InputStream(data: data)
        stream.open()
        defer { stream.close() }

        let reader = SplatPLYSceneReader(stream)
        let content = ContentStorage()
        reader.read(to: content)
        XCTAssertTrue(content.didFinish)
        XCTAssertFalse(content.didFail)
        return content.points
    }

    func testPLYMaterialFloat32() throws {
        let ascii = """
        ply
        format ascii 1.0
        element vertex 1
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float base_color_r
        property float base_color_g
        property float base_color_b
        property float metallic
        property float roughness
        end_header
        0 0 0 0 0 1 0 0 0 0 1 1 1 1 0 0 0 0.25 0.5 0.75 0.2 0.8
        """

        let points = try readPoints(fromASCII: ascii)
        XCTAssertEqual(points.count, 1)
        let point = try XCTUnwrap(points.first)
        XCTAssertEqual(point.albedo.x, 0.25, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.y, 0.5, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.z, 0.75, accuracy: 1e-5)
        XCTAssertEqual(point.metallic, 0.2, accuracy: 1e-5)
        XCTAssertEqual(point.roughness, 0.8, accuracy: 1e-5)
        XCTAssertEqual(point.normal.x, 0.0, accuracy: 1e-5)
        XCTAssertEqual(point.normal.y, 0.0, accuracy: 1e-5)
        XCTAssertEqual(point.normal.z, 1.0, accuracy: 1e-5)
        XCTAssertTrue(point.hasSerializedNormal)
    }

    func testPLYMaterialFloat32WithoutNormalsFallsBack() throws {
        let ascii = """
        ply
        format ascii 1.0
        element vertex 1
        property float x
        property float y
        property float z
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float albedo_r
        property float albedo_g
        property float albedo_b
        property float metallic
        property float roughness
        end_header
        0 0 0 0 0 0 0 1 1 1 0 0 0 0 0.25 0.5 0.75 0.2 0.8
        """

        let points = try readPoints(fromASCII: ascii)
        XCTAssertEqual(points.count, 1)
        let point = try XCTUnwrap(points.first)
        XCTAssertEqual(point.albedo.x, 0.25, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.y, 0.5, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.z, 0.75, accuracy: 1e-5)
        XCTAssertEqual(point.metallic, 0.2, accuracy: 1e-5)
        XCTAssertEqual(point.roughness, 0.8, accuracy: 1e-5)
        XCTAssertEqual(point.normal.x, 0.0, accuracy: 1e-5)
        XCTAssertEqual(point.normal.y, 0.0, accuracy: 1e-5)
        XCTAssertEqual(point.normal.z, 1.0, accuracy: 1e-5)
        XCTAssertFalse(point.hasSerializedNormal)
    }

    func testPLYMaterialFloat32GIRRaw() throws {
        let ascii = """
        ply
        format ascii 1.0
        element vertex 1
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float albedo_r
        property float albedo_g
        property float albedo_b
        property float metallic
        property float roughness
        end_header
        0 0 0 0 0 1 0 0 0 0 1 1 1 1 0 0 0 -0.25 0 0.25 0.2 0.8
        """

        let points = try readPoints(fromASCII: ascii)
        XCTAssertEqual(points.count, 1)
        let point = try XCTUnwrap(points.first)
        XCTAssertEqual(point.albedo.x, 0.25, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.y, 0.5, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.z, 0.75, accuracy: 1e-5)
        XCTAssertEqual(point.metallic, 0.2, accuracy: 1e-5)
        XCTAssertEqual(point.roughness, 0.8, accuracy: 1e-5)
        XCTAssertEqual(point.normal.x, 0.0, accuracy: 1e-5)
        XCTAssertEqual(point.normal.y, 0.0, accuracy: 1e-5)
        XCTAssertEqual(point.normal.z, 1.0, accuracy: 1e-5)
        XCTAssertTrue(point.hasSerializedNormal)
    }

    func testPLYMaterialFloat32Times256() throws {
        let ascii = """
        ply
        format ascii 1.0
        element vertex 1
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float base_color_r
        property float base_color_g
        property float base_color_b
        property float metallic
        property float roughness
        end_header
        0 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 64 128 192 128 64
        """

        let points = try readPoints(fromASCII: ascii)
        XCTAssertEqual(points.count, 1)
        let point = try XCTUnwrap(points.first)
        XCTAssertEqual(point.albedo.x, 64.0 / 256.0, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.y, 128.0 / 256.0, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.z, 192.0 / 256.0, accuracy: 1e-5)
        XCTAssertEqual(point.metallic, 128.0 / 256.0, accuracy: 1e-5)
        XCTAssertEqual(point.roughness, 64.0 / 256.0, accuracy: 1e-5)
        let expectedNormal = simd_normalize(SIMD3<Float>(1, 1, 1))
        XCTAssertEqual(point.normal.x, expectedNormal.x, accuracy: 1e-5)
        XCTAssertEqual(point.normal.y, expectedNormal.y, accuracy: 1e-5)
        XCTAssertEqual(point.normal.z, expectedNormal.z, accuracy: 1e-5)
        XCTAssertTrue(point.hasSerializedNormal)
    }

    func testPLYMaterialUInt8() throws {
        let ascii = """
        ply
        format ascii 1.0
        element vertex 1
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property uchar albedo_r
        property uchar albedo_g
        property uchar albedo_b
        property uchar metallic
        property uchar roughness
        end_header
        0 0 0 0 1 0 0 0 0 0 1 1 1 1 0 0 0 64 128 255 32 224
        """

        let points = try readPoints(fromASCII: ascii)
        XCTAssertEqual(points.count, 1)
        let point = try XCTUnwrap(points.first)
        XCTAssertEqual(point.albedo.x, 64.0 / 255.0, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.y, 128.0 / 255.0, accuracy: 1e-5)
        XCTAssertEqual(point.albedo.z, 255.0 / 255.0, accuracy: 1e-5)
        XCTAssertEqual(point.metallic, 32.0 / 255.0, accuracy: 1e-5)
        XCTAssertEqual(point.roughness, 224.0 / 255.0, accuracy: 1e-5)
        let expectedNormal = SIMD3<Float>(0, 1, 0)
        XCTAssertEqual(point.normal.x, expectedNormal.x, accuracy: 1e-5)
        XCTAssertEqual(point.normal.y, expectedNormal.y, accuracy: 1e-5)
        XCTAssertEqual(point.normal.z, expectedNormal.z, accuracy: 1e-5)
    }
}

extension SplatScenePoint {
    enum Tolerance {
        static let position: Float = 1e-10
        static let color: Float = 1.0 / 256
        static let opacity: Float = 1.0 / 256
        static let scale: Float = 1e-10
        static let rotation: Float = 2.0 / 128
        static let albedo: Float = 1.0 / 256
        static let metallic: Float = 1.0 / 255
        static let roughness: Float = 1.0 / 255
        static let normal: Float = 1e-6
    }

    public static func ~= (lhs: SplatScenePoint, rhs: SplatScenePoint) -> Bool {
        (lhs.position - rhs.position).isWithin(tolerance: Tolerance.position) &&
        lhs.color ~= rhs.color &&
        lhs.opacity ~= rhs.opacity &&
        lhs.scale ~= rhs.scale &&
        (lhs.rotation.normalized.vector - rhs.rotation.normalized.vector).isWithin(tolerance: Tolerance.rotation) &&
        (lhs.albedo - rhs.albedo).isWithin(tolerance: Tolerance.albedo) &&
        abs(lhs.metallic - rhs.metallic) <= Tolerance.metallic &&
        abs(lhs.roughness - rhs.roughness) <= Tolerance.roughness &&
        (lhs.normal - rhs.normal).isWithin(tolerance: Tolerance.normal)
    }
}

extension SplatScenePoint.Color {
    public static func ~= (lhs: SplatScenePoint.Color, rhs: SplatScenePoint.Color) -> Bool {
        (lhs.asLinearFloat - rhs.asLinearFloat).isWithin(tolerance: SplatScenePoint.Tolerance.color)
    }
}

extension SplatScenePoint.Opacity {
    public static func ~= (lhs: SplatScenePoint.Opacity, rhs: SplatScenePoint.Opacity) -> Bool {
        abs(lhs.asLinearFloat - rhs.asLinearFloat) <= SplatScenePoint.Tolerance.opacity
    }
}

extension SplatScenePoint.Scale {
    public static func ~= (lhs: SplatScenePoint.Scale, rhs: SplatScenePoint.Scale) -> Bool {
        (lhs.asLinearFloat - rhs.asLinearFloat).isWithin(tolerance: SplatScenePoint.Tolerance.scale)
    }
}

extension SIMD3 where Scalar: Comparable & SignedNumeric {
    public func isWithin(tolerance: Scalar) -> Bool {
        abs(x) <= tolerance && abs(y) <= tolerance && abs(z) <= tolerance
    }
}

extension SIMD4 where Scalar: Comparable & SignedNumeric {
    public func isWithin(tolerance: Scalar) -> Bool {
        abs(x) <= tolerance && abs(y) <= tolerance && abs(z) <= tolerance && abs(w) <= tolerance
    }
}

private class DataOutputStream: OutputStream {
    var data = Data()

    override func open() {}
    override func close() {}
    override var hasSpaceAvailable: Bool { true }

    override func write(_ buffer: UnsafePointer<UInt8>, maxLength length: Int) -> Int {
        data.append(buffer, count: length)
        return length
    }
}

private extension SIMD3 where Scalar == Float {
    var magnitude: Scalar {
        sqrt(x*x + y*y + z*z)
    }
}

private extension SIMD4 where Scalar == Float {
    var magnitude: Scalar {
        sqrt(x*x + y*y + z*z + w*w)
    }
}
