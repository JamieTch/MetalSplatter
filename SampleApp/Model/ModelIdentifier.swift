import Foundation

enum ModelIdentifier: Equatable, Hashable, Codable, CustomStringConvertible {
    case gaussianSplat(URL)
    case mesh(URL)
    case sampleBox

    var description: String {
        switch self {
        case .gaussianSplat(let url):
            let filename = url.deletingPathExtension().lastPathComponent
            return "Gaussian Splat: \(filename)"
        case .mesh(let url):
            let filename = url.deletingPathExtension().lastPathComponent
            return "Mesh: \(filename)"
        case .sampleBox:
            return("Sample Box")
        }
    }

    var calibrationKey: String {
        switch self {
        case .gaussianSplat(let url):
            let baseName = url.deletingPathExtension().lastPathComponent
            let canonicalPath = url.standardizedFileURL.absoluteString
            var hash: UInt64 = 5381
            for byte in canonicalPath.utf8 {
                hash = ((hash << 5) &+ hash) &+ UInt64(byte)
            }
            let suffix = String(format: "%016llx", hash)
            return "\(baseName)-\(suffix)"
        case .mesh(let url):
            let baseName = url.deletingPathExtension().lastPathComponent
            let canonicalPath = url.standardizedFileURL.absoluteString
            var hash: UInt64 = 5381
            for byte in canonicalPath.utf8 {
                hash = ((hash << 5) &+ hash) &+ UInt64(byte)
            }
            let suffix = String(format: "%016llx", hash)
            return "mesh-\(baseName)-\(suffix)"
        case .sampleBox:
            return "sample-box"
        }
    }

    var supportsCalibration: Bool {
        switch self {
        case .gaussianSplat, .mesh:
            return true
        case .sampleBox:
            return false
        }
    }
}
