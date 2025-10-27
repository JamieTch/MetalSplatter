import Foundation

enum ModelIdentifier: Equatable, Hashable, Codable, CustomStringConvertible {
    case gaussianSplat(URL)
    case sampleBox

    var description: String {
        switch self {
        case .gaussianSplat(let url):
            let filename = url.deletingPathExtension().lastPathComponent
            return "Gaussian Splat: \(filename)"
        case .sampleBox:
            return("Sample Box")
        }
    }
}
