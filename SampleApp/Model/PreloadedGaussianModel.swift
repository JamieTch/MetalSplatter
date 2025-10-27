import Foundation

enum PreloadedGaussianModel: String, CaseIterable, Identifiable {
    case holzablage
    case klavierbank
    case glastisch
    case holztisch
    case lampe

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .holzablage:
            return "Holzablage"
        case .klavierbank:
            return "Klavierbank"
        case .glastisch:
            return "Glastisch"
        case .holztisch:
            return "Holztisch"
        case .lampe:
            return "Lampe"
        }
    }

    var resourceFileName: String {
        "GIR_\(rawValue)_30000"
    }

    var bundleURL: URL? {
        Bundle.main.url(forResource: resourceFileName, withExtension: "ply")
    }

    var resourceMissingMessage: String {
        "Could not find \(resourceFileName).ply in the bundled Gaussian Splats resources. Add the file to SampleApp/Resources/GaussianSplats before building."
    }
}
