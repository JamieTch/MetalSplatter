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

    private var resourceSubdirectory: String { "GaussianSplats" }

    var bundleURL: URL? {
        Bundle.main.url(
            forResource: resourceFileName,
            withExtension: "ply",
            subdirectory: resourceSubdirectory
        )
    }

    var resourceMissingMessage: String {
        "Could not find \(resourceFileName).ply in the bundled Gaussian Splats resources folder (\(resourceSubdirectory)). Add the file to SampleApp/Resources/\(resourceSubdirectory) before building."
    }
}
