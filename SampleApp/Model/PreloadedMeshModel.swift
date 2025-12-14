import Foundation

enum PreloadedMeshModel: String, CaseIterable, Identifiable {
    case cube

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .cube:
            return "Sample Mesh Cube"
        }
    }

    private var resourceSubdirectory: String { "Meshes" }

    var bundleURL: URL? {
        Bundle.main.url(forResource: "Cube", withExtension: "obj", subdirectory: resourceSubdirectory)
    }
}
