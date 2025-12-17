import Combine
import Foundation
import MetalSplatter

final class RendererSettings: ObservableObject {
    enum CalibrationMode {
        case idle
        case running
    }

    enum CalibrationCommand {
        case start
        case confirm
        case cancel
        case reset
    }

    @Published var debugViewMode: SplatRenderer.DebugViewMode = .albedo
    @Published var handInteractionEnabled: Bool = true
    @Published var activeModel: ModelIdentifier? {
        didSet {
            if oldValue != activeModel {
                calibrationMode = .idle
                handInteractionEnabled = true
            }
        }
    }
    @Published var calibrationMode: CalibrationMode = .idle

    let calibrationCommands = PassthroughSubject<CalibrationCommand, Never>()
    let primaryDebugModes: [SplatRenderer.DebugViewMode] = [
        .shaded,
        .albedo,
        .normal,
        .roughness,
        .metallic,
        .ambientOcclusion,
        .depth,
        .coverage
    ]
}

extension SplatRenderer.DebugViewMode {
    var displayName: String {
        switch self {
        case .coverage:
            return "Coverage"
        case .albedo:
            return "Albedo"
        case .normal:
            return "Normal"
        case .roughness:
            return "Roughness"
        case .metallic:
            return "Metallic"
        case .ambientOcclusion:
            return "AO"
        case .depth:
            return "Depth"
        case .shaded:
            return "Shaded"
        case .shadedAmbientOcclusionUnity:
            return "Shaded (AO=1)"
        case .environmentReflection:
            return "Reflection"
        case .normalViewRelationship:
            return "N·V"
        case .environmentPanorama:
            return "Env Panorama"
        case .lambert:
            return "Lambert"
        case .brdfLookup:
            return "BRDF LUT"
        case .environmentFixedLod0:
            return "Env LOD0"
        case .environmentFixedMaxLod:
            return "Env Max LOD"
        case .albedoLinear:
            return "Albedo (Linear)"
        case .roughnessSweep:
            return "Roughness (Alt)"
        case .metallicSweep:
            return "Metallic (Alt)"
        case .normalSweep:
            return "Normal (Alt)"
        case .normalRaw:
            return "Normal Raw"
        case .normalDifference:
            return "Normal Δ"
        case .normalDotComparison:
            return "N·V Compare"
        }
    }
}
