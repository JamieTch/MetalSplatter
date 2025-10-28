import Foundation
import MetalSplatter

final class RendererSettings: ObservableObject {
    @Published var debugViewMode: SplatRenderer.DebugViewMode = .albedo
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

    var advancedDebugModes: [SplatRenderer.DebugViewMode] {
        SplatRenderer.DebugViewMode.allCases.filter { mode in
            !primaryDebugModes.contains(mode)
        }
    }
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
        case .sphericalHarmonicsDiffuse:
            return "SH Diffuse"
        case .sphericalHarmonicsSpecular:
            return "SH Specular"
        }
    }
}
