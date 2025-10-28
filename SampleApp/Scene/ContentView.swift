import SwiftUI
import RealityKit

struct ContentView: View {
    @State private var isPickingFile = false
    @EnvironmentObject private var rendererSettings: RendererSettings
    @State private var missingModelAlert: PreloadedGaussianModel?

#if os(macOS)
    @Environment(\.openWindow) private var openWindow
#elseif os(iOS)
    @State private var navigationPath = NavigationPath()

    private func openWindow(value: ModelIdentifier) {
        navigationPath.append(value)
    }
#elseif os(visionOS)
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State var immersiveSpaceIsShown = false

    private func openWindow(value: ModelIdentifier) {
        Task {
            switch await openImmersiveSpace(value: value) {
            case .opened:
                immersiveSpaceIsShown = true
            case .error, .userCancelled:
                break
            @unknown default:
                break
            }
        }
    }
#endif

    var body: some View {
#if os(macOS) || os(visionOS)
        mainView
#elseif os(iOS)
        NavigationStack(path: $navigationPath) {
            mainView
                .navigationDestination(for: ModelIdentifier.self) { modelIdentifier in
                    MetalKitSceneView(modelIdentifier: modelIdentifier)
                        .navigationTitle(modelIdentifier.description)
                        .environmentObject(rendererSettings)
                }
        }
#endif // os(iOS)
    }

    @ViewBuilder
    var mainView: some View {
        VStack {
            Spacer()

            Text("MetalSplatter SampleApp")

            Spacer()

            // Debug view selector
            HStack(spacing: 12) {
                HStack(spacing: 8) {
                    ForEach(rendererSettings.primaryDebugModes, id: \.self) { mode in
                        Button {
                            rendererSettings.debugViewMode = mode
                        } label: {
                            Text(mode.displayName)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .foregroundColor(rendererSettings.debugViewMode == mode ? .white : .primary)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .fill(rendererSettings.debugViewMode == mode ? Color.accentColor : Color.primary.opacity(0.08))
                                )
                        }
                        .buttonStyle(.plain)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(rendererSettings.debugViewMode == mode ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 1)
                        )
                    }
                }
                .frame(maxWidth: .infinity)

                Menu {
                    ForEach(rendererSettings.advancedDebugModes, id: \.self) { mode in
                        Button(mode.displayName) {
                            rendererSettings.debugViewMode = mode
                        }
                    }
                } label: {
                    HStack {
                        Text(rendererSettings.advancedDebugModes.contains(rendererSettings.debugViewMode) ? rendererSettings.debugViewMode.displayName : "Advanced")
                        Image(systemName: "chevron.down")
                    }
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .foregroundColor(.primary)
                    .frame(minWidth: 120)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.primary.opacity(0.08))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                    )
                }
            }
            .padding(.horizontal)

            // Read a scene file from disk
            Button("Read Scene File") {
                isPickingFile = true
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(isPickingFile)

            Divider().padding(.vertical, 4)

            // Preloaded models from bundle
            VStack(alignment: .leading, spacing: 12) {
                ForEach(PreloadedGaussianModel.allCases) { model in
                    Button(model.displayName) {
                        guard let url = model.bundleURL else {
                            missingModelAlert = model
                            return
                        }
                        openWindow(value: ModelIdentifier.gaussianSplat(url))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal)
                    .buttonStyle(.borderedProminent)
#if os(visionOS)
                    .disabled(immersiveSpaceIsShown)
#endif
                }
            }
            .padding(.horizontal)

            Spacer(minLength: 24)

            Button("Show Sample Box") {
                openWindow(value: ModelIdentifier.sampleBox)
            }
            .padding()
            .buttonStyle(.borderedProminent)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif

            Spacer()

#if os(visionOS)
            Button("Dismiss Immersive Space") {
                Task {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
            .disabled(!immersiveSpaceIsShown)

            Spacer()
#endif // os(visionOS)
        }
        .alert(item: $missingModelAlert) { model in
            Alert(
                title: Text("Model Not Found"),
                message: Text(model.resourceMissingMessage),
                dismissButton: .default(Text("OK"))
            )
        }
    }
}
