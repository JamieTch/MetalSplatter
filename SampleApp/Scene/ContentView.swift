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
            Picker("Debug View", selection: $rendererSettings.debugViewMode) {
                ForEach(rendererSettings.primaryDebugModes, id: \.self) { mode in
                    Text(mode.displayName).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            Toggle("Rotate Normals by Quaternion", isOn: $rendererSettings.rotateNormalsByQuaternion)
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
