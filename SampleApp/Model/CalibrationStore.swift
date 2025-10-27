import Foundation

#if os(visionOS)

final class CalibrationStore {
    private let fileManager: FileManager
    private let directoryURL: URL

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let baseURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        directoryURL = baseURL.appendingPathComponent("Calibrations", isDirectory: true)
        if !fileManager.fileExists(atPath: directoryURL.path) {
            try? fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        }
    }

    func loadCalibration(forKey key: String) -> ModelCalibration? {
        let url = calibrationURL(for: key)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        return try? decoder.decode(ModelCalibration.self, from: data)
    }

    func saveCalibration(_ calibration: ModelCalibration, forKey key: String) throws {
        if !fileManager.fileExists(atPath: directoryURL.path) {
            try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        let data = try encoder.encode(calibration)
        let url = calibrationURL(for: key)
        try data.write(to: url, options: .atomic)
    }

    private func calibrationURL(for key: String) -> URL {
        directoryURL.appendingPathComponent("\(key).json")
    }
}

#else

final class CalibrationStore {
    init() {}

    func loadCalibration(forKey key: String) -> ModelCalibration? { nil }

    func saveCalibration(_ calibration: ModelCalibration, forKey key: String) throws {}
}

#endif
