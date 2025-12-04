// Dataset Url: https://zenodo.org/records/7233404
import ArgumentParser
import Foundation
import Torch
import _Differentiation

/// CLI-adjustable hyperparameters and runtime options for the MNIST example.
struct TrainingConfig {
    /// Number of full passes over the training split.
    var epochs: Int = 3
    /// Mini-batch size used during training.
    var batchSize: Int = 128
    /// Mini-batch size used during evaluation.
    var evalBatchSize: Int = 1024
    /// Frequency (in optimizer steps) at which progress is logged.
    var logInterval: Int = 100
    /// Optimizer learning rate.
    var learningRate: Double = 1e-3
    /// Seed for data shuffling so runs are reproducible.
    var shuffleSeed: UInt64 = 0xfeed_cafe
    /// Optional cap on the number of batches processed per epoch (useful for debugging).
    var maxBatchesPerEpoch: Int? = nil
}

public struct ImageDataset<Element>: Dataset, RandomAccessCollection {
    public typealias Index = Int
    public let elements: [Element]

    public init(_ elements: [Element]) {
        self.elements = elements
    }

    // Collection
    public var startIndex: Int { elements.startIndex }
    public var endIndex: Int { elements.endIndex }
    public func index(after i: Int) -> Int { elements.index(after: i) }
    public func index(before i: Int) -> Int { elements.index(before: i) }
    public subscript(_ index: Int) -> Element { elements[index] }

    // Dataset
    public var count: Int { elements.count }
}

enum SolarError: Error {
    case URLNotValidError(string)
}

@main
struct SolarPanelsExample: ParsableCommand {
    @Option(name: .shortAndLong, help: "The number of full passes through the training dataset")
    var epochs: Int = 3

    @Option(name: .shortAndLong, help: "The mini-batch size used during training")
    var batchSize: Int = 32

    @Option(name: .shortAndLong, help: "The mini-batch size used during evaluation")
    var evalBatchSize: Int = 32

    @Option(
        name: .shortAndLong, help: "Frequency (in optimizer steps) at which progress is logged.")
    var logInterval: Int = 100

    @Option(name: .shortAndLong, help: "Optimizer learning rate.")
    var learningRate: Double = 1e-3

    @Option(name: .shortAndLong, help: "Seed for data shuffling so runs are reproducible.")
    var shuffleSeed: UInt64 = 0xfeed_cafe

    @Option(
        name: .shortAndLong,
        help: "Optional cap on the number of batches processed per epoch (useful for debugging")
    var maxBatchesPerEpoch: Int? = nil

    mutating func run() throws {
        let trainingConfig = TrainingConfig(
            epochs: epochs,
            batchSize: batchSize,
            evalBatchSize: evalBatchSize,
            logInterval: logInterval,
            learningRate: learningRate,
            shuffleSeed: shuffleSeed,
            maxBatchesPerEpoch: maxBatchesPerEpoch
        )

        guard
            let imageZipUrl = URL(
                string:
                    "https://zenodo.org/records/7551799/files/DeepStat-WP5-dataset.zip?download=1")
        else {
            throw SolarError.URLNotValidError("Invalid url: \(imageZipUrl)")
        }

        let zippedImages = Data(contentsOf: imageZipUrl)
    }

}
