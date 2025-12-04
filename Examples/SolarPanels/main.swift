// Dataset Url: https://zenodo.org/records/7233404
import ArgumentParser

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

}
