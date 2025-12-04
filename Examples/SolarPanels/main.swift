// Dataset Url: https://zenodo.org/records/7233404
import ArgumentParser
import Foundation
import Torch
import ZIPFoundation
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

extension URL {
    var fileSize: Int? {  // in bytes
        do {
            let val = try self.resourceValues(forKeys: [
                .totalFileAllocatedSizeKey, .fileAllocatedSizeKey,
            ])
            return val.totalFileAllocatedSize ?? val.fileAllocatedSize
        } catch {
            print(error)
            return nil
        }
    }
}

extension FileManager {
    func directorySize(_ dir: URL) -> Int? {  // in bytes
        if let enumerator = self.enumerator(
            at: dir,
            includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey],
            options: [],
            errorHandler: { (_, error) -> Bool in
                print(error)
                return false
            })
        {
            var bytes = 0
            for case let url as URL in enumerator {
                bytes += url.fileSize ?? 0
            }
            return bytes
        } else {
            return nil
        }
    }
}

enum SolarError: Error {
    case URLNotValidError(String)
}

struct SolarPanelData {
    let train: ImageDataset
    let test: ImageDataset
}

func buildVGG16Model() -> Sequential {
    return Sequential {
        // ── Block 1 ──────────────────────────────────────────────────────────────
        Conv2D(
            kaimingUniformInChannels: 1, outChannels: 32,
            kernelSize: (3, 3), padding: (1, 1))
        Dropout(probability: 0.025)  // Flax applies dropout before BN here
        BatchNorm(featureCount: 32)  // NCHW => axis 1
        ReLU()
        AvgPool2D(kernelSize: (2, 2), stride: (2, 2))

        // ── Block 2 ──────────────────────────────────────────────────────────────
        Conv2D(
            kaimingUniformInChannels: 32, outChannels: 64,
            kernelSize: (3, 3), padding: (1, 1))
        BatchNorm(featureCount: 64)
        ReLU()
        AvgPool2D(kernelSize: (2, 2), stride: (2, 2))

        // ── Head ─────────────────────────────────────────────────────────────────
        Flatten(startDim: 1)  // [N, 64*7*7] = [N, 3136]
        Linear(inputSize: 64 * 7 * 7, outputSize: 256)
        Dropout(probability: 0.025)
        ReLU()
        Linear(inputSize: 256, outputSize: 10)
    }
}

/// Packs a batch of `MNISTExample` samples into dense tensors.
/// - Parameter batch: Collection of MNIST examples to stack.
/// - Returns: Tuple containing image tensors (`[batch, 1, 28, 28]`) and integer labels.
func makeBatch(_ batch: [MNISTExample]) -> (images: Tensor, labels: Tensor) {
    let images = Tensor.stack(batch.map { $0.image }, dim: 0)
    let labelScalars = batch.map { Int64($0.label) }
    let labels = Tensor(array: labelScalars, shape: [labelScalars.count], dtype: .int64)
    return (images, labels)
}

/// Computes the number of correct predictions in a batch.
/// - Parameters:
///   - logits: Raw model outputs shaped `[batch, numClasses]`.
///   - labels: Ground-truth class indices shaped `[batch]`.
/// - Returns: Tuple containing the count of correct predictions and total samples.
func batchAccuracy(logits: Tensor, labels: Tensor) -> (correct: Int, total: Int) {
    let predictions = logits.argmax(dim: 1)
    let matches = predictions.eq(labels)
    let correctTensor = matches.to(dtype: .int32).sum()
    let correct = Int(correctTensor.toArray(as: Int32.self)[0])
    return (correct, labels.shape[0])
}

func evaluate<Model: Layer>(
    _ model: Model,
    loader: DataLoader<ImageDataset>
) -> (loss: Double, accuracy: Double) {
    var totalLoss: Double = 0
    var totalCorrect = 0
    var totalSamples = 0

    for batch in loader {
        let (images, labels) = makeBatch(batch)
        let logits = model(images as! Model.Input)
        let loss = softmaxCrossEntropy(logits: logits as! Tensor, labels: labels)
        let lossValue = loss.toArray(as: Float.self)[0]
        let (correct, batchTotal) = batchAccuracy(logits: logits as! Tensor, labels: labels)

        totalLoss += Double(lossValue) * Double(batchTotal)
        totalCorrect += correct
        totalSamples += batchTotal
    }

    let meanLoss = totalLoss / Double(totalSamples)
    let accuracy = Double(totalCorrect) / Double(totalSamples)
    return (meanLoss, accuracy)
}

func trainModel(model: Sequential, config: TrainingConfig, data: SolarPanelData) {
    let testLoader = DataLoader(
        dataset: data.test,
        batchSize: config.evalBatchSize,
        shuffle: false,
        dropLast: false,
        seed: nil
    )

    /// Adam optimizer configured with the requested learning rate.
    let optimizer = Adam(for: model, learningRate: Float(config.learningRate))
    let stepsPerEpoch = (data.train.count + config.batchSize - 1) / config.batchSize
    let startTime = Date()

    for epoch in 1...config.epochs {
        /// Shuffled loader that feeds training batches for the current epoch.
        let trainLoader = DataLoader(
            dataset: data.train,
            batchSize: config.batchSize,
            shuffle: true,
            dropLast: false,
            seed: config.shuffleSeed &+ UInt64(epoch)
        )

        var runningLoss: Double = 0
        var runningCorrect = 0
        var runningSamples = 0
        var blockLoss: Double = 0
        var blockCorrect = 0
        var blockSamples = 0
        var step = 0

        for batch in trainLoader {
            step += 1
            if let limit = config.maxBatchesPerEpoch, step > limit { break }
            let (images, labels) = makeBatch(batch)

            let (lossTensor, pullback) = valueWithPullback(at: model) { current -> Tensor in
                let logits = current(images)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }
            let grad = pullback(Tensor(1.0, dtype: .float32))

            let logits = model(images)
            let (correct, batchTotal) = batchAccuracy(logits: logits, labels: labels)
            let lossValue = lossTensor.toArray(as: Float.self)[0]

            optimizer.update(&model, along: grad)

            runningLoss += Double(lossValue) * Double(batchTotal)
            runningCorrect += correct
            runningSamples += batchTotal

            blockLoss += Double(lossValue) * Double(batchTotal)
            blockCorrect += correct
            blockSamples += batchTotal

            if step % config.logInterval == 0 {
                let avgLoss = blockLoss / Double(blockSamples)
                let avgAcc = Double(blockCorrect) / Double(blockSamples)
                let elapsed = Date().timeIntervalSince(startTime)
                print(
                    String(
                        format: "epoch %d • step %d/%d • loss %.4f • acc %.2f%% • %.1fs",
                        epoch, step, stepsPerEpoch, avgLoss, avgAcc * 100, elapsed))
                blockLoss = 0
                blockCorrect = 0
                blockSamples = 0
            }
        }

        if blockSamples > 0 {
            let avgLoss = blockLoss / Double(blockSamples)
            let avgAcc = Double(blockCorrect) / Double(blockSamples)
            let elapsed = Date().timeIntervalSince(startTime)
            print(
                String(
                    format: "epoch %d • step %d/%d • loss %.4f • acc %.2f%% • %.1fs",
                    epoch, step, stepsPerEpoch, avgLoss, avgAcc * 100, elapsed))
        }

        let epochLoss = runningLoss / Double(runningSamples)
        let epochAcc = Double(runningCorrect) / Double(runningSamples)
        let (valLoss, valAcc) = evaluate(model, loader: testLoader)
        print(
            String(
                format:
                    "epoch %d done • train loss %.4f • train acc %.2f%% • val loss %.4f • val acc %.2f%%",
                epoch, epochLoss, epochAcc * 100, valLoss, valAcc * 100))
    }

    let totalElapsed = Date().timeIntervalSince(startTime)
    print(String(format: "Training finished in %.1fs", totalElapsed))
}

@main
struct SolarPanelsExample: ParsableCommand {
    @Option(name: .shortAndLong, help: "The number of full passes through the training dataset")
    var epochs: Int = 3

    @Option(name: .shortAndLong, help: "The mini-batch size used during training")
    var batchSize: Int = 32

    @Option(name: .shortAndLong, help: "The mini-batch size used during evaluation")
    var testBatchSize: Int = 32

    @Option(
        name: [.customShort("i"), .long],
        help: "Frequency (in optimizer steps) at which progress is logged.")
    var logInterval: Int = 100

    @Option(name: .shortAndLong, help: "Optimizer learning rate.")
    var learningRate: Double = 1e-3

    @Option(name: .shortAndLong, help: "Seed for data shuffling so runs are reproducible.")
    var shuffleSeed: UInt64 = 0xfeed_cafe

    @Option(
        name: .shortAndLong,
        help: "Optional cap on the number of batches processed per epoch (useful for debugging")
    var maxBatchesPerEpoch: Int? = nil

    @Option(
        name: .shortAndLong,
        help: "Download dataset again"
    )
    var download: Bool = false

    mutating func run() throws {
        let trainingConfig = TrainingConfig(
            epochs: epochs,
            batchSize: batchSize,
            evalBatchSize: testBatchSize,
            logInterval: logInterval,
            learningRate: learningRate,
            shuffleSeed: shuffleSeed,
            maxBatchesPerEpoch: maxBatchesPerEpoch
        )

        let imageUrlString =
            "https://zenodo.org/records/7551799/files/DeepStat-WP5-dataset.zip?download=1"
        guard
            let imageZipUrl = URL(
                string: imageUrlString
            )
        else {
            throw SolarError.URLNotValidError("Invalid url: \(imageUrlString)")
        }

        let rootDir = (DataHome.root).appendingPathComponent(
            "deepstat-wp5", isDirectory: true)
        try Downloader.ensureDir(rootDir)

        try ensureFileExists(at: rootDir, remote: imageZipUrl, downloadAgain: download)

        var model = buildVGG16Model()
        trainModel(model: model, config: trainingConfig)
    }

    func ensureFileExists(at rawPath: URL, remote: URL, downloadAgain: Bool) throws {
        if !downloadAgain && FileManager.default.fileExists(atPath: rawPath.path) {
            print("[SolarPanels] Using cached \(rawPath.lastPathComponent)")
            return
        }
        let unzippedPath = rawPath.appending(path: "unzipped")
        print("[SolarPanels] Preparing \(rawPath.lastPathComponent)")
        let zippedFile = try Downloader.fetch(url: remote, to: rawPath)
        print(
            "[SolarPanels] Decompressing \(rawPath.lastPathComponent) → \(rawPath.lastPathComponent)"
        )
        try FileManager.default.unzipItem(at: zippedFile, to: unzippedPath)

        let sizeMB = Double(FileManager.default.directorySize(unzippedPath) ?? 0) / 1_048_576.0
        print(
            String(format: "[SolarPanels] Wrote %@ (%.1f MB)", rawPath.lastPathComponent, sizeMB))
    }

}
