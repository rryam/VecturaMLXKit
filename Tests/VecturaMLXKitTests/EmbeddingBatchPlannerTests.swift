import Foundation
import Testing
@testable import VecturaMLXKit

@Suite("Embedding Batch Planner")
struct EmbeddingBatchPlannerTests {
  @Test("Single batch for compact inputs")
  func keepsSingleBatchForCompactInputs() {
    let tokenizedInputs = Array(
      repeating: Array(repeating: 1, count: 32),
      count: 16
    )

    let plans = EmbeddingBatchPlanner.makePlans(tokenizedInputs: tokenizedInputs, maxBatchSize: 32)
    #expect(plans.count == 1)
    #expect(plans[0].maxTokenLength == 32)
    #expect(plans[0].originalIndices.count == tokenizedInputs.count)
  }

  @Test("Adaptive batches reduce padding for skewed lengths")
  func adaptiveBatchesReducePaddingForSkewedLengths() {
    var tokenizedInputs: [[Int]] = []
    tokenizedInputs.reserveCapacity(256)

    for _ in 0..<240 {
      tokenizedInputs.append(Array(repeating: 1, count: 32))
    }
    for _ in 0..<16 {
      tokenizedInputs.append(Array(repeating: 1, count: 768))
    }

    let singlePlan = [EmbeddingBatchPlan(
      originalIndices: Array(tokenizedInputs.indices),
      maxTokenLength: 768
    )]
    let adaptivePlans = EmbeddingBatchPlanner.makePlans(
      tokenizedInputs: tokenizedInputs,
      maxBatchSize: 32
    )

    let baselinePaddedTokens = EmbeddingBatchPlanner.paddedTokenCount(for: singlePlan)
    let adaptivePaddedTokens = EmbeddingBatchPlanner.paddedTokenCount(for: adaptivePlans)
    let reduction = Double(baselinePaddedTokens - adaptivePaddedTokens) / Double(baselinePaddedTokens)

    #expect(adaptivePaddedTokens < baselinePaddedTokens)
    #expect(reduction > 0.5)
  }

  @Test("Adaptive plans preserve all source indices exactly once")
  func adaptivePlansPreserveAllIndicesExactlyOnce() {
    let tokenizedInputs = (0..<97).map { index in
      Array(repeating: 1, count: 8 + (index % 53))
    }

    let plans = EmbeddingBatchPlanner.makePlans(tokenizedInputs: tokenizedInputs, maxBatchSize: 16)
    let flattened = plans.flatMap(\.originalIndices).sorted()

    #expect(flattened == Array(tokenizedInputs.indices))
  }

  @Test("Benchmark: adaptive batching vs single max-length padding")
  func benchmarkAdaptiveBatching() {
    let tokenizedInputs = Self.makeSyntheticTokenizedInputs(seed: 20260305, count: 512)
    let iterations = 200

    let baselinePaddingOnly = Self.benchmarkSingleBatchPadding(
      tokenizedInputs: tokenizedInputs,
      iterations: iterations,
      simulateInferenceCost: false
    )
    let adaptivePaddingOnly = Self.benchmarkAdaptivePadding(
      tokenizedInputs: tokenizedInputs,
      iterations: iterations,
      simulateInferenceCost: false
    )
    let baselineWeighted = Self.benchmarkSingleBatchPadding(
      tokenizedInputs: tokenizedInputs,
      iterations: iterations,
      simulateInferenceCost: true
    )
    let adaptiveWeighted = Self.benchmarkAdaptivePadding(
      tokenizedInputs: tokenizedInputs,
      iterations: iterations,
      simulateInferenceCost: true
    )

    let weightedSpeedup = baselineWeighted.avgMs / max(adaptiveWeighted.avgMs, 0.000_001)
    let paddedReduction = Double(baselineWeighted.paddedTokens - adaptiveWeighted.paddedTokens)
      / Double(baselineWeighted.paddedTokens)

    print("\nEmbedding Batch Planner Benchmark")
    print("  Iterations: \(iterations)")
    print(String(format: "  Padding-only single-batch avg: %.3f ms", baselinePaddingOnly.avgMs))
    print(String(format: "  Padding-only adaptive avg: %.3f ms", adaptivePaddingOnly.avgMs))
    print(String(format: "  Workload-weighted single-batch avg: %.3f ms", baselineWeighted.avgMs))
    print(String(format: "  Workload-weighted adaptive avg: %.3f ms", adaptiveWeighted.avgMs))
    print(String(format: "  Workload-weighted speedup: %.2fx", weightedSpeedup))
    print("  Single-batch padded tokens: \(baselineWeighted.paddedTokens)")
    print("  Adaptive padded tokens: \(adaptiveWeighted.paddedTokens)")
    print(String(format: "  Padded-token reduction: %.1f%%", paddedReduction * 100))

    #expect(adaptiveWeighted.paddedTokens < baselineWeighted.paddedTokens)
  }

  private static func benchmarkSingleBatchPadding(
    tokenizedInputs: [[Int]],
    iterations: Int,
    simulateInferenceCost: Bool
  ) -> (avgMs: Double, paddedTokens: Int) {
    let start = DispatchTime.now().uptimeNanoseconds
    var paddedTokens = 0
    var sink: UInt64 = 0

    for _ in 0..<iterations {
      let maxLength = tokenizedInputs.map(\.count).max() ?? 0
      var tokenCount = 0

      for tokens in tokenizedInputs {
        let deficit = maxLength - tokens.count
        tokenCount += tokens.count + max(0, deficit)
        _ = tokens + Array(repeating: 0, count: max(0, deficit))
      }

      if simulateInferenceCost {
        sink &+= simulatedInferenceUnits(tokenCount)
      }
      paddedTokens = tokenCount
    }

    let elapsedNs = DispatchTime.now().uptimeNanoseconds - start
    let avgMs = Double(elapsedNs) / Double(iterations) / 1_000_000.0
    _ = sink
    return (avgMs, paddedTokens)
  }

  private static func benchmarkAdaptivePadding(
    tokenizedInputs: [[Int]],
    iterations: Int,
    simulateInferenceCost: Bool
  ) -> (avgMs: Double, paddedTokens: Int) {
    let start = DispatchTime.now().uptimeNanoseconds
    var paddedTokens = 0
    var sink: UInt64 = 0

    for _ in 0..<iterations {
      let plans = EmbeddingBatchPlanner.makePlans(tokenizedInputs: tokenizedInputs, maxBatchSize: 32)
      var tokenCount = 0

      for plan in plans {
        for index in plan.originalIndices {
          var tokens = tokenizedInputs[index]
          if tokens.count < plan.maxTokenLength {
            tokens.reserveCapacity(plan.maxTokenLength)
            tokens.append(contentsOf: repeatElement(0, count: plan.maxTokenLength - tokens.count))
          }
          tokenCount += tokens.count
        }
      }

      if simulateInferenceCost {
        sink &+= simulatedInferenceUnits(tokenCount)
      }
      paddedTokens = tokenCount
    }

    let elapsedNs = DispatchTime.now().uptimeNanoseconds - start
    let avgMs = Double(elapsedNs) / Double(iterations) / 1_000_000.0
    _ = sink
    return (avgMs, paddedTokens)
  }

  private static func makeSyntheticTokenizedInputs(seed: UInt64, count: Int) -> [[Int]] {
    var generator = SeededGenerator(seed: seed)
    var outputs: [[Int]] = []
    outputs.reserveCapacity(count)

    for _ in 0..<count {
      let probability = Double.random(in: 0..<1, using: &generator)
      let length: Int
      if probability < 0.95 {
        length = Int.random(in: 16...64, using: &generator)
      } else {
        length = Int.random(in: 256...1024, using: &generator)
      }
      outputs.append(Array(repeating: 1, count: length))
    }
    return outputs
  }

  private static func simulatedInferenceUnits(_ tokenCount: Int) -> UInt64 {
    var accumulator: UInt64 = 0
    for value in 0..<tokenCount {
      accumulator = accumulator &+ UInt64((value & 7) + 1)
    }
    return accumulator
  }
}

private struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64

  init(seed: UInt64) {
    state = seed
  }

  mutating func next() -> UInt64 {
    state = state &* 6364136223846793005 &+ 1
    return state
  }
}
