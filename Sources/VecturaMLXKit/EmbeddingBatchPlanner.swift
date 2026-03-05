import Foundation

struct EmbeddingBatchPlan {
  let originalIndices: [Int]
  let maxTokenLength: Int
}

enum EmbeddingBatchPlanner {
  static let defaultMaxBatchSize = 32
  static let minimumPaddingReduction = 0.15

  static func makePlans(
    tokenizedInputs: [[Int]],
    maxBatchSize: Int = defaultMaxBatchSize,
    minimumPaddingReduction: Double = minimumPaddingReduction
  ) -> [EmbeddingBatchPlan] {
    precondition(maxBatchSize > 0, "maxBatchSize must be greater than zero")
    guard !tokenizedInputs.isEmpty else {
      return []
    }

    let lengths = tokenizedInputs.map(\.count)
    let allIndices = Array(lengths.indices)
    let singlePlan = [EmbeddingBatchPlan(
      originalIndices: allIndices,
      maxTokenLength: lengths.max() ?? 0
    )]

    guard tokenizedInputs.count > maxBatchSize else {
      return singlePlan
    }

    let sortedIndices = allIndices.sorted { lhs, rhs in
      lengths[lhs] < lengths[rhs]
    }

    var candidatePlans: [EmbeddingBatchPlan] = []
    candidatePlans.reserveCapacity((sortedIndices.count + maxBatchSize - 1) / maxBatchSize)

    var start = 0
    while start < sortedIndices.count {
      let end = min(start + maxBatchSize, sortedIndices.count)
      let batchIndices = Array(sortedIndices[start..<end])
      let maxTokenLength = batchIndices.reduce(into: 0) { currentMax, index in
        currentMax = max(currentMax, lengths[index])
      }
      candidatePlans.append(EmbeddingBatchPlan(
        originalIndices: batchIndices,
        maxTokenLength: maxTokenLength
      ))
      start = end
    }

    let baselinePaddedTokenCount = paddedTokenCount(for: singlePlan)
    guard baselinePaddedTokenCount > 0 else {
      return singlePlan
    }

    let candidatePaddedTokenCount = paddedTokenCount(for: candidatePlans)
    let reduction = Double(baselinePaddedTokenCount - candidatePaddedTokenCount) / Double(baselinePaddedTokenCount)

    guard reduction >= minimumPaddingReduction else {
      return singlePlan
    }
    return candidatePlans
  }

  static func paddedTokenCount(for plans: [EmbeddingBatchPlan]) -> Int {
    plans.reduce(into: 0) { total, plan in
      total += plan.maxTokenLength * plan.originalIndices.count
    }
  }
}
