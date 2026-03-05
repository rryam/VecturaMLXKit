// Test script for VecturaMLXKit README examples
import Foundation
import VecturaKit
import VecturaMLXKit
import MLXEmbedders

@main
struct TestMLXExamples {
  static func main() async throws {
    if ProcessInfo.processInfo.environment["MLX_RUNTIME_BENCH"] == "1" {
      try await runRuntimeBenchmark()
      return
    }

    debugPrint("Testing VecturaMLXKit Examples")

    // 2. Initialize Database
    debugPrint("2. Initialize Database")

    let config = try VecturaConfig(
      name: "test-mlx-vector-db"
      // Dimension will be auto-detected from the model
    )
    let vectorDB = try await VecturaKit(
      config: config,
      embedder: try await MLXEmbedder(configuration: .nomic_text_v1_5)
    )
    debugPrint("MLX Database initialized successfully")
    debugPrint("Document count: \(try await vectorDB.documentCount)")

    // 3. Add Documents
    debugPrint("3. Add Documents")

    let texts = [
      "First document text",
      "Second document text",
      "Third document text"
    ]
    let documentIds = try await vectorDB.addDocuments(texts: texts)
    debugPrint("Documents added with IDs: \(documentIds)")
    debugPrint("Total document count: \(try await vectorDB.documentCount)")

    // 4. Search Documents
    debugPrint("4. Search Documents")

    let results = try await vectorDB.search(
      query: "document text",
      numResults: 5,      // Optional
      threshold: 0.8     // Optional
    )

    debugPrint("Search found \(results.count) results:")
    for result in results {
      debugPrint("ID: \(result.id)")
      debugPrint("Text: \(result.text)")
      debugPrint("Score: \(result.score)")
      debugPrint("Created At: \(result.createdAt)")
    }

    // 5. Document Management
    debugPrint("5. Document Management")

    // Update document:
    guard let documentToUpdate = documentIds.first else {
      debugPrint("No documents to update")
      return
    }
    debugPrint("Updating document...")
    try await vectorDB.updateDocument(
      id: documentToUpdate,
      newText: "Updated text"
    )
    debugPrint("Document updated")

    // Verify update by searching
    let updatedResults = try await vectorDB.search(query: "Updated text", threshold: 0.0)
    debugPrint("Verification: Found \(updatedResults.count) documents with 'Updated text'")

    // Delete documents:
    debugPrint("Deleting documents...")
    let idsToDelete = documentIds.count >= 2
      ? [documentToUpdate, documentIds[1]]
      : [documentToUpdate]
    try await vectorDB.deleteDocuments(ids: idsToDelete)
    debugPrint("Documents deleted")
    debugPrint("Document count after deletion: \(try await vectorDB.documentCount)")

    // Reset database:
    debugPrint("Resetting database...")
    try await vectorDB.reset()
    debugPrint("Database reset")
    debugPrint("Document count after reset: \(try await vectorDB.documentCount)")
  }

  private static func runRuntimeBenchmark() async throws {
    let adaptiveEnabled = ProcessInfo.processInfo.environment["VECTURA_MLX_ADAPTIVE_BATCHING"]
      .map { !$0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().matchesAny(of: ["0", "false", "no", "off"]) }
      ?? true
    let modeLabel = adaptiveEnabled ? "adaptive" : "single-batch"

    let embedder = try await MLXEmbedder(configuration: .nomic_text_v1_5)
    let corpus = makeSyntheticTexts(seed: 20260305, count: 384)
    let iterations = 5

    _ = try await embedder.embed(texts: Array(corpus.prefix(8)))

    var durations: [Double] = []
    durations.reserveCapacity(iterations)

    for _ in 0..<iterations {
      let start = DispatchTime.now().uptimeNanoseconds
      _ = try await embedder.embed(texts: corpus)
      let elapsedNs = DispatchTime.now().uptimeNanoseconds - start
      durations.append(Double(elapsedNs) / 1_000_000_000.0)
    }

    let avgSeconds = durations.reduce(0, +) / Double(durations.count)
    let p95Seconds = percentile(0.95, values: durations)
    let throughput = Double(corpus.count) / max(avgSeconds, 0.000_001)

    print("\nMLX Runtime Benchmark (\(modeLabel))")
    print("  corpus_size=\(corpus.count)")
    print("  iterations=\(iterations)")
    print(String(format: "  avg_seconds=%.4f", avgSeconds))
    print(String(format: "  p95_seconds=%.4f", p95Seconds))
    print(String(format: "  throughput_texts_per_second=%.2f", throughput))
  }

  private static func percentile(_ target: Double, values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let index = min(Int(Double(sorted.count) * target), sorted.count - 1)
    return sorted[index]
  }

  private static func makeSyntheticTexts(seed: UInt64, count: Int) -> [String] {
    var generator = SeededGenerator(seed: seed)
    let vocabulary = [
      "vector", "search", "embedding", "database", "token", "swift", "metal",
      "index", "semantic", "query", "document", "batch", "runtime", "latency",
      "throughput", "adaptive", "storage", "result", "benchmark", "profile",
    ]

    var outputs: [String] = []
    outputs.reserveCapacity(count)

    for i in 0..<count {
      let p = Double.random(in: 0..<1, using: &generator)
      let wordCount: Int
      if p < 0.95 {
        wordCount = Int.random(in: 12...48, using: &generator)
      } else {
        wordCount = Int.random(in: 220...520, using: &generator)
      }

      var words: [String] = []
      words.reserveCapacity(wordCount)
      for _ in 0..<wordCount {
        let term = vocabulary.randomElement(using: &generator) ?? "text"
        words.append(term)
      }
      outputs.append("doc\(i) " + words.joined(separator: " "))
    }

    return outputs
  }
}

private extension String {
  func matchesAny(of values: [String]) -> Bool {
    values.contains(self)
  }
}

private struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64

  init(seed: UInt64) {
    self.state = seed
  }

  mutating func next() -> UInt64 {
    state = state &* 6364136223846793005 &+ 1
    return state
  }
}
