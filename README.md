# VecturaMLXKit

MLX-based embeddings for [VecturaKit](https://github.com/rryam/VecturaKit) — GPU-accelerated on-device vector search using Apple's MLX framework.

<p align="center">
  <img src="https://img.shields.io/badge/Swift-6.0+-fa7343?style=flat&logo=swift&logoColor=white" alt="Swift 6.0+">
  <br>
  <img src="https://img.shields.io/badge/iOS-18.0+-000000?style=flat&logo=apple&logoColor=white" alt="iOS 18.0+">
  <img src="https://img.shields.io/badge/macOS-15.0+-000000?style=flat&logo=apple&logoColor=white" alt="macOS 15.0+">
  <img src="https://img.shields.io/badge/tvOS-18.0+-000000?style=flat&logo=apple&logoColor=white" alt="tvOS 18.0+">
  <img src="https://img.shields.io/badge/visionOS-2.0+-000000?style=flat&logo=apple&logoColor=white" alt="visionOS 2.0+">
  <img src="https://img.shields.io/badge/watchOS-11.0+-000000?style=flat&logo=apple&logoColor=white" alt="watchOS 11.0+">
</p>

## Overview

VecturaMLXKit provides `MLXEmbedder`, an embedding provider that uses Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework for GPU-accelerated inference on Apple Silicon. It conforms to VecturaKit's `VecturaEmbedder` protocol, so you can use it as a drop-in replacement for `SwiftEmbedder` or `NLContextualEmbedder`.

This package also includes `vectura-mlx-cli`, a command-line tool for managing a VecturaKit database with MLX embeddings.

## Installation

Add both VecturaKit and VecturaMLXKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "5.0.0"),
    .package(url: "https://github.com/rryam/VecturaMLXKit.git", from: "2.0.0"),
],
```

`VecturaMLXKit` 2.x tracks the `VecturaKit` 5.x release line and shares the same minimum deployment targets:
macOS 15, iOS 18, tvOS 18, visionOS 2, and watchOS 11.

Or add them via Xcode's **File > Add Package Dependencies** UI — no special configuration needed.

## Usage

### Import

```swift
import VecturaKit
import VecturaMLXKit
import MLXEmbedders
```

### Initialize Database with MLX

```swift
let config = VecturaConfig(
  name: "my-mlx-vector-db",
  dimension: nil  // Auto-detect dimension from MLX embedder
)

// Create MLX embedder
let embedder = try await MLXEmbedder(configuration: .nomic_text_v1_5)
let vectorDB = try await VecturaKit(config: config, embedder: embedder)
```

### Add Documents

```swift
let texts = [
  "First document text",
  "Second document text",
  "Third document text"
]
let documentIds = try await vectorDB.addDocuments(texts: texts)
```

### Search Documents

```swift
let results = try await vectorDB.search(
    query: "search query",
    numResults: 5,
    threshold: 0.8
)

for result in results {
    print("Document ID: \(result.id)")
    print("Text: \(result.text)")
    print("Similarity Score: \(result.score)")
}
```

### Document Management

```swift
// Update
try await vectorDB.updateDocument(id: documentId, newText: "Updated text")

// Delete
try await vectorDB.deleteDocuments(ids: [documentId1, documentId2])

// Reset
try await vectorDB.reset()
```

## CLI Tool (`vectura-mlx-cli`)

```bash
# Add documents
vectura-mlx add "First document" "Second document" --db-name "my-mlx-db"

# Search documents
vectura-mlx search "search query" --db-name "my-mlx-db" --threshold 0.7 --num-results 5

# Update document
vectura-mlx update <document-uuid> "Updated text" --db-name "my-mlx-db"

# Delete documents
vectura-mlx delete <document-uuid-1> <document-uuid-2> --db-name "my-mlx-db"

# Reset database
vectura-mlx reset --db-name "my-mlx-db"

# Run demo with sample data
vectura-mlx mock --db-name "my-mlx-db"
```

Options:

-   `--db-name, -d`: Database name (default: "vectura-mlx-cli-db")
-   `--dimension, -v`: Vector dimension (auto-detected by default)
-   `--threshold, -t`: Minimum similarity threshold (default: 0.7)
-   `--num-results, -n`: Number of results to return (default: 10)

## Local Validation

For fast compile checks, `swift build` and `swift test` are fine. For MLX runtime verification on macOS, use Xcode or `xcodebuild` instead of `swift run`.

`swift run vectura-mlx-cli ...` can compile successfully and still fail at runtime if the MLX Metal bundle is not packaged into the SwiftPM CLI build. The reliable path is an Xcode-style build that emits `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`.

Recommended local smoke test:

```bash
xcrun --find metal
xcrun --find metallib

xcodebuild \
  -scheme vectura-mlx-cli \
  -destination 'platform=macOS' \
  -derivedDataPath /tmp/VecturaMLXKit-xc \
  build

find /tmp/VecturaMLXKit-xc/Build/Products/Debug -name default.metallib

/tmp/VecturaMLXKit-xc/Build/Products/Debug/vectura-mlx-cli mock --db-name qa-db
```

Expected behavior:

- `default.metallib` is present under `mlx-swift_Cmlx.bundle`
- the CLI gets past `Setting up database...`
- the mock flow resets the database, adds sample documents, searches, updates, deletes, and exits successfully

## Dependencies

- [VecturaKit](https://github.com/rryam/VecturaKit): Core vector database
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm): MLX-based embeddings
- [swift-argument-parser](https://github.com/apple/swift-argument-parser): CLI framework

## License

VecturaMLXKit is released under the MIT License. See the [LICENSE](LICENSE) file for details.
