// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "VecturaMLXKit",
  platforms: [
    .macOS(.v15),
    .iOS(.v18),
    .tvOS(.v18),
    .visionOS(.v2),
    .watchOS(.v11),
  ],
  products: [
    .library(
      name: "VecturaMLXKit",
      targets: ["VecturaMLXKit"]
    ),
    .executable(
      name: "vectura-mlx-cli",
      targets: ["VecturaMLXCLI"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "6.0.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm/", exact: "3.31.3"),
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", exact: "0.4.3"),
    .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
  ],
  targets: [
    .target(
      name: "VecturaMLXKit",
      dependencies: [
        .product(name: "VecturaKit", package: "VecturaKit"),
        .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "Tokenizers", package: "swift-tokenizers"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
      ]
    ),
    .executableTarget(
      name: "VecturaMLXCLI",
      dependencies: [
        .product(name: "VecturaKit", package: "VecturaKit"),
        "VecturaMLXKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .executableTarget(
      name: "TestMLXExamples",
      dependencies: ["VecturaMLXKit"]
    ),
    .testTarget(
      name: "VecturaMLXKitTests",
      dependencies: ["VecturaMLXKit"]
    ),
  ]
)
