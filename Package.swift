// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "VecturaMLXKit",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
    .tvOS(.v17),
    .visionOS(.v1),
    .watchOS(.v10),
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
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "3.0.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.3"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
  ],
  targets: [
    .target(
      name: "VecturaMLXKit",
      dependencies: [
        .product(name: "VecturaKit", package: "VecturaKit"),
        .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
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
