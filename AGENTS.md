# Repository Guidelines

## Project Structure & Module Organization
VecturaMLXKit ships as a Swift package with library target `VecturaMLXKit` and executable target `VecturaMLXCLI`. The MLX embedder lives in `Sources/VecturaMLXKit/MLXEmbedder.swift`, the CLI entry point lives in `Sources/VecturaMLXCLI`, and end-to-end sample code lives in `Sources/TestMLXExamples`. Tests live under `Tests/VecturaMLXKitTests`.

## Build, Test, and Development Commands
- `swift build` compiles the package and is useful for fast compile checks.
- `swift test --no-parallel` runs the Swift Testing suite.
- `xcodebuild -scheme vectura-mlx-cli -destination 'platform=macOS' build` is the canonical local runtime validation path for MLX.
- `./DerivedData/.../Build/Products/Debug/vectura-mlx-cli mock --db-name qa-db` is the preferred smoke test after an Xcode or `xcodebuild` build.

## MLX Validation Notes
For MLX-backed runtime verification, do not rely on `swift run vectura-mlx-cli ...` alone. SwiftPM CLI builds can compile successfully while still failing at runtime because the MLX `default.metallib` bundle is not available on that path. Use Xcode or `xcodebuild` for runtime validation so `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` is emitted and packaged correctly.

Recommended verification flow:
1. `xcrun --find metal`
2. `xcrun --find metallib`
3. `xcodebuild -scheme vectura-mlx-cli -destination 'platform=macOS' -derivedDataPath /tmp/VecturaMLXKit-xc build`
4. `find /tmp/VecturaMLXKit-xc/Build/Products/Debug -name default.metallib`
5. `/tmp/VecturaMLXKit-xc/Build/Products/Debug/vectura-mlx-cli mock --db-name qa-db`

## Coding Style & Naming Conventions
Follow Swift 6 defaults. Keep types UpperCamelCase, members lowerCamelCase, and prefer concise `///` comments for public APIs when behavior is not obvious. Preserve the current async/await style and avoid introducing legacy XCTest patterns into new tests.

## Testing Guidelines
Use Swift Testing for new tests. MLX-heavy tests may be environment-gated, so compile-only success is not enough for runtime signoff. Before closing a change that touches MLX loading or packaging, verify the Xcode-built CLI gets past `Setting up database...` and that the mock flow completes successfully.
