// swift-tools-version: 5.9
import PackageDescription

// Unified Package.swift for both macOS and Linux builds.
// - macOS: Full GUI app with SwiftUI/SpriteKit
// - Linux: Headless CLI only for ML training

let package = Package(
    name: "HackMatrix",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "HackMatrix",
            targets: ["HackMatrix"]
        )
    ],
    targets: [
        // Core game logic + GUI (GUI files use #if canImport(SwiftUI))
        .target(
            name: "HackMatrixCore",
            path: "HackMatrix",
            exclude: [
                "Info.plist",
                "Assets.xcassets"
            ]
        ),
        // Entry point (launches GUI on macOS, CLI on Linux)
        .executableTarget(
            name: "HackMatrix",
            dependencies: ["HackMatrixCore"],
            path: "Sources/SPMMain"
        )
    ]
)
