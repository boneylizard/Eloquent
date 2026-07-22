// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MiridAppleRunner",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "mirid-apple-runner", targets: ["MiridAppleRunner"]),
    ],
    targets: [
        .executableTarget(name: "MiridAppleRunner"),
    ]
)
