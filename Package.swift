// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "tempShader",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "gpuComputeShader", targets: ["gpuComputeShader"])
    ],
    dependencies: [
        // Add dependencies here if needed
    ],
    targets: [
        .executableTarget(
            name: "gpuComputeShader",
            path: "Sources/gpuComputeShader",
        )
    ],
    
)