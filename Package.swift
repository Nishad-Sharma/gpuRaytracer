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
        .target(
            name: "CShaderTypes",
            path: "Sources/CHeaders",
            sources: ["shaderTypes.h"],
            publicHeadersPath: "."
        ),
        .executableTarget(
            name: "gpuComputeShader",
            dependencies: ["CShaderTypes"],
            path: "Sources/gpuComputeShader"
        )
    ],
    
)