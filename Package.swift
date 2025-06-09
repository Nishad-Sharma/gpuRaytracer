// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "tempShader",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "gpuRaytracer", targets: ["gpuRaytracer"])
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
            name: "gpuRaytracer",
            dependencies: ["CShaderTypes"],
            path: "Sources/gpuRaytracer",
            exclude: ["shaders.metal", "shaders.air", "MyMetalLib.metallib"]
        )
    ],
    
)