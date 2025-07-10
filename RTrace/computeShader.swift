//
//  computeShader.swift
//  gpuRaytracer
//
//  Created by Nishad Sharma on 5/6/2025.
//
import Metal
import Foundation
import simd
import MetalKit
//import shaderTypes

func convertMaterial(material: Material) -> MaterialGPU {
    return MaterialGPU(
        diffuse: material.diffuse,
        metallic: material.metallic,
        roughness: material.roughness,
        emissive: material.emissive
    )
}

func convertTriangle(triangles: [Triangle]) -> [TriangleGPU] {
    return triangles.map { triangle in
        var gpuTriangle = TriangleGPU()
        gpuTriangle.vertices.0 = triangle.vertices[0]
        gpuTriangle.vertices.1 = triangle.vertices[1]
        gpuTriangle.vertices.2 = triangle.vertices[2]
        gpuTriangle.material = convertMaterial(material: triangle.material)
        return gpuTriangle
    }
}

func convertSquareLight(squareLight: SquareLight) -> SquareLightGPU {
    return SquareLightGPU(
        center: squareLight.center,
        color: squareLight.material.diffuse,
        emittedRadiance: squareLight.emittedLuminance,
        width: squareLight.width,
        depth: squareLight.depth
    )
}

// must be setup in separate command to actual ray trace calc. ray trace needs complete accel structure anyway.
// uses different command encoder than compute or render
func setupAccelerationStructures(device: MTLDevice, triangles: [Triangle]) -> MTLAccelerationStructure {

    // flatten triangle verts into one array
    var allVertices: [simd_float3] = []
    for triangle in triangles {
        allVertices.append(contentsOf: triangle.vertices)
    }
    // vertex buffer
    let vertexData = allVertices.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
    let vertexBuffer = device.makeBuffer(bytes: vertexData.withUnsafeBytes { $0.bindMemory(to: Float.self).baseAddress! },
                                        length: vertexData.count,
                                        options: [])!
    
    // triangle geometry descriptor, describes set of triangles for building acceleration structure
    // tells gpu where triangle data is in memory, how many triangles and how to interpret data.
    let geometryDescriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
    geometryDescriptor.vertexBuffer = vertexBuffer
    geometryDescriptor.vertexBufferOffset = 0
    geometryDescriptor.vertexStride = MemoryLayout<simd_float3>.stride
    geometryDescriptor.triangleCount = triangles.count

    // primitive acceleration structure descriptor, tells gpu which geometry prims (triangles),
    // how geometry is in memory which will be used to make the acceleration structure
    let primitiveDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
    primitiveDescriptor.geometryDescriptors = [geometryDescriptor]

    // get sizes for acceleration structure
    let accelSizes = device.accelerationStructureSizes(descriptor: primitiveDescriptor)
    // actual acceleration structure of size
    let accelerationStructure = device.makeAccelerationStructure(size: accelSizes.accelerationStructureSize)!
    // create scratch buffer for temp memory during construction of accel structure on gpu
    let scratchBuffer = device.makeBuffer(length: accelSizes.buildScratchBufferSize, options: .storageModeShared)!

    let commandQueue = device.makeCommandQueue()
    let commandBuffer = commandQueue?.makeCommandBuffer()
    // make command encoder for accel structure operations. different from computecommandEncoder
    // which is used for compute operations
    let encoder = commandBuffer!.makeAccelerationStructureCommandEncoder()!
    
    // tells gpu to actually build structure
    encoder.build(accelerationStructure: accelerationStructure,
                  descriptor: primitiveDescriptor,
                  scratchBuffer: scratchBuffer,
                  scratchBufferOffset: 0)
    encoder.endEncoding()

    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()

    return accelerationStructure
}

func drawTriangle(device: MTLDevice, cameras: [Camera], triangles: [Triangle], squareLights: [SquareLight], pixels: [UInt8],
accelerationStructure : MTLAccelerationStructure) -> [UInt8] {
    let camerasGPU = convertCameras(cameras: cameras)
    // let trianglesGPU = convertTriangle(triangles: triangles)
    let materialsGPU = triangles.map { convertMaterial(material: $0.material) }
    let squareLightsGPU = [convertSquareLight(squareLight: squareLights[0])]
    // triangle verts needed for normal calc - cant pull from accel structure
    var allVertices: [simd_float3] = []
    for triangle in triangles {
        allVertices.append(contentsOf: triangle.vertices)
    }


//    let metalLibURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
//        .appendingPathComponent("Sources/gpuRaytracer/MyMetalLib.metallib")
//
//    guard let defaultLibrary = try? device.makeLibrary(URL: metalLibURL) else {
//        fatalError("Could not load Metal library from \(metalLibURL.path)")
//    }
    guard let metalLibURL = Bundle.main.url(forResource: "default", withExtension: "metallib") else {
        fatalError("Could not find MyMetalLib.metallib in app bundle")
    }
    guard let defaultLibrary = try? device.makeLibrary(URL: metalLibURL) else {
        fatalError("Could not load Metal library from \(metalLibURL.path)")
    }

    guard let drawFunction = defaultLibrary.makeFunction(name: "drawTriangle") else {
        fatalError("Could not find 'drawTriangle' kernel in Metal library")
    }

    // intersection function table maybe not needed??
    let intersectionFunctionTableDescriptor = MTLIntersectionFunctionTableDescriptor()
    intersectionFunctionTableDescriptor.functionCount = 1

    let computePipeline: MTLComputePipelineState
    do {
        computePipeline = try device.makeComputePipelineState(function: drawFunction)
    } catch {
        print("Failed to create compute pipeline state: \(error)")
        return pixels // maybe bad return fix?
    }
    
    if #available(macOS 13.0, iOS 16.0, *) {
        if device.supportsRaytracing {
            // Safe to use MTLRayTracingPipelineState and ray tracing features
            print("yes good to go")
        } else {
            print("Ray tracing not supported on this device.")
        }
    } else {
        print("Ray tracing requires macOS 13/iOS 16 or later.")
    }

//    let raytracePipeline: MTLRayTracingPipelineState
    
    
    let commandQueue = device.makeCommandQueue()
    let commandBuffer = commandQueue?.makeCommandBuffer()
    
    let cameraBuffer = device.makeBuffer(bytes: camerasGPU,
    length: camerasGPU.count * MemoryLayout<CameraGPU>.size, options: .storageModeShared)
    let materialsBuffer = device.makeBuffer(bytes: materialsGPU,
    length: materialsGPU.count * MemoryLayout<MaterialGPU>.size, options: .storageModeShared)
    let pixelsBuffer = device.makeBuffer(length: pixels.count * MemoryLayout<UInt8>.size,
    options: .storageModeShared)
    let squareLightsBuffer = device.makeBuffer(bytes: squareLightsGPU,
    length: squareLightsGPU.count * MemoryLayout<SquareLightGPU>.size, options: .storageModeShared)
    let vertexBuffer = device.makeBuffer(bytes: allVertices,
    length: allVertices.count * MemoryLayout<simd_float3>.size, options: .storageModeShared)
    let debugBuffer = device.makeBuffer(length: pixels.count *
    MemoryLayout<simd_float3>.size, options: .storageModeShared)

    let computeCommandEncoder = commandBuffer?.makeComputeCommandEncoder()
    computeCommandEncoder?.setComputePipelineState(computePipeline)
    computeCommandEncoder?.setBuffer(cameraBuffer, offset: 0, index: 0)
    computeCommandEncoder?.setBuffer(materialsBuffer, offset: 0, index: 1)
    computeCommandEncoder?.setBuffer(squareLightsBuffer, offset: 0, index: 2)
    computeCommandEncoder?.setBuffer(vertexBuffer, offset: 0, index: 3)
    computeCommandEncoder?.setBuffer(pixelsBuffer, offset: 0, index: 4)
    computeCommandEncoder?.setAccelerationStructure(accelerationStructure, bufferIndex: 5)
    computeCommandEncoder?.setBuffer(debugBuffer, offset: 0, index: 6)

    // sort out threads, can yoyu just do 32*32 even if it doesnt divide evenly?
    let width = Int(camerasGPU[0].resolution.x)
    let height = Int(camerasGPU[0].resolution.y)
    let gridSize = MTLSize(width: width, height: height, depth: 80)

    // let threadsPerThreadGroup = MTLSize(width: 32, height: 32, depth: 1)
    // let threadsPerThreadGroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadsPerThreadGroup = MTLSize(width: 8, height: 8, depth: 1)
    // let threadsPerThreadGroup = MTLSize(width: 4, height: 4, depth: 1)

    computeCommandEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadGroup)
    computeCommandEncoder?.endEncoding()
    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()

    let debugPtr = debugBuffer!.contents()
    let debugCount = pixels.count
    let debugBufferPointer = debugPtr.bindMemory(to: simd_float3.self, capacity: debugCount)
    let debugArray = Array(UnsafeBufferPointer(start: debugBufferPointer, count: debugCount))

    writeDebugArrayToFile(debugArray: debugArray, width: width, height: height, fileName: "debugOutput.txt")

    let pixelPtr = pixelsBuffer!.contents()
    let count = pixels.count
    let bufferPointer = pixelPtr.bindMemory(to: UInt8.self, capacity: count)
    let pixelArray = Array(UnsafeBufferPointer(start: bufferPointer, count: count))

    return pixelArray
}

func writeDebugArrayToFile(debugArray: [simd_float3], width: Int, height: Int, fileName: String) {
    let fileURL = URL(fileURLWithPath: fileName)
    var lines: [String] = []
    for y in 0..<height {
        var rowSum = simd_float3(0, 0, 0)
        for x in 0..<width {
            let idx = y * width + x
            rowSum += debugArray[idx]
        }
        let avg = rowSum / Float(width)
        lines.append("\(avg.x),\(avg.y),\(avg.z)")
    }
    do {
        let data = lines.joined(separator: "\n")
        try data.write(to: fileURL, atomically: true, encoding: .utf8)
        print("Debug output written to \(fileURL.path)")
    } catch {
        print("Failed to write debug output: \(error)")
    }
}

func convertSpheres(spheres: [Sphere]) -> [SphereGPU] {
    return spheres.map { sphere in
        SphereGPU(
            center: sphere.center,
            material: convertMaterial(material: sphere.material),
            radius: sphere.radius
        )
    }
}

func convertCameras(cameras: [Camera]) -> [CameraGPU] {
    return cameras.map { camera in
        CameraGPU(
            position: camera.position,
            direction: camera.direction,
            up: camera.up,
            resolution: camera.resolution,
            horizontalFov: camera.horizontalFov,
            ev100: camera.ev100
        )
    }
}

func generateRandomArray(size: Int) -> [Float] {
    var array: [Float] = []
    for _ in 0..<size {
        array.append(Float.random(in: 0..<10))
    }
    return array
}
