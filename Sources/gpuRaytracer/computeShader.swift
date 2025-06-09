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
import CShaderTypes

func convertSpheres(spheres: [Sphere]) -> [SphereGPU] {
    return spheres.map { sphere in
        SphereGPU(
            center: sphere.center,
            material: convertMaterial(material: sphere.material),
            radius: sphere.radius
        )
    }
}

func convertMaterial(material: Material) -> MaterialGPU {
    return MaterialGPU(
        diffuse: material.diffuse,
        metallic: material.metallic,
        roughness: material.roughness
    )
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

func convertLights(lights: [SphereLight]) -> [SphereLightGPU] {
    return lights.map { light in
        SphereLightGPU(
            center: light.center,
            color: light.color,
            emittedRadiance: light.emittedRadiance,
            radius: light.radius
        )
    }
}

func gpuIntersect(cameras: [Camera], spheres: [Sphere], pixels: [UInt8], lights: [SphereLight], ambientLight: simd_float4) -> [UInt8] {
    let spheresGPU = convertSpheres(spheres: spheres)
    let camerasGPU = convertCameras(cameras: cameras)
    let lightsGPU = convertLights(lights: lights)

    let device = MTLCreateSystemDefaultDevice()!
    let metalLibURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("Sources/gpuRaytracer/MyMetalLib.metallib")

    guard let defaultLibrary = try? device.makeLibrary(URL: metalLibURL) else {
        fatalError("Could not load Metal library from \(metalLibURL.path)") 
    }

    guard let intersectFunction = defaultLibrary.makeFunction(name: "draw") else {
        fatalError("Could not find 'draw' kernel in Metal library")
    }

    let computePipeline: MTLComputePipelineState
    do {
        computePipeline = try device.makeComputePipelineState(function: intersectFunction)
    } catch {
        print("Failed to create compute pipeline state: \(error)")
        return pixels // maybe bad return fix?
    }

    let commandQueue = device.makeCommandQueue()
    let commandBuffer = commandQueue?.makeCommandBuffer()

    let cameraBuffer = device.makeBuffer(bytes: camerasGPU, length: camerasGPU.count * MemoryLayout<CameraGPU>.size, options: .storageModeShared)
    let spheresBuffer = device.makeBuffer(bytes: spheresGPU, length: spheresGPU.count * MemoryLayout<SphereGPU>.size, options: .storageModeShared)
    var sphereCount = UInt32(spheresGPU.count)
    let pixelsBuffer = device.makeBuffer(length: pixels.count * MemoryLayout<UInt8>.size, options: .storageModeShared)
    let lightsBuffer = device.makeBuffer(bytes: lightsGPU, length: lightsGPU.count * MemoryLayout<SphereLightGPU>.size, options: .storageModeShared)
    var lightCount = UInt32(lightsGPU.count)
    let ambientLightBuffer = device.makeBuffer(bytes: [ambientLight], length: MemoryLayout<simd_float4>.size, options: .storageModeShared)

    let computeCommandEncoder = commandBuffer?.makeComputeCommandEncoder()
    computeCommandEncoder?.setComputePipelineState(computePipeline)
    computeCommandEncoder?.setBuffer(cameraBuffer, offset: 0, index: 0)
    computeCommandEncoder?.setBuffer(spheresBuffer, offset: 0, index: 1)
    computeCommandEncoder?.setBytes(&sphereCount, length: MemoryLayout<UInt32>.size, index: 2)
    computeCommandEncoder?.setBuffer(pixelsBuffer, offset: 0, index: 3)
    computeCommandEncoder?.setBuffer(lightsBuffer, offset: 0, index: 4)
    computeCommandEncoder?.setBytes(&lightCount, length: MemoryLayout<UInt32>.size, index: 5)
    computeCommandEncoder?.setBuffer(ambientLightBuffer, offset: 0, index: 6)

    // sort out threads, can yoyu just do 32*32 even if it doesnt divide evenly?
    let width = Int(camerasGPU[0].resolution.x)
    let height = Int(camerasGPU[0].resolution.y)
    let gridSize = MTLSize(width: width, height: height, depth: 1)

    // let threadsPerThreadGroup = MTLSize(width: 32, height: 32, depth: 1)
    let threadsPerThreadGroup = MTLSize(width: 16, height: 16, depth: 1)
    // let threadsPerThreadGroup = MTLSize(width: 8, height: 8, depth: 1)

    computeCommandEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadGroup)
    computeCommandEncoder?.endEncoding()
    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()

    let pixelPtr = pixelsBuffer!.contents()
    let count = pixels.count
    let bufferPointer = pixelPtr.bindMemory(to: UInt8.self, capacity: count)
    let pixelArray = Array(UnsafeBufferPointer(start: bufferPointer, count: count))
    return pixelArray
}


func gpuAddArrays() {
    let device = MTLCreateSystemDefaultDevice()

    // collection of precompiled gpu shader functions in this case we will specify our add function.
    let defaultLibrary = device?.makeDefaultLibrary()

    let addFunction = defaultLibrary?.makeFunction(name: "addArrays")
    // this MTL function object isnt executable yet. need to make a computePipelineState to specify the what the gpu will do. in this case, use our addArray function
    // need try because makeComputerPipelineState is a throw func - error if it fails
    let computePipeline: MTLComputePipelineState
    do {
        computePipeline = try device!.makeComputePipelineState(function: addFunction!)
    } catch {
        print("Failed to create compute pipeline state: \(error)")
        return
    }

    // queue of commands to be sent to gpu
    let commandQueue = device?.makeCommandQueue()

    // buffer of commands to be run on gpu, when we commit to queue - gpu starts running said commands. in this case will just be addArray command
    let commandBuffer = commandQueue?.makeCommandBuffer()

    // MTL buffers to be added and result buffer. allocd memory that can be accessed by cpu and gpu. gpu will read the first two and write to 3rd
    let arraySize = 1000
    let mBuff1 = device?.makeBuffer(bytes: generateRandomArray(size: arraySize), length: arraySize * MemoryLayout<Float>.size, options: MTLResourceOptions.storageModeShared) // shared storagemode for cpu + gpu access
    let mBuff2 = device?.makeBuffer(bytes: generateRandomArray(size: arraySize), length: arraySize * MemoryLayout<Float>.size, options: MTLResourceOptions.storageModeShared)
    let mBuff3 = device?.makeBuffer(length: arraySize * MemoryLayout<Float>.size, options: MTLResourceOptions.storageModeShared)

    // now need to write commands to command buffer with encoder
    // command encoder translates commands (like setting buffers, specifying pipelines, and dispatching compute threads) into instructions for gpu to execute
    // this is different from a commandEncoder which is generic and can do render, compute and blit (blit is just memory copy). computeCmdEncoder just does the compute
    let computeCommandEncoder = commandBuffer?.makeComputeCommandEncoder()

    // now we set computePipelineState and buffers to be used
    computeCommandEncoder?.setComputePipelineState(computePipeline)
    computeCommandEncoder?.setBuffer(mBuff1, offset: 0, index: 0)
    computeCommandEncoder?.setBuffer(mBuff2, offset: 0, index: 1)
    computeCommandEncoder?.setBuffer(mBuff3, offset: 0, index: 2)

    // print out threadgroupsize info
    let pipelineThreadGroupSize = computePipeline.maxTotalThreadsPerThreadgroup
    let deviceThreadGroupSize = device?.maxThreadsPerThreadgroup
    let deviceThreadGroupMemLength = device?.maxThreadgroupMemoryLength
    print("threads per thread group: ")
    print("pipeline: " + String(describing: pipelineThreadGroupSize)) // 1024 threads per threadgroup
    print("device: " + String(describing: deviceThreadGroupSize)) // 1024 * 1024 * 1024 dunno what this means, maybe max width/height/depth possible for grid size?
    print("device thread group mem length: " + String(describing: deviceThreadGroupMemLength)) // 32768 = 1024 * 32 (means 32 threadgroups?)

    // now specify threads and shape
    let gridSize = MTLSize(width: arraySize, height: 1, depth: 1) // shape of data in buffer - will need an arraySize No. of threads to compute the whole calc
    var threadGroupSize = computePipeline.maxTotalThreadsPerThreadgroup
    //var threadGroupSize = computePipeline?.maxTotalThreadsPerThreadgroup > arraySize ? arraySize : computePipeline?.maxTotalThreadsPerThreadgroup
    if (threadGroupSize > arraySize) {
        threadGroupSize = arraySize
    }
    // the way the data will be split for the gpu. since gpu has 1024 max per threadgroup cant send more.
    // so if we had an arraySize of 2048, it would be sent in 2 batches of size (1024, 1, 1)
    let threadsPerThreadGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)

    // encode command to use the gridsize and split specified
    computeCommandEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadGroup)
    // done encoding commands
    computeCommandEncoder?.endEncoding()
    // sends completed command buffer to the queue which then sends to gpu
    commandBuffer?.commit()

    commandBuffer?.waitUntilCompleted()

    // ptr to each buffer
    var b1Ptr = mBuff1?.contents()
    var b2Ptr = mBuff2?.contents()
    var b3Ptr = mBuff3?.contents()

    for _ in 0...9 {
        let a = UnsafeRawPointer(b1Ptr!).load(as: Float.self)
        let b = UnsafeRawPointer(b2Ptr!).load(as: Float.self)
        let c = UnsafeRawPointer(b3Ptr!).load(as: Float.self)
        print("\(a) + \(b) = \(c)")
        if ((a + b) != c) {
            print("ERROR")
        }
        b1Ptr = b1Ptr?.advanced(by: MemoryLayout<Float>.size)
        b2Ptr = b2Ptr?.advanced(by: MemoryLayout<Float>.size)
        b3Ptr = b3Ptr?.advanced(by: MemoryLayout<Float>.size)
    }

}

func generateRandomArray(size: Int) -> [Float] {
    var array: [Float] = []
    for _ in 0..<size {
        array.append(Float.random(in: 0..<10))
    }
    return array
}
