//
//  renderer.swift
//  RTrace
//
//  Created by Nishad Sharma on 9/7/2025.
//
import Metal

public class Renderer {
    let device: MTLDevice
//    let library: MTLLibrary
    let computePipeline: MTLComputePipelineState
    let commandQueue: MTLCommandQueue
    
    let cameraBuffer: MTLBuffer
    let materialsBuffer: MTLBuffer
    let pixelsBuffer: MTLBuffer
    let lightsBuffer: MTLBuffer
    let vertexBuffer: MTLBuffer
    let debugBuffer: MTLBuffer
    
    let accelerationStructure: MTLAccelerationStructure
    
    let camera: Camera
    
    init() throws {
        device = MTLCreateSystemDefaultDevice()!
        
        guard let metalLibURL = Bundle.main.url(forResource: "default", withExtension: "metallib") else {
            fatalError("Could not find MyMetalLib.metallib in app bundle")
        }
        guard let defaultLibrary = try? device.makeLibrary(URL: metalLibURL) else {
            fatalError("Could not load Metal library from \(metalLibURL.path)")
        }

        guard let drawFunction = defaultLibrary.makeFunction(name: "pathTrace") else {
            fatalError("Could not find 'drawTriangle' kernel in Metal library")
        }
        
        self.computePipeline = try device.makeComputePipelineState(function: drawFunction)
        self.commandQueue = device.makeCommandQueue()!
        
        let cornellBoxScene = initCornellBox()
        
        
        let camerasGPU = convertCameras(cameras: [cornellBoxScene.camera])
        // let trianglesGPU = convertTriangle(triangles: triangles)
        let materialsGPU = cornellBoxScene.triangles.map { convertMaterial(material: $0.material) }
        let squareLightsGPU = [convertSquareLight(squareLight: cornellBoxScene.light)]
        // triangle verts needed for normal calc - cant pull from accel structure
        var allVertices: [simd_float3] = []
        for triangle in cornellBoxScene.triangles {
            allVertices.append(contentsOf: triangle.vertices)
        }
        
        camera = cornellBoxScene.camera
        
        cameraBuffer = device.makeBuffer(bytes: camerasGPU,
        length: camerasGPU.count * MemoryLayout<CameraGPU>.size, options: .storageModeShared)!
        materialsBuffer = device.makeBuffer(bytes: materialsGPU,
        length: materialsGPU.count * MemoryLayout<MaterialGPU>.size, options: .storageModeShared)!
        pixelsBuffer = device.makeBuffer(length: camera.pixelCount() * 4 * MemoryLayout<UInt8>.size,
        options: .storageModeShared)!
        lightsBuffer = device.makeBuffer(bytes: squareLightsGPU,
        length: squareLightsGPU.count * MemoryLayout<SquareLightGPU>.size, options: .storageModeShared)!
        vertexBuffer = device.makeBuffer(bytes: allVertices,
        length: allVertices.count * MemoryLayout<simd_float3>.size, options: .storageModeShared)!
        debugBuffer = device.makeBuffer(length: camera.pixelCount() *
        MemoryLayout<simd_float3>.size, options: .storageModeShared)!
        
        accelerationStructure = setupAccelerationStructures(device: device, triangles: cornellBoxScene.triangles)

    }
    
    func draw() {
        let commandBuffer = commandQueue.makeCommandBuffer()
        
        let computeCommandEncoder = commandBuffer?.makeComputeCommandEncoder()
        computeCommandEncoder?.setComputePipelineState(computePipeline)
        computeCommandEncoder?.setBuffer(cameraBuffer, offset: 0, index: 0)
        computeCommandEncoder?.setBuffer(materialsBuffer, offset: 0, index: 1)
        computeCommandEncoder?.setBuffer(lightsBuffer, offset: 0, index: 2)
        computeCommandEncoder?.setBuffer(vertexBuffer, offset: 0, index: 3)
        computeCommandEncoder?.setBuffer(pixelsBuffer, offset: 0, index: 4)
        computeCommandEncoder?.setAccelerationStructure(accelerationStructure, bufferIndex: 5)
        computeCommandEncoder?.setBuffer(debugBuffer, offset: 0, index: 6)

        // sort out threads, can yoyu just do 32*32 even if it doesnt divide evenly?
        let width = Int(camera.resolution.x)
        let height = Int(camera.resolution.y)
        let pixels = [UInt8](repeating: 0, count: camera.pixelCount() * 4)
        
        let gridSize = MTLSize(width: width, height: height, depth: 1)
        // let threadsPerThreadGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerThreadGroup = MTLSize(width: 8, height: 8, depth: 1)

        computeCommandEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadGroup)
        computeCommandEncoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        
        let pixelPtr = pixelsBuffer.contents()
        let count = pixels.count
        let bufferPointer = pixelPtr.bindMemory(to: UInt8.self, capacity: count)
        let pixelArray = Array(UnsafeBufferPointer(start: bufferPointer, count: count))

        savePixelArrayToImage(pixels: pixelArray, width: width, height: height, fileName: outputFileName)
    }
}
