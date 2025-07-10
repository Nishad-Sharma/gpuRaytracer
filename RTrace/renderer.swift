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
//    let pixelsBuffer: MTLBuffer
    let lightsBuffer: MTLBuffer
    let vertexBuffer: MTLBuffer
    let debugBuffer: MTLBuffer

    let randomTexture: MTLTexture
    let renderTexture: MTLTexture
    
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
        // pixelsBuffer = device.makeBuffer(length: camera.pixelCount() * 4 * MemoryLayout<UInt8>.size,
        // options: .storageModeShared)!
        lightsBuffer = device.makeBuffer(bytes: squareLightsGPU,
        length: squareLightsGPU.count * MemoryLayout<SquareLightGPU>.size, options: .storageModeShared)!
        vertexBuffer = device.makeBuffer(bytes: allVertices,
        length: allVertices.count * MemoryLayout<simd_float3>.size, options: .storageModeShared)!
        debugBuffer = device.makeBuffer(length: camera.pixelCount() *
        MemoryLayout<simd_float3>.size, options: .storageModeShared)!

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: Int(camera.resolution.x),
            height: Int(camera.resolution.y),
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        textureDescriptor.storageMode = .shared
        renderTexture = device.makeTexture(descriptor: textureDescriptor)!
        
        let randomTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Uint,
            width: Int(camera.resolution.x),
            height: Int(camera.resolution.y),
            mipmapped: false
        )
        randomTextureDescriptor.usage = [.shaderRead]
        randomTextureDescriptor.storageMode = .shared
        randomTexture = device.makeTexture(descriptor: randomTextureDescriptor)!
    
        let width = Int(camera.resolution.x)
        let height = Int(camera.resolution.y)
        var randomData = [UInt32](repeating: 0, count: width * height)
        
        for i in 0..<(width * height) {
            // Using rand() as requested. Note: arc4random_uniform is often preferred in Swift.
            randomData[i] = UInt32(arc4random() % (1024 * 1024))
        }
        
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: width, height: height, depth: 1))
        let bytesPerRow = width * MemoryLayout<UInt32>.size
        
        randomTexture.replace(region: region,
                              mipmapLevel: 0,
                              withBytes: randomData,
                              bytesPerRow: bytesPerRow)
                
        
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
        // computeCommandEncoder?.setBuffer(pixelsBuffer, offset: 0, index: 4)
        computeCommandEncoder?.setAccelerationStructure(accelerationStructure, bufferIndex: 5)
        computeCommandEncoder?.setBuffer(debugBuffer, offset: 0, index: 6)
        computeCommandEncoder?.setTexture(renderTexture, index: 0)
        computeCommandEncoder?.setTexture(randomTexture, index: 1)

        // sort out threads, can yoyu just do 32*32 even if it doesnt divide evenly?
        let width = Int(camera.resolution.x)
        let height = Int(camera.resolution.y)
//        let pixels = [UInt8](repeating: 0, count: camera.pixelCount() * 4)
        
        let gridSize = MTLSize(width: width, height: height, depth: 1)
        // let threadsPerThreadGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerThreadGroup = MTLSize(width: 8, height: 8, depth: 1)

        computeCommandEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadGroup)
        computeCommandEncoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()

        saveTextureToImage(texture: renderTexture, fileName: outputFileName)
        
        // let pixelPtr = pixelsBuffer.contents()
        // let count = pixels.count
        // let bufferPointer = pixelPtr.bindMemory(to: UInt8.self, capacity: count)
        // let pixelArray = Array(UnsafeBufferPointer(start: bufferPointer, count: count))

//        let bytesPerRow = width * 4
//        let bytesPerImage = bytesPerRow * height
//        let pixelCount = width * height * 4
//        let tempBuffer = device.makeBuffer(length: pixelCount, options: .storageModeShared)!

//        // Create a blit command buffer to copy texture data to buffer
//        let blitCommandBuffer = commandQueue.makeCommandBuffer()
//        let blitEncoder = blitCommandBuffer?.makeBlitCommandEncoder()
//
//        let origin = MTLOrigin(x: 0, y: 0, z: 0)
//        let size = MTLSize(width: width, height: height, depth: 1)
//
//        blitEncoder?.copy(
//            from: renderTexture,
//            sourceSlice: 0,
//            sourceLevel: 0,
//            sourceOrigin: origin,
//            sourceSize: size,
//            to: tempBuffer,
//            destinationOffset: 0,
//            destinationBytesPerRow: bytesPerRow,
//            destinationBytesPerImage: bytesPerImage
//        )
//
//        blitEncoder?.endEncoding()
//        blitCommandBuffer?.commit()
//        blitCommandBuffer?.waitUntilCompleted()
//
//        let pixelPtr = tempBuffer.contents()
//        let bufferPointer = pixelPtr.bindMemory(to: UInt8.self, capacity: pixelCount)
//        let pixelArray = Array(UnsafeBufferPointer(start: bufferPointer, count: pixelCount))
//
//        savePixelArrayToImage(pixels: pixelArray, width: width, height: height, fileName: outputFileName)
    }
}
