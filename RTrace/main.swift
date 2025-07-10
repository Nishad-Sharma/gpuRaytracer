//
//  main.swift
//  gpuRaytracer
//
//  Created by Nishad Sharma on 5/6/2025.
//

import Foundation
import Metal
import simd
import MetalKit

let outputFileName: String = {
    let args = CommandLine.arguments
    if args.count > 1 {
        return args[1]
    } else {
        let sourceFileURL = URL(fileURLWithPath: #file)
        
        // Navigate up to the project directory
        let projectDir = sourceFileURL.deletingLastPathComponent().path
        
        // Create output path in the project directory
        return "\(projectDir)/output.png"
    }
}()

let renderer = try! Renderer()

renderer.draw()


//// Position camera to look into the room from the front
//let direction = simd_normalize(simd_float3(0, 0, -2.5) - simd_float3(0, 0, 9))
//let camera = Camera(position: simd_float3(0, 0, 9), direction: direction,
//    resolution: simd_int2(800, 600), horizontalFov: Float.pi / 4.0)
//
//let roomSize: Float = 5.0
//let half = roomSize / 2.0
//
//let lightWidth: Float = 1.5
//let lightDepth: Float = 1.5
//let lightY: Float = half - 0.01
//let lightCenter = simd_float3(0, lightY, 0)
//let halfW = lightWidth / 2
//let halfD = lightDepth / 2
//
//// Vertices for the two triangles (rectangle split into two)
//let v0 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z - halfD)
//let v1 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z - halfD)
//let v2 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z + halfD)
//let v3 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z + halfD)
//
//// Emissive material for the light triangles
//var lightMaterial = Material(
//    diffuse: simd_float4(1.0, 0.95, 0.9, 1.0),
//    metallic: 0.0,
//    roughness: 0.0,
//    emissive: simd_float3(1.0, 1.0, 1.0) // White light
//    // emissive: simd_float3(20, 20, 20) // Adjust intensity as needed
//)
//
//var squareLight = SquareLight(
//    center: lightCenter,
//    vertices: [v0, v1, v2, v3],
//    material: lightMaterial,
//    LightType: .bulb(luminousEfficacy: 100.0, watts: 12), // Example values
//    // LightType: .bulb(efficacy: 150, watts: 200), // Example values
//    width: lightWidth,
//    depth: lightDepth
//)
//
//var cornellBoxTriangles = createCornellBoxScene()
//
//// add triangles for light
//cornellBoxTriangles.append(Triangle(vertices: [v0, v1, v2], material: lightMaterial))
//cornellBoxTriangles.append(Triangle(vertices: [v0, v2, v3], material: lightMaterial))
//
//
//let width = Int(camera.resolution.x)
//let height = Int(camera.resolution.y)
//let pixelCount = width * height
//var pixels = [UInt8](repeating: 0, count: pixelCount * 4)
//
//let device = MTLCreateSystemDefaultDevice()!
//guard device.supportsRaytracing else {
//    print("Ray tracing not supported on this device")
//    exit(1)
//}
//
//let accelerationStructure = setupAccelerationStructures(device: device, triangles: cornellBoxTriangles)
//// dont need render pipeline since we aren't rendering to screen - send to compute pipe
//let startTime = DispatchTime.now()
//
//pixels = drawTriangle(device: device, cameras: [camera], triangles: cornellBoxTriangles, squareLights: [squareLight], pixels: pixels, accelerationStructure: accelerationStructure)
//
//savePixelArrayToImage(pixels: pixels, width: width, height: height, fileName: outputFileName)
//let endTime = DispatchTime.now()
//let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
//let timeInterval = Double(nanoTime) / 1_000_000_000 // Convert to seconds
//
//print("Render completed in \(String(format: "%.2f", timeInterval)) seconds")
//



