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


// Replace your cube setup with Cornell Box
let cornellBoxTriangles = createCornellBoxScene()

// Position camera to look into the room from the front
let direction = simd_normalize(simd_float3(0, 0, -2.5) - simd_float3(0, 0, 9))
let camera = Camera(position: simd_float3(0, 0, 9), direction: direction,
    resolution: simd_int2(800, 600), horizontalFov: Float.pi / 4.0)

let roomSize: Float = 5.0
let half = roomSize / 2.0

// var boxLight = BoxLight(
//     center: simd_float3(0, half - 0.05, 0), // Slightly below ceiling
//     material: Material(
//         diffuse: simd_float4(1.0, 0.95, 0.9, 1.0),
//         metallic: 0.0,
//         roughness: 0.0,
//         emissive: simd_float3(20, 20, 20) // Will be calculated from LightType
//     ),
//     LightType: .bulb(efficacy: 120, watts: 200),
//     width: 1.5,
//     height: 0.1,
//     depth: 1.5
// )

// setup squareLight
// Emissive material for the light triangles
let lightMaterial = Material(
    diffuse: simd_float4(1.0, 0.95, 0.9, 1.0),
    metallic: 0.0,
    roughness: 0.0,
    emissive: simd_float3(20, 20, 20) // Adjust intensity as needed
)

let lightWidth: Float = 1.5
let lightDepth: Float = 1.5
let lightY: Float = half - 0.01
let lightCenter = simd_float3(0, lightY, 0)
let halfW = lightWidth / 2
let halfD = lightDepth / 2

// Vertices for the two triangles (rectangle split into two)
let v0 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z - halfD)
let v1 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z - halfD)
let v2 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z + halfD)
let v3 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z + halfD)

var squareLight = SquareLight(
    center: lightCenter,
    vertices: [v0, v1, v2, v3],
    material: lightMaterial,
    LightType: .bulb(efficacy: 120, watts: 200), // Example values
    width: lightWidth,
    depth: lightDepth
)
// squareLight.material.emissive = squareLight.emittedRadiance;

// boxLight.material.emissive = boxLight.emittedRadiance // Use calculated radiance

let width = Int(camera.resolution.x)
let height = Int(camera.resolution.y)
let pixelCount = width * height
var pixels = [UInt8](repeating: 0, count: pixelCount * 4)

let device = MTLCreateSystemDefaultDevice()!
guard device.supportsRaytracing else {
    print("Ray tracing not supported on this device")
    exit(1)
}

let accelerationStructure = setupAccelerationStructures(device: device, triangles: cornellBoxTriangles)
// dont need render pipeline since we aren't rendering to screen - send to compute pipe
let startTime = DispatchTime.now()

pixels = drawTriangle(device: device, cameras: [camera], triangles: cornellBoxTriangles, squareLights: [squareLight], pixels: pixels, accelerationStructure: accelerationStructure)

savePixelArrayToImage(pixels: pixels, width: width, height: height, fileName: "cornell.png")
let endTime = DispatchTime.now()
let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
let timeInterval = Double(nanoTime) / 1_000_000_000 // Convert to seconds

print("Render completed in \(String(format: "%.2f", timeInterval)) seconds")

func createCornellBoxScene() -> [Triangle] {
    var triangles: [Triangle] = []
    
    // Cornell Box dimensions
    let roomSize: Float = 5.0
    let half = roomSize / 2.0
    
    // Materials for Cornell Box
    let redMaterial = Material(diffuse: simd_float4(0.9, 0.0, 0.0, 1), metallic: 0.3, roughness: 0.6)    // Left wall
    let greenMaterial = Material(diffuse: simd_float4(0.0, 0.7, 0.0, 1), metallic: 0.3, roughness: 0.6)  // Right wall
    let whiteMaterial = Material(diffuse: simd_float4(0.9, 0.9, 0.9, 1), metallic: 0.3, roughness: 0.6)  // Other walls
    let diffuseBoxMaterial = Material(diffuse: simd_float4(0.9, 0.9, 0.9, 1), metallic: 0.1, roughness: 0.8)  
    let specularBoxMaterial = Material(diffuse: simd_float4(0.9, 0.9, 0.9, 1), metallic: 0.9, roughness: 0.1)  
    // let blueMaterial = Material(diffuse: simd_float4(0.25, 0.25, 0.75, 1), metallic: 0.3, roughness: 0.6)   // Tall box
    // let yellowMaterial = Material(diffuse: simd_float4(0.75, 0.75, 0.25, 1), metallic: 0.3, roughness: 0.6) // Short cube
    
    // Back wall (z = -half) - facing INTO room (positive Z direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // bottom-left
        simd_float3(half, half, -half),    // top-right
        simd_float3(-half, half, -half)  // top-left
    ], material: whiteMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // bottom-left
        simd_float3(half, -half, -half),   // bottom-right
        simd_float3(half, half, -half)   // top-right
    ], material: whiteMaterial))
    
    // Left wall (x = -half) - RED - facing INTO room (positive X direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-bottom
        simd_float3(-half, half, half),    // front-top
        simd_float3(-half, -half, half)  // front-bottom
    ], material: redMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-bottom
        simd_float3(-half, half, -half),   // back-top
        simd_float3(-half, half, half)   // front-top
    ], material: redMaterial))
    
    // Right wall (x = +half) - GREEN - facing INTO room (negative X direction)
    triangles.append(Triangle(vertices: [
        simd_float3(half, -half, -half),  // back-bottom
        simd_float3(half, half, half),     // front-top
        simd_float3(half, half, -half)   // back-top
    ], material: greenMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(half, -half, -half),  // back-bottom
        simd_float3(half, -half, half),    // front-bottom
        simd_float3(half, half, half)    // front-top
    ], material: greenMaterial))
    
    // Floor (y = -half) - facing UP into room (positive Y direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-left
        simd_float3(half, -half, half),    // front-right
        simd_float3(half, -half, -half)  // back-right
    ], material: whiteMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-left
        simd_float3(-half, -half, half),   // front-left
        simd_float3(half, -half, half)   // front-right
    ], material: whiteMaterial))
    
    // Ceiling (y = +half) - facing DOWN into room (negative Y direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, half, -half),  // back-left
        simd_float3(half, half, half),     // front-right
        simd_float3(-half, half, half)   // front-left
    ], material: whiteMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, half, -half),  // back-left
        simd_float3(half, half, -half),    // back-right
        simd_float3(half, half, half)    // front-right
    ], material: whiteMaterial))
  
    // back rectanglular prism
    let tallBoxWidth: Float = 1.2
    let tallBoxHeight: Float = 2.8
    let tallBoxDepth: Float = 1.2
    let tallBoxPosition = simd_float3(-1, -half + tallBoxHeight/2 + 0.01, -1.5) // Back left
    let tallBoxRotationY: Float = Float.pi / 2.4 
    
    let tallBoxVertices = createRotatedBoxVertices(
        center: tallBoxPosition,
        width: tallBoxWidth,
        height: tallBoxHeight,
        depth: tallBoxDepth,
        rotationY: tallBoxRotationY
    )
    
    triangles.append(contentsOf: createBoxTriangles(vertices: tallBoxVertices, material: diffuseBoxMaterial))
    
    // front cube
    let shortBoxSize: Float = 1.2
    let shortBoxHeight: Float = 1.2
    let shortBoxPosition = simd_float3(0.7, -half + shortBoxHeight/2 + 0.01, 1.2) // Front right
    let shortBoxRotationY: Float = -Float.pi / 2.5 
    
    let shortBoxVertices = createRotatedBoxVertices(
        center: shortBoxPosition,
        width: shortBoxSize,
        height: shortBoxHeight,
        depth: shortBoxSize,
        rotationY: shortBoxRotationY
    )
    
    triangles.append(contentsOf: createBoxTriangles(vertices: shortBoxVertices, material: specularBoxMaterial))
    // triangles.append(contentsOf: createBoxTriangles(vertices: shortBoxVertices, material: diffuseBoxMaterial))

    // // Create BoxLight for the ceiling
    // let ceilingLight = BoxLight(
    //     center: simd_float3(0, half - 0.05, 0), // Slightly below ceiling
    //     material: Material(
    //         diffuse: simd_float4(1.0, 1.0, 1.0, 1.0),
    //         metallic: 0.0,
    //         roughness: 0.0,
    //         emissive: simd_float3(20, 20, 20) // Will be calculated from LightType
    //     ),
    //     LightType: .bulb(efficacy: 120, watts: 200),
    //     width: 1.5,
    //     height: 0.1,
    //     depth: 1.5
    // )
    // // ceilingLight.material.emissive = ceilingLight.emittedRadiance // Use calculated radiance
    
    // // because material emmisive set poorly
    // let lightMaterial = Material(
    //     diffuse: ceilingLight.material.diffuse,
    //     metallic: ceilingLight.material.metallic,
    //     roughness: ceilingLight.material.roughness,
    //     emissive: ceilingLight.material.emissive // Use calculated radiance - poorly done atm fix later
    // )

    // let lightBoxVertices = createRotatedBoxVertices(
    //     center: ceilingLight.center,
    //     width: ceilingLight.width,
    //     height: ceilingLight.height,
    //     depth: ceilingLight.depth,
    //     rotationY: 0
    // )

    // triangles.append(contentsOf: createBoxTriangles(vertices: lightBoxVertices, material: lightMaterial))

    //setup emmisive triangles
    let lightWidth: Float = 1.5
    let lightHeight: Float = 1.5
    let lightY: Float = half - 0.01
    let lightCenter = simd_float3(0, lightY, 0)
    let halfW: Float = lightWidth / 2
    let halfD: Float = lightHeight / 2

    // have to define lightmaterial here as well
    let lightMaterial = Material(
        diffuse: simd_float4(1.0, 0.95, 0.9, 1.0),
        metallic: 0.0,
        roughness: 0.0,
        emissive: simd_float3(20, 20, 20) // Adjust intensity as needed
    )

    // Vertices for the two triangles (rectangle split into two)
    let v0 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z - halfD)
    let v1 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z - halfD)
    let v2 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z + halfD)
    let v3 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z + halfD)

    // add the two triangles (winding counter clock so normal faces down into the room)
    triangles.append(Triangle(vertices: [v0, v1, v2], material: lightMaterial))
    triangles.append(Triangle(vertices: [v0, v2, v3], material: lightMaterial))

    return triangles
}

func createRotatedBoxVertices(center: simd_float3, width: Float, height: Float, depth: Float, rotationY: Float) -> [simd_float3] {
    let halfW = width / 2
    let halfH = height / 2
    let halfD = depth / 2
    
    // Create box vertices centered at origin
    let baseVertices = [
        simd_float3(-halfW, -halfH, -halfD), // 0
        simd_float3( halfW, -halfH, -halfD), // 1
        simd_float3( halfW,  halfH, -halfD), // 2
        simd_float3(-halfW,  halfH, -halfD), // 3
        simd_float3(-halfW, -halfH,  halfD), // 4
        simd_float3( halfW, -halfH,  halfD), // 5
        simd_float3( halfW,  halfH,  halfD), // 6
        simd_float3(-halfW,  halfH,  halfD), // 7
    ]
    
    // Rotation matrix around Y-axis
    let cosY = cos(rotationY)
    let sinY = sin(rotationY)
    let rotationMatrix = simd_float4x4(
        simd_float4(cosY, 0, sinY, 0),
        simd_float4(0, 1, 0, 0),
        simd_float4(-sinY, 0, cosY, 0),
        simd_float4(0, 0, 0, 1)
    )
    
    // Apply rotation and translation
    return baseVertices.map { vertex in
        let homogeneous = simd_float4(vertex.x, vertex.y, vertex.z, 1.0)
        let rotated = rotationMatrix * homogeneous
        return simd_float3(rotated.x + center.x, rotated.y + center.y, rotated.z + center.z)
    }
}

func createBoxTriangles(vertices: [simd_float3], material: Material) -> [Triangle] {
    var triangles: [Triangle] = []
    
    // Back face 
    triangles.append(Triangle(vertices: [vertices[0], vertices[2], vertices[1]], material: material))
    triangles.append(Triangle(vertices: [vertices[0], vertices[3], vertices[2]], material: material))
    
    // Front face 
    triangles.append(Triangle(vertices: [vertices[4], vertices[5], vertices[6]], material: material))
    triangles.append(Triangle(vertices: [vertices[4], vertices[6], vertices[7]], material: material))
    
    // Left face 
    triangles.append(Triangle(vertices: [vertices[0], vertices[4], vertices[7]], material: material))
    triangles.append(Triangle(vertices: [vertices[0], vertices[7], vertices[3]], material: material))
    
    // Right face 
    triangles.append(Triangle(vertices: [vertices[1], vertices[6], vertices[5]], material: material))
    triangles.append(Triangle(vertices: [vertices[1], vertices[2], vertices[6]], material: material))
    
    // Bottom face 
    triangles.append(Triangle(vertices: [vertices[0], vertices[5], vertices[4]], material: material)) 
    triangles.append(Triangle(vertices: [vertices[0], vertices[1], vertices[5]], material: material))
    
    // Top face 
    triangles.append(Triangle(vertices: [vertices[3], vertices[6], vertices[2]], material: material)) 
    triangles.append(Triangle(vertices: [vertices[3], vertices[7], vertices[6]], material: material))
    
    return triangles
}


// let direction = simd_normalize(simd_float3(0, 0, 0) - simd_float3(5, 0, 0))
// let camera = Camera(position: simd_float3(5, 0, 0), direction: direction,
//     resolution: simd_int2(400, 300), horizontalFov: Float.pi / 4.0)

// let direction = simd_normalize(simd_float3(0, 0, 0) - simd_float3(5, 3, 5))
// let camera = Camera(position: simd_float3(5, 3, 5), direction: direction,
//     resolution: simd_int2(400, 300), horizontalFov: Float.pi / 4.0)

// // cube setup
// let cubeSize: Float = 1
// let mat = Material(diffuse: simd_float4(1, 0, 0, 1), metallic: 0.0, roughness: 0.5)
// let vertices = [
//     simd_float3(-cubeSize, -cubeSize, -cubeSize), // 0: left  bottom back
//     simd_float3( cubeSize, -cubeSize, -cubeSize), // 1: right bottom back
//     simd_float3( cubeSize,  cubeSize, -cubeSize), // 2: right top    back
//     simd_float3(-cubeSize,  cubeSize, -cubeSize), // 3: left  top    back
//     simd_float3(-cubeSize, -cubeSize,  cubeSize), // 4: left  bottom front
//     simd_float3( cubeSize, -cubeSize,  cubeSize), // 5: right bottom front
//     simd_float3( cubeSize,  cubeSize,  cubeSize), // 6: right top    front
//     simd_float3(-cubeSize,  cubeSize,  cubeSize), // 7: left  top    front
// ]


// //setup triangles for cube
// var triangles: [Triangle] = []
// // Back face (z = -half)
// triangles.append(Triangle(vertices: [vertices[0], vertices[1], vertices[2]], material: mat))
// triangles.append(Triangle(vertices: [vertices[0], vertices[2], vertices[3]], material: mat))

// // Front face (z = +half)
// triangles.append(Triangle(vertices: [vertices[4], vertices[6], vertices[5]], material: mat))
// triangles.append(Triangle(vertices: [vertices[4], vertices[7], vertices[6]], material: mat))

// // Left face (x = -half)
// triangles.append(Triangle(vertices: [vertices[0], vertices[3], vertices[7]], material: mat))
// triangles.append(Triangle(vertices: [vertices[0], vertices[7], vertices[4]], material: mat))

// // Right face (x = +half)
// triangles.append(Triangle(vertices: [vertices[1], vertices[5], vertices[6]], material: mat))
// triangles.append(Triangle(vertices: [vertices[1], vertices[6], vertices[2]], material: mat))

// // Bottom face (y = -half)
// triangles.append(Triangle(vertices: [vertices[0], vertices[4], vertices[5]], material: mat))
// triangles.append(Triangle(vertices: [vertices[0], vertices[5], vertices[1]], material: mat))

// // Top face (y = +half)
// triangles.append(Triangle(vertices: [vertices[3], vertices[2], vertices[6]], material: mat))
// triangles.append(Triangle(vertices: [vertices[3], vertices[6], vertices[7]], material: mat))

// // let triangle = Triangle(
// //     vertices: [
// //         simd_float3(0, -1, -1), // bottom left
// //         simd_float3(0,  1, -1), // top left
// //         simd_float3(0,  0,  1)  // right
// //     ],  
// //     material: Material(diffuse: simd_float4(1, 0, 0, 1), metallic: 0.0, roughness: 0.5)
// // )

// let width = Int(camera.resolution.x)
// let height = Int(camera.resolution.y)
// let pixelCount = width * height
// // Pre-allocate pixels array with black transparent pixels
// var pixels = [UInt8](repeating: 0, count: pixelCount * 4)

// // Create a Metal device
// let device = MTLCreateSystemDefaultDevice()!
// // Check if ray tracing is supported
// guard device.supportsRaytracing else {
//     print("Ray tracing not supported on this device")
//     exit(1)
// }
// // make accel structures
// let accelerationStructures   = setupAccelerationStructures(device: device, triangles: triangles)
// // dont need render pipeline since we aren't rendering to screen - send to compute pipe
// pixels = drawTriangle(device: device, cameras: [camera], pixels: pixels, accelerationStructure: accelerationStructures)

// savePixelArrayToImage(pixels: pixels, width: width, height: height, fileName: "triangle.png")




struct Triangle {
    var vertices: [simd_float3]
    var material: Material
}

//setup scene
func rayTraceScene() {
    let direction = simd_normalize(simd_float3(0, 0, 0) - simd_float3(13, 2, 3))
    let camera = Camera(position: simd_float3(13, 2, 3), direction: direction,
        resolution: simd_int2(400, 300), horizontalFov: Float.pi / 4.0)
    // let camera = Camera(position: simd_float3(13, 6, 3), direction: direction, // higher cam angle
    //     resolution: simd_int2(400, 300), horizontalFov: Float.pi / 4.0)
    let cameras = [camera]

    let lightBulb = SphereLight(center: simd_float3(3, 3, 3), color: simd_float4(1, 0.9, 0.7, 1), 
        LightType: .bulb(efficacy: 15, watts: 60), radius: 0.1)
    // let lightBulb = SphereLight(center: simd_float3(3, 3, 3), color: simd_float4(1, 0.8, 0.4, 1),  // warmer light
        // LightType: .bulb(efficacy: 15, watts: 60), radius: 0.1)
    let lights = [lightBulb]
    let ambientLight = simd_float4(0.53, 0.81, 0.92, 1) 

    var spheres: [Sphere] = []
    let sphere1 = Sphere(center: simd_float3(4, 2, 0), 
    material: Material(diffuse: simd_float4(1, 0, 0, 1), metallic: 0.9, roughness: 0.1), 
    radius: 1.0) // front sphere
    let sphere2 = Sphere(center: simd_float3(0, 2, 0), 
    material: Material(diffuse: simd_float4(0, 1, 0, 1), metallic: 0.05, roughness: 0.1), 
    radius: 1.0) // middle sphere
    let sphere3 = Sphere(center: simd_float3(-4, 2, 0), 
    material: Material(diffuse: simd_float4(0, 0, 1, 1), metallic: 0.1, roughness: 0.9), 
    radius: 1.0) // back sphere
    let ground = Sphere(center: simd_float3(0, -100, 0), 
    material: Material(diffuse: simd_float4(0.2, 0.2, 0.2, 1), metallic: 0.0, roughness: 0.9), 
    radius: 100.0) // ground sphere
    spheres.append(sphere1)
    spheres.append(sphere2)
    spheres.append(sphere3)
    spheres.append(ground)

    render(cameras: cameras, spheres: spheres, lights: lights, ambientLight: ambientLight)
}

func render(cameras: [Camera] = [], spheres: [Sphere] = [], lights: [SphereLight] = [], ambientLight: simd_float4 = simd_float4(0.53, 0.81, 0.92, 1)) {
    let startTime = DispatchTime.now()
    let camera = cameras[0]
    let width = Int(camera.resolution.x)
    let height = Int(camera.resolution.y)
    let pixelCount = width * height
    // Pre-allocate pixels array with black transparent pixels
    var pixels = [UInt8](repeating: 0, count: pixelCount * 4)

    pixels = gpuIntersect(cameras: cameras, spheres: spheres, pixels: pixels, lights: lights, ambientLight: ambientLight)

    savePixelArrayToImage(pixels: pixels, width: width, height: height, fileName: "gradient.png")
    let endTime = DispatchTime.now()
    let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
    let timeInterval = Double(nanoTime) / 1_000_000_000 // Convert to seconds
    
    print("Render completed in \(String(format: "%.2f", timeInterval)) seconds")
}

struct BoxLight {
    var center: simd_float3
    var material: Material
    var LightType: LightType
    var width: Float
    var height: Float
    var depth: Float

    // Convert to radiance for outgoing rays
    var emittedRadiance: simd_float3 {
        switch LightType {
        case .bulb(let efficacy, let watts):
            let luminousFlux = efficacy * watts // lm
            let radiantFlux = luminousFlux / 683.0 // Convert lumens to watts (assuming 555 nm peak sensitivity)
            return calculateRadiance(radiantFlux: radiantFlux)
        case .radiometic(let radiantFlux):
            return calculateRadiance(radiantFlux: radiantFlux)
        }
    }

    func calculateRadiance(radiantFlux: Float) -> simd_float3 {
        let surfaceArea = 2.0 * (width * height + width * depth + height * depth) // m²
        let radiantExitance = radiantFlux / surfaceArea // W/m²
        
        // For Lambertian emitter: radiance = exitance / π
        let radiance = radiantExitance / Float.pi // W/(sr·m²)
        
        return simd_float3(material.diffuse.x, material.diffuse.y, material.diffuse.z) * radiance
    }
}

// square light made up of two triangles
struct SquareLight {
    var center: simd_float3
    var vertices: [simd_float3]
    var material: Material
    var LightType: LightType
    var width: Float
    var depth: Float

    // Convert to radiance for outgoing rays
    var emittedRadiance: simd_float3 {
        switch LightType {
        case .bulb(let efficacy, let watts):
            let luminousFlux = efficacy * watts // lm
            let radiantFlux = luminousFlux / 683.0 // Convert lumens to watts (assuming 555 nm peak sensitivity)
            return calculateRadiance(radiantFlux: radiantFlux)
        case .radiometic(let radiantFlux):
            return calculateRadiance(radiantFlux: radiantFlux)
        }
    }

    func calculateRadiance(radiantFlux: Float) -> simd_float3 {
        let area = width * depth // m²
        let radiantExitance = radiantFlux / area // W/m²
        // For Lambertian emitter: radiance = exitance / π
        let radiance = radiantExitance / Float.pi // W/(sr·m²)
        
        return simd_float3(material.diffuse.x, material.diffuse.y, material.diffuse.z) * radiance
    }
}

enum LightType {
    case bulb(efficacy: Float, watts: Float)
    case radiometic(radiantFlux: Float) 
}

struct SphereLight {
    var center: simd_float3
    var color: simd_float4
    var LightType: LightType
    var radius: Float

    // Convert to radiance for outgoing rays
    var emittedRadiance: simd_float3 {
        switch LightType {
        case .bulb(let efficacy, let watts):
            let luminousFlux = efficacy * watts // lm
            let radiantFlux = luminousFlux / 683.0 // Convert lumens to watts (assuming 555 nm peak sensitivity)
            return calculateRadiance(radiantFlux: radiantFlux)
        case .radiometic(let radiantFlux):
            return calculateRadiance(radiantFlux: radiantFlux)
        }
    }

    func calculateRadiance(radiantFlux: Float) -> simd_float3 {
        let surfaceArea = 4.0 * Float.pi * radius * radius
        let radiantExitance = radiantFlux / surfaceArea // W/m²
        
        // For Lambertian emitter: radiance = exitance / π
        let radiance = radiantExitance / Float.pi // W/(sr·m²)
        
        return simd_float3(color.x, color.y, color.z) * radiance
    }
}

struct Material {
    var diffuse: simd_float4
    var metallic: Float
    var roughness: Float
    var emissive: simd_float3 = simd_float3(0, 0, 0)
}

struct Sphere {
    var center: simd_float3
    var material: Material
    var radius: Float
}

struct Camera {
    var position: simd_float3
    var direction: simd_float3
    var up: simd_float3 = simd_float3(0, 1, 0) // assuming camera's up vector is positive y-axis
    var resolution: simd_int2
    var horizontalFov: Float // field of view in radians
    var ev100: Float = -1.0 // lower ev100 makes light brighter, doesnt seem to impact rest of scene
    
    // func exposure() -> Float {
    //     return 1.0 / pow(2.0, ev100 * 1.2)
    // }
    
    // func generateRays() -> [Ray] {
    //     var rays: [Ray] = []
    //     let aspectRatio = Float(resolution.x / resolution.y)
    //     let halfWidth = tan(horizontalFov / 2.0)
    //     let halfHeight = halfWidth / aspectRatio
        
    //     // Create camera coordinate system
    //     let w = -simd_normalize(direction)  // Forward vector
    //     let u = simd_normalize(simd_cross(up, w))  // Right vector
    //     let v = simd_normalize(simd_cross(w, u))  // Up vector (normalized)
        
    //     for y in 0..<resolution.y {
    //         for x in 0..<resolution.x {
    //             let s = (Float(x) / Float(resolution.x)) * 2.0 - 1.0
    //             // Flip the t coordinate by negating it
    //             let t = -((Float(y) / Float(resolution.y)) * 2.0 - 1.0)
                
    //             // Calculate ray direction in camera space
    //             let dir = simd_float3(
    //                 Float(s * halfWidth) * u +
    //                 Float(t * halfHeight) * v -
    //                 w
    //             )
    //             rays.append(Ray(origin: position, direction: simd_normalize(dir)))
    //         }
    //     }
    //     return rays
    // }
}

// struct Ray {
//     var origin: simd_float3
//     var direction: simd_float3    
// }

// func intersect(ray: Ray, sphere: Sphere) -> Intersection {
//     let cent3 = simd_float3(sphere.center.x, sphere.center.y, sphere.center.z)
//     let oc = ray.origin - cent3
//     let a = simd_dot(ray.direction, ray.direction)
//     let b = 2.0 * simd_dot(oc, ray.direction)
//     let c = simd_dot(oc, oc) - Float(sphere.radius * sphere.radius)
//     let discriminant = b * b - 4 * a * c

//     if discriminant > 0 {
//         let t1 = (-b - sqrt(discriminant)) / (2.0 * a)
//         let t2 = (-b + sqrt(discriminant)) / (2.0 * a)
//         if t1 > 0 || t2 > 0 {
// //            let hitPoint = ray.origin + ray.direction * min(t1, t2)
// //            let normal = simd_normalize(hitPoint - sphere.center)
            
//                 // Offset the hit point slightly along the normal to prevent self-intersection
// //            let epsilon: Float = 1e-4
//             // let offsetHitPoint = hitPoint + normal * epsilon
//             return .hit
//             // return .hit(point: offsetHitPoint, color: sphere.material.diffuse, material: sphere.material, ray: ray, normal: simd_normalize(hitPoint - sphere.center))
//         }
//     }
//     return .miss
// }

// enum Intersection {
//     // case hit(point: simd_float3, color: simd_float3, ray: Ray, normal: simd_float3)
//     case hit
//     case hitLight(point: simd_float3, color: simd_float3, radiance: simd_float3)
//     case miss
// }

