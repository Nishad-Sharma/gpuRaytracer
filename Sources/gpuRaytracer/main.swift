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

let direction = simd_normalize(simd_float3(0, 0, 0) - simd_float3(5, 0, 0))
let camera = Camera(position: simd_float3(5, 0, 0), direction: direction,
    resolution: simd_int2(400, 300), horizontalFov: Float.pi / 4.0)

let triangle = Triangle(
    vertices: [
        simd_float3(0, -1, -1), // bottom left
        simd_float3(0,  1, -1), // top left
        simd_float3(0,  0,  1)  // right
    ],  
    material: Material(diffuse: simd_float4(1, 0, 0, 1), metallic: 0.0, roughness: 0.5)
)

let width = Int(camera.resolution.x)
let height = Int(camera.resolution.y)
let pixelCount = width * height
// Pre-allocate pixels array with black transparent pixels
var pixels = [UInt8](repeating: 0, count: pixelCount * 4)

// Create a Metal device
let device = MTLCreateSystemDefaultDevice()!
// Check if ray tracing is supported
guard device.supportsRaytracing else {
    print("Ray tracing not supported on this device")
    exit(1)
}
// make accel structures
let accelerationStructures = setupAccelerationStructures(device: device, triangles: [triangle])
// dont need render pipeline since we aren't rendering to screen - send to compute pipe
pixels = drawTriangle(device: device, cameras: [camera], pixels: pixels, accelerationStructure: accelerationStructures)

savePixelArrayToImage(pixels: pixels, width: width, height: height, fileName: "triangle.png")

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
    var lights = [lightBulb]
    var ambientLight = simd_float4(0.53, 0.81, 0.92, 1) 

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
    var ev100: Float = 0.5 // lower ev100 makes light brighter, doesnt seem to impact rest of scene
    
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

