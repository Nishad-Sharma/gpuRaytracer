//
//  main.swift
//  gpuComputeShader
//
//  Created by Nishad Sharma on 5/6/2025.
//

import Foundation
import Metal
import simd
import MetalKit

//setup scene

let direction = simd_normalize(simd_float4(0, 0, 0, 0) - simd_float4(5, 5, 0, 0))

let camera = Camera(position: simd_float4(5, 5, 0, 0), direction: direction,
    horizontalFov: Float.pi / 4.0, resolution: simd_int2(400, 300))

let sphere = Sphere(center: simd_float4(0, 0, 0, 0), radius: 1.0)

render()

func render() {
    let startTime = DispatchTime.now()

    let width = Int(camera.resolution.x)
    let height = Int(camera.resolution.y)
    let pixelCount = width * height
    // Pre-allocate pixels array with black transparent pixels
    var pixels = [UInt8](repeating: 0, count: pixelCount * 4)

    pixels = gpuIntersect(cameras: [camera], spheres: [sphere], pixels: pixels)

    // let rays = camera.generateRays()
    // for (index, ray) in rays.enumerated() {
    //     let intersection = intersect(ray: ray, sphere: sphere)
    //     switch intersection {
    //         case .hit:
    //             let color = simd_float3(1,0,0)
    //             let pixelOffset = index * 4
    //             pixels[pixelOffset + 0] = UInt8(color.x * 255)  // R
    //             pixels[pixelOffset + 1] = UInt8(color.y * 255)  // G
    //             pixels[pixelOffset + 2] = UInt8(color.z * 255)  // B
    //             pixels[pixelOffset + 3] = 255     
    //         case _:
    //             let color = simd_float3(0,0,0)
    //             let pixelOffset = index * 4
    //             pixels[pixelOffset + 0] = UInt8(color.x * 255)  // R
    //             pixels[pixelOffset + 1] = UInt8(color.y * 255)  // G
    //             pixels[pixelOffset + 2] = UInt8(color.z * 255)  // B
    //             pixels[pixelOffset + 3] = 255     
    //     }
    // }
    savePixelArrayToImage(pixels: pixels, width: width, height: height, fileName: "/Users/nishadsharma/tempShader/gradient.png")
    let endTime = DispatchTime.now()
    let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
    let timeInterval = Double(nanoTime) / 1_000_000_000 // Convert to seconds
    
    print("Render completed in \(String(format: "%.2f", timeInterval)) seconds")
}

struct Sphere {
    var center: simd_float4
    var radius: Float
    var _padding: (Float, Float, Float) = (0, 0, 0) // 12 bytes padding
}


struct Camera {
    var position: simd_float4
    var direction: simd_float4
    var horizontalFov: Float // field of view in radians
    var resolution: simd_int2
    var up: simd_float4 = simd_float4(0, 1, 0, 0) // assuming camera's up vector is positive y-axis
    var ev100: Float = 0.01
    var _padding: (Float, Float, Float) = (0, 0, 0) // 12 bytes padding
    
    func exposure() -> Float {
        return 1.0 / pow(2.0, ev100 * 1.2)
    }
    
    func generateRays() -> [Ray] {
        var rays: [Ray] = []
        let aspectRatio = Float(resolution.x / resolution.y)
        let halfWidth = tan(horizontalFov / 2.0)
        let halfHeight = halfWidth / aspectRatio
        
        // Create camera coordinate system
        let dir3 = simd_float3(direction.x, direction.y, direction.z)
        let w = -simd_normalize(dir3)  // Forward vector
        let up3 = simd_float3(up.x, up.y, up.z)
        let u = simd_normalize(simd_cross(up3, w))  // Right vector
        let v = simd_normalize(simd_cross(w, u))  // Up vector (normalized)
        
        for y in 0..<resolution.y {
            for x in 0..<resolution.x {
                let s = (Float(x) / Float(resolution.x)) * 2.0 - 1.0
                // Flip the t coordinate by negating it
                let t = -((Float(y) / Float(resolution.y)) * 2.0 - 1.0)
                
                // Calculate ray direction in camera space
                let dir = simd_float3(
                    Float(s * halfWidth) * u +
                    Float(t * halfHeight) * v -
                    w
                )
                let pos3 = simd_float3(position.x, position.y, position.z)
                rays.append(Ray(origin: pos3, direction: simd_normalize(dir)))
            }
        }
        return rays
    }
}

struct Ray {
    var origin: simd_float3
    var direction: simd_float3    
}

func intersect(ray: Ray, sphere: Sphere) -> Intersection {
    let cent3 = simd_float3(sphere.center.x, sphere.center.y, sphere.center.z)
    let oc = ray.origin - cent3
    let a = simd_dot(ray.direction, ray.direction)
    let b = 2.0 * simd_dot(oc, ray.direction)
    let c = simd_dot(oc, oc) - Float(sphere.radius * sphere.radius)
    let discriminant = b * b - 4 * a * c

    if discriminant > 0 {
        let t1 = (-b - sqrt(discriminant)) / (2.0 * a)
        let t2 = (-b + sqrt(discriminant)) / (2.0 * a)
        if t1 > 0 || t2 > 0 {
//            let hitPoint = ray.origin + ray.direction * min(t1, t2)
//            let normal = simd_normalize(hitPoint - sphere.center)
            
                // Offset the hit point slightly along the normal to prevent self-intersection
//            let epsilon: Float = 1e-4
            // let offsetHitPoint = hitPoint + normal * epsilon
            return .hit
            // return .hit(point: offsetHitPoint, color: sphere.material.diffuse, material: sphere.material, ray: ray, normal: simd_normalize(hitPoint - sphere.center))
        }
    }
    return .miss
}

enum Intersection {
    // case hit(point: simd_float3, color: simd_float3, ray: Ray, normal: simd_float3)
    case hit
    case hitLight(point: simd_float3, color: simd_float3, radiance: simd_float3)
    case miss
}

