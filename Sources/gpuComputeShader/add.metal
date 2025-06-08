//
//  add.metal
//  gpuComputeShader
//
//  Created by Nishad Sharma on 5/6/2025.
//

// kernel makes this a public gpu function. also defines it as a compute function/kernel which allows parallel calcuation with threads
kernel void addArrays(device const float* arr1, device const float* arr2, device float* arr3, uint index [[thread_position_in_grid]]) {
    arr3[index] = arr1[index] + arr2[index];
}

// kernel void intersect(device const Camera* camera, device const Sphere* spheres, device UInt8* pixels uint2 index [[thread_position_in_grid]]) {

// }

#include <metal_stdlib>
using namespace metal;
#include "shaderTypes.h"

Intersection intersect(Ray ray, Sphere sphere) {
    float3 sphereCenter3 = float3(sphere.center.x, sphere.center.y, sphere.center.z);
    float3 oc = ray.origin - sphereCenter3;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant > 0.0) {
        return Hit;
    }
    return Miss;
}

kernel void intersect(device const Camera* cameras, device const Sphere* spheres, device uchar* pixels, uint2 index [[thread_position_in_grid]]) {
    // gen rays
    Camera camera = cameras[0];
    float aspectRatio = float(camera.resolution.x / camera.resolution.y);
    float halfWidth = tan(camera.horizontalFov / 2.0);
    float halfHeight = halfWidth / aspectRatio;
    
    // Camera coord system
    float3 up3 = float3(camera.up.x, camera.up.y, camera.up.z);
    float3 dir3 = float3(camera.direction.x, camera.direction.y, camera.direction.z);
    float3 w = -normalize(dir3);
    float3 u = normalize(cross(up3, w));
    float3 v = normalize(cross(w, u));
    
    int x = index.x;
    int y = index.y;
    if (x >= camera.resolution.x || y >= camera.resolution.y) return;
    
    float s = (float(x) / float(camera.resolution.x)) * 2.0 - 1.0;
    float t = -((float(y) / float(camera.resolution.y)) * 2.0 - 1.0);

    float3 dir = normalize(s * halfWidth * u + t * halfHeight * v - w);
    
    float3 pos3 = float3(camera.position.x, camera.position.y, camera.position.z);
    Ray ray;
    ray.origin = pos3;
    ray.direction = dir;

    Sphere sphere = spheres[0];
    int pixelOffset = (y * camera.resolution.x + x) * 4;

    if (intersect(ray, sphere) == Hit) {
        pixels[pixelOffset + 0] = uchar(1.0 * 255); // R
        pixels[pixelOffset + 1] = uchar(0.0 * 255); // G
        pixels[pixelOffset + 2] = uchar(0.0 * 255); // B
        pixels[pixelOffset + 3] = 255;              // A
    } else {
        pixels[pixelOffset + 0] = uchar(0.0 * 255); // R
        pixels[pixelOffset + 1] = uchar(0.0 * 255); // G
        pixels[pixelOffset + 2] = uchar(0.0 * 255); // B
        pixels[pixelOffset + 3] = 255;              // A
    }
}
