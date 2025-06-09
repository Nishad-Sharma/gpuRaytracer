//
//  add.metal
//  gpuRaytracer
//
//  Created by Nishad Sharma on 5/6/2025.
//

#include <metal_stdlib>
using namespace metal;
#include "shaderTypes.h"

// kernel makes this a public gpu function. also defines it as a compute function/kernel which allows parallel calcuation with threads
kernel void addArrays(device const float* arr1, device const float* arr2, device float* arr3, uint index [[thread_position_in_grid]]) {
    arr3[index] = arr1[index] + arr2[index];
}

IntersectionGPU intersect(RayGPU ray, SphereGPU sphere) {
    float3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0 * a * c;

    IntersectionGPU result;
    result.type = Miss;

    if (discriminant > 0.0) {
        float t1 = (-b - sqrt(discriminant)) / (2.0 * a);
        float t2 = (-b + sqrt(discriminant)) / (2.0 * a);
        if ((t1 > 0.0) || (t2 > 0.0)) {
            float3 hitPoint = ray.origin + ray.direction * min(t1, t2);
            float3 normal = normalize(hitPoint - sphere.center);
            float epsilon = 1e-4;
            float3 offsetHitPoint = hitPoint + epsilon * normal;

            result.type = Hit;
            result.ray = ray;
            result.point = offsetHitPoint;
            result.normal = normal;
            result.diffuse = sphere.diffuse;
        }
    }
    return result;
}

IntersectionGPU getClosestIntersection(RayGPU ray, device const SphereGPU* spheres, uint sphereCount) {
    IntersectionGPU closestIntersection = IntersectionGPU();
    closestIntersection.type = Miss;
    float closestDistance = INFINITY;

    for (uint i = 0; i < sphereCount; i++) {
        IntersectionGPU result = intersect(ray, spheres[i]);
        if (result.type == Hit) {
            float distance = length(result.point - ray.origin);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestIntersection = result;
            }
        }
    }
    return closestIntersection;
}

kernel void intersect(device const CameraGPU* cameras, device const SphereGPU* spheres, constant uint& sphereCount, device uchar* pixels, uint2 index [[thread_position_in_grid]]) {
    // gen rays
    CameraGPU camera = cameras[0];
    float aspectRatio = float(camera.resolution.x / camera.resolution.y);
    float halfWidth = tan(camera.horizontalFov / 2.0);
    float halfHeight = halfWidth / aspectRatio;
    
    // Camera coord system
    float3 w = -normalize(camera.direction);
    float3 u = normalize(cross(camera.up, w));
    float3 v = normalize(cross(w, u));
    
    int x = index.x;
    int y = index.y;
    if (x >= camera.resolution.x || y >= camera.resolution.y) return;
    
    float s = (float(x) / float(camera.resolution.x)) * 2.0 - 1.0;
    float t = -((float(y) / float(camera.resolution.y)) * 2.0 - 1.0);

    float3 dir = normalize(s * halfWidth * u + t * halfHeight * v - w);
    
    RayGPU ray;
    ray.origin = camera.position;
    ray.direction = dir;

    int pixelOffset = (y * camera.resolution.x + x) * 4;

    IntersectionGPU closestIntersection = getClosestIntersection(ray, spheres, sphereCount);
    if (closestIntersection.type == Hit) {
        pixels[pixelOffset + 0] = uchar(closestIntersection.diffuse.x * 255); // R
        pixels[pixelOffset + 1] = uchar(closestIntersection.diffuse.y * 255); // G
        pixels[pixelOffset + 2] = uchar(closestIntersection.diffuse.z * 255); // B
        pixels[pixelOffset + 3] = uchar(closestIntersection.diffuse.w * 255); // A
    } else {
        pixels[pixelOffset + 0] = uchar(0.0 * 255); // R
        pixels[pixelOffset + 1] = uchar(0.0 * 255); // G
        pixels[pixelOffset + 2] = uchar(0.0 * 255); // B
        pixels[pixelOffset + 3] = 255;              // A
    }
}
