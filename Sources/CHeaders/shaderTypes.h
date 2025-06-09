//
//  shaderTypes.h
//  gpuRaytracer
//
//  Created by Nishad Sharma on 7/6/2025.
//

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

typedef struct {
    simd_float3 center;
    simd_float4 diffuse;
    float radius;
} SphereGPU;

typedef struct {
    simd_float3 position;
    simd_float3 direction;
    simd_float3 up;
    simd_int2 resolution;
    float horizontalFov;
    float ev100;
} CameraGPU;

typedef struct {
    simd_float3 origin;
    simd_float3 direction;
} RayGPU;

typedef enum {
    Hit,
    Miss
} IntersectionTypeGPU;

typedef struct {
    IntersectionTypeGPU type;
    simd_float3 point;
    RayGPU ray;
    simd_float3 normal;
    simd_float4 diffuse;
} IntersectionGPU;

#endif /* ShaderTypes_h */
