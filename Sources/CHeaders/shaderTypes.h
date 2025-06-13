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
    simd_float4 diffuse; 
    float metallic;     
    float roughness;      
} MaterialGPU;

typedef struct {
    simd_float3 vertices[3];
    MaterialGPU material;
} TriangleGPU;

typedef struct {
    simd_float3 center;
    MaterialGPU material;
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
    simd_float3 center;
    simd_float4 color;
    simd_float3 emittedRadiance;
    float radius;
} SphereLightGPU;

typedef struct {
    simd_float3 origin;
    simd_float3 direction;
} RayGPU;

typedef enum {
    Hit,
    HitLight,
    Miss
} IntersectionTypeGPU;

typedef struct {
    IntersectionTypeGPU type;
    simd_float3 point;
    RayGPU ray;
    simd_float3 normal;
    MaterialGPU material;
    simd_float3 radiance; // this is just for light - find better way to do intersections!
} IntersectionGPU;

typedef struct {
    simd_float3 direction;
    float pdf;
    simd_float3 radiance;
} SampleResultGPU;

typedef struct {
    simd_float3 tangent;
    simd_float3 bitangent;
} OrthonormalBasisGPU;

#endif /* ShaderTypes_h */
