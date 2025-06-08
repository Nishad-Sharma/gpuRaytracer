//
//  shaderTypes.h
//  gpuComputeShader
//
//  Created by Nishad Sharma on 7/6/2025.
//

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

typedef struct {
    simd_float4 center;
    float radius;
    simd_float4 diffuse;
    float _padding[3];
} Sphere;

typedef struct {
    simd_float4 position;
    simd_float4 direction;
    float horizontalFov;
    simd_int2 resolution;
    simd_float4 up;
    float ev100;
    float _padding[3];
} Camera;

typedef struct {
    simd_float3 origin;
    simd_float3 direction;
} Ray;

typedef enum {
    Hit,
    Miss
} IntersectionType;

typedef struct {
    IntersectionType type;
    simd_float3 point;
    Ray ray;
    simd_float3 normal;
    simd_float4 diffuse;
} Intersection;

#endif /* ShaderTypes_h */
