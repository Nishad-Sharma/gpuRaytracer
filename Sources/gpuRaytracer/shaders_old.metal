//
//  add.metal
//  gpuRaytracer
//
//  Created by Nishad Sharma on 5/6/2025.
//

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
#include "shaderTypes.h"

// kernel makes this a public gpu function. also defines it as a compute function/kernel which allows parallel calcuation with threads
kernel void addArrays(device const float* arr1, device const float* arr2, device float* arr3, uint index [[thread_position_in_grid]]) {
    arr3[index] = arr1[index] + arr2[index];
}

// cast shadow ray from point towards direction to see if it hits light source
float3 traceTriangleLightRay(float3 origin, float3 direction, 
                       primitive_acceleration_structure accelerationStructure,
                       device const MaterialGPU* materials) {
    ray r;
    r.origin = origin + direction * 1e-4; // Small offset to avoid self-intersection
    r.direction = direction;
    r.min_distance = 0.001f;
    r.max_distance = 1000.0f;
    
    intersection_params params;
    intersection_query<triangle_data> i;
    params.assume_geometry_type(geometry_type::triangle);
    params.force_opacity(forced_opacity::opaque);
    params.accept_any_intersection(false);
    
    i.reset(r, accelerationStructure, params);
    i.next();
    
    intersector<triangle_data>::result_type intersection;
    intersection.type = i.get_committed_intersection_type();
    intersection.primitive_id = i.get_committed_primitive_id();
    
    if (intersection.type == intersection_type::triangle) {
        MaterialGPU material = materials[intersection.primitive_id];
        // Check if we hit a light (emissive material)
        if (length(material.emissive) > 0.0) {
            return material.emissive; // Hit a light triangle
        }
    }
    // TODO: fix return type??
    return float3(-1.0, -1.0, -1.0); // Miss or hit non-emissive surface
}

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float randomFloat(uint seed) {
    return float(hash(seed)) / (float(0xffffffffU) + 1.0);
}

// Radical inverse function for base 2 (Van der Corput sequence)
float radicalInverse2(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Hammersley point generation (2D low-discrepancy sequence)
// good for 2d sampling e.g. area light, texture sampling,
// screen-space sampling, any correlated 2d points
float2 hammersley2D(uint i, uint N) {
    return float2(float(i) / float(N), radicalInverse2(i));
}

// Hammersley-based random float (much better distribution)
// better for 3d/4d sampling, independant random numbers
float hammersleyFloat(uint index, uint dimension, uint totalSamples) {
    if (dimension == 0) {
        return float(index) / float(totalSamples);
    } else if (dimension == 1) {
        return radicalInverse2(index);
    } else {
        // For higher dimensions, use scrambled radical inverse
        uint scrambledIndex = hash(index + dimension * 12345);
        return radicalInverse2(scrambledIndex);
    }
}

// Power heuristic for MIS
float powerHeuristic(float pdf1, float pdf2, float pdf3, float beta) {
    float p1 = pow(pdf1, beta);
    float sum = p1 + pow(pdf2, beta) + pow(pdf3, beta);
    return p1 / (sum + 1e-6);  // epsilon to avoid division by zero
}

float powHeuristic(float pdf1, float pdf2, float beta) {
    return pow(pdf1, beta) / (pow(pdf1, beta) + pow(pdf2, beta) + 1e-6); // epsilon to avoid division by zero
}

IntersectionGPU intersectSphere(RayGPU ray, SphereGPU sphere) {
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
            result.material = sphere.material;
            result.radiance = float3(0.0, 0.0, 0.0);
        }
    }
    return result;
}

IntersectionGPU intersectLight(RayGPU ray, SphereLightGPU light) {
    float3 oc = ray.origin - light.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(oc, ray.direction);
    float c = dot(oc, oc) - light.radius * light.radius;
    float discriminant = b * b - 4.0 * a * c;

    IntersectionGPU result;
    result.type = Miss;

    if (discriminant > 0.0) {
        float t1 = (-b - sqrt(discriminant)) / (2.0 * a);
        float t2 = (-b + sqrt(discriminant)) / (2.0 * a);
        if ((t1 > 0.0) || (t2 > 0.0)) {
            float3 hitPoint = ray.origin + ray.direction * min(t1, t2);
            float3 normal = normalize(hitPoint - light.center);
            float epsilon = 1e-4;
            float3 offsetHitPoint = hitPoint + epsilon * normal;

            result.type = HitLight;
            result.ray = ray;
            result.point = offsetHitPoint;
            result.normal = normal;
            // MaterialGPU material;
            // material.diffuse = light.color;
            // material.roughness = 0.0; // Lights are not rough
            // material.specular = float3(0.0, 0.0, 0.0); // Lights do not have specular component
            // result.material = material;
            result.radiance = light.emittedRadiance;
        }
    }
    return result;
}

IntersectionGPU getClosestIntersection(RayGPU ray, device const SphereGPU* spheres, uint sphereCount, 
device const SphereLightGPU* lights, uint lightCount) {
    IntersectionGPU closestIntersection;
    closestIntersection.type = Miss;
    float closestDistance = INFINITY;

    for (uint i = 0; i < lightCount; i++) {
        IntersectionGPU result = intersectLight(ray, lights[i]);
        if (result.type == HitLight) {
            float distance = length(result.point - ray.origin);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestIntersection = result;
            }
        }
    }

    for (uint i = 0; i < sphereCount; i++) {
        IntersectionGPU result = intersectSphere(ray, spheres[i]);
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

// TODO
float cameraExposure(CameraGPU camera) {
    return 1.0 / pow(2.0, camera.ev100 * 1.2);
    // return 1.0 / (78.0 * pow(2.0, camera.ev100));
}

float4 reinhartToneMapping(float3 color) {
    float3 finalColor = color / (color + float3(1.0, 1.0, 1.0));
    finalColor = clamp(pow(finalColor, float3(1.0 / 2.2)), 0.0, 1.0);
    return float4(finalColor, 1.0);
}

OrthonormalBasisGPU buildOrthonormalBasis(float3 normal) {
    float3 tangent;
    if (abs(normal.x) > 0.9) {
        tangent = normalize(float3(0, 1, 0) - dot(float3(0, 1, 0), normal) * normal);
    } else {
        tangent = normalize(float3(1, 0, 0) - dot(float3(1, 0, 0), normal) * normal);
    }
    float3 bitangent = cross(normal, tangent);
    
    OrthonormalBasisGPU basis;
    basis.tangent = tangent;
    basis.bitangent = bitangent;
    return basis;
}

uint nextPowerOfTwo(uint n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

SampleResultGPU sampleSquareLight(SquareLightGPU light, float3 point, uint2 index, uint sampleIndex, uint samples, uint i) {
    // Build orthonormal basis for the rectangle's plane
    // float3 normal = light.normal; // hard code for now?
    float3 normal = float3(0, -1, 0); // TODO: hard code for now?
    OrthonormalBasisGPU basis = buildOrthonormalBasis(normal);

    // Rectangle corners in local space
    float3 center = light.center;

    uint pow2 = nextPowerOfTwo(samples);
    float2 u = hammersley2D(i, pow2);

    // Uniformly sample a point on the rectangle
    float x = (u.x - 0.5) * light.width;
    float y = (u.y - 0.5) * light.depth;
    float3 samplePos = center + basis.tangent * x + basis.bitangent * y;

    // Direction from shading point to light sample
    float3 toLight = samplePos - point;
    float distance = length(toLight);
    float3 lightDir = toLight / distance;

    // PDF for area light: (distance^2) / (area * cosTheta)
    float area = light.width * light.depth;
    float cosTheta = max(0.0, dot(-lightDir, normal));
    float pdf = (distance * distance) / (area * cosTheta + 1e-6);

    SampleResultGPU result;
    result.direction = lightDir;
    result.pdf = pdf;
    result.radiance = light.emittedRadiance;
    return result;
}

float calculateSquareLightPdf(SquareLightGPU light, float3 point, float3 direction) {
    // Calculate the distance from the point to the light center
    float3 lightNormal = float3(0, -1, 0); // TODO: either store or gen normal properly
    float3 toLight = light.center - point;
    float distanceToLight = length(toLight);
    
    // Calculate the cosine of the angle between the direction and the light normal
    float cosTheta = max(0.0, dot(-direction, lightNormal));
    
    // Area of the square light
    float area = light.width * light.depth;
    
    // PDF for area light: (distance^2) / (area * cosTheta)
    return (distanceToLight * distanceToLight) / (area * cosTheta + 1e-6);
}

// doesnt sample visible faces only but apparently this is fine since we do shadow ray test later
SampleResultGPU sampleBoxLight(BoxLightGPU light, float3 point, uint2 index, uint sampleIndex, uint samples, uint i) {
    uint pow2 = nextPowerOfTwo(samples); // find smallest pow2 greater than number
    // uint scrambledId = sampleIndex ^ hash(index.x + index.y * 800);
    // hammersly2D better results with first argument value between 0 to (sampleCount per pixel - 1)
    // so we pass i instead of sampleIndex or scramledID
    float2 u = hammersley2D(i, pow2);
    float u1 = u.x;
    float u2 = u.y;
    // float u3 = hammersleyFloat(sampleIndex, 2, pow2); 
    // also expectting value between 0 - (sampleCount per pixel - 1)
    float u3 = hammersleyFloat(i, 2, pow2); 

    // Box light dimensions
    float halfWidth = light.width * 0.5;
    float halfHeight = light.height * 0.5;
    float halfDepth = light.depth * 0.5;
    
    // Calculate surface areas for each face
    float areaXY = light.width * light.height;  // Front/Back faces
    float areaXZ = light.width * light.depth;   // Top/Bottom faces
    float areaYZ = light.height * light.depth;  // Left/Right faces
    float totalArea = 2.0 * (areaXY + areaXZ + areaYZ);
    
    // Normalized cumulative areas for face selection
    float prob1 = (2.0 * areaXY) / totalArea;           // Front + Back
    float prob2 = prob1 + (2.0 * areaXZ) / totalArea;   // + Top + Bottom
    // prob3 = 1.0 (Left + Right faces)
    
    float3 randomPoint;
    float3 faceNormal;
    float faceArea;
    
    if (u3 < prob1) {
        // Sample front or back face (XY planes)
        if (u3 < prob1 * 0.5) {
            // Front face (positive Z)
            randomPoint = light.center + float3(
                (u1 - 0.5) * light.width,
                (u2 - 0.5) * light.height,
                halfDepth
            );
            faceNormal = float3(0, 0, 1);
        } else {
            // Back face (negative Z)
            randomPoint = light.center + float3(
                (u1 - 0.5) * light.width,
                (u2 - 0.5) * light.height,
                -halfDepth
            );
            faceNormal = float3(0, 0, -1);
        }
        faceArea = areaXY;
    } else if (u3 < prob2) {
        // Sample top or bottom face (XZ planes)
        float adjustedU3 = (u3 - prob1) / (prob2 - prob1);
        if (adjustedU3 < 0.5) {
            // Top face (positive Y)
            randomPoint = light.center + float3(
                (u1 - 0.5) * light.width,
                halfHeight,
                (u2 - 0.5) * light.depth
            );
            faceNormal = float3(0, 1, 0);
        } else {
            // Bottom face (negative Y)
            randomPoint = light.center + float3(
                (u1 - 0.5) * light.width,
                -halfHeight,
                (u2 - 0.5) * light.depth
            );
            faceNormal = float3(0, -1, 0);
        }
        faceArea = areaXZ;
    } else {
        // Sample left or right face (YZ planes)
        float adjustedU3 = (u3 - prob2) / (1.0 - prob2);
        if (adjustedU3 < 0.5) {
            // Right face (positive X)
            randomPoint = light.center + float3(
                halfWidth,
                (u1 - 0.5) * light.height,
                (u2 - 0.5) * light.depth
            );
            faceNormal = float3(1, 0, 0);
        } else {
            // Left face (negative X)
            randomPoint = light.center + float3(
                -halfWidth,
                (u1 - 0.5) * light.height,
                (u2 - 0.5) * light.depth
            );
            faceNormal = float3(-1, 0, 0);
        }
        faceArea = areaYZ;
    }
    
    // Calculate direction from surface point to light sample point
    float3 toLight = randomPoint - point;
    float distanceToLight = length(toLight);
    float3 lightDir = toLight / distanceToLight;
    
    // Calculate PDF based on solid angle
    // For area lights: PDF = (distance^2) / (area * cos(theta_light))
    float cosTheta = max(0.0, dot(-lightDir, faceNormal));
    float pdf = (distanceToLight * distanceToLight) / (totalArea * cosTheta + 1e-6);
    
    SampleResultGPU result;
    result.direction = lightDir;
    result.pdf = pdf;
    result.radiance = light.emittedRadiance;
    
    return result;
}

SampleResultGPU sampleSphereLight(SphereLightGPU light, float3 point, uint2 index, uint sampleIndex) {
    // Calculate vectors to light center
    float3 toLight = light.center - point;
    float distanceToLight = length(toLight);
    float3 lightDir = toLight / distanceToLight;
    
    // Calculate visible solid angle of the light
    float sinThetaMax = min(light.radius / distanceToLight, 1.0);
    float cosThetaMax = sqrt(1.0 - sinThetaMax * sinThetaMax);
    
    // Sample visible cone
    uint seed1 = hash(index.x + index.y * 1920 + sampleIndex * 3840);
    uint seed2 = hash(index.y + index.x * 1080 + sampleIndex * 7680 + 12345);

    float u1 = randomFloat(seed1);
    float u2 = randomFloat(seed2);
    
    // Convert uniform random samples to cone coordinates
    float cosTheta = 1.0 - u1 * (1.0 - cosThetaMax);  // Sample visible portion only
    // float cosTheta = (1.0 - u1) + u1 * cosThetaMax; // sample uniformly on spherical cap
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float phi = 2.0 * M_PI_F * u2;

    OrthonormalBasisGPU basis = buildOrthonormalBasis(lightDir);

    // TODO: find which is right??
    // float3 sampleDir = normalize(
    //     cos(phi) * basis.tangent * sinTheta + 
    //     sin(phi) * basis.bitangent * sinTheta + 
    //     cosTheta * lightDir
    // );
    float3 sampleDir = normalize(
        basis.tangent * (cos(phi) * sinTheta) +
        basis.bitangent * (sin(phi) * sinTheta) +
        lightDir * cosTheta
    );

    float pdf = 1.0 / (2.0 * M_PI_F * (1.0 - cosThetaMax));

    SampleResultGPU result;
    result.direction = sampleDir;
    result.pdf = pdf;
    result.radiance = light.emittedRadiance;

    return result;
}

// uniform hemisphere sampling
SampleResultGPU sampleUniformHemisphere(float3 normal, uint2 index, uint sampleIndex, uint samples, uint i) {
    uint pow2 = nextPowerOfTwo(samples);
    float2 u = hammersley2D(i, pow2);
    float u1 = u.x;
    float u2 = u.y;
    
    // Uniform hemisphere sampling
    float phi = 2.0 * M_PI_F * u1;
    float cosTheta = u2; // Uniform in cosine
    float sinTheta = sqrt(1.0 - u2 * u2);
    
    OrthonormalBasisGPU basis = buildOrthonormalBasis(normal);
    float3 direction = normalize(
        basis.tangent * (cos(phi) * sinTheta) +
        basis.bitangent * (sin(phi) * sinTheta) +
        normal * cosTheta
    );
    
    // PDF for uniform hemisphere sampling
    float pdf = 1.0 / (2.0 * M_PI_F);
    
    SampleResultGPU result;
    result.direction = direction;
    result.pdf = pdf;
    result.radiance = float3(0.0, 0.0, 0.0);
    
    return result;
}

// Cosine-weighted hemisphere sampling
SampleResultGPU sampleCosineWeighted(float3 normal, uint2 index, uint sampleIndex, uint samples, uint i) {
    uint pow2 = nextPowerOfTwo(samples); // find smallest pow2 greater than number
    float2 u = hammersley2D(i, pow2); // Use Hammersley sequence for better distribution
    float u1 = u.x;
    float u2 = u.y;
    
    float phi = 2.0 * M_PI_F * u1;
    float cosTheta = sqrt(u2);
    float sinTheta = sqrt(1.0 - u2);
    
    OrthonormalBasisGPU basis = buildOrthonormalBasis(normal);
    float3 direction = normalize(
        basis.tangent * (cos(phi) * sinTheta) +
        basis.bitangent * (sin(phi) * sinTheta) +
        normal * cosTheta
    );
    
    // PDF for cosine-weighted sampling
    float pdf = cosTheta / M_PI_F;
    
    SampleResultGPU result;
    result.direction = direction;
    result.pdf = pdf;
    result.radiance = float3(0.0, 0.0, 0.0);
    
    return result;
}

float smithG1_GGX(float NoV, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NoV2 = NoV * NoV;
    return 2.0 / (1.0 + sqrt(1.0 + a2 * (1.0 - NoV2) / NoV2));
}

float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float f = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (M_PI_F * f * f);
}

// TODO: replace with just brdf
// VNDF sampling
SampleResultGPU sampleVNDF(float3 viewDir, float3 normal, float roughness, uint2 index, uint sampleIndex, 
uint samples, uint i) {
    float alpha = roughness * roughness;
    
    // Step 1: Transform view direction to local space and stretch it
    OrthonormalBasisGPU basis = buildOrthonormalBasis(normal);
    float3 Ve = normalize(float3(
        alpha * dot(viewDir, basis.tangent),
        alpha * dot(viewDir, basis.bitangent),
        dot(viewDir, normal)
    ));
    
    // Step 2: Build orthonormal basis aligned with Ve
    float3 T1 = normalize(float3(Ve.z, 0, -Ve.x));
    float3 T2 = cross(Ve, T1);
    
    // // Step 3: Sample point with polar coordinates
    // uint seed1 = hash(index.x + index.y * 1920 + sampleIndex * 3840);
    // uint seed2 = hash(index.y + index.x * 1080 + sampleIndex * 7680 + 12345);
    
    // float u1 = randomFloat(seed1);
    // float u2 = randomFloat(seed2);
    uint pow2 = nextPowerOfTwo(samples); // find smallest pow2 greater than number
    // float u1 = hammersleyFloat(sampleIndex, 0, samples);
    // float u2 = hammersleyFloat(sampleIndex, 1, samples);
    
    // float2 u = hammersley2D(i, samples); // also works, might be better // probably faster
    // might be best in future to hard code this value based on sampleSize
    float2 u = hammersley2D(i, pow2); // where i is your per-pixel sample index in [0, samples-1]
    float u1 = u.x;
    float u2 = u.y;

    float phi = 2.0 * M_PI_F * u1;
    
    // Transform Ve to the hemisphere configuration
    float lenVe = length(Ve);
    float cosThetaMax = lenVe / sqrt(1.0 + lenVe * lenVe);
    float cosTheta = cosThetaMax + (1.0 - cosThetaMax) * u2;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    // Step 4: Compute normalized half-vector in stretched space
    float3 h = normalize(
        T1 * (cos(phi) * sinTheta) +
        T2 * (sin(phi) * sinTheta) +
        Ve * cosTheta
    );
    
    // Step 5: Unstretch
    float3 Nh = normalize(float3(
        alpha * h.x,
        alpha * h.y,
        max(0.0, h.z)
    ));
    
    // Step 6: Transform half-vector to world space
    float3 worldH = normalize(
        basis.tangent * Nh.x +
        basis.bitangent * Nh.y +
        normal * Nh.z
    );
    
    // Step 7: Reflect view direction around half-vector
    float3 direction = reflect(-viewDir, worldH);
    
    // Calculate PDF
    float NoV = abs(dot(normal, viewDir));
    float NoH = abs(dot(normal, worldH));
    float VoH = abs(dot(viewDir, worldH));
    float D = D_GGX(NoH, roughness);
    float G1 = smithG1_GGX(NoV, roughness);
    float pdf = (D * G1 * VoH) / (4.0 * NoV);
    
    SampleResultGPU result;
    result.direction = direction;
    result.pdf = pdf;
    result.radiance = float3(0.0, 0.0, 0.0);
    
    return result;
}

float calculateVNDFPdf(float3 viewDir, float3 normal, float3 lightDir, float roughness) {
    float3 h = normalize(viewDir + lightDir);
    float NoH = abs(dot(normal, h));
    float VoH = abs(dot(viewDir, h));
    float NoV = abs(dot(normal, viewDir));
    float D = D_GGX(NoH, roughness);
    float G1 = smithG1_GGX(NoV, roughness);
    return (D * G1 * VoH) / (4.0 * NoV); // This should match your sampleVNDF PDF
}

float calculateLightPdf(SphereLightGPU light, float3 point, float3 direction) {
    float3 toLight = light.center - point;
    float distanceToLight = length(toLight);
    float sinThetaMax = min(light.radius / distanceToLight, 1.0);
    float cosThetaMax = sqrt(1.0 - sinThetaMax * sinThetaMax);
    return 1.0 / (2.0 * M_PI_F * (1.0 - cosThetaMax));
}

float calculateBoxLightPdf(BoxLightGPU light, float3 point, float3 direction) {
    float3 lightCenter = light.center;
    float halfWidth = light.width * 0.5;
    float halfHeight = light.height * 0.5;
    float halfDepth = light.depth * 0.5;
    
    // Box bounds
    float3 boxMin = lightCenter - float3(halfWidth, halfHeight, halfDepth);
    float3 boxMax = lightCenter + float3(halfWidth, halfHeight, halfDepth);
    
    // Ray-box intersection
    float3 invDir = float3(
        abs(direction.x) > 1e-8 ? 1.0 / direction.x : 1e8,
        abs(direction.y) > 1e-8 ? 1.0 / direction.y : 1e8,
        abs(direction.z) > 1e-8 ? 1.0 / direction.z : 1e8
    );
    float3 t1 = (boxMin - point) * invDir;
    float3 t2 = (boxMax - point) * invDir;
    
    float3 tMin = min(t1, t2);
    float3 tMax = max(t1, t2);
    
    float tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar = min(min(tMax.x, tMax.y), tMax.z);
    
    if (tNear <= tFar && tFar > 0.0) {
        float t = (tNear > 0.0) ? tNear : tFar;
        
        if (t > 0.0) {
            float3 hitPoint = point + direction * t;
            
            // Determine which face was hit by checking which coordinate is at the boundary
            float3 faceNormal;
            if (abs(hitPoint.x - boxMin.x) < 1e-5) faceNormal = float3(-1, 0, 0);
            else if (abs(hitPoint.x - boxMax.x) < 1e-5) faceNormal = float3(1, 0, 0);
            else if (abs(hitPoint.y - boxMin.y) < 1e-5) faceNormal = float3(0, -1, 0);
            else if (abs(hitPoint.y - boxMax.y) < 1e-5) faceNormal = float3(0, 1, 0);
            else if (abs(hitPoint.z - boxMin.z) < 1e-5) faceNormal = float3(0, 0, -1);
            else faceNormal = float3(0, 0, 1);
            
            float cosTheta = abs(dot(-direction, faceNormal));
            float distanceSquared = t * t;
            
            // Total surface area of the box
            float totalArea = 2.0 * (light.width * light.height + light.width * light.depth + light.height * light.depth);
            
            return distanceSquared / (totalArea * cosTheta + 1e-6);
        }
    }
    
    return 0.0;
}

float3 traceLightRay(float3 origin, float3 direction, device const SphereGPU* spheres, 
constant uint& sphereCount, device const SphereLightGPU* lights, constant uint& lightCount) {
    RayGPU shadowRay;
    shadowRay.origin = origin + direction * 1e-4;
    shadowRay.direction = direction;
    IntersectionGPU intersection = getClosestIntersection(shadowRay, spheres, sphereCount, lights, lightCount);

    if (intersection.type == HitLight) {
        return intersection.radiance; 
    }
    return float3(-1.0, -1.0, -1.0);
}

float3 F_Schlick(float LoH, float3 f0) {
    return f0 + (float3(1.0, 1.0, 1.0) - f0) * pow(1.0 - LoH, 5.0);
}

float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float GGXL = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
    float GGXV = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
    return 0.5 / (GGXV + GGXL);
}

float Fd_Lambert() {
    return 1.0 / M_PI_F;
}

float3 calculateBRDFContribution(RayGPU ray, float3 point, float3 normal, MaterialGPU material, float3 lightDir, float3 lightValue) {
    float3 v = -normalize(ray.direction); // surface to view direction 
    float3 n = normal; // surface normal
    float3 l = lightDir; // light direction normalized already
    float3 h = normalize(v + l); // half vector between view and light direction
    
    float NoV = abs(dot(n, v)) + 1e-5;  // visibility used for fresnel + shadow
    float NoL = min(1.0, max(0.0, dot(n, l))); // shadowing + light attenuation
    float NoH = min(1.0, max(0.0, dot(n, h)));  // used for microfacet distribution
    float LoH = min(1.0, max(0.0, dot(l, h)));  // used for fresnel

    float3 dielectricF0 = float3(0.04); // Default F0 for dielectric materials
    float3 f0 = mix(dielectricF0, material.diffuse.rgb, material.metallic);

    float D = D_GGX(NoH, material.roughness);
    float3 F = F_Schlick(LoH, f0);
    float G = V_SmithGGXCorrelated(NoV, NoL, material.roughness);

    float3 Fr = (D * G) * F / (4.0 * NoV * NoL + 1e-7);

    // Diffuse BRDF
    float3 energyCompensation = float3(1.0) - F;  // Amount of light not reflected
    float3 Fd = material.diffuse.rgb * Fd_Lambert() * energyCompensation * (1.0 - material.metallic);
    
    // Combine both terms and apply light properties
    float3 BRDF = (Fd + Fr);
    
    float3 finalColor = BRDF * lightValue * NoL;

    return finalColor;
}

float3 calculateLighting(IntersectionGPU incomingIntersection, uint samples, RayGPU ray, uint2 index, SquareLightGPU light, 
primitive_acceleration_structure accelerationStructure, device const MaterialGPU* materials, device const float3* vertices, intersection_params params, intersection_query<triangle_data> intersectionQuery, uint maxBounces, uint remainingBounces) {
    if (remainingBounces == 0) {
        return float3(0.0, 0.0, 0.0); // No more bounces left
    } else {
        // only one light atm
        float3 totalLight = float3(0.0, 0.0, 0.0);
        float3 bounceLight = float3(0.0, 0.0, 0.0);
        uint samplesPerStrategy = samples / 3;
        // float scaleFactor = 3.0; // Scale factor to balance contributions from different strategies
        float beta = 2.0; // Power heuristic exponent
        uint imageWidth = 800; 
        for (uint i = 0; i < samplesPerStrategy; i++) {
            // Sample box light
            uint sampleId = (index.y * imageWidth + index.x) * (samplesPerStrategy) + i;
            // SampleResultGPU lightSample = sampleBoxLight(light, intersection.point, index, sampleId, samplesPerStrategy, i);
            SampleResultGPU lightSample = sampleSquareLight(light, incomingIntersection.point, index, sampleId, samplesPerStrategy, i);

            // Calculate PDFs for other strategies
            float cosineWeightedPdf = max(dot(incomingIntersection.normal, lightSample.direction), 0.0) / M_PI_F;
            float vndfPdf = calculateVNDFPdf(-ray.direction, incomingIntersection.normal, lightSample.direction, 
            incomingIntersection.material.roughness);

            float3 radiance = traceTriangleLightRay(incomingIntersection.point, lightSample.direction, accelerationStructure, materials);
            if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
                float weight = powerHeuristic(lightSample.pdf, cosineWeightedPdf, vndfPdf, beta);
                float3 brdf = calculateBRDFContribution(ray, incomingIntersection.point, incomingIntersection.normal,
                incomingIntersection.material, lightSample.direction, radiance);
                totalLight += brdf * weight / lightSample.pdf;
            }
        }
        for (uint i = 0; i < samplesPerStrategy; i++) {
            // Sample cosine-weighted hemisphere
            uint sampleId = (index.y * imageWidth + index.x) * (samplesPerStrategy) + i;
            SampleResultGPU cosineSample = sampleCosineWeighted(incomingIntersection.normal, index, sampleId, 
            samplesPerStrategy, i + samplesPerStrategy); // use i+ samplesPerStrategy to avoid correlation between two strategies
            
            // Calculate PDFs for other strategies
            // float lightPdf = calculateBoxLightPdf(light, intersection.point, cosineSample.direction);
            float lightPdf = calculateSquareLightPdf(light, incomingIntersection.point, cosineSample.direction);
            float vndfPdf = calculateVNDFPdf(-ray.direction, incomingIntersection.normal, cosineSample.direction, 
            incomingIntersection.material.roughness);

            float3 radiance = traceTriangleLightRay(incomingIntersection.point, cosineSample.direction, accelerationStructure, materials);
            if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
                float weight = powerHeuristic(cosineSample.pdf, lightPdf, vndfPdf, beta);
                float3 brdf = calculateBRDFContribution(ray, incomingIntersection.point, incomingIntersection.normal,
                incomingIntersection.material, cosineSample.direction, radiance);
                totalLight += brdf * weight / cosineSample.pdf;
            }
            if (remainingBounces > 1) {
                // accumulate bounce light
                struct ray newRay;
                newRay.origin = incomingIntersection.point + incomingIntersection.normal * 1e-4;
                newRay.direction = cosineSample.direction;
                newRay.min_distance = 0.001f;
                newRay.max_distance = 1000.0f;

                intersectionQuery.reset(newRay, accelerationStructure, params);
                intersectionQuery.next();

                // pull required info about the committed intersection.
                intersector<triangle_data>::result_type intersection;
                intersection.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
                intersection.distance = intersectionQuery.get_committed_distance();
                intersection.primitive_id = intersectionQuery.get_committed_primitive_id();
                MaterialGPU material = materials[intersection.primitive_id];

                if (intersection.type != intersection_type::triangle) {
                    // bounceLight += float3(0.0, 0.0, 0.0); // Miss
                    continue;
                } 
                else if (length(material.emissive) > 0.0) {
                    // check hit light source
                    continue;
                    bounceLight += material.emissive;
                } 
                else {
                    uint triangleIndex = intersection.primitive_id;
                    float3 v0 = vertices[triangleIndex * 3 + 0];
                    float3 v1 = vertices[triangleIndex * 3 + 1];
                    float3 v2 = vertices[triangleIndex * 3 + 2];
                    float3 edge1 = v1 - v0;
                    float3 edge2 = v2 - v0;
                    float3 triangleNormal = normalize(cross(edge1, edge2));
                    
                    RayGPU nray;
                    nray.origin = newRay.origin;
                    nray.direction = newRay.direction;

                    IntersectionGPU nextIntersection;
                    nextIntersection.type = Hit;
                    nextIntersection.point = newRay.origin + newRay.direction * intersection.distance;
                    nextIntersection.ray = nray;
                    nextIntersection.normal = triangleNormal;
                    nextIntersection.material = material;
                    float3 brdfCosine = calculateBRDFContribution(ray, incomingIntersection.point, incomingIntersection.normal, material, newRay.direction, float3(1.0));
                    float3 throughput = brdfCosine / (cosineSample.pdf + 1e-6);

                    bounceLight += throughput * calculateLighting(nextIntersection, 30, nray, index, light,
                    accelerationStructure, materials, vertices, params, intersectionQuery, maxBounces, remainingBounces - 1);
                    // / float (samples);
                }
            }
        }
        for (uint i = 0; i < samplesPerStrategy; i++) {
            // Sample VNDF
            uint sampleId = (index.y * imageWidth + index.x) * (samplesPerStrategy) + i;
            SampleResultGPU vndfSample = sampleVNDF(-ray.direction, incomingIntersection.normal, 
            incomingIntersection.material.roughness, index, sampleId, samplesPerStrategy, i + 2 * samplesPerStrategy);
            
            // Calculate PDFs for other strategies
            // float lightPdf = calculateBoxLightPdf(light, intersection.point, vndfSample.direction);
            float lightPdf = calculateSquareLightPdf(light, incomingIntersection.point, vndfSample.direction);
            float cosineWeightedPdf = max(dot(incomingIntersection.normal, vndfSample.direction), 0.0) / M_PI_F;

            float3 radiance = traceTriangleLightRay(incomingIntersection.point, vndfSample.direction, accelerationStructure, materials);
            if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
                float weight = powerHeuristic(vndfSample.pdf, lightPdf, cosineWeightedPdf, beta);
                float3 brdf = calculateBRDFContribution(ray, incomingIntersection.point, incomingIntersection.normal,
                incomingIntersection.material, vndfSample.direction, radiance);
                totalLight += brdf * weight / vndfSample.pdf;
            }
            if (remainingBounces > 1) {
                // accumulate bounce light
                struct ray newRay;
                newRay.origin = incomingIntersection.point + incomingIntersection.normal * 1e-4;
                newRay.direction = vndfSample.direction;
                newRay.min_distance = 0.001f;
                newRay.max_distance = 1000.0f;

                intersectionQuery.reset(newRay, accelerationStructure, params);
                intersectionQuery.next();

                // pull required info about the committed intersection.
                intersector<triangle_data>::result_type intersection;

                intersection.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
                intersection.distance = intersectionQuery.get_committed_distance();
                intersection.primitive_id = intersectionQuery.get_committed_primitive_id();
                MaterialGPU material = materials[intersection.primitive_id];

                if (intersection.type != intersection_type::triangle) {
                    bounceLight += float3(0.0, 0.0, 0.0); // Miss
                    // continue;
                } 
                else if (length(material.emissive) > 0.0) {
                    // check hit light source
                    continue;
                    bounceLight += material.emissive;
                } 
                else {
                    uint triangleIndex = intersection.primitive_id;
                    float3 v0 = vertices[triangleIndex * 3 + 0];
                    float3 v1 = vertices[triangleIndex * 3 + 1];
                    float3 v2 = vertices[triangleIndex * 3 + 2];
                    float3 edge1 = v1 - v0;
                    float3 edge2 = v2 - v0;
                    float3 triangleNormal = normalize(cross(edge1, edge2));
                    
                    RayGPU nray;
                    nray.origin = newRay.origin;
                    nray.direction = newRay.direction;

                    IntersectionGPU nextIntersection;
                    nextIntersection.type = Hit;
                    nextIntersection.point = newRay.origin + newRay.direction * intersection.distance;
                    nextIntersection.ray = nray;
                    nextIntersection.normal = triangleNormal;
                    nextIntersection.material = material;
                    float3 brdfCosine = calculateBRDFContribution(ray, incomingIntersection.point, incomingIntersection.normal, material, newRay.direction, float3(1.0));
                    float3 throughput = brdfCosine / (vndfSample.pdf + 1e-6);

                    bounceLight += throughput * calculateLighting(nextIntersection, 30, nray, index, light,
                    accelerationStructure, materials, vertices, params, intersectionQuery, maxBounces, remainingBounces - 1);
                    // / float (samples);
                }
            }
        }
        // return (totalLight + bounceLight) / float (samplesPerStrategy * 3);
        return totalLight / float(samplesPerStrategy * 3) + bounceLight / float(60);
        // return totalLight / float(samplesPerStrategy * 3) + bounceLight / float(90);
    }
}

// TODO: make proper recursive 
float3 recursiveLightingCalculation(ray r, uint2 index, SquareLightGPU light, primitive_acceleration_structure accelerationStructure, device const MaterialGPU* materials, device const float3* vertices, uint remainingBounces, uint totalBounces, uint samples) {
    if (remainingBounces <= 0) {
        return float3(0.0, 0.0, 0.0); // No more bounces
    }
    // intersection params and query object
    // acceleration structure = bvh structure that stores the geometry
    // we are intersecting with
    // triangle_data = type of geometry we are intersecting with

    // configures how intersection test behaves
    intersection_params params;
    // intersection query object that can be initialised with ray, accel struct and params
    // performs actual intersection test. stores results of test
    intersection_query<triangle_data> intersectionQuery;
    params.assume_geometry_type(geometry_type::triangle); 
    params.force_opacity(forced_opacity::opaque); // treats geometry as solid not transparent
    params.accept_any_intersection(false); // find closest intersection

    // init the intersection query with data
    intersectionQuery.reset(r, accelerationStructure, params);

    // performs the actual intersection - can call multiple times to find multiple intersections along ray.
    // right now we just get closest one, metal automatically handles bvh intersections
    intersectionQuery.next();

    // intersector is metals built in ray tracing class that performs ray/geometry intersections
    // triangle_data template param that tells us what kind of geometry we are intersecting
    // different geometries have different result structures so we need to get the type
    intersector<triangle_data>::result_type intersection;
    // pull required info about the committed intersection.
    intersection.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    intersection.distance = intersectionQuery.get_committed_distance();
    intersection.primitive_id = intersectionQuery.get_committed_primitive_id();
    // intersection.triangle_barycentric_coord = intersectionQuery.get_committed_triangle_barycentric_coord();

    // check miss
    if (intersection.type != intersection_type::triangle) {
        return float3(0.0, 0.0, 0.0);
    }

    MaterialGPU material = materials[intersection.primitive_id];
    // check hit light source
    if (length(material.emissive) > 0.0) {
        return material.emissive * material.diffuse.rgb;
    }

    // now handle normal triangle hit
    // calc triangle normal
    uint triangleIndex = intersection.primitive_id;
    float3 v0 = vertices[triangleIndex * 3 + 0];
    float3 v1 = vertices[triangleIndex * 3 + 1];
    float3 v2 = vertices[triangleIndex * 3 + 2];
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 triangleNormal = normalize(cross(edge1, edge2)); // just face normal atm (fine for boxes)
    
    RayGPU gRay;
    gRay.origin = r.origin;
    gRay.direction = r.direction;
    IntersectionGPU closestIntersection;
    closestIntersection.type = Hit;
    closestIntersection.point = gRay.origin + gRay.direction * intersection.distance;
    closestIntersection.ray = gRay;
    closestIntersection.normal = triangleNormal;
    closestIntersection.material = material;

    float3 sampledLight = calculateLighting(closestIntersection, samples, gRay, index, 
    light, accelerationStructure, materials, vertices, params, intersectionQuery, 2, 2);

    return sampledLight;

    // float3 bounceLight = float3(0.0, 0.0, 0.0);
    // uint spawns = 30;
    // uint spawnsPerStrategy = spawns / 3;

    // // do mis sampling to generate new directions for 3 strategies
    // uint imageWidth = 800;
    // // direct light case
    // for (uint i = 0; i < spawnsPerStrategy; i++) {
    //     uint sampleId = (index.y * imageWidth + index.x) * (spawnsPerStrategy) + i;
    //     // SampleResultGPU lightSample = sampleBoxLight(light, closestIntersection.point, index, sampleId, spawnsPerStrategy, i);
    //     SampleResultGPU lightSample = sampleSquareLight(light, closestIntersection.point, index, sampleId, spawnsPerStrategy, i);
    //     float3 newDirection = lightSample.direction;

    //     ray newRay;
    //     newRay.origin = closestIntersection.point + triangleNormal * 1e-4;
    //     newRay.direction = newDirection;
    //     newRay.min_distance = r.min_distance;
    //     newRay.max_distance = r.max_distance;

    //     intersectionQuery.reset(newRay, accelerationStructure, params);
    //     intersectionQuery.next();

    //     // pull required info about the committed intersection.
    //     intersection.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    //     intersection.distance = intersectionQuery.get_committed_distance();
    //     intersection.primitive_id = intersectionQuery.get_committed_primitive_id();
    //     material = materials[intersection.primitive_id];

    //     if (intersection.type != intersection_type::triangle) {
    //         bounceLight += float3(0.0, 0.0, 0.0); // Miss
    //         // continue;
    //     } 
    //     else if (length(material.emissive) > 0.0) {
    //         // check hit light source
    //         continue;
    //         bounceLight += material.emissive;
    //     } 
    //     else {
    //         // now handle normal triangle hit
    //         // calc triangle normal
    //         triangleIndex = intersection.primitive_id;
    //         v0 = vertices[triangleIndex * 3 + 0];
    //         v1 = vertices[triangleIndex * 3 + 1];
    //         v2 = vertices[triangleIndex * 3 + 2];
    //         edge1 = v1 - v0;
    //         edge2 = v2 - v0;
    //         triangleNormal = normalize(cross(edge1, edge2)); // just face normal atm (fine for boxes)
            
    //         RayGPU nray;
    //         nray.origin = newRay.origin;
    //         nray.direction = newRay.direction;

    //         IntersectionGPU nextIntersection;
    //         nextIntersection.type = Hit;
    //         nextIntersection.point = newRay.origin + newRay.direction * intersection.distance;
    //         nextIntersection.ray = nray;
    //         nextIntersection.normal = triangleNormal;
    //         nextIntersection.material = material;

    //         bounceLight += calculateLighting(nextIntersection, 100, nray, index, 
    //         light, accelerationStructure, materials);
    //     }
    // }
    // // cosineweighted case
    // for (uint i = 0; i < spawnsPerStrategy; i++) {
    //     uint sampleId = (index.y * imageWidth + index.x) * (spawnsPerStrategy) + i;
    //     SampleResultGPU cosineSample = sampleCosineWeighted(closestIntersection.normal, index, sampleId, 
    //     spawnsPerStrategy, i + spawnsPerStrategy);
    //     float3 newDirection = cosineSample.direction;

    //     ray newRay;
    //     newRay.origin = closestIntersection.point + triangleNormal * 1e-4;
    //     newRay.direction = newDirection;
    //     newRay.min_distance = r.min_distance;
    //     newRay.max_distance = r.max_distance;

    //     intersectionQuery.reset(newRay, accelerationStructure, params);
    //     intersectionQuery.next();

    //     // pull required info about the committed intersection.
    //     intersection.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    //     intersection.distance = intersectionQuery.get_committed_distance();
    //     intersection.primitive_id = intersectionQuery.get_committed_primitive_id();
    //     material = materials[intersection.primitive_id];

    //     if (intersection.type != intersection_type::triangle) {
    //         // bounceLight += float3(0.0, 0.0, 0.0); // Miss
    //         continue;
    //     } 
    //     else if (length(material.emissive) > 0.0) {
    //         // check hit light source
    //         continue;
    //         bounceLight += material.emissive;
    //     } 
    //     else {
    //         // now handle normal triangle hit
    //         // calc triangle normal
    //         triangleIndex = intersection.primitive_id;
    //         v0 = vertices[triangleIndex * 3 + 0];
    //         v1 = vertices[triangleIndex * 3 + 1];
    //         v2 = vertices[triangleIndex * 3 + 2];
    //         edge1 = v1 - v0;
    //         edge2 = v2 - v0;
    //         triangleNormal = normalize(cross(edge1, edge2)); // just face normal atm (fine for boxes)
            
    //         RayGPU nray;
    //         nray.origin = newRay.origin;
    //         nray.direction = newRay.direction;

    //         IntersectionGPU nextIntersection;
    //         nextIntersection.type = Hit;
    //         nextIntersection.point = newRay.origin + newRay.direction * intersection.distance;
    //         nextIntersection.ray = nray;
    //         nextIntersection.normal = triangleNormal;
    //         nextIntersection.material = material;

    //         bounceLight += calculateLighting(nextIntersection, 100, nray, index, 
    //         light, accelerationStructure, materials);
    //     }
    // }
    // // vndf case
    // for (uint i = 0; i < spawnsPerStrategy; i++) {
    //     uint sampleId = (index.y * imageWidth + index.x) * (spawnsPerStrategy) + i;
    //     SampleResultGPU vndfSample = sampleVNDF(-gRay.direction, closestIntersection.normal, 
    //     closestIntersection.material.roughness, index, sampleId, spawnsPerStrategy, i + 2 * spawnsPerStrategy);
    //     float3 newDirection = vndfSample.direction;

    //     ray newRay;
    //     newRay.origin = closestIntersection.point + triangleNormal * 1e-4;
    //     newRay.direction = newDirection;
    //     newRay.min_distance = r.min_distance;
    //     newRay.max_distance = r.max_distance;

    //     intersectionQuery.reset(newRay, accelerationStructure, params);
    //     intersectionQuery.next();

    //     // pull required info about the committed intersection.
    //     intersection.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    //     intersection.distance = intersectionQuery.get_committed_distance();
    //     intersection.primitive_id = intersectionQuery.get_committed_primitive_id();
    //     material = materials[intersection.primitive_id];

    //     if (intersection.type != intersection_type::triangle) {
    //         // bounceLight += float3(0.0, 0.0, 0.0); // Miss
    //         continue;
    //     } 
    //     else if (length(material.emissive) > 0.0) {
    //         // check hit light source
    //         continue;
    //         bounceLight += material.emissive;
    //     } 
    //     else {
    //         // now handle normal triangle hit
    //         // calc triangle normal
    //         triangleIndex = intersection.primitive_id;
    //         v0 = vertices[triangleIndex * 3 + 0];
    //         v1 = vertices[triangleIndex * 3 + 1];
    //         v2 = vertices[triangleIndex * 3 + 2];
    //         edge1 = v1 - v0;
    //         edge2 = v2 - v0;
    //         triangleNormal = normalize(cross(edge1, edge2)); // just face normal atm (fine for boxes)
            
    //         RayGPU nray;
    //         nray.origin = newRay.origin;
    //         nray.direction = newRay.direction;

    //         IntersectionGPU nextIntersection;
    //         nextIntersection.type = Hit;
    //         nextIntersection.point = newRay.origin + newRay.direction * intersection.distance;
    //         nextIntersection.ray = nray;
    //         nextIntersection.normal = triangleNormal;
    //         nextIntersection.material = material;

    //         bounceLight += calculateLighting(nextIntersection, 100, nray, index, 
    //         light, accelerationStructure, materials);
    //     }
    // }
    // bounceLight /= float(spawnsPerStrategy * 3);
    // return sampledLight + bounceLight;



    // now spawn bounce rays and calc
    // for (uint i = 0; i < spawns; i++) {
        
    //     // generate new ray dir
    //     // SampleResultGPU sample = sampleUniformHemisphere(triangleNormal, index, i, spawns, i);
    //     SampleResultGPU sample = sampleCosineWeighted(triangleNormal, index, i, spawns, i);
    //     float3 newDirection = sample.direction;
    //     // float pdf = sample.pdf;
    //     // float3 brdf = material.diffuse.rgb * Fd_Lambert();
    //     // float cosTheta = max(0.0, dot(triangleNormal, newDirection));
    //     // float3 weight = brdf * cosTheta / (pdf + 1e-6);

    //     ray newRay;
    //     newRay.origin = closestIntersection.point + triangleNormal * 1e-4;
    //     newRay.direction = newDirection;
    //     newRay.min_distance = r.min_distance;
    //     newRay.max_distance = r.max_distance;

    //     // float3 bounceResult = recursiveLightingCalculation(
    //     //     newRay, index, light, accelerationStructure, materials, vertices,
    //     //     remainingBounces - 1, totalBounces, samples
    //     // );
    //     // bounceLight += bounceResult;

    //     intersectionQuery.reset(newRay, accelerationStructure, params);
    //     intersectionQuery.next();

    //     // pull required info about the committed intersection.
    //     intersection.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    //     intersection.distance = intersectionQuery.get_committed_distance();
    //     intersection.primitive_id = intersectionQuery.get_committed_primitive_id();
    //     material = materials[intersection.primitive_id];

    //     if (intersection.type != intersection_type::triangle) {
    //         // bounceLight += float3(0.0, 0.0, 0.0); // Miss
    //         continue;
    //     } 
    //     else if (length(material.emissive) > 0.0) {
    //         // check hit light source
    //         continue;
    //         bounceLight += material.emissive;
    //     } 
    //     else {
    //         // now handle normal triangle hit
    //         // calc triangle normal
    //         triangleIndex = intersection.primitive_id;
    //         v0 = vertices[triangleIndex * 3 + 0];
    //         v1 = vertices[triangleIndex * 3 + 1];
    //         v2 = vertices[triangleIndex * 3 + 2];
    //         edge1 = v1 - v0;
    //         edge2 = v2 - v0;
    //         triangleNormal = normalize(cross(edge1, edge2)); // just face normal atm (fine for boxes)
            
    //         RayGPU nray;
    //         nray.origin = newRay.origin;
    //         nray.direction = newRay.direction;

    //         IntersectionGPU nextIntersection;
    //         nextIntersection.type = Hit;
    //         nextIntersection.point = newRay.origin + newRay.direction * intersection.distance;
    //         nextIntersection.ray = nray;
    //         nextIntersection.normal = triangleNormal;
    //         nextIntersection.material = material;

    //         bounceLight += calculateLighting(nextIntersection, spawns, nray, index, 
    //         light, accelerationStructure, materials);
    //     }
    // }  

    // // bounceLight /= (float(spawns) * float(samples)); // average bounce light over number of samples and spawns
    // bounceLight /= float(spawns); // average bounce light over number of samples and spawns

    // return sampledLight + bounceLight;
}

// can typedef the intersection result type to make it easier to use
// typedef intersector<triangle_data>::result_type IntersectionResult;
// need explicity specify the accelerationstructure buffer
kernel void drawTriangle(device const CameraGPU* cameras, device const MaterialGPU * materials, 
device const SquareLightGPU* squareLights, device const float3* vertices, device uchar* pixels, 
primitive_acceleration_structure accelerationStructure [[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    CameraGPU camera = cameras[0];
    SquareLightGPU light = squareLights[0];

    float aspectRatio = float(camera.resolution.x / camera.resolution.y);
    float halfWidth = tan(camera.horizontalFov / 2.0);
    float halfHeight = halfWidth / aspectRatio;
    
    // Camera coord system
    float3 w = -normalize(camera.direction);
    float3 u = normalize(cross(camera.up, w));
    float3 v = normalize(cross(w, u));
    
    // pixel/screen coordinates
    int x = index.x;
    int y = index.y;
    if (x >= camera.resolution.x || y >= camera.resolution.y) return;
    
    float s = (float(x) / float(camera.resolution.x)) * 2.0 - 1.0;
    float t = -((float(y) / float(camera.resolution.y)) * 2.0 - 1.0);

    // dir to pixel
    float3 dir = normalize(s * halfWidth * u + t * halfHeight * v - w);
    
    // use metal ray since it the metal intersect needs min/max dist
    ray r;
    r.origin = camera.position;
    r.direction = dir;
    r.min_distance = 0.001f; // small epsilon to avoid self-intersection
    r.max_distance = 1000.0f; 

    int maxBounces = 2;

    int pixelOffset = (y * camera.resolution.x + x) * 4;

    float3 totalColor = recursiveLightingCalculation(r, index, light, accelerationStructure, materials,
    vertices, maxBounces, maxBounces, 99);
    // float3 sampledLight = calculateLighting(closestIntersection, samples, gRay, index, 
    // light, accelerationStructure, materials, vertices, params, intersectionQuery, 2, 2);

    float4 color = reinhartToneMapping(totalColor * cameraExposure(camera));
    pixels[pixelOffset + 0] = uchar(color.r * 255);
    pixels[pixelOffset + 1] = uchar(color.g * 255);
    pixels[pixelOffset + 2] = uchar(color.b * 255);
    pixels[pixelOffset + 3] = 255;

    // // dont for loop here?? this can go maybe
    // for (int bounce = 0; bounce < maxBounces; bounce++) {
    //     // intersection params and query object
    //     // acceleration structure is the bvh structure that stores the geometry
    //     // we are intersecting with, can be used to accelerate intersection tests
    //     // triangle_data is the type of geometry we are intersecting with

    //     // configures how intersecttion test behaves
    //     intersection_params params;
    //     // intersection query object that can be initialised with ray, accel struct and params
    //     // performs actual intersection test. stores results of test
    //     intersection_query<triangle_data> i;
    //     params.assume_geometry_type(geometry_type::triangle); 
    //     params.force_opacity(forced_opacity::opaque); // treats geometry as solid not transparent
    //     params.accept_any_intersection(false); // find closest intersection

    //     // init the intersection query with data
    //     i.reset(r, accelerationStructure, params);

    //     // performs the actual intersection - can call multiple times to find multiple intersections along ray.
    //     // right now we just get closest one, metal automatically handles bvh intersections
    //     i.next();
    //     // to get past invis wall?
    //     // i.next();

    //     // intersector is metals built in ray tracing class that performs ray/geometry intersections
    //     // triangle_data template param that tells us what kind of geometry we are intersecting
    //     // different geometries have different result structures so we need to get the type
    //     intersector<triangle_data>::result_type intersection;
    //     // pull required info about the committed intersection.
    //     intersection.type = i.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    //     intersection.distance = i.get_committed_distance();
    //     intersection.primitive_id = i.get_committed_primitive_id();
    //     intersection.triangle_barycentric_coord = i.get_committed_triangle_barycentric_coord();

    //     // check miss
    //     if (intersection.type != intersection_type::triangle) {
    //         break;
    //     }

    //     MaterialGPU material = materials[intersection.primitive_id];

    //     // check hit light source
    //     if (length(material.emissive) > 0.0) {
    //         totalColor += throughput * material.emissive;
    //         break; // no further bounce once light hit
    //     }

        // // now handle normal triangle hit
        // // calc triangle normal
        // uint triangleIndex = intersection.primitive_id;
        // float3 v0 = vertices[triangleIndex * 3 + 0];
        // float3 v1 = vertices[triangleIndex * 3 + 1];
        // float3 v2 = vertices[triangleIndex * 3 + 2];
        // float3 edge1 = v1 - v0;
        // float3 edge2 = v2 - v0;
        // float3 triangleNormal = normalize(cross(edge1, edge2)); // just face normal atm (fine for boxes)

        // uint samples = 100;
        // RayGPU gRay;
        // gRay.origin = r.origin;
        // gRay.direction = r.direction;

        // IntersectionGPU closestIntersection;
        // closestIntersection.type = Hit;
        // closestIntersection.point = gRay.origin + gRay.direction * intersection.distance;
        // closestIntersection.ray = gRay;
        // closestIntersection.normal = triangleNormal;
        // closestIntersection.material = material;
        // float3 sampledLight = calculateLighting(closestIntersection, samples, gRay, index, 
        // light, accelerationStructure, materials);
        // totalColor += throughput * sampledLight;
        
        //spawn new directions for next bounce
    //     float3 indirectLight = float3(0.0, 0.0, 0.0);
    //     uint bounceSplit = 100;
    //     for (uint i = 0; i < bounceSplit; i++) {
    //         // Sample a new direction in the hemisphere
    //         SampleResultGPU sample = sampleUniformHemisphere(triangleNormal, index, i, bounceSplit, i);
    //         float3 newDirection = sample.direction;

    //         // Calculate BRDF contribution
    //         float3 brdf = material.diffuse.rgb * Fd_Lambert(); // lambertian diffuse BRDF
    //         float cosTheta = max(0.0, dot(triangleNormal, newDirection));
    //         // tracks how much impact light energy has from bounce to bounce
    //         float3 bounceThroughput = throughput * brdf * cosTheta;

    //         // Create new ray for next bounce
    //         ray newRay;
    //         newRay.min_distance = r.min_distance; // keep same min distance
    //         newRay.max_distance = r.max_distance; // keep same max distance
    //         newRay.origin = gRay.origin + gRay.direction * intersection.distance + triangleNormal * 1e-4; // avoid self intersect
    //         newRay.direction = newDirection;
    //         float3 traceResult = traceBounceRay(newRay, accelerationStructure, materials, vertices, light, 
    //         index, maxBounces - 1, bounceThroughput);
    //         indirectLight += bounceThroughput * traceResult;
    //     }
    //     totalColor += indirectLight / float(bounceSplit); // average the indirect light contributions
    //     break; // break quick since we are handlings bounces other way
    // }

    // float4 color = reinhartToneMapping(totalColor * cameraExposure(camera));
    // pixels[pixelOffset + 0] = uchar(color.r * 255);
    // pixels[pixelOffset + 1] = uchar(color.g * 255);
    // pixels[pixelOffset + 2] = uchar(color.b * 255);
    // pixels[pixelOffset + 3] = 255;
}