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
// check could be improved mayybe?
// Hammersley and Halton Points
float randomFloat(uint seed) {
    return float(hash(seed)) / (float(0xffffffffU) + 1.0);
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

SampleResultGPU sampleBoxLight(BoxLightGPU light, float3 point, uint2 index, uint sampleIndex) {
    uint seed1 = hash(index.x + index.y * 1920 + sampleIndex * 3840);
    uint seed2 = hash(index.y + index.x * 1080 + sampleIndex * 7680 + 12345);
    uint seed3 = hash(index.x * 2 + index.y * 3 + sampleIndex * 5432 + 54321);
    
    float u1 = randomFloat(seed1);
    float u2 = randomFloat(seed2);
    float u3 = randomFloat(seed3);
    
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

// Cosine-weighted hemisphere sampling
SampleResultGPU sampleCosineWeighted(float3 normal, uint2 index, uint sampleIndex) {
    uint seed1 = hash(index.x + index.y * 1920 + sampleIndex * 3840);
    uint seed2 = hash(index.y + index.x * 1080 + sampleIndex * 7680 + 12345);
    
    float u1 = randomFloat(seed1);
    float u2 = randomFloat(seed2);
    
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

// VNDF sampling
SampleResultGPU sampleVNDF(float3 viewDir, float3 normal, float roughness, uint2 index, uint sampleIndex) {
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
    
    // Step 3: Sample point with polar coordinates
    uint seed1 = hash(index.x + index.y * 1920 + sampleIndex * 3840);
    uint seed2 = hash(index.y + index.x * 1080 + sampleIndex * 7680 + 12345);
    
    float u1 = randomFloat(seed1);
    float u2 = randomFloat(seed2);
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

float3 calculateTotalLighting(IntersectionGPU intersection, uint samples, RayGPU ray, uint2 index, 
device const SphereGPU* spheres, constant uint& sphereCount, device const SphereLightGPU* lights, 
constant uint& lightCount) {
    SphereLightGPU light = lights[0]; // one light only atm - fix later
    float3 totalLight = float3(0.0, 0.0, 0.0);
    uint samplesPerStrategy = samples / 3;
    // float scaleFactor = 3.0; // Scale factor to balance contributions from different strategies
    float beta = 2.0; // Power heuristic exponent

    for (uint i = 0; i < samplesPerStrategy; i++) {
        // Sample direct light
        SampleResultGPU lightSample = sampleSphereLight(light, intersection.point, index, i);

        // Calculate PDFs for other strategies
        float cosineWeightedPdf = max(dot(intersection.normal, lightSample.direction), 0.0) / M_PI_F;
        float vndfPdf = calculateVNDFPdf(-ray.direction, intersection.normal, lightSample.direction, intersection.material.roughness);

        float3 radiance = traceLightRay(intersection.point, lightSample.direction, 
        spheres, sphereCount, lights, lightCount);
        if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
            float weight = powerHeuristic(lightSample.pdf, cosineWeightedPdf, vndfPdf, beta);
            float3 brdf = calculateBRDFContribution(ray, intersection.point, intersection.normal,
            intersection.material, lightSample.direction, radiance);
            totalLight += brdf * weight / lightSample.pdf;
        }
    }
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // Sample cosine-weighted hemisphere
        SampleResultGPU cosineSample = sampleCosineWeighted(intersection.normal, index, i + samplesPerStrategy);
        
        // Calculate PDFs for other strategies
        float lightPdf = calculateLightPdf(light, intersection.point, cosineSample.direction);
        float vndfPdf = calculateVNDFPdf(-ray.direction, intersection.normal, cosineSample.direction, intersection.material.roughness);

        float3 radiance = traceLightRay(intersection.point, cosineSample.direction, 
        spheres, sphereCount, lights, lightCount);
        if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
            float weight = powerHeuristic(cosineSample.pdf, lightPdf, vndfPdf, beta);
            float3 brdf = calculateBRDFContribution(ray, intersection.point, intersection.normal,
            intersection.material, cosineSample.direction, radiance);
            totalLight += brdf * weight / cosineSample.pdf;
        }
    }
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // Sample VNDF
        SampleResultGPU vndfSample = sampleVNDF(-ray.direction, intersection.normal, 
        intersection.material.roughness, index, i + 2 * samplesPerStrategy);
        
        // Calculate PDFs for other strategies
        float lightPdf = calculateLightPdf(light, intersection.point, vndfSample.direction);
        float cosineWeightedPdf = max(dot(intersection.normal, vndfSample.direction), 0.0) / M_PI_F;

        float3 radiance = traceLightRay(intersection.point, vndfSample.direction, 
        spheres, sphereCount, lights, lightCount);
        if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
            float weight = powerHeuristic(vndfSample.pdf, lightPdf, cosineWeightedPdf, beta);
            float3 brdf = calculateBRDFContribution(ray, intersection.point, intersection.normal,
            intersection.material, vndfSample.direction, radiance);
            totalLight += brdf * weight / vndfSample.pdf;
        }
    }
    return totalLight;
    // return totalLight / 3;
    // return totalLight / float (samplesPerStrategy * 3); 
    // return totalLight / float (samplesPerStrategy); 
}

kernel void draw(device const CameraGPU* cameras, device const SphereGPU* spheres, 
constant uint& sphereCount, device uchar* pixels, device const SphereLightGPU* lights, 
constant uint& lightCount, device const float4* ambientLight, uint2 index [[thread_position_in_grid]]) {
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

    IntersectionGPU closestIntersection = getClosestIntersection(ray, spheres, sphereCount, lights, lightCount);
    switch (closestIntersection.type) {
        case Hit: {
            uint samples = 20;
            float3 totalLight = calculateTotalLighting(closestIntersection, samples, ray, index, 
            spheres, sphereCount, lights, lightCount);
            float4 color = reinhartToneMapping(totalLight * cameraExposure(camera));
            pixels[pixelOffset + 0] = uchar(color.r * 255); // R
            pixels[pixelOffset + 1] = uchar(color.g * 255); // G
            pixels[pixelOffset + 2] = uchar(color.b * 255); // B
            pixels[pixelOffset + 3] = uchar(color.a * 255); // A
            break;

            // // Simple Lambert lighting for debugging
            // float3 lightPos = float3(3, 3, 3); // Your light position
            // float3 lightDir = normalize(lightPos - closestIntersection.point);
            // float lambert = max(0.0, dot(closestIntersection.normal, lightDir));
            
            // float3 ambient = closestIntersection.material.diffuse.rgb * 0.2;
            // float3 diffuse = closestIntersection.material.diffuse.rgb * lambert;
            // float3 finalColor = ambient + diffuse;
            
            // pixels[pixelOffset + 0] = uchar(clamp(finalColor.r, 0.0, 1.0) * 255);
            // pixels[pixelOffset + 1] = uchar(clamp(finalColor.g, 0.0, 1.0) * 255);
            // pixels[pixelOffset + 2] = uchar(clamp(finalColor.b, 0.0, 1.0) * 255);
            // pixels[pixelOffset + 3] = 255;
            // break;
        }
        case HitLight: {
            float3 totalLight = closestIntersection.radiance;
            float4 color = reinhartToneMapping(totalLight * cameraExposure(camera));
            pixels[pixelOffset + 0] = uchar(color.r * 255); // R
            pixels[pixelOffset + 1] = uchar(color.g * 255); // G
            pixels[pixelOffset + 2] = uchar(color.b * 255); // B
            pixels[pixelOffset + 3] = uchar(color.a * 255); // A
            break;
        }
        case Miss:
        default:
            pixels[pixelOffset + 0] = uchar(ambientLight[0].r * 255); // R
            pixels[pixelOffset + 1] = uchar(ambientLight[0].g * 255); // G
            pixels[pixelOffset + 2] = uchar(ambientLight[0].b * 255); // B
            pixels[pixelOffset + 3] = uchar(ambientLight[0].a * 255); // A
            break;
    }
}


float3 calculateLighting(IntersectionGPU intersection, uint samples, RayGPU ray, uint2 index, BoxLightGPU light, 
primitive_acceleration_structure accelerationStructure, device const MaterialGPU* materials) {
    // only one light atm
    float3 totalLight = float3(0.0, 0.0, 0.0);
    uint samplesPerStrategy = samples / 3;
    // float scaleFactor = 3.0; // Scale factor to balance contributions from different strategies
    float beta = 2.0; // Power heuristic exponent
    uint imageWidth = 800; 
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // Sample box light
        uint sampleId = (index.y * imageWidth + index.x) * samples + i;
        SampleResultGPU lightSample = sampleBoxLight(light, intersection.point, index, sampleId);

        // Calculate PDFs for other strategies
        float cosineWeightedPdf = max(dot(intersection.normal, lightSample.direction), 0.0) / M_PI_F;
        float vndfPdf = calculateVNDFPdf(-ray.direction, intersection.normal, lightSample.direction, intersection.material.roughness);

        float3 radiance = traceTriangleLightRay(intersection.point, lightSample.direction, accelerationStructure, materials);
        if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
            float weight = powerHeuristic(lightSample.pdf, cosineWeightedPdf, vndfPdf, beta);
            float3 brdf = calculateBRDFContribution(ray, intersection.point, intersection.normal,
            intersection.material, lightSample.direction, radiance);
            totalLight += brdf * weight / lightSample.pdf;
        }
    }
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // Sample cosine-weighted hemisphere
        uint sampleId = (index.y * imageWidth + index.x) * samples + i;
        SampleResultGPU cosineSample = sampleCosineWeighted(intersection.normal, index, sampleId);
        
        // Calculate PDFs for other strategies
        float lightPdf = calculateBoxLightPdf(light, intersection.point, cosineSample.direction);
        float vndfPdf = calculateVNDFPdf(-ray.direction, intersection.normal, cosineSample.direction, intersection.material.roughness);

        float3 radiance = traceTriangleLightRay(intersection.point, cosineSample.direction, accelerationStructure, materials);
        if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
            float weight = powerHeuristic(cosineSample.pdf, lightPdf, vndfPdf, beta);
            float3 brdf = calculateBRDFContribution(ray, intersection.point, intersection.normal,
            intersection.material, cosineSample.direction, radiance);
            totalLight += brdf * weight / cosineSample.pdf;
        }
    }
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // Sample VNDF
        uint sampleId = (index.y * imageWidth + index.x) * samples + i;
        SampleResultGPU vndfSample = sampleVNDF(-ray.direction, intersection.normal, 
        intersection.material.roughness, index, sampleId);
        
        // Calculate PDFs for other strategies
        float lightPdf = calculateBoxLightPdf(light, intersection.point, vndfSample.direction);
        float cosineWeightedPdf = max(dot(intersection.normal, vndfSample.direction), 0.0) / M_PI_F;

        float3 radiance = traceTriangleLightRay(intersection.point, vndfSample.direction, accelerationStructure, materials);
        if (radiance.x != -1.0 || radiance.y != -1.0 || radiance.z != -1.0) {
            float weight = powerHeuristic(vndfSample.pdf, lightPdf, cosineWeightedPdf, beta);
            float3 brdf = calculateBRDFContribution(ray, intersection.point, intersection.normal,
            intersection.material, vndfSample.direction, radiance);
            totalLight += brdf * weight / vndfSample.pdf;
        }
    }
    // return totalLight;
    // return totalLight / 3;
    return totalLight / float (samplesPerStrategy * 3); 
    // return totalLight / float (samplesPerStrategy); 
}

// can typedef the intersection result type to make it easier to use
// typedef intersector<triangle_data>::result_type IntersectionResult;
// need explicity specify the accelerationstructure buffer
kernel void drawTriangle(device const CameraGPU* cameras, device const MaterialGPU * materials, 
device const BoxLightGPU* boxLights, device const float3* vertices, device uchar* pixels, 
primitive_acceleration_structure accelerationStructure [[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    CameraGPU camera = cameras[0];
    BoxLightGPU light = boxLights[0];

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
    
    // use metal ray since it the metal intersect needs min/max dist
    ray r;
    r.origin = camera.position;
    r.direction = dir;
    r.min_distance = 0.001f; // small epsilon to avoid self-intersection
    r.max_distance = 1000.0f; 

    // configures intersecttion test behaves
    intersection_params params;
    // intersection query object that can be initialised with ray, accel struct and params
    // performs actual intersection test. stores results of test
    intersection_query<triangle_data> i;
    params.assume_geometry_type(geometry_type::triangle); 
    params.force_opacity(forced_opacity::opaque); // treats geometry as solid not transparent
    params.accept_any_intersection(false); // find closests intersection not any hit

    // init the intersection query with data
    i.reset(r, accelerationStructure, params);

    // performs the actual intersection - can call multiple times to find multiple intersections along ray.
    // right now we just get closest one, metal automatically handles bvh intersections
    i.next();

    // intersector is metals built in ray tracing class that performs ray/geometry intersections
    // triangle_data template param that tells us what kind of geometry we are intersecting
    // different geometries have different result structures so we need to get the type
    intersector<triangle_data>::result_type intersection;

    // IntersectionResult intersection;
    
    // pull required info about the committed intersection.
    intersection.type = i.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    intersection.distance = i.get_committed_distance();
    intersection.primitive_id = i.get_committed_primitive_id();
    // intersection.normal = normalize(i.get_committed_normal());
    // intersection.geometry_id = i.get_committed_geometry_id();
    intersection.triangle_barycentric_coord = i.get_committed_triangle_barycentric_coord();

    // intersection.instance_id = i.get_committed_instance_id();
    // intersection.object_to_world_transform = i.get_committed_object_to_world_transform();

    int pixelOffset = (y * camera.resolution.x + x) * 4;
    if (intersection.type == intersection_type::triangle) {
        MaterialGPU material = materials[intersection.primitive_id];

        if (length(material.emissive) > 0.0) {
            // hit a light source triangle
            float3 totalLight = material.emissive;
            float4 color = reinhartToneMapping(totalLight * cameraExposure(camera));
            pixels[pixelOffset + 0] = uchar(color.r * 255); // R
            pixels[pixelOffset + 1] = uchar(color.g * 255); // G
            pixels[pixelOffset + 2] = uchar(color.b * 255); // B
            pixels[pixelOffset + 3] = 255; // A
        } else {
            // calc triangle normal
            uint triangleIndex = intersection.primitive_id;
            float3 v0 = vertices[triangleIndex * 3 + 0];
            float3 v1 = vertices[triangleIndex * 3 + 1];
            float3 v2 = vertices[triangleIndex * 3 + 2];
            
            // calc face normal
            float3 edge1 = v1 - v0;
            float3 edge2 = v2 - v0;
            float3 triangleNormal = normalize(cross(edge1, edge2)); // just face normal atm (fine for boxes)
            float3 viewDirection = -r.direction; // Ray direction points away from camera
            if (dot(triangleNormal, viewDirection) < 0.0) {
                triangleNormal = -triangleNormal; // Flip normal to face camera
            } // triangle winding should be fixed

            // hit a regular triangle
            int samples = 100;
            IntersectionGPU closestIntersection;
            closestIntersection.type = Hit;
            closestIntersection.point = r.origin + r.direction * intersection.distance;
            RayGPU ray;
            ray.origin = r.origin;
            ray.direction = r.direction;
            closestIntersection.ray = ray;
            closestIntersection.normal = triangleNormal;
            closestIntersection.material = material;

            float3 totalLight = calculateLighting(closestIntersection, samples, ray, index, 
            light, accelerationStructure, materials);
            float4 color = reinhartToneMapping(totalLight * cameraExposure(camera));
            pixels[pixelOffset + 0] = uchar(color.r * 255); // R
            pixels[pixelOffset + 1] = uchar(color.g * 255); // G
            pixels[pixelOffset + 2] = uchar(color.b * 255); // B
            pixels[pixelOffset + 3] = uchar(255); // A

            // float3 lightPos = light.center;
            // float3 lightDir = normalize(lightPos - closestIntersection.point);
            // float lambert = max(0.1, dot(triangleNormal, lightDir)); // 10% ambient minimum

            // float3 color = material.diffuse.rgb * lambert;
            // pixels[pixelOffset + 0] = uchar(clamp(color.r, 0.0, 1.0) * 255);
            // pixels[pixelOffset + 1] = uchar(clamp(color.g, 0.0, 1.0) * 255);
            // pixels[pixelOffset + 2] = uchar(clamp(color.b, 0.0, 1.0) * 255);
            // pixels[pixelOffset + 3] = 255;

        }
    } else {
        // miss - black
        pixels[pixelOffset + 0] = 0;   // R
        pixels[pixelOffset + 1] = 0;   // G
        pixels[pixelOffset + 2] = 0;   // B
        pixels[pixelOffset + 3] = 255; // A
    }
}