//
//  shaders2.metal
//  gpuRaytracer
//
//  Created by Nishad Sharma on 21/6/2025.
//

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
#include "shaderTypes.h"

typedef struct {
    IntersectionTypeGPU type;
    float3 point;
    ray ray;
    float3 normal;
    MaterialGPU material;
} IntersectionGPU;

constant unsigned int primes[] = {
    2,   3,  5,  7,
    11, 13, 17, 19,
    23, 29, 31, 37,
    41, 43, 47, 53,
    59, 61, 67, 71,
    73, 79, 83, 89
};

// index i, dimension d.
float halton(unsigned int i, unsigned int d) {
    unsigned int b = primes[d];

    float f = 1.0f;
    float invB = 1.0f / b;

    float r = 0;

    while (i > 0) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }

    return r;
}

float2 haltonRandom(uint i, uint d) {
    // Generate two independent Halton sequences
    float u1 = halton(i, 0 + d); // First dimension
    float u2 = halton(i, 1 + d); // Second dimension

    // Return as a 2D vector
    return float2(u1, u2);
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

float2 hashRandom(uint2 index, uint i) {
    // TODO: add camera to func so we can pull screen width easier
    uint sampleId = (index.y * 800 + index.x) * i; // Assuming 800 is the width of the image
    
    // uint seed1 = hash(index.x + index.y * 800 + sampleId * 1600);
    // uint seed2 = hash(index.y + index.x * 600 + sampleId * 3200 + 12345);
    uint seed1 = hash(index.x + index.y * 800 + sampleId );
    uint seed2 = hash(index.y + index.x * 600 + sampleId + 12345);
    
    // Generate two random floats
    float u1 = randomFloat(seed1);
    float u2 = randomFloat(seed2);
    
    return float2(u1, u2);
}

float2 shiftRandomPoints(float2 u) {
    float2 result;
    // Double the values
    result.x = u.x * 2.0;
    result.y = u.y * 2.0;
    
    // If any value exceeds 1.0, subtract 1.0 (wrap around)
    if (result.x >= 1.0) result.x -= 1.0;
    if (result.y >= 1.0) result.y -= 1.0;
    
    return result;
}

// radical inverse function for base 2 (Van der Corput sequence)
float radicalInverse2(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // 0x100000000
}

// hammersley point generation (2D low-discrepancy sequence)
// good for 2d sampling e.g. area light, texture sampling,
// screen-space sampling, any correlated 2d points
float2 hammersley2D(uint i, uint N) {
    return float2(float(i) / float(N), radicalInverse2(i));
}

// hammersley-based random float (much better distribution)
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
float powerHeuristic(float pdf1, float pdf2, float pdf3, uint samplesPerStrategy, float beta = 2.0) {
    float n = float(samplesPerStrategy);
    float p1 = pow(n * pdf1, beta);
    float sum = p1 + pow(n * pdf2, beta) + pow(n * pdf3, beta);
    return p1 / (sum + 1e-6);  // epsilon to avoid division by zero
}

// two strategy power heuristic for MIS
float twoStrategyPowerHeuristic(float pdf1, float pdf2, float beta = 2.0) {
    return pow(pdf1, beta) / (pow(pdf1, beta) + pow(pdf2, beta) + 1e-6); // epsilon to avoid division by zero
}

// TODO: figure out which one is better
float cameraExposure(CameraGPU camera) {
    float maxLuminance = 1.2 * pow(2.0, camera.ev100 );
    // float maxLuminance = pow(2.0, camera.ev100 - 3.0); // 3.0 is a constant to adjust exposure);
    return 1.0 / maxLuminance;
    // return 1.0 / (78.0 * pow(2.0, camera.ev100));
}

float3 reinhartToneMapping(float3 color) {
    float3 finalColor = color / (color + float3(1.0, 1.0, 1.0));
    // Don't apply gamma here - just clamp
    finalColor = clamp(finalColor, 0.0, 1.0);
    return finalColor;
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

ray generateCameraRay(CameraGPU camera, uint2 index, float2 pixelJitter) {
    // pixel/screen coordinates
    // look into temporal accumulation, use frame counter to accumulate results over time
    //store the running average in separate buffer and blend result with accumulated.

    int x = index.x;
    int y = index.y;
    float aspectRatio = float(camera.resolution.x / camera.resolution.y);
    float halfWidth = tan(camera.horizontalFov / 2.0);
    float halfHeight = halfWidth / aspectRatio;
    
    // Camera coord system
    float3 w = -normalize(camera.direction);
    float3 u = normalize(cross(camera.up, w));
    float3 v = normalize(cross(w, u));
    
    float s = ((float(x) + pixelJitter.x) / float(camera.resolution.x)) * 2.0 - 1.0;
    float t = -(((float(y) + pixelJitter.y) / float(camera.resolution.y)) * 2.0 - 1.0);
    // float s = (float(x) / float(camera.resolution.x)) * 2.0 - 1.0;
    // float t = -((float(y) / float(camera.resolution.y)) * 2.0 - 1.0);

    // dir to pixel
    float3 dir = normalize(s * halfWidth * u + t * halfHeight * v - w);
    
    // use metal ray since it the metal intersect needs min/max dist
    // intial ray from camera position to pixel direction
    ray r;
    r.origin = camera.position;
    r.direction = dir;
    r.min_distance = 0.001f; // small epsilon to avoid self-intersection
    r.max_distance = 1000.0f; 
    return r;
}

void writeToPixelBuffer(device uchar* pixels, CameraGPU camera, uint2 index, float3 color) {
    int x = index.x;
    int y = index.y;
    int pixelOffset = (y * camera.resolution.x + x) * 4;
    pixels[pixelOffset + 0] = uchar(color.r * 255);
    pixels[pixelOffset + 1] = uchar(color.g * 255);
    pixels[pixelOffset + 2] = uchar(color.b * 255);
    pixels[pixelOffset + 3] = 255; // alpha
    return;
}

float3 calculateBRDFContribution(ray ray, float3 normal, MaterialGPU material, float3 lightDir) {
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
    float3 Fd = material.diffuse.rgb * Fd_Lambert();

    float3 energyCompensation = float3(1.0) - F;  // Amount of light not reflected
    float3 kD = energyCompensation * (1.0 - material.metallic);
    
    // Combine both terms and apply light properties
    float3 BRDF = kD * (Fd + Fr);
    
    float3 finalColor = BRDF * NoL;

    return finalColor;
}

ray directSquareLightRay(float3 origin, SquareLightGPU light, float2 randomPoints) {
    float3 lightNormal = float3(0, -1, 0); // TODO: hardcoded for now fix up later.
    OrthonormalBasisGPU basis = buildOrthonormalBasis(lightNormal);

    float3 center = light.center;
    // Uniformly sample a point on the rectangle
    float x = (randomPoints.x - 0.5) * light.width;
    float y = (randomPoints.y - 0.5) * light.depth;
    float3 samplePos = center + basis.tangent * x + basis.bitangent * y;

    // Direction from shading point to light sample
    float3 toLight = samplePos - origin;
    float distance = length(toLight);
    float3 lightDir = toLight / distance;

    ray r;
    r.origin = origin;
    r.direction = lightDir;
    r.min_distance = 0.001;
    r.max_distance = distance;

    return r;
}

float calculateSquareLightPdf(float3 origin, SquareLightGPU light, float3 direction) {
    // Calculate the distance from the point to the light center
    float3 lightNormal = float3(0, -1, 0); // TODO: either store or gen normal properly
    float3 toLight = light.center - origin;
    float distance = length(toLight);
    // Calculate the cosine of the angle between the direction and the light normal
    float cosTheta = max(0.0, dot(-direction, lightNormal));
    // Area of the square light
    float area = light.width * light.depth;
    // PDF for area light: (distance^2) / (area * cosTheta)
    return (distance * distance) / (area * cosTheta + 1e-6);

    // float denom = dot(direction, lightNormal);
    // if (denom >= -1e-6) return 0.0; // Ray parallel to or pointing away from light plane
    
    // // Find distance to intersection with the light's plane
    // float t = dot(light.center - origin, lightNormal) / denom;
    // if (t <= 0.0) return 0.0; // Intersection behind origin
    
    // // 2. Find the hit point
    // float3 hitPoint = origin + t * direction;
    
    // // 3. Check if hit point is within light bounds
    // OrthonormalBasisGPU basis = buildOrthonormalBasis(lightNormal);
    // float3 toHitPoint = hitPoint - light.center;
    // float dx = abs(dot(toHitPoint, basis.tangent));
    // float dy = abs(dot(toHitPoint, basis.bitangent));
    
    // if (dx > light.width/2.0 || dy > light.depth/2.0) {
    //     return 0.0; // Hit point outside light bounds
    // }
    
    // // 4. Calculate the actual PDF
    // float area = light.width * light.depth;
    // float cosTheta = max(0.0, -dot(direction, lightNormal));
    
    // // This is the correct PDF calculation - using t (distance to actual hit point)
    // return (t * t) / (area * cosTheta + 1e-6);
}

ray cosineWeightedRay(float3 origin, float3 normal, float2 randomPoint) {
    float phi = 2.0 * M_PI_F * randomPoint.x;
    float cosTheta = sqrt(randomPoint.y);
    float sinTheta = sqrt(1.0 - randomPoint.y);
    
    OrthonormalBasisGPU basis = buildOrthonormalBasis(normal);
    float3 direction = normalize(
        basis.tangent * (cos(phi) * sinTheta) +
        basis.bitangent * (sin(phi) * sinTheta) +
        normal * cosTheta
    );
    
    ray r;
    r.origin = origin;
    r.direction = direction;
    r.min_distance = 0.001;
    r.max_distance = 1000.0;
    
    return r;
}

float calculateCosineWeightedPdf(float3 normal, float3 direction) {
    // Cosine-weighted PDF: cos(theta) / pi
    float cosTheta = max(0.0, dot(normal, direction));
    return cosTheta / M_PI_F;
}

ray vndfRay(float3 origin, float3 viewDir, float3 normal, float roughness, float2 randomPoints) {
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
    
    float phi = 2.0 * M_PI_F * randomPoints.x; // angle in the hemisphere
    
    // Transform Ve to the hemisphere configuration
    float lenVe = length(Ve);
    float cosThetaMax = lenVe / sqrt(1.0 + lenVe * lenVe);
    float cosTheta = cosThetaMax + (1.0 - cosThetaMax) * randomPoints.y;
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

    ray r;
    r.origin = origin; 
    r.direction = direction;
    r.min_distance = 0.001;
    r.max_distance = 1000.0;
    return r;
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

float3 getTriangleNormal(device const float3* vertices, uint primitiveId) {
    // Assuming vertices are in the order v0, v1, v2 for each triangle
    float3 v0 = vertices[primitiveId * 3 + 0];
    float3 v1 = vertices[primitiveId * 3 + 1];
    float3 v2 = vertices[primitiveId * 3 + 2];
    
    // Calculate the normal using cross product
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    return normalize(cross(edge1, edge2));
}

IntersectionGPU getClosestIntersection(primitive_acceleration_structure accelerationStructure, 
device const MaterialGPU* materials, device const float3* vertices, ray incomingRay) {
    // configures how intersection test behaves
    intersection_params intersectionParams;
    intersectionParams.assume_geometry_type(geometry_type::triangle); 
    intersectionParams.force_opacity(forced_opacity::opaque); // treats geometry as solid not transparent
    intersectionParams.accept_any_intersection(false); // find closest intersection

    // intersection query object that can be initialised with ray, accel struct and params
    // performs actual intersection test. stores results of test
    intersection_query<triangle_data> intersectionQuery;
    // init the intersection query with data
    intersectionQuery.reset(incomingRay, accelerationStructure, intersectionParams);
    // performs the actual intersection
    intersectionQuery.next();

    // intersector is metals built in ray tracing class that performs ray/geometry intersections
    // triangle_data template param that tells us what kind of geometry we are intersecting
    // different geometries have different result structures so we need to get the type
    intersector<triangle_data>::result_type intersectionResult;
    // pull required info about the committed intersection.
    intersectionResult.type = intersectionQuery.get_committed_intersection_type(); // triangle, boundingbox, nothing, curve if implemented
    intersectionResult.distance = intersectionQuery.get_committed_distance();
    intersectionResult.primitive_id = intersectionQuery.get_committed_primitive_id();

    MaterialGPU intersectionMaterial = materials[intersectionResult.primitive_id];

    IntersectionGPU intersection;

    // if no intersection / miss
    if (intersectionResult.type != intersection_type::triangle) {
        intersection.type = Miss;
        return intersection;
    } else {
        // check if light intersection
        if (length(intersectionMaterial.emissive) > 0.0) {
            intersection.type = HitLight;
            intersection.material = intersectionMaterial;
            intersection.point = incomingRay.origin + incomingRay.direction * intersectionResult.distance;
            return intersection;
        } else {
            // hit geometry
            intersection.type = Hit;
            intersection.point = incomingRay.origin + incomingRay.direction * intersectionResult.distance;
            intersection.ray = incomingRay;
            intersection.normal = getTriangleNormal(vertices, intersectionResult.primitive_id);
            intersection.material = intersectionMaterial;
            return intersection;
        }
    }
}

float balancedHeuristic(float pdf1, float pdf2, float pdf3) {
    // Balanced heuristic for three strategies
    if (pdf1 == 0.0) return 0.0;
    float sum = pdf1 + pdf2 + pdf3;
    return pdf1 / sum;
}


float3 calculateDirectLightSamplingContribution(device const MaterialGPU* materials, SquareLightGPU light,
device const float3* vertices, primitive_acceleration_structure accelerationStructure, uint2 index,
IntersectionGPU incomingIntersection, float2 randomPoints, uint samplesPerStrategy, bool usePowerHeuristic = true) {
    float3 directLight = float3(0.0, 0.0, 0.0);
    ray lightRay = directSquareLightRay(incomingIntersection.point + incomingIntersection.normal * 1e-4, light, randomPoints);
    float directLightPDF = calculateSquareLightPdf(incomingIntersection.point, light, lightRay.direction);
    float cosinePDF = calculateCosineWeightedPdf(incomingIntersection.normal, lightRay.direction);
    float vndfPDF = calculateVNDFPdf(-incomingIntersection.ray.direction, incomingIntersection.normal, lightRay.direction, incomingIntersection.material.roughness);

    IntersectionGPU lightIntersection = getClosestIntersection(accelerationStructure, materials, vertices, 
    lightRay);
    if (lightIntersection.type == HitLight) {
        float3 brdfContribution = calculateBRDFContribution(incomingIntersection.ray, incomingIntersection.normal, incomingIntersection.material, lightRay.direction);
        if (usePowerHeuristic) {
            // float weight = balancedHeuristic(directLightPDF, cosinePDF, vndfPDF);
            float weight = powerHeuristic(directLightPDF, cosinePDF, vndfPDF, samplesPerStrategy, 1.0);
            directLight += weight * brdfContribution * light.emittedRadiance / directLightPDF;
        } else {
            directLight += brdfContribution * light.emittedRadiance / directLightPDF;
        }
    }
    return directLight;
}

float3 recursiveMultiImportanceSampling(device const MaterialGPU* materials, SquareLightGPU light, 
device const float3* vertices, primitive_acceleration_structure accelerationStructure, uint2 index, 
IntersectionGPU incomingIntersection, uint samples, uint bounces, float3 throughput = float3(1.0, 1.0, 1.0)) {
    uint samplesPerStrategy = samples / 3;

    float3 directLight = float3(0.0, 0.0, 0.0);
    float3 cosine = float3(0.0, 0.0, 0.0);
    float3 vndf = float3(0.0, 0.0, 0.0);

    // direct light sampling
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // generate random points for sampling
        // float2 u = hashRandom(index, i);
        float2 u = haltonRandom(i, 0);
        // float2 u = haltonRandom(index.y * 800 + index.x + i, 0);
        directLight += calculateDirectLightSamplingContribution(materials, light, vertices, accelerationStructure, index, incomingIntersection, u, samplesPerStrategy);

    }
    // cosine-weighted hemisphere sampling
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // float2 u = hashRandom(index, i + samplesPerStrategy);
        float2 u = haltonRandom(i + samplesPerStrategy, 2);
        // float2 u = haltonRandom(index.y * 800 + index.x + i + samplesPerStrategy, 2);

        // generate cosine-weighted ray using random points
        ray lightRay = cosineWeightedRay(incomingIntersection.point + incomingIntersection.normal * 1e-4, incomingIntersection.normal, u);
        float cosinePDF = calculateCosineWeightedPdf(incomingIntersection.normal, lightRay.direction);
        float directLightPDF = calculateSquareLightPdf(incomingIntersection.point, light, lightRay.direction);
        float vndfPDF = calculateVNDFPdf(-incomingIntersection.ray.direction, incomingIntersection.normal, lightRay.direction, incomingIntersection.material.roughness);

        // float weight = balancedHeuristic(cosinePDF, directLightPDF, vndfPDF);
        float weight = powerHeuristic(cosinePDF, directLightPDF, vndfPDF, samplesPerStrategy, 1.0);

        IntersectionGPU cosineIntersection = getClosestIntersection(accelerationStructure, materials, vertices,
        lightRay);
        float3 brdfContribution = calculateBRDFContribution(incomingIntersection.ray, incomingIntersection.normal, incomingIntersection.material, lightRay.direction);
        if (cosineIntersection.type == HitLight) {
            // cosine += brdfContribution * cosineIntersection.material.emissive / cosinePDF;
            cosine += weight * brdfContribution * light.emittedRadiance / cosinePDF;
        } else if (cosineIntersection.type == Hit) {
            // float2 u2 = haltonRandom(i + samplesPerStrategy, 6);
            float2 u2 = haltonRandom(i, 6);
            // float2 u2 = haltonRandom(index.y * 800 + index.x + i, 6);
            // float2 u2 = hashRandom(index, i + 3 * samplesPerStrategy);


            cosine += brdfContribution / cosinePDF * calculateDirectLightSamplingContribution(materials, light, vertices, accelerationStructure, index, cosineIntersection, u2, 1, false);
        }
    }
    // VNDF sampling
    for (uint i = 0; i < samplesPerStrategy; i++) {
        // float2 u = hashRandom(index, i + 2 * samplesPerStrategy);
        float2 u = haltonRandom(i + 2 * samplesPerStrategy, 4);
        // float2 u = haltonRandom(i + 2 * samplesPerStrategy, 4);
        // float2 u = haltonRandom(index.y * 800 + index.x + i + 2 * samplesPerStrategy, 4);

        ray lightRay = vndfRay(incomingIntersection.point + incomingIntersection.normal * 1e-4, -incomingIntersection.ray.direction,
        incomingIntersection.normal, incomingIntersection.material.roughness, u);
        float vndfPDF = calculateVNDFPdf(-incomingIntersection.ray.direction, incomingIntersection.normal,
        lightRay.direction, incomingIntersection.material.roughness);
        float cosinePDF = calculateCosineWeightedPdf(incomingIntersection.normal, lightRay.direction);
        float directLightPDF = calculateSquareLightPdf(incomingIntersection.point, light, lightRay.direction);
        // float weight = balancedHeuristic(vndfPDF, cosinePDF, directLightPDF);
        float weight = powerHeuristic(vndfPDF, directLightPDF, cosinePDF, samplesPerStrategy, 1.0);

        IntersectionGPU vndfIntersection = getClosestIntersection(accelerationStructure, materials, vertices, 
        lightRay);
        float3 brdfContribution = calculateBRDFContribution(incomingIntersection.ray, incomingIntersection.normal, incomingIntersection.material, lightRay.direction);
        if (vndfIntersection.type == HitLight) {
            // vndf += brdfContribution * vndfIntersection.material.emissive / vndfPDF; 
            vndf += weight * brdfContribution * light.emittedRadiance / vndfPDF; 
        } else if (vndfIntersection.type == Hit) {
            // float2 u2 = haltonRandom(i + 2 * samplesPerStrategy, 6);
            // float2 u2 = haltonRandom(i + 2 * samplesPerStrategy, 5);
            float2 u2 = haltonRandom(i + samplesPerStrategy, 6);
            // float2 u2 = haltonRandom(index.y * 800 + index.x + i, 6);
            // float2 u2 = hashRandom(index, i + 4 * samplesPerStrategy);

            vndf += brdfContribution / vndfPDF * calculateDirectLightSamplingContribution(materials, light, vertices, accelerationStructure, index, vndfIntersection, u2, 1, false);
        }
    }
    return (directLight + cosine + vndf) / float(samplesPerStrategy); // combine all contributions
}

void writeToTextBuffer(device float* textBuffer, float3 color, uint2 index, uint width) {
    // Each pixel needs 3 floats (R, G, B)
    int pixelOffset = (index.y * width + index.x) * 3;
    textBuffer[pixelOffset + 0] = color.r;
    textBuffer[pixelOffset + 1] = color.g;
    textBuffer[pixelOffset + 2] = color.b;
}

kernel void drawTriangle(device const CameraGPU* cameras, device const MaterialGPU * materials, 
device const SquareLightGPU* squareLights, device const float3* vertices, device uchar* pixels, 
primitive_acceleration_structure accelerationStructure [[buffer(5)]], device float* textBuffer [[buffer(6)]], 
uint2 index [[thread_position_in_grid]]) {
    CameraGPU camera = cameras[0];
    SquareLightGPU light = squareLights[0];
    // check if index is within camera resolution bounds
    if (int(index.x) >= camera.resolution.x || int(index.y) >= camera.resolution.y) return;

    uint cameraRaysPerPixel = 6;
    float3 accumulatedColor = float3(0.0, 0.0, 0.0);

    // ray r = generateCameraRay(camera, index, jitter);
    uint misSamples = 300;
    uint bounces = 2;


    for (uint i = 0; i < cameraRaysPerPixel; i++) {
        //get jitter values
        float2 pixelJitter = hashRandom(index, i);
        // float2 pixelJitter = hammersley2D(i + 1 * cameraRaysPerPixel, cameraRaysPerPixel);
        // gen intial ray
        ray r = generateCameraRay(camera, index, pixelJitter);

        // handle camera ray intersection
        IntersectionGPU intersection = getClosestIntersection(accelerationStructure, materials, vertices, r);
        // float4 finalColor = float4(0.0, 0.0, 0.0, 1.0);
        // float3 col = (8 + intersection.point.y) / 14.0; // just for testing, remove later
        // float4 finalColor = float4(col, 1.0); // just for testing, remove later

        if (intersection.type == Miss) {
            // no intersection leave color black.
        } else if (intersection.type == HitLight) {
            // hit light, set pixel to light color
            float3 lightColor = light.emittedRadiance;
            accumulatedColor += lightColor;
            // finalColor = reinhartToneMapping(lightColor * cameraExposure(camera));
        } else {
            // hit geometry, recursive ray trace
            float3 sampledColor = recursiveMultiImportanceSampling(materials, light, vertices, accelerationStructure, 
            index, intersection, misSamples, bounces);
            accumulatedColor += sampledColor;
            // finalColor = reinhartToneMapping(sampledColor * cameraExposure(camera));
        }
    }

    // float3 hdrColor = accumulatedColor / float(cameraRaysPerPixel);
    // float luminance = dot(hdrColor, float3(0.2126, 0.7152, 0.0722));
    // float exposedLuminance = 683 * luminance * cameraExposure(camera);
    // float toneMappedLuminance = exposedLuminance / (exposedLuminance + 1.0); // Reinhard on luminance
    // float3 toneMappedColor = (hdrColor / (luminance + 1e-6)) * toneMappedLuminance;

    // accumulated color gets mapped into 
    float3 exposedColor = accumulatedColor / float(cameraRaysPerPixel) * cameraExposure(camera);
    float3 toneMappedColor = reinhartToneMapping(exposedColor);
    

    // float3 sRGBlo = exposedColor * 12.92;
    // float3 sRGBHi = (pow(abs(exposedColor), 1.0/2.4) * 1.055) - 0.055;
    // // float3 sRGB = (toneMappedColor <= float3(0.0031308)) ? sRGBlo : sRGBHi;
    // float3 sRGB;
    // if (exposedColor.x <= 0.0031308 || exposedColor.y <= 0.0031308 || exposedColor.z <= 0.0031308) {
    //     sRGB = sRGBlo;
    // } else {
    //     sRGB = sRGBHi;
    // }

    float gamma = 2.2;
    float3 gammaCorrectedColor = pow(toneMappedColor, float3(1.0 / gamma, 1.0 / gamma, 1.0 / gamma));
    
    writeToTextBuffer(textBuffer, accumulatedColor, index, camera.resolution.x);
    writeToPixelBuffer(pixels, camera, index, gammaCorrectedColor);
}
