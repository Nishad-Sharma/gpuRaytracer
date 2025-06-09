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