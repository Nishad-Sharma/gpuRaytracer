//
//  sampling.metal
//  RTrace
//
//  Created by Nishad Sharma on 9/7/2025.
//

#include <metal_stdlib>
#include <metal_raytracing>
#import "shaderTypes.h"

using namespace metal;
using namespace raytracing;


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

inline float3 sampleCosineWeightedHemisphere(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;

    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);

    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

inline float3 alignHemisphereWithNormal(float3 sample, float3 normal) {
    // Set the "up" vector to the normal
    float3 up = normal;

    // Find an arbitrary direction perpendicular to the normal, which becomes the
    // "right" vector.
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));

    // Find a third vector perpendicular to the previous two, which becomes the
    // "forward" vector.
    float3 forward = cross(right, up);

    // Map the direction on the unit hemisphere to the coordinate system aligned
    // with the normal.
    return sample.x * right + sample.y * up + sample.z * forward;
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

float2 hashRandom3D(uint3 index, uint i) {
    // Generate unique seeds from the 3D coordinates
    // Use prime multipliers to avoid patterns
    uint sampleId = (index.z * 1013 + index.y * 809 + index.x) * i;
    
    // Create two different seeds with different mixing patterns
    uint seed1 = hash(index.x + index.y * 809 + index.z * 929 + sampleId);
    uint seed2 = hash(index.z + index.x * 613 + index.y * 743 + sampleId + 12345);
    
    // Generate two random floats
    float u1 = randomFloat(seed1);
    float u2 = randomFloat(seed2);
    
    return float2(u1, u2);
}

ray generateCameraRay(CameraGPU camera, uint3 index, float2 pixelJitter) {
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
//     float s = (float(x) / float(camera.resolution.x)) * 2.0 - 1.0;
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

inline float3 sampleAreaLight(SquareLightGPU light,
                            float2 u,
                            float3 position,
                            thread float3 & lightDirection)
{
    // Map to -1..1
    u = u * 2.0f - 1.0f;

    float3 normal = float3(0.0f, -1.0f, 0.0f);
    float3 right = float3(0.25f, 0.0f, 0.0f);
    float3 up = float3(0.0f, 0.0f, 0.25f);
    // Transform into the light's coordinate system.
    float3 samplePosition = light.center +
                            right * u.x +
                            up * u.y;

    // Compute the vector from sample point on  the light source to intersection point.
    lightDirection = samplePosition - position;

    float lightDistance = length(lightDirection);

    float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);

    // Normalize the light direction.
    lightDirection *= inverseLightDistance;

    // Start with the light's color.
    float3 lightColor = light.color.xyz;

    // Light falls off with the inverse square of the distance to the intersection point.
    lightColor *= (inverseLightDistance * inverseLightDistance);

    // Light also falls off with the cosine of the angle between the intersection point
    // and the light source.
    lightColor *= saturate(dot(-lightDirection, normal));
    
    return lightColor;
}

//float3 calculateDirectLightSamplingContribution(device const MaterialGPU* materials, SquareLightGPU light,
//device const float3* vertices, primitive_acceleration_structure accelerationStructure, uint2 index,
//IntersectionGPU incomingIntersection, float2 randomPoints, uint samplesPerStrategy, bool usePowerHeuristic = true) {
//    float3 directLight = float3(0.0, 0.0, 0.0);
//    ray lightRay = directSquareLightRay(incomingIntersection.point + incomingIntersection.normal * 1e-4, light, randomPoints);
//    float directLightPDF = calculateSquareLightPdf(incomingIntersection.point, light, lightRay.direction);
//    float cosinePDF = calculateCosineWeightedPdf(incomingIntersection.normal, lightRay.direction);
//    float vndfPDF = calculateVNDFPdf(-incomingIntersection.ray.direction, incomingIntersection.normal, lightRay.direction, incomingIntersection.material.roughness);
//
//    IntersectionGPU lightIntersection = getClosestIntersection(accelerationStructure, materials, vertices,
//    lightRay);
//    if (lightIntersection.type == HitLight) {
//        float3 brdfContribution = calculateBRDFContribution(incomingIntersection.ray, incomingIntersection.normal, incomingIntersection.material, lightRay.direction);
//        if (usePowerHeuristic) {
//            // float weight = balancedHeuristic(directLightPDF, cosinePDF, vndfPDF);
//            float weight = powerHeuristic(directLightPDF, cosinePDF, vndfPDF, samplesPerStrategy, 1.0);
//            directLight += weight * brdfContribution * light.emittedRadiance / directLightPDF;
//        } else {
//            directLight += brdfContribution * light.emittedRadiance / directLightPDF;
//        }
//    }
//    return directLight;
//}

bool checkShadowRay() {
    intersector<triangle_data> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    i.accept_any_intersection(false);
    
    
    
    return false;
}

float3 calculateDirectLightContribution() {
    float3 lightContribution = float3(0.0);
//    ray toLight = directSquareLightRay(<#float3 origin#>, <#SquareLightGPU light#>, <#float2 randomPoints#>);
    
    
    return lightContribution;
}

