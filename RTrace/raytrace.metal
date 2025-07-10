//
//  raytrace.metal
//  RTrace
//
//  Created by Nishad Sharma on 9/7/2025.
//
#pragma once
#import "sampling.metal"
#import "shaderTypes.h"

kernel void pathTrace(device const CameraGPU* cameras, device const MaterialGPU * materials,
                      device const SquareLightGPU* squareLights, device const float3* vertices, device uchar* pixels,
                      primitive_acceleration_structure accelerationStructure [[buffer(5)]], device float* textBuffer [[buffer(6)]],
                      uint3 index [[thread_position_in_grid]]) {
    
    SquareLightGPU light = squareLights[0];
    
    uint samples = 1000;
    uint bounces = 3;
    intersector<triangle_data> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    i.accept_any_intersection(false);
    typename intersector<triangle_data>::result_type intersection;
    
    float3 luminance = float3(0.0);
    // gen $samples camera rays
    for (uint n = 0; n < samples; n++) {
        // generate one camera ray
        
        float2 uv = hashRandom3D(index, 1);
        
        ray r = generateCameraRay(cameras[0], index, uv);
        
        float3 accumulatedColor = float3(1.0);
        
        for (int bounce = 0; bounce < 3; bounce++) {
            intersection = i.intersect(r, accelerationStructure);
            
            if (intersection.type == intersection_type::none) {
                break;
            }
                
            MaterialGPU intersectionMaterial = materials[intersection.primitive_id];
            
            if (length(intersectionMaterial.emissive) > 0.0) {
                // hit light, sample and stop
                luminance += intersectionMaterial.emissive * accumulatedColor;
                break;
            } else {
                // hit non light
                // sample light
                // update ray using MIS
                
                //sample cosine only
                float2 u = hashRandom3D(index, bounce);
                float3 sampleDirection = sampleCosineWeightedHemisphere(u);
                float3 normal = getTriangleNormal(vertices, intersection.primitive_id);
                sampleDirection = alignHemisphereWithNormal(sampleDirection, normal);
                
                accumulatedColor *= intersectionMaterial.diffuse.xyz;
                // calc brdfcontribution and sample at this point
                //update throughput
//                float3 brdfContribution = calculateBRDFContribution(r, normal, intersectionMaterial, sampleDirection);
                
                r.origin = r.origin + r.direction * intersection.distance + normal * 1e-3f;
                r.direction = sampleDirection;
            }
        }
    }
    // divide luminance by samples
    luminance /= samples;
    //tonemap?
    //write to buffer
    writeToPixelBuffer(pixels, cameras[0], index.xy, luminance);
    
}
