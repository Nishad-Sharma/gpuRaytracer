//
//  raytrace.metal
//  RTrace
//
//  Created by Nishad Sharma on 9/7/2025.
//
#pragma once
#import "sampling.metal"
#import "shaderTypes.h"

kernel void pathTrace(device const CameraGPU* cameras [[buffer(0)]], 
    device const MaterialGPU* materials [[buffer(1)]], 
    device const SquareLightGPU* squareLights [[buffer(2)]], 
    device const float3* vertices [[buffer(3)]],
    texture2d<float, access::write> outputTexture [[texture(0)]],
    texture2d<unsigned int, access::read> randomTexture [[texture(1)]],
    primitive_acceleration_structure accelerationStructure [[buffer(5)]],
    device float* debugBuffer [[buffer(6)]],
    uint3 index [[thread_position_in_grid]])
    {
    
    SquareLightGPU light = squareLights[0];
    
    uint samples = 400;
    int bounces = 3;
    intersector<triangle_data> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    
    typename intersector<triangle_data>::result_type intersection;
    
    float3 luminance = float3(0.0);
    // gen $samples camera rays
    for (uint n = 0; n < samples; n++) {
        // generate one camera ray
//        float2 uv = hashRandom3D(index, 1);
        unsigned int offset = randomTexture.read(index.xy).x;
        
        float2 uv = float2(halton(offset + n, 0),
                           halton(offset + n, 1));
        
        ray r = generateCameraRay(cameras[0], index, uv);
        
        float3 accumulatedColor = float3(0.0);
        float3 color = float3(1.0);
        
        for (int bounce = 0; bounce < bounces; bounce++) {
            i.accept_any_intersection(false);
            intersection = i.intersect(r, accelerationStructure);
            
            if (intersection.type == intersection_type::none) {
                break;
            }
                
            MaterialGPU intersectionMaterial = materials[intersection.primitive_id];
            
            if (length(intersectionMaterial.emissive) > 0.0) {
                // hit light, sample and stop
                accumulatedColor = intersectionMaterial.emissive;
                break;
            } else {
                // hit non light
                // sample light
                // update ray using MIS
                
                float3 normal = getTriangleNormal(vertices, intersection.primitive_id);
                float3 intersectionPoint = r.origin + r.direction * intersection.distance + normal * 1e-3f;
                
                // sample light
                float3 lightDirection;
                float lightDistance;
                float2 w = float2(halton(offset + n, 2 + bounce * 5 + 0),
                                  halton(offset + n, 2 + bounce * 5 + 1));
                float3 lightColor = sampleAreaLight(light, w, intersectionPoint, lightDirection, lightDistance);
                lightColor *= saturate(dot(normal, lightDirection));
                color *= intersectionMaterial.diffuse.xyz;
                
                // add shadow ray
                ray shadowRay;
                shadowRay.origin = intersectionPoint;
                shadowRay.direction = lightDirection;
                shadowRay.max_distance = lightDistance - 1e-3f;
                i.accept_any_intersection(true);
                
                intersection = i.intersect(shadowRay, accelerationStructure);
                
                if (intersection.type == intersection_type::none) {
                    accumulatedColor += lightColor * color;
                }
                
                //sample cosine only
//                float2 u = hashRandom3D(index, bounce);
                float2 u = float2(halton(offset + n, 2 + bounce * 5 + 2),
                                  halton(offset + n, 2 + bounce * 5 + 3));
                float3 sampleDirection = sampleCosineWeightedHemisphere(u);
                
                sampleDirection = alignHemisphereWithNormal(sampleDirection, normal);
                
                r.origin = intersectionPoint;
                r.direction = sampleDirection;
            }
        }
        luminance += accumulatedColor;
    }
    // divide luminance by samples
    luminance /= samples;
    //tonemap?
    //write to buffer
    outputTexture.write(float4(luminance, 1.0), index.xy);
    
}
