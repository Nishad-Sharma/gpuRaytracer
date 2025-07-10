//
//  scene.swift
//  RTrace
//
//  Created by Nishad Sharma on 9/7/2025.
//

struct Scene {
    let camera: Camera
    let light: SquareLight
    let triangles: [Triangle]
}

func initCornellBox() -> Scene {
    // Position camera to look into the room from the front
    let direction = simd_normalize(simd_float3(0, 0, -2.5) - simd_float3(0, 0, 9))
    let camera = Camera(position: simd_float3(0, 0, 9), direction: direction,
        resolution: simd_int2(800, 600), horizontalFov: Float.pi / 4.0)

    let roomSize: Float = 5.0
    let half = roomSize / 2.0

    let lightWidth: Float = 1.0
    let lightDepth: Float = 1.0
    let lightY: Float = half - 0.01
    let lightCenter = simd_float3(0, lightY, 0)
    let halfW = lightWidth / 2
    let halfD = lightDepth / 2

    // Vertices for the two triangles (rectangle split into two)
    let v0 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z - halfD)
    let v1 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z - halfD)
    let v2 = simd_float3(lightCenter.x + halfW, lightY, lightCenter.z + halfD)
    let v3 = simd_float3(lightCenter.x - halfW, lightY, lightCenter.z + halfD)

    // Emissive material for the light triangles
    let lightMaterial = Material(
        diffuse: simd_float4(1.0, 0.95, 0.9, 1.0),
        metallic: 0.0,
        roughness: 0.0,
        emissive: simd_float3(1.0, 1.0, 1.0) // White light
        // emissive: simd_float3(20, 20, 20) // Adjust intensity as needed
    )

    let squareLight = SquareLight(
        center: lightCenter,
        vertices: [v0, v1, v2, v3],
        material: lightMaterial,
        LightType: .bulb(luminousEfficacy: 100.0, watts: 12), // Example values
        // LightType: .bulb(efficacy: 150, watts: 200), // Example values
        width: lightWidth,
        depth: lightDepth
    )

    var cornellBoxTriangles = createCornellBoxScene()

    // add triangles for light
    cornellBoxTriangles.append(Triangle(vertices: [v0, v1, v2], material: lightMaterial))
    cornellBoxTriangles.append(Triangle(vertices: [v0, v2, v3], material: lightMaterial))
    
    return Scene(camera: camera, light: squareLight, triangles: cornellBoxTriangles)
}

func createCornellBoxScene() -> [Triangle] {
    var triangles: [Triangle] = []
    
    // Cornell Box dimensions
    let roomSize: Float = 5.0
    let half = roomSize / 2.0
    
    // Materials for Cornell Box
    let redMaterial = Material(diffuse: simd_float4(0.9, 0.0, 0.0, 1), metallic: 0.05, roughness: 0.3)    // Left wall
    let greenMaterial = Material(diffuse: simd_float4(0.0, 0.7, 0.0, 1), metallic: 0.05, roughness: 0.8)  // Right wall
    let whiteMaterial = Material(diffuse: simd_float4(0.9, 0.9, 0.9, 1), metallic: 0.05, roughness: 0.8)  // Other walls
    let diffuseBoxMaterial = Material(diffuse: simd_float4(0.9, 0.9, 0.9, 1), metallic: 0.05, roughness: 0.3)
    let specularBoxMaterial = Material(diffuse: simd_float4(0.9, 0.9, 0.9, 1), metallic: 0.9, roughness: 0.3)
    // let blueMaterial = Material(diffuse: simd_float4(0.25, 0.25, 0.75, 1), metallic: 0.3, roughness: 0.6)   // Tall box
    // let yellowMaterial = Material(diffuse: simd_float4(0.75, 0.75, 0.25, 1), metallic: 0.3, roughness: 0.6) // Short cube
    
    // Back wall (z = -half) - facing INTO room (positive Z direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // bottom-left
        simd_float3(half, half, -half),    // top-right
        simd_float3(-half, half, -half)  // top-left
    ], material: whiteMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // bottom-left
        simd_float3(half, -half, -half),   // bottom-right
        simd_float3(half, half, -half)   // top-right
    ], material: whiteMaterial))
    
    // Left wall (x = -half) - RED - facing INTO room (positive X direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-bottom
        simd_float3(-half, half, half),    // front-top
        simd_float3(-half, -half, half)  // front-bottom
    ], material: redMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-bottom
        simd_float3(-half, half, -half),   // back-top
        simd_float3(-half, half, half)   // front-top
    ], material: redMaterial))
    
    // Right wall (x = +half) - GREEN - facing INTO room (negative X direction)
    triangles.append(Triangle(vertices: [
        simd_float3(half, -half, -half),  // back-bottom
        simd_float3(half, half, half),     // front-top
        simd_float3(half, half, -half)   // back-top
    ], material: greenMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(half, -half, -half),  // back-bottom
        simd_float3(half, -half, half),    // front-bottom
        simd_float3(half, half, half)    // front-top
    ], material: greenMaterial))
    
    // Floor (y = -half) - facing UP into room (positive Y direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-left
        simd_float3(half, -half, half),    // front-right
        simd_float3(half, -half, -half)  // back-right
    ], material: whiteMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, -half, -half), // back-left
        simd_float3(-half, -half, half),   // front-left
        simd_float3(half, -half, half)   // front-right
    ], material: whiteMaterial))
    
    // Ceiling (y = +half) - facing DOWN into room (negative Y direction)
    triangles.append(Triangle(vertices: [
        simd_float3(-half, half, -half),  // back-left
        simd_float3(half, half, half),     // front-right
        simd_float3(-half, half, half)   // front-left
    ], material: whiteMaterial))
    triangles.append(Triangle(vertices: [
        simd_float3(-half, half, -half),  // back-left
        simd_float3(half, half, -half),    // back-right
        simd_float3(half, half, half)    // front-right
    ], material: whiteMaterial))
  
    // back rectanglular prism
    let tallBoxWidth: Float = 1.2
    let tallBoxHeight: Float = 2.8
    let tallBoxDepth: Float = 1.2
    let tallBoxPosition = simd_float3(-1, -half + tallBoxHeight/2 - 0.05, -1.5) // Back left
    let tallBoxRotationY: Float = Float.pi / 2.4
    
    let tallBoxVertices = createRotatedBoxVertices(
        center: tallBoxPosition,
        width: tallBoxWidth,
        height: tallBoxHeight,
        depth: tallBoxDepth,
        rotationY: tallBoxRotationY
    )
    // triangles.append(contentsOf: createBoxTriangles(vertices: tallBoxVertices, material: specularBoxMaterial))
    triangles.append(contentsOf: createBoxTriangles(vertices: tallBoxVertices, material: diffuseBoxMaterial))
    
    // front cube
    let shortBoxSize: Float = 1.2
    let shortBoxHeight: Float = 1.2
    let shortBoxPosition = simd_float3(0.7, -half + shortBoxHeight/2 - 0.05, 1.2) // Front right
    let shortBoxRotationY: Float = -Float.pi / 2.5
    
    let shortBoxVertices = createRotatedBoxVertices(
        center: shortBoxPosition,
        width: shortBoxSize,
        height: shortBoxHeight,
        depth: shortBoxSize,
        rotationY: shortBoxRotationY
    )
    
    // triangles.append(contentsOf: createBoxTriangles(vertices: shortBoxVertices, material: specularBoxMaterial))
    triangles.append(contentsOf: createBoxTriangles(vertices: shortBoxVertices, material: diffuseBoxMaterial))

    return triangles
}

func createRotatedBoxVertices(center: simd_float3, width: Float, height: Float, depth: Float, rotationY: Float) -> [simd_float3] {
    let halfW = width / 2
    let halfH = height / 2
    let halfD = depth / 2
    
    // Create box vertices centered at origin
    let baseVertices = [
        simd_float3(-halfW, -halfH, -halfD), // 0
        simd_float3( halfW, -halfH, -halfD), // 1
        simd_float3( halfW,  halfH, -halfD), // 2
        simd_float3(-halfW,  halfH, -halfD), // 3
        simd_float3(-halfW, -halfH,  halfD), // 4
        simd_float3( halfW, -halfH,  halfD), // 5
        simd_float3( halfW,  halfH,  halfD), // 6
        simd_float3(-halfW,  halfH,  halfD), // 7
    ]
    
    // Rotation matrix around Y-axis
    let cosY = cos(rotationY)
    let sinY = sin(rotationY)
    let rotationMatrix = simd_float4x4(
        simd_float4(cosY, 0, sinY, 0),
        simd_float4(0, 1, 0, 0),
        simd_float4(-sinY, 0, cosY, 0),
        simd_float4(0, 0, 0, 1)
    )
    
    // Apply rotation and translation
    return baseVertices.map { vertex in
        let homogeneous = simd_float4(vertex.x, vertex.y, vertex.z, 1.0)
        let rotated = rotationMatrix * homogeneous
        return simd_float3(rotated.x + center.x, rotated.y + center.y, rotated.z + center.z)
    }
}

func createBoxTriangles(vertices: [simd_float3], material: Material) -> [Triangle] {
    var triangles: [Triangle] = []
    
    // Back face
    triangles.append(Triangle(vertices: [vertices[0], vertices[2], vertices[1]], material: material))
    triangles.append(Triangle(vertices: [vertices[0], vertices[3], vertices[2]], material: material))
    
    // Front face
    triangles.append(Triangle(vertices: [vertices[4], vertices[5], vertices[6]], material: material))
    triangles.append(Triangle(vertices: [vertices[4], vertices[6], vertices[7]], material: material))
    
    // Left face
    triangles.append(Triangle(vertices: [vertices[0], vertices[4], vertices[7]], material: material))
    triangles.append(Triangle(vertices: [vertices[0], vertices[7], vertices[3]], material: material))
    
    // Right face
    triangles.append(Triangle(vertices: [vertices[1], vertices[6], vertices[5]], material: material))
    triangles.append(Triangle(vertices: [vertices[1], vertices[2], vertices[6]], material: material))
    
    // Bottom face
    triangles.append(Triangle(vertices: [vertices[0], vertices[5], vertices[4]], material: material))
    triangles.append(Triangle(vertices: [vertices[0], vertices[1], vertices[5]], material: material))
    
    // Top face
    triangles.append(Triangle(vertices: [vertices[3], vertices[6], vertices[2]], material: material))
    triangles.append(Triangle(vertices: [vertices[3], vertices[7], vertices[6]], material: material))
    
    return triangles
}

struct Triangle {
    var vertices: [simd_float3]
    var material: Material
}

// square light made up of two triangles
struct SquareLight {
    var center: simd_float3
    var vertices: [simd_float3]
    var material: Material
    var LightType: LightType
    var width: Float
    var depth: Float

    // Convert to radiance for outgoing rays
    var emittedLuminance: simd_float3 {
        switch LightType {
        case .bulb(let luminousEfficacy, let watts):
            let luminousFlux = luminousEfficacy * watts// lm
            return calculateLuminance(luminousFlux : luminousFlux)
        }
    }

    func calculateLuminance(luminousFlux : Float) -> simd_float3 {
        let area = width * depth // m²
        let luminanceExitance = luminousFlux  / area // lm/m²
        let luminance = luminanceExitance / Float.pi // lm/(sr·m²)
        return simd_float3(material.diffuse.x, material.diffuse.y, material.diffuse.z) * luminance
    }
}

enum LightType {
    case bulb(luminousEfficacy: Float, watts: Float)
}

struct Material {
    var diffuse: simd_float4
    var metallic: Float
    var roughness: Float
    var emissive: simd_float3 = simd_float3(0, 0, 0)
}

struct Sphere {
    var center: simd_float3
    var material: Material
    var radius: Float
}
    
struct Camera {
    var position: simd_float3
    var direction: simd_float3
    var up: simd_float3 = simd_float3(0, 1, 0) // assuming camera's up vector is positive y-axis
    var resolution: simd_int2
    var horizontalFov: Float // field of view in radians
    var ev100: Float = 5.0 // lower ev100 makes light brighter, doesnt seem to impact rest of scene
    
    func pixelCount() -> Int {
        return Int(resolution.x * resolution.y)
    }
}
