//
//  image.swift
//  gpuRaytracer
//
//  Created by Nishad Sharma on 5/6/2025.
//

import CoreGraphics
import ImageIO
import Foundation
import UniformTypeIdentifiers

import Metal

func saveTextureToImage(texture: MTLTexture, fileName: String) {
    let width = texture.width
    let height = texture.height
    
    // Ensure the texture format is what we expect.
    guard texture.pixelFormat == .rgba16Float else {
        print("Error: saveTextureToImage expects a texture with pixel format rgba16Float, but got \(texture.pixelFormat).")
        return
    }
    
    // 1. Create a buffer to hold the 16-bit float data, matching the texture format.
    let bytesPerPixel = MemoryLayout<Float16>.size * 4 // 4 channels (R,G,B,A) of 16-bit floats
    let bytesPerRow = width * bytesPerPixel
    let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: width, height: height, depth: 1))
    
    // Create an array of Float16 to receive the data.
    var float16Data = [Float16](repeating: 0, count: width * height * 4)
    
    // Copy texture data into the Float16 array.
    texture.getBytes(&float16Data,
                     bytesPerRow: bytesPerRow,
                     from: region,
                     mipmapLevel: 0)
    
    // 2. Convert float data to 8-bit with simple exposure and gamma correction
    let exposure: Float = 4.0  // Adjust based on your scene brightness
    let gamma: Float = 2.2
    
    var rgbaData = [UInt8](repeating: 0, count: width * height * 4)
    
    for i in 0..<(width * height) {
        let pixelIndex = i * 4
        
        for c in 0..<3 {  // R, G, B channels
            // Convert Float16 to Float for calculations
            var value = Float(float16Data[pixelIndex + c])
            
            // Apply exposure and tonemapping/gamma correction
            value *= exposure
            value = value / (value + 1.0) // Simple Reinhard tonemapping
            value = pow(value, 1.0 / gamma)
            
            // Clamp to 0-1 range and convert to 8-bit
            value = max(0.0, min(1.0, value))
            rgbaData[pixelIndex + c] = UInt8(value * 255.0)
        }
        
        // Alpha channel (always 1.0)
        rgbaData[pixelIndex + 3] = 255
    }
    
    // 3. Create a CGImage
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    
    guard let provider = CGDataProvider(data: Data(rgbaData) as CFData),
          let cgImage = CGImage(width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bitsPerPixel: 32,
                              bytesPerRow: width * 4,
                              space: colorSpace,
                              bitmapInfo: bitmapInfo,
                              provider: provider,
                              decode: nil,
                              shouldInterpolate: false,
                              intent: .defaultIntent) else {
        print("Failed to create CGImage")
        return
    }
    
    // 4. Save to file
    let fileURL = URL(fileURLWithPath: fileName)
    
    if let destination = CGImageDestinationCreateWithURL(fileURL as CFURL, UTType.png.identifier as CFString, 1, nil) {
        CGImageDestinationAddImage(destination, cgImage, nil)
        if CGImageDestinationFinalize(destination) {
            print("Image saved successfully to \(fileName)")
        } else {
            print("Failed to save image")
        }
    } else {
        print("Failed to create image destination")
    }
}

func savePixelArrayToImage(pixels: [UInt8], width: Int, height: Int, fileName: String) {
    let bitsPerComponent = 8
    let bytesPerPixel = 4 // Assuming RGBA
    let bytesPerRow = width * bytesPerPixel
    
    // Create color space
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
        print("Failed to create color space")
        return
    }
    
    // Create data provider from pixel array
    let data = Data(pixels)
    guard let dataProvider = CGDataProvider(data: data as CFData) else {
        print("Failed to create data provider")
        return
    }
    
    // Create CGImage from raw pixel data
    guard let image = CGImage(
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bitsPerPixel: bitsPerComponent * bytesPerPixel,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: dataProvider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    ) else {
        print("Failed to create CGImage")
        return
    }

    // Save to file
    let fileURL = URL(fileURLWithPath: fileName)
    guard let destination = CGImageDestinationCreateWithURL(
        fileURL as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        print("Failed to create destination")
        return
    }
    
    CGImageDestinationAddImage(destination, image, nil)
    
    if CGImageDestinationFinalize(destination) {
        print("Image saved to: \(fileURL.path)")
    } else {
        print("Failed to save image")
    }
}

// Example usage with a simple gradient
func createGradientPixels(width: Int, height: Int) -> [UInt8] {
    var pixels: [UInt8] = []
    
    for y in 0..<height {
        for x in 0..<width {
            let red = UInt8(255 * x / width)     // R
            let green = UInt8(255 * y / height)  // G
            let blue: UInt8 = 128                // B
            let alpha: UInt8 = 255               // A
            
            pixels.append(red)
            pixels.append(green)
            pixels.append(blue)
            pixels.append(alpha)
        }
    }
    
    return pixels
}

// // Usage
// let width = 256
// let height = 256
// let pixels = createGradientPixels(width: width, height: height)
// savePixelArrayToImage(pixels: pixels, width: width, height: height, fileName: "/tmp/gradient.png")
