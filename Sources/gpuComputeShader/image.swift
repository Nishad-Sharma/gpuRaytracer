//
//  image.swift
//  gpuComputeShader
//
//  Created by Nishad Sharma on 5/6/2025.
//

import CoreGraphics
import ImageIO
import Foundation
import UniformTypeIdentifiers

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
