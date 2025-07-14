 ![cornell raytrace image](https://github.com/Nishad-Sharma/gpuRaytracer/blob/main/Sources/gpuRaytracer/example.png)

Physically based, GPU acclerated path tracer using MIS and one bounce. 

run the following inside "Sources/gpuRaytracer" to compile a metallib:

```xcrun -sdk macosx metal -I../CHeaders -c shaders.metal -o shaders.air```

```xcrun -sdk macosx metallib shaders.air -o MyMetalLib.metallib```

then you can 

```swift run```

OR just run

```./build_and_run.sh```

OR run the program via vscode.
