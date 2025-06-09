#!/bin/bash
set -e
cd Sources/gpuRaytracer
xcrun -sdk macosx metal -I../CHeaders -c shaders.metal -o shaders.air
xcrun -sdk macosx metallib shaders.air -o MyMetalLib.metallib
cd ../../
swift run