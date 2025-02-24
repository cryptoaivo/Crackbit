#!/bin/bash

set -e  # Exit immediately if a command fails

# Create and enter the build directory
mkdir -p build && cd build

# Run CMake with Release build type
cmake .. -DCMAKE_BUILD_TYPE=Release

# Run Make with optimal parallel jobs
make -j$(nproc)

echo "✅ Build complete!"
