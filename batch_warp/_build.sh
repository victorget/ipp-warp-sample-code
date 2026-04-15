#!/bin/bash
# Build script for batch_warp Python extension using Cython based on Intel(R) Integrated Performance Primitives (Intel(R) IPP)
rm -f batch_warp*.so

set -e  # Exit on error
echo "=== Building Batch WarpPerspective Python Extension ==="

# Check if Intel(R) IPP is sourced
if [ -z "$IPPROOT" ]; then
    . /opt/intel/oneapi/setvars.sh
fi

echo "Using Intel(R) IPP from: $IPPROOT"

# Check for required Python packages
echo ""
echo "Checking Python dependencies..."
echo "Checking OpenCV..."
python3 -c "import cv2" 2>/dev/null || { echo "OpenCV not found. Installing..."; pip3 install opencv-python; }
echo "Checking Cython..."
python3 -c "import Cython" 2>/dev/null || { echo "Cython not found. Installing..."; pip3 install Cython; }


# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo ""
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . -j${NUM_CORES}

# Copy the built module to the parent directory for easy import
echo ""
echo "Copying module to source directory..."
mv batch_warp*.so ../ 2>/dev/null || mv batch_warp*.pyd ../ 2>/dev/null || true

cd ..

echo ""
echo "=== Build Complete ==="
echo "The batch_warp module is ready to use."
echo ""
echo "To test the module, run:"
echo "  python3 warp_sample.py"
