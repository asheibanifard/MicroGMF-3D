#!/bin/bash
# Build script for separate forward and backward CUDA extensions

set -e

echo "==========================================="
echo "Building CUDA extensions for hisnegs"
echo "==========================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null
then
    echo "ERROR: nvcc (CUDA compiler) not found!"
    echo "Please install CUDA toolkit or check your PATH"
    exit 1
fi

echo "Found CUDA compiler: $(which nvcc)"
echo "CUDA version: $(nvcc --version | grep release)"
echo ""

# Build forward kernel
echo "Building forward kernel..."
python setup_forward.py install

if [ $? -eq 0 ]; then
    echo "✓ Forward kernel built successfully"
else
    echo "✗ Forward kernel build failed"
    exit 1
fi

echo ""

# Build backward kernel
echo "Building backward kernel..."
python setup_backward.py install

if [ $? -eq 0 ]; then
    echo "✓ Backward kernel built successfully"
else
    echo "✗ Backward kernel build failed"
    exit 1
fi

echo ""
echo "==========================================="
echo "✓ All CUDA extensions built successfully!"
echo "==========================================="
echo ""
echo "The extensions are ready for use."
echo "Run training: python run.py --config config.yml"
