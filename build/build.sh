#!/bin/bash
# Build script for BackgroundRemove (Linux/Mac cross-compilation or testing)
# Run this script from the project root directory

echo "============================================================"
echo "  BackgroundRemove Build Script"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check/create virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate

    echo "Installing dependencies..."
    pip install -r build/requirements-build.txt
    pip install -e .
fi

# Run build script
echo ""
echo "Starting build..."
python build/build_windows.py "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Build complete!"
echo "Output: dist/BackgroundRemove/"
