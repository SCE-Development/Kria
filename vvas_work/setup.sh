#!/bin/bash
# Rebuild the face-results plugin in a fresh smartcam container.
set -e

cd "$(dirname "$0")"

apt update
apt install -y build-essential pkg-config libjansson-dev
pkg-config --exists opencv4 || apt install -y libopencv-dev

# Let the Vitis AI library find the face-recognition model by name
mkdir -p /usr/share/vitis_ai_library/models
ln -sfn /opt/xilinx/kv260-smartcam/share/vitis_ai_library/models/facerec_resnet20 \
        /usr/share/vitis_ai_library/models/facerec_resnet20

make clean
make

echo "--- exported functions ---"
nm -D libvvas_results.so | grep -E "xlnx_kernel|face_embedder"

echo "--- missing libraries (should be empty) ---"
ldd libvvas_results.so | grep "not found" || true

echo "Done. libvvas_results.so is ready."