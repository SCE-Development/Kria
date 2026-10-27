# execute if a a new container of the pipeline is created

# rebuild face-results in a fresh smartcam container
set -e


# work from the folder this script lives in
cd "$(dirname "$0")"

apt install -y build-essential pkg-config libjansson-dev

# build plugin
make clean
make EXTRA_LIBS="-lgstvvasinfermeta-2.0"


# checks
echo "-exported functions-"
nm -D libvvas_results.so | grep xlnx_kernel

echo "-missing libraries (should be empty-"
ldd libvvas_results.so | grep "not found" || true

echo "Done. Results are ready."