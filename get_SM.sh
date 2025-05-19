#!/bin/bash 
#
# Prints the compute capability of the first CUDA device installed
# on the system, or alternatively the device whose index is the
# first command-line argument

device_index=${1:-0}
timestamp=$(date +%s.%N)
gcc_binary=${CMAKE_CXX_COMPILER:-$(which c++)}

# Derive CUDA paths from nvcc location
nvcc_path=$(which nvcc)
if [ -z "$nvcc_path" ]; then
  echo "nvcc not found in PATH" >&2
  exit 0
fi

CUDA_BIN_DIR=$(dirname "$nvcc_path")
CUDA_HOME=$(dirname "$CUDA_BIN_DIR")
CUDA_INCLUDE_DIRS=${CUDA_HOME}/include
CUDA_LIB_DIR=${CUDA_HOME}/lib64
CUDA_CUDART_LIBRARY=${CUDA_LIB_DIR}/libcudart.so

if [ ! -f "$CUDA_CUDART_LIBRARY" ]; then
  echo "ERROR: libcudart.so not found at $CUDA_CUDART_LIBRARY" >&2
  exit 0
fi


generated_binary="/tmp/cuda-compute-version-helper-$$-$timestamp"
# create a 'here document' that is code we compile and use to probe the card
source_code="$(cat << EOF 
#include <stdio.h>
#include <cuda_runtime_api.h>
int main()
{
	cudaDeviceProp prop;
	cudaError_t status;
	int device_count;
	status = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	if (${device_index} >= device_count) {
		fprintf(stderr, "Specified device index %d exceeds the maximum (the device count on this system is %d)\n", ${device_index}, device_count);
		return -1;
	}
	status = cudaGetDeviceProperties(&prop, ${device_index});
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceProperties() for device ${device_index} failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	int v = prop.major * 10 + prop.minor;
	printf("%d\\n", v);
}
EOF
)"
echo "$source_code" | $gcc_binary -x c++ -I"$CUDA_INCLUDE_DIRS" -o "$generated_binary" - -x none "$CUDA_CUDART_LIBRARY"

# probe the card and cleanup

$generated_binary
rm $generated_binary