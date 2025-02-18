#pragma once

#include <cuda.h>
#include "logger.cuh"

#define CUDA_RUNTIME(ans)                 \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
    cudaDeviceSynchronize();              \
  }

#define execKernel(kernel, gridSize, blockSize, deviceId, verbose, ...)                       \
  {                                                                                           \
    dim3 grid(gridSize);                                                                      \
    dim3 block(blockSize);                                                                    \
                                                                                              \
    CUDA_RUNTIME(cudaSetDevice(deviceId));                                                    \
    if (verbose)                                                                              \
      Log(info, "Launching %s with nblocks: %u, blockDim: %u", #kernel, gridSize, blockSize); \
    kernel<<<grid, block>>>(__VA_ARGS__);                                                     \
    CUDA_RUNTIME(cudaGetLastError());                                                         \
  }

#define assert_d(X)                                           \
  if (!(X))                                                   \
  {                                                           \
    printf("Assertion failed: %s, %d\n", __FILE__, __LINE__); \
    assert(X);                                                \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{

  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

    /*if (abort) */ exit(1);
  }
}

// Wrapper using atomicAdd to perform an atomic read.
template <typename T = int>
__device__ __forceinline__ T atomicRead(T *addr)
{
  // We cast away the const because the CUDA atomic functions require a non-const pointer.
  // The addition of T(0) is guaranteed to leave the value unchanged.
  return atomicAdd(addr, static_cast<T>(0));
}

// Returns the device ID of the pointer, or -1 if pointer is on host
int getPointerAtt(void *ptr)
{
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  if (err == cudaSuccess)
  {
    if (attributes.type == cudaMemoryTypeDevice)
    {
      return attributes.device; // Return device ID
    }
    else if (attributes.type == cudaMemoryTypeHost)
    {
      return -1; // Host memory
    }
  }
  return -1; // Unrecognized pointer
}
