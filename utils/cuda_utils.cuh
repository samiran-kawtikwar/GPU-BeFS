#pragma once
#include <cuda.h>
#include "logger.cuh"

#define CUDA_RUNTIME(ans)                 \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{

  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

    /*if (abort) */ exit(1);
  }
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
    CUDA_RUNTIME(cudaDeviceSynchronize());                                                    \
  }
