#pragma once
#include "config.h"
#include <cuda_runtime.h>

struct problem_info
{
  uint psize;

  cost_type *distances;
  cost_type *flows;
  uint opt_objective;

  void allocate()
  {
    CUDA_RUNTIME(cudaMallocManaged(&distances, psize * psize * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMallocManaged(&flows, psize * psize * sizeof(cost_type)));
  }
  void free()
  {
    CUDA_RUNTIME(cudaFree(distances));
    CUDA_RUNTIME(cudaFree(flows));
  }
  ~problem_info()
  {
    CUDA_RUNTIME(cudaFree(distances));
    CUDA_RUNTIME(cudaFree(flows));
  }
};