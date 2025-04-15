#pragma once
#include "config.h"
#include <cuda_runtime.h>

struct problem_info
{
  uint psize;

  cost_type *distances;
  cost_type *flows;
  uint opt_objective;

  problem_info(uint user_n)
  {
    psize = user_n;
  }
  ~problem_info()
  {
    // Free distances pointer
    if (distances != nullptr)
    {
      cudaPointerAttributes attributes;
      cudaError_t err = cudaPointerGetAttributes(&attributes, distances);
      // Check if the pointer is a device pointer.
      if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice)
      {
        cudaFree(distances);
      }
      else
      {
        // If not on device (or error occurred), assume host allocation.
        delete[] distances;
      }
    }

    // Free flows pointer
    if (flows != nullptr)
    {
      cudaPointerAttributes attributes;
      cudaError_t err = cudaPointerGetAttributes(&attributes, flows);
      if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice)
      {
        cudaFree(flows);
      }
      else
      {
        delete[] flows;
      }
    }
  }
};