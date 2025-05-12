#pragma once
#include "config.h"

struct problem_info
{
  uint psize, ncommodities;
  cost_type *costs;     // cost of assigning
  weight_type *weights; // weight of each commodity
  weight_type *budgets; // capacity of each commodity

  ~problem_info()
  {
    CUDA_RUNTIME(cudaFree(costs));
    CUDA_RUNTIME(cudaFree(weights));
    CUDA_RUNTIME(cudaFree(budgets));
  }
};