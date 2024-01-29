#pragma once

#include "../utils/logger.cuh"
#include "../utils/cuda_utils.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../defs.cuh"

__device__ __forceinline__ void feas_check(const problem_info *pinfo, const node *a, bnb_stats *stats, bool &feasible)
{
  const uint psize = pinfo->psize, ncommmodities = pinfo->ncommodities;
  if (threadIdx.x == 0)
  {
    feasible = true;
  }
  __syncthreads();

  for (uint i = 0; i < ncommmodities; i++)
  {
    __shared__ weight_type budget;
    if (threadIdx.x == 0)
      budget = 0;
    __syncthreads();
    for (uint tid = threadIdx.x; tid < psize; tid += blockDim.x)
    {
      if (a[0].value->fixed_assignments[tid] != 0)
        atomicAdd(&budget, pinfo->weights[i * psize * psize + tid * psize + a[0].value->fixed_assignments[tid] - 1]);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      if (budget > pinfo->budgets[i])
      {
        feasible = false;
        atomicAdd(&stats->nodes_pruned_infeasible, 1);
      }
    }
    __syncthreads();
    if (!feasible)
      break;
  }
  __syncthreads();
}

__device__ __forceinline__ void update_bounds(const problem_info *pinfo, const node *a)
{
  const uint psize = pinfo->psize;
  for (uint i = threadIdx.x; i < psize; i += blockDim.x)
  {
    if (a[0].value->fixed_assignments[i] != 0)
      atomicAdd(&a[0].value->LB, pinfo->costs[i * psize + (a[0].value->fixed_assignments[i] - 1)]);
  }
  __syncthreads();
}