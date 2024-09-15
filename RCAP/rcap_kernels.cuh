#pragma once

#include "../utils/logger.cuh"
#include "../utils/cuda_utils.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../defs.cuh"
#include "subgrad_solver.cuh"

__device__ __forceinline__ void feas_check_naive(const problem_info *pinfo, const node *a, int *col_fa,
                                                 float *lap_costs, bnb_stats *stats, bool &feasible)
{
  const uint psize = pinfo->psize;
  const uint ncommodities = pinfo->ncommodities;
  __shared__ float budget;
  if (threadIdx.x == 0)
    feasible = true;
  __syncthreads();
  for (uint i = 1; i < ncommodities; i++)
  {
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
}

__device__ __forceinline__ void feas_check(const problem_info *pinfo, const node *a, int *col_fa,
                                           float *lap_costs, bnb_stats *stats, bool &feasible,
                                           GLOBAL_HANDLE<float> &gh, SHARED_HANDLE &sh)
{
  const uint psize = pinfo->psize, ncommmodities = pinfo->ncommodities;
  const int *row_fa = a->value->fixed_assignments;
  if (threadIdx.x == 0)
  {
    feasible = true;
    gh.cost = lap_costs;
  }
  __syncthreads();

  // set col_fa using row_fa
  for (uint i = threadIdx.x; i < psize; i += blockDim.x)
  {
    col_fa[i] = 0;
  }
  __syncthreads();
  for (uint i = threadIdx.x; i < psize; i += blockDim.x)
  {
    if (row_fa[i] > 0)
      col_fa[row_fa[i] - 1] = i + 1;
  }
  __syncthreads();

  for (uint k = 0; k < ncommmodities; k++)
  {
    // copy weights to lap_costs for further operations
    for (uint i = threadIdx.x; i < psize * psize; i += blockDim.x)
    {
      lap_costs[i] = float(pinfo->weights[k * psize * psize + i]);
    }
    __syncthreads();

    BHA_fa<float>(gh, sh, a->value->fixed_assignments, col_fa, 1);
    __syncthreads();
    get_objective_block(gh);
    if (threadIdx.x == 0)
    {
      float used_budget = gh.objective[0];
      if (used_budget > pinfo->budgets[k])
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

// Simple bounds based on fixed assignments
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

__device__ void update_bounds_subgrad(const problem_info *pinfo, subgrad_space *space,
                                      float &UB, node *a, int *col_fa,
                                      GLOBAL_HANDLE<float> &gh, SHARED_HANDLE &sh)
{
  __shared__ int *row_fa;
  if (threadIdx.x == 0)
    row_fa = a[0].value->fixed_assignments;
  __syncthreads();
  // Update UB using the current fixed assignments
  for (int i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    if (row_fa[i] != 0)
    {
      atomicAdd(&UB, (float)pinfo->costs[i * SIZE + (row_fa[i] - 1)]);
    }
  }
  __syncthreads();

  subgrad_solver_block(pinfo, space, UB, row_fa, col_fa, gh, sh);
  __syncthreads();
  a[0].value->LB = space->max_LB[blockIdx.x];
  __syncthreads();
}