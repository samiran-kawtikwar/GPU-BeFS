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

__device__ __forceinline__ void feas_check(const problem_info *pinfo, TILE tile,
                                           const node *a, int *col_fa,
                                           float *lap_costs, bnb_stats *stats, bool &feasible,
                                           PARTITION_HANDLE<float> &ph)
{
  const uint psize = pinfo->psize, ncommmodities = pinfo->ncommodities;
  const int *row_fa = a->value->fixed_assignments;
  const uint local_id = tile.thread_rank();

  if (local_id == 0)
  {
    feasible = true;
    ph.cost = lap_costs;
  }
  tile.sync();

  // set col_fa using row_fa
  for (uint i = local_id; i < psize; i += TileSize)
  {
    col_fa[i] = 0;
  }
  tile.sync();

  for (uint i = local_id; i < psize; i += TileSize)
  {
    if (row_fa[i] > 0)
      col_fa[row_fa[i] - 1] = i + 1;
  }
  tile.sync();

  for (uint k = 0; k < ncommmodities; k++)
  {
    // copy weights to lap_costs for further operations
    for (uint i = local_id; i < psize * psize; i += TileSize)
    {
      lap_costs[i] = float(pinfo->weights[k * psize * psize + i]);
    }
    tile.sync();

    PHA_fa<float>(tile, ph, a->value->fixed_assignments, col_fa, 1);
    tile.sync();
    get_objective(tile, ph);
    if (tile.thread_rank() == 0)
    {
      float used_budget = ph.objective[0];
      if (used_budget > pinfo->budgets[k])
      {
        feasible = false;
        atomicAdd(&stats->nodes_pruned_infeasible, 1);
      }
    }
    tile.sync();
    if (!feasible)
      break;
  }
  tile.sync();
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

__device__ void update_bounds_subgrad(const problem_info *pinfo, TILE tile,
                                      subgrad_space *space, float &UB, node *a, int *col_fa,
                                      PARTITION_HANDLE<float> &ph)
{
  __shared__ int *row_fa[TilesPerBlock];
  const uint tile_id = tile.meta_group_rank();
  if (tile.thread_rank() == 0)
    row_fa[tile_id] = a[0].value->fixed_assignments;
  tile.sync();
  // Update UB using the current fixed assignments
  for (int i = tile.thread_rank(); i < SIZE; i += TileSize)
  {
    if (row_fa[tile_id][i] != 0)
    {
      atomicAdd(&UB, (float)pinfo->costs[i * SIZE + (row_fa[tile_id][i] - 1)]);
    }
  }
  tile.sync();

  subgrad_solver_tile(pinfo, tile, space, UB, row_fa[tile_id], col_fa, ph);
  tile.sync();
  if (tile.thread_rank())
    a[0].value->LB = space->max_LB[blockIdx.x * TilesPerBlock + tile.meta_group_rank()];
  tile.sync();
}