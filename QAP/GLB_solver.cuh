#pragma once

#include <cooperative_groups.h>
#include "../defs.cuh"
#include "LAP/Hung_Tlap.cuh"
#include "../utils/logger.cuh"
#include "problem_info.h"

namespace cg = cooperative_groups;

struct glb_space
{
  cost_type *z;         // psize*psize
  TLAP<cost_type> tlap; // nsubworkers
  int *la;              // psize

  __host__ void allocate(uint psize, uint nsubworkers = TilesPerBlock, uint devID = 0)
  {
    // allocate space for z, th, la
    CUDA_RUNTIME(cudaMalloc((void **)&z, psize * psize * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&la, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(z, 0, psize * psize * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMemset(la, -1, psize * sizeof(int)));

    tlap = TLAP<cost_type>(nsubworkers, psize, devID);
  }
  __host__ void clear()
  {
    CUDA_RUNTIME(cudaFree(z));
    CUDA_RUNTIME(cudaFree(la));
    tlap.clear();
  }
  static void allocate_all(glb_space *&d_glb_space, size_t nworkers, size_t psize, uint dev_ = 0)
  {
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_glb_space, nworkers * sizeof(glb_space)));
    for (size_t i = 0; i < nworkers; i++)
      d_glb_space[i].allocate(psize, TilesPerBlock, dev_);
  }
  static void free_all(glb_space *d_glb_space, size_t nworkers)
  {
    for (size_t i = 0; i < nworkers; i++)
      d_glb_space[i].clear();
    CUDA_RUNTIME(cudaFree(d_glb_space));
  }
};

__device__ __forceinline__ void print_matrix(const cost_type *matrix, uint N)
{
  if (threadIdx.x == 0)
  {
    DLog(info, "Z matrix: \n");
    for (uint i = 0; i < N; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        if (matrix[i * N + j] == cost_type(MAX_DATA))
          printf("M ");
        else
          printf("%u ", matrix[i * N + j]);
      }
      printf("\n");
    }
  }
}

// Called by a single tile (partition)
__device__ __forceinline__ void populate_costs(cg::thread_block_tile<TileSize> tile, const int *fa, const int *la,
                                               const uint i, const uint k,
                                               const uint N, const cost_type *flows, const cost_type *distances,
                                               cost_type *cost)
{
  for (uint idx = tile.thread_rank(); idx < N * N; idx += TileSize)
  {
    const uint j = idx / N, l = idx % N;
    if (fa[j] > -1)
    {
      // Fixed assignment mapping: only one location is valid.
      if (l == uint(fa[j]))
        cost[idx] = flows[i * N + j] * distances[k * N + l];
      else
        cost[idx] = (cost_type)MAX_DATA;
    }
    else
    {
      // When facility j is unassigned.
      if (la[l] == -1)
        cost[idx] = flows[i * N + j] * distances[k * N + l];
      else
        cost[idx] = (cost_type)MAX_DATA;

      // Override cost values when additional restrictions apply.
      if ((j == i && k != l) || (l == k && i != j))
        cost[idx] = (cost_type)MAX_DATA;
    }
  }
}

__device__ void glb_getZ(TILE tile, glb_space &glb_space, const int *fa, const int *la, const problem_info *pinfo)
{
  const uint psize = pinfo->psize;
  const cost_type *flows = pinfo->flows;
  const cost_type *distances = pinfo->distances;
  const uint tId = tile.meta_group_rank();
  __shared__ PARTITION_HANDLE<cost_type> ph[TilesPerBlock];
  __shared__ uint global_id, tile_id[TilesPerBlock];
  set_handles(tile, ph[tId], glb_space.tlap.th);

  if (tile.thread_rank() == 0)
  {
    tile_id[tId] = 0;
    if (threadIdx.x == 0)
      global_id = 0;
  }
  __syncthreads();
  while (global_id < psize * psize)
  {
    if (tile.thread_rank() == 0)
    {
      tile_id[tId] = atomicAdd(&global_id, 1);
    }
    sync(tile);
    if (tile_id[tId] >= psize * psize)
      break;
    sync(tile);
    const uint i = tile_id[tId] / psize, k = tile_id[tId] % psize;
    if ((fa[i] == -1 && la[k] == -1) || (fa[i] > -1 && la[k] == i))
    {
      populate_costs(tile, fa, la, i, k, psize, flows, distances, ph[tId].cost);
      sync(tile);
      PHA(tile, ph[tId]);
      sync(tile);
      get_objective(tile, ph[tId]);
      sync(tile);
      if (tile.thread_rank() == 0)
        glb_space.z[i * psize + k] = ph[tId].objective[0];
    }
    else
    {
      if (tile.thread_rank() == 0)
        glb_space.z[i * psize + k] = (cost_type)MAX_DATA;
    }
    sync(tile);
  }
}

// Called by a single thread block
__device__ void glb_solve(glb_space &glb_space, const int *fa, int *la,
                          const problem_info *pinfo, cost_type &LB)
{
  const uint psize = pinfo->psize;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);

  for (uint i = threadIdx.x; i < psize; i += blockDim.x)
  {
    if (fa[i] > -1)
      la[fa[i]] = i;
  }
  __syncthreads();
  // if (threadIdx.x == 0)
  // {
  //   printf("new fa: ");
  //   for (uint j = 0; j < psize; j++)
  //     printf("%d ", fa[j]);
  //   printf("\n");
  //   printf("la: ");
  //   for (uint j = 0; j < psize; j++)
  //     printf("%d ", la[j]);
  //   printf("\n");
  // }
  // __syncthreads();
  START_TIME(GET_Z);
  glb_getZ(tile, glb_space, fa, la, pinfo);
  __syncthreads();
  END_TIME(GET_Z);
  // print Z
  // print_matrix(glb_space.z, psize);
  // __syncthreads();
  // Found z, now find LB with BHA
  START_TIME(SOLVE_Z);
  __shared__ PARTITION_HANDLE<cost_type> bha_handle;
  set_handles(tile, bha_handle, glb_space.tlap.th);
  if (tile.meta_group_rank() == 0)
  {
    if (tile.thread_rank() == 0)
      bha_handle.cost = glb_space.z; // bypass the cost matrix
    sync(tile);
    PHA(tile, bha_handle);
    sync(tile);
    get_objective(tile, bha_handle);
  }
  __syncthreads();
  END_TIME(SOLVE_Z);
  if (threadIdx.x == 0)
    LB = bha_handle.objective[0];
  // reset la
  for (uint i = threadIdx.x; i < psize; i += blockDim.x)
    la[i] = -1;
  __syncthreads();
  return;
}
