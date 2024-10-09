#pragma once

#include "subgrad_utils.cuh"

struct subgrad_space
{
  float *mult, *g, *lap_costs, *LB, *real_obj, *max_LB;
  int *X;
  int *col_fixed_assignments;
  TLAP<float> T;
  __host__ void allocate(uint N, uint K, uint nworkers = 0, uint devID = 0)
  {
    nworkers = (nworkers == 0) ? N : nworkers;
    // allocate space for mult, g, lap_costs, LB, LB_old, X, and th
    CUDA_RUNTIME(cudaMalloc((void **)&mult, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&g, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&lap_costs, nworkers * N * N * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&X, nworkers * N * N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&LB, nworkers * MAX_ITER * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&max_LB, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&real_obj, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_fixed_assignments, nworkers * N * sizeof(int)));

    CUDA_RUNTIME(cudaMemset(mult, 0, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(g, 0, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(lap_costs, 0, nworkers * N * N * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(X, 0, nworkers * N * N * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(LB, 0, nworkers * MAX_ITER * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(max_LB, 0, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(real_obj, 0, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(col_fixed_assignments, 0, nworkers * N * sizeof(int)));

    T = TLAP<float>(nworkers, N, devID);
    // T.allocate(nworkers, N, devID);
  };
  __host__ void clear()
  {
    CUDA_RUNTIME(cudaFree(mult));
    CUDA_RUNTIME(cudaFree(g));
    CUDA_RUNTIME(cudaFree(lap_costs));
    CUDA_RUNTIME(cudaFree(LB));
    CUDA_RUNTIME(cudaFree(real_obj));
    CUDA_RUNTIME(cudaFree(max_LB));
    CUDA_RUNTIME(cudaFree(X));
    CUDA_RUNTIME(cudaFree(col_fixed_assignments));
    T.clear();
  }
};

// Solve subgradient with a block
__device__ void subgrad_solver_tile(const problem_info *pinfo, TILE tile,
                                    subgrad_space *space, float &UB,
                                    int *row_fa, int *col_fa,
                                    PARTITION_HANDLE<float> &ph)
{
  const uint N = pinfo->psize, K = pinfo->ncommodities;
  const uint tile_id = tile.meta_group_rank();
  const uint worker_id = blockIdx.x * TilesPerBlock + tile_id;
  float *mult, *g, *lap_costs, *LB, *real_obj;
  int *X;

  mult = &space->mult[worker_id * K];
  g = &space->g[worker_id * K];
  lap_costs = &space->lap_costs[worker_id * N * N];
  LB = &space->LB[worker_id * MAX_ITER];
  real_obj = &space->real_obj[worker_id];
  X = &space->X[worker_id * N * N];

  __shared__ float lrate[TilesPerBlock], denom[TilesPerBlock], feas[TilesPerBlock], neg[TilesPerBlock];
  __shared__ bool restart[TilesPerBlock], terminate[TilesPerBlock];
  __shared__ uint t[TilesPerBlock];
  sync(tile);

  // Initialize
  init(tile, mult, g, LB,
       restart[tile_id], terminate[tile_id], lrate[tile_id], t[tile_id], K);

  if (tile.thread_rank() == 0)
    ph.cost = lap_costs;
  sync(tile);

  while (t[tile_id] < MAX_ITER)
  {
    reset(tile, g, mult, denom[tile_id], feas[tile_id], neg[tile_id], restart[tile_id], K);

    update_lap_costs(tile, lap_costs, pinfo,
                     mult, neg[tile_id],
                     N, K);

    // Solve the LAP
    PHA_fa<float>(tile, ph, row_fa, col_fa);

    get_objective(tile, ph);

    if (tile.thread_rank() == 0)
      LB[t[tile_id]] = ph.objective[0] - neg[tile_id];
    sync(tile);

    get_X(tile, ph, X);

    // Find the difference between the sum of the costs and the budgets
    get_denom(pinfo, tile, g, real_obj, X, denom[tile_id],
              feas[tile_id], N, K);

    check_feasibility(pinfo, tile, ph, terminate[tile_id], feas[tile_id]);
    if (terminate[tile_id])
      break;

    // Update multipliers according to subgradient rule
    update_mult(tile, mult, g, lrate[tile_id],
                denom[tile_id], LB[t[tile_id]], UB, K);

    if (tile.thread_rank() == 0)
    {
      // DLog(info, "Iteration %d, LB: %.3f, UB: %.3f, lrate: %.3f, Infeasibility: %.3f\n", t, LB[t], UB, lrate, feas);
      if ((t[tile_id] > 0 && t[tile_id] < 5 && LB[t[tile_id]] < LB[t[tile_id] - 1]) || LB[t[tile_id]] < 0)
      {
        // DLog(debug, "Initial Step size too large, restart with smaller step size\n");
        lrate[tile_id] /= 2;
        t[tile_id] = 0;
        restart[tile_id] = true;
      }
      if ((t[tile_id] + 1) % 5 == 0 && LB[t[tile_id]] <= LB[t[tile_id] - 4])
        lrate[tile_id] /= 2;
      if (lrate[tile_id] < 0.005)
        terminate[tile_id] = true;
      t[tile_id]++;
    }
    sync(tile);
    if (terminate[tile_id])
      break;
  }
  sync(tile);
  // Use cub to take the max of the LB array
  get_LB(tile, LB, space->max_LB[worker_id]);

  // if (threadIdx.x == 0)
  // {
  // DLog(debug, "Block %u finished subgrad solver with LB: %.3f\n", blockIdx.x, space->max_LB[blockIdx.x]);
  //   DLog(info, "Max LB: %.3f\n", space->max_LB[blockIdx.x]);
  //   DLog(info, "Subgrad Solver Gap: %.3f%%\n", (UB - space->max_LB[blockIdx.x]) * 100 / UB);
  // }
  sync(tile);
}

__global__ void g_subgrad_solver(const problem_info *pinfo, subgrad_space *space, float UB)
{
  __shared__ PARTITION_HANDLE<float> ph;
  cg::thread_block block = cg::this_thread_block();
  TILE tile = cg::tiled_partition<TileSize>(block);

  set_handles(tile, ph, space->T.th);
  subgrad_solver_tile(pinfo, tile, space, UB, nullptr, nullptr, ph);
}
