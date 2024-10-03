#pragma once

#include "../utils/logger.cuh"
#include "stdio.h"
#include "gurobi_solver.h"
#include <sstream>

#include "../LAP/Hung_Tlap.cuh"

__device__ __forceinline__ void init(TILE tile, float *mult, float *g, float *LB,
                                     bool &restart, bool &terminate, float &lrate, uint &t,
                                     const uint K)
{
  // reset mult, g to zero
  for (size_t k = tile.thread_rank(); k < K; k += TileSize)
  {
    mult[k] = 0;
    g[k] = 0;
  }
  tile.sync();
  // reset LB to zero
  for (size_t t = tile.thread_rank(); t < MAX_ITER; t += TileSize)
  {
    LB[t] = 0;
  }
  tile.sync();
  if (tile.thread_rank() == 0)
  {
    restart = false;
    terminate = false;
    lrate = 2;
    t = 0;
  }
  tile.sync();
}

__device__ __forceinline__ void reset(TILE tile, float *g, float *mult,
                                      float &denom, float &feas, float &neg, bool &restart,
                                      const uint K)
{
  for (int k = tile.thread_rank(); k < K; k += TileSize)
  {
    g[k] = 0;
    if (restart)
      mult[k] = 0;
  }
  tile.sync();
  if (tile.thread_rank() == 0)
  {
    denom = 0;
    restart = false;
    feas = 0;
    neg = 0;
  }
  tile.sync();
}

__device__ __forceinline__ void update_lap_costs(TILE tile, float *lap_costs, const problem_info *pinfo,
                                                 float *mult, float &neg,
                                                 const uint N, const uint K)
{
  for (size_t i = tile.thread_rank(); i < N * N; i += TileSize)
  {
    lap_costs[i] = pinfo->costs[i];
    float sum = 0;
    for (size_t k = 0; k < K; k++)
    {
      sum += mult[k] * pinfo->weights[k * N * N + i];
    }
    lap_costs[i] += sum;
  }
  tile.sync();
  for (size_t k = tile.thread_rank(); k < K; k += TileSize)
  {
    atomicAdd(&neg, mult[k] * pinfo->budgets[k]);
  }
  tile.sync();
}

__device__ __forceinline__ void get_denom(const problem_info *pinfo, TILE tile,
                                          float *g, float *real_obj, int *X,
                                          float &denom, float &feas,
                                          const uint N, const uint K)
{

  for (int k = 0; k < K; k++)
  {
    float sum = 0, real = 0;
    for (int i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
    {
      // atomicAdd(&g[k], float(X[i] * pinfo->weights[k * N * N + i]));
      sum += float(X[i] * pinfo->weights[k * N * N + i]);
      real += float(X[i] * pinfo->costs[i]);
    }
    sum = tileReduce(tile, sum, cub::Sum());
    tile.sync();
    real = tileReduce(tile, real, cub::Sum());
    if (tile.thread_rank() == 0)
    {
      g[k] = sum;
      g[k] -= float(pinfo->budgets[k]);
      denom += g[k] * g[k];
      feas += max(float(0), g[k]);
      real_obj[0] = real;
    }
    tile.sync();
  }
}

__device__ __forceinline__ void update_mult(TILE tile, float *mult, float *g, float lrate,
                                            float denom, float LB, float UB, uint K)
{
  for (int k = tile.thread_rank(); k < K; k += TileSize)
  {
    mult[k] += max(float(0), lrate * (g[k] * (UB - LB)) / denom);
  }
  tile.sync();
}

__device__ __forceinline__ float round_to(float val, int places)
{
  float factor = pow(10, places);
  return round(val * factor) / factor;
}

__device__ __forceinline__ void get_LB(TILE tile, float *LB, float &max_LB)
{
  float val = 0;
  for (size_t i = tile.thread_rank(); i < MAX_ITER; i += TileSize)
    val = max(val, LB[i]);
  tile.sync();
  float max_ = tileReduce(tile, val, cub::Max());
  tile.sync();
  if (tile.thread_rank() == 0)
    max_LB = ceil(round_to(max_, 3));
  tile.sync();
}

__device__ __forceinline__ void check_feasibility(const problem_info *pinfo, TILE tile, PARTITION_HANDLE<float> &ph,
                                                  bool &terminate, const float feas)
{
  if (tile.thread_rank() == 0)
  {
    if (feas < eps)
    {
      // DLog(debug, "Found feasible solution!\n");
      // Solution need not be optimal
      // TODO: Update UB and save this solution

      float obj = 0;
      for (uint r = 0; r < SIZE; r++)
      {
        int c = ph.column_of_star_at_row[r];
        obj += pinfo->costs[c * SIZE + r];
      }
      terminate = true;
    }
  }
  tile.sync();
}