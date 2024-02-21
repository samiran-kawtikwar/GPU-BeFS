#pragma once

#include "../utils/logger.cuh"
#include "stdio.h"
#include "gurobi_solver.h"
#include <sstream>

#include "../LAP/Hung_lap.cuh"

__device__ __forceinline__ void init(float *mult, float *g, float *LB,
                                     bool &restart, bool &terminate, float &lrate, uint &t,
                                     const uint K)
{
  // reset mult, g to zero
  for (size_t k = threadIdx.x; k < K; k += blockDim.x)
  {
    mult[k] = 0;
    g[k] = 0;
  }
  __syncthreads();
  // reset LB to zero
  for (size_t t = threadIdx.x; t < MAX_ITER; t += blockDim.x)
  {
    LB[t] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    restart = false;
    terminate = false;
    lrate = 2;
    t = 0;
  }
  __syncthreads();
}

__device__ __forceinline__ void reset(float *g, float *mult,
                                      float &denom, float &feas, float &neg, bool &restart,
                                      const uint K)
{
  for (int k = threadIdx.x; k < K; k += blockDim.x)
  {
    g[k] = 0;
    if (restart)
      mult[k] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    denom = 0;
    restart = false;
    feas = 0;
    neg = 0;
  }
  __syncthreads();
}

__device__ __forceinline__ void update_lap_costs(float *lap_costs, const problem_info *pinfo,
                                                 float *mult, float &neg,
                                                 const uint N, const uint K)
{
  for (size_t i = threadIdx.x; i < N * N; i += blockDim.x)
  {
    lap_costs[i] = pinfo->costs[i];
    float sum = 0;
    for (size_t k = 0; k < K; k++)
    {
      sum += mult[k] * pinfo->weights[k * N * N + i];
    }
    lap_costs[i] += sum;
  }
  __syncthreads();
  for (size_t k = threadIdx.x; k < K; k += blockDim.x)
  {
    atomicAdd(&neg, mult[k] * pinfo->budgets[k]);
  }
}

__device__ __forceinline__ void get_denom(float *g, float *real_obj, int *X,
                                          float &denom, float &feas,
                                          const problem_info *pinfo, const uint N, const uint K)
{
  typedef cub::BlockReduce<float, n_threads_reduction> BR;
  __shared__ typename BR::TempStorage temp_storage;
  for (int k = 0; k < K; k++)
  {
    float sum = 0, real = 0;
    for (int i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
    {
      // atomicAdd(&g[k], float(X[i] * pinfo->weights[k * N * N + i]));
      sum += float(X[i] * pinfo->weights[k * N * N + i]);
      real += float(X[i] * pinfo->costs[i]);
    }
    sum = BR(temp_storage).Reduce(sum, cub::Sum());
    real = BR(temp_storage).Reduce(real, cub::Sum());
    if (threadIdx.x == 0)
    {
      g[k] = sum;
      g[k] -= float(pinfo->budgets[k]);
      denom += g[k] * g[k];
      feas += max(float(0), g[k]);
      real_obj[0] = real;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void update_mult(float *mult, float *g, const float lrate,
                                            float &denom, const float LB,
                                            const float &UB, const uint K)
{
  for (int k = threadIdx.x; k < K; k += blockDim.x)
  {
    mult[k] += max(float(0), lrate * (g[k] * (UB - LB)) / denom);
  }
  __syncthreads();
}

__device__ __forceinline__ void get_LB(float *LB, float &max_LB)
{
  typedef cub::BlockReduce<float, n_threads_reduction> BR;
  __shared__ typename BR::TempStorage temp_storage;
  for (size_t i = threadIdx.x; i < MAX_ITER; i += blockDim.x)
    LB[threadIdx.x] = max(LB[threadIdx.x], LB[i]);
  float max_ = BR(temp_storage).Reduce(LB[threadIdx.x], cub::Max());
  __syncthreads();
  if (threadIdx.x == 0)
    max_LB = ceil(max_);
  __syncthreads();
}

__device__ __forceinline__ void check_feasibility(const problem_info *pinfo, GLOBAL_HANDLE<float> &gh,
                                                  float &LB, bool &terminate, const float feas)
{
  if (threadIdx.x == 0)
  {
    if (feas < eps)
    {
      // DLog(debug, "Found feasible solution!\n");
      // Solution need not be optimal
      // TODO: Update UB and save this solution

      float obj = 0;
      for (uint r = 0; r < SIZE; r++)
      {
        int c = gh.column_of_star_at_row[r];
        obj += pinfo->costs[c * SIZE + r];
      }
      terminate = true;
    }
  }
  __syncthreads();
}