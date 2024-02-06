#pragma once

#include "../utils/logger.cuh"
#include "stdio.h"
#include "gurobi_solver.h"
#include <sstream>

#include "../LAP/Hung_lap.cuh"

template <typename cost_type, typename weight_type>
weight_type subgrad_solver(const cost_type *original_costs, cost_type upper, const weight_type *weights, weight_type *budgets, uint N, uint K, uint dev_ = 0)
{
  Log(info, "Starting subgrad solver");
  // For root nodew
  float LB = 0, UB = float(upper), LB_old;
  float *mult = new float[K];
  int *X = new int[N * N];
  float *g = new float[K];
  std::fill(mult, mult + K, 0);
  std::fill(X, X + N * N, 0);

  float *lap_costs;
  CUDA_RUNTIME(cudaMallocManaged((void **)&lap_costs, N * N * sizeof(float)));
  float lrate = 0.5;
  for (size_t t = 0; t < 100; t++)
  {
    LB_old = LB;
    std::fill(lap_costs, lap_costs + N * N, 0);
    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        lap_costs[i * N + j] = original_costs[i * N + j];
        float sum = 0;
        for (size_t k = 0; k < K; k++)
        {
          sum += mult[k] * weights[k * N * N + i * N + j];
        }
        lap_costs[i * N + j] += sum;
      }
    }
    LAP<float> lap = LAP<float>(lap_costs, N, dev_);
    LB = lap.full_solve();
    // lap.print_solution();
    lap.get_X(X);

    // Find the difference between the sum of the costs and the budgets

    std::fill(g, g + K, 0);
    for (size_t k = 0; k < K; k++)
    {
      for (size_t i = 0; i < N; i++)
      {
        for (size_t j = 0; j < N; j++)
        {
          g[k] += float(X[i * N + j] * weights[k * N * N + i * N + j]);
        }
      }
      g[k] -= float(budgets[k]);
    }

    float denom = 0;
    for (size_t k = 0; k < K; k++)
    {
      denom += g[k] * g[k];
    }
    // Update multipliers according to subgradient rule
    for (size_t k = 0; k < K; k++)
    {
      mult[k] += max(float(0), lrate * (g[k] * (UB - LB)) / denom);
    }
    Log(info, "Iteration %d, LB: %.3f, UB: %.3f, lrate: %f", t, LB, UB, lrate);

    if (LB_old > LB)
      lrate /= 2;
    if (abs(LB_old - LB) / LB < 1E-6)
      break;
    if (LB > UB + 1 && t < 5)
    {
      Log(info, "Initial Step size too large, restart with smaller step size");
      lrate /= 2;
      t = 0;
      std::fill(mult, mult + K, 0);
      LB = 0;
    }
  }
  delete[] mult;
  delete[] X;
  delete[] g;
  CUDA_RUNTIME(cudaFree(lap_costs));
  return LB;
}

// Solve subgradient with a block
__device__ void subgrad_solver_block(const problem_info *pinfo, subgrad_space *space, float UB, node *a)
{
  // Assume a is a root node
  const uint N = pinfo->psize, K = pinfo->ncommodities;
  float *mult = space->mult, *g = space->g, *lap_costs = space->lap_costs;
  int *X = space->X;
  float LB = space->LB, LB_old = space->LB_old;
  float lrate = 0.5;

  // reset mult, g to zero
  for (size_t k = threadIdx.x; k < K; k += blockDim.x)
  {
    mult[k] = 0;
    g[k] = 0;
  }
  __syncthreads();
  for (size_t t = 0; t < 100; t++)
  {
    LB_old = LB;
    for (size_t i = threadIdx.x; i < N * N; i += blockDim.x)
    {
      // uint r = i / N, c = i % N;
      lap_costs[i] = pinfo->costs[i];
      float sum = 0;
      for (size_t k = 0; k < K; k++)
      {
        sum += mult[k] * pinfo->weights[k * N * N + i];
      }
      lap_costs[i] += sum;
    }
    __syncthreads();
    // solve block-LAP
  }
}