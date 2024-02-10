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
  float *LB = new float[100], UB = float(upper), max_LB;
  float *mult = new float[K];
  int *X = new int[N * N];
  float *g = new float[K];
  std::fill(mult, mult + K, 0);
  std::fill(X, X + N * N, 0);
  std::fill(LB, LB + 100, 0);
  float *lap_costs;
  CUDA_RUNTIME(cudaMallocManaged((void **)&lap_costs, N * N * sizeof(float)));
  float lrate = 2;
  for (size_t t = 0; t < 100; t++)
  {
    std::fill(lap_costs, lap_costs + N * N, 0);
    std::fill(g, g + K, 0);

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
    float neg = 0;
    for (size_t k = 0; k < K; k++)
    {
      neg += mult[k] * budgets[k];
    }

    LAP<float> lap = LAP<float>(lap_costs, N, dev_);
    LB[t] = lap.full_solve() - neg;
    // lap.print_solution();
    lap.get_X(X);
    float my_UB = 0;
    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        my_UB += float(X[i * N + j] * original_costs[i * N + j]);
      }
    }
    // Find the difference between the sum of the costs and the budgets
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

    float denom = 0, feas = 0;
    for (size_t k = 0; k < K; k++)
    {
      denom += g[k] * g[k];
      feas += max(float(0), g[k]);
    }
    Log(info, "Iteration %d, LB: %.3f, UB: %.3f, lrate: %.3f, real: %.3f, Infeasibility: %.3f", t, LB[t], UB, lrate, my_UB, feas);
    if (feas < eps)
    {
      Log(debug, "Found feasible solution");
      break;
    }
    // Update multipliers according to subgradient rule
    for (size_t k = 0; k < K; k++)
    {
      mult[k] += max(float(0), lrate * (g[k] * (UB - LB[t])) / denom);
    }
    // print lap_costs
    // for (size_t i = 0; i < N; i++)
    // {
    //   for (size_t j = 0; j < N; j++)
    //   {
    //     printf("%.2f ", lap_costs[i * N + j]);
    //   }
    //   printf("\n");
    // }

    // print mult and g
    // for (size_t i = 0; i < K; i++)
    //   printf("%.2f ", mult[i]);
    // printf("\n");
    // for (size_t i = 0; i < K; i++)
    //   printf("%.2f ", g[i]);
    // printf("\n");
    // printf("denom: %.2f\n", denom);
    if ((t > 0 && t < 5 && LB[t] < LB[t - 1]) || LB[t] < 0)
    {
      Log(debug, "Initial Step size too large, restart with smaller step size");
      lrate /= 2;
      t = 0;
      std::fill(mult, mult + K, 0);
    }

    if ((t + 1) % 5 == 0 && LB[t] <= LB[t - 4])
      lrate /= 2;
    if (lrate < 0.1)
      break;
  }
  // max_LB = max(LB)
  max_LB = *std::max_element(LB, LB + 100);
  delete[] mult;
  delete[] X;
  delete[] g;
  delete[] LB;
  CUDA_RUNTIME(cudaFree(lap_costs));
  return uint(ceil(max_LB));
}

// Solve subgradient with a block
__device__ void subgrad_solver_block(const problem_info *pinfo, subgrad_space *space, float UB)
{
  // Assume a is a root node
  const uint N = pinfo->psize, K = pinfo->ncommodities;
  float *mult = space->mult, *g = space->g, *lap_costs = space->lap_costs;
  int *X = space->X;
  float *LB = &space->LB, *LB_old = &space->LB_old;
  UB += 2;
  __shared__ float lrate, denom;
  __shared__ bool restart, terminate;
  // reset mult, g to zero
  for (size_t k = threadIdx.x; k < K; k += blockDim.x)
  {
    mult[k] = 0;
    g[k] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    restart = false;
    terminate = false;
    lrate = 0.25;
  }
  __syncthreads();
  __shared__ GLOBAL_HANDLE<float> gh;
  __shared__ SHARED_HANDLE sh;
  set_handles(gh, space->T.th);
  for (size_t t = 0; t < 1000; t++)
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
      LB_old[0] = LB[0];
      denom = 0;
      restart = false;
    }
    __syncthreads();
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
    gh.cost = lap_costs;

    BHA<float>(gh, sh);
    get_objective_block(gh);
    if (threadIdx.x == 0)
    {
      LB[0] = gh.objective[0];
    }
    __syncthreads();
    get_X(gh, X);

    // Find the difference between the sum of the costs and the budgets
    for (int k = 0; k < K; k++)
    {
      for (int i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
      {
        atomicAdd(&g[k], float(X[i] * pinfo->weights[k * N * N + i]));
      }
      __syncthreads();
      if (threadIdx.x == 0)
      {
        g[k] -= float(pinfo->budgets[k]);
        denom += g[k] * g[k];
      }
      __syncthreads();
    }
    // Update multipliers according to subgradient rule
    for (size_t k = threadIdx.x; k < K; k += blockDim.x)
    {
      mult[k] += max(float(0), lrate * (g[k] * (UB - LB[0])) / denom);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      DLog(info, "Iteration %d, LB: %.3f, UB: %.3f, lrate: %f\n", t, LB[0], UB, lrate);

      // Print gh.costs
      // for (size_t i = 0; i < N; i++)
      // {
      //   for (size_t j = 0; j < N; j++)
      //     printf("%.2f ", gh.cost[i * N + j]);
      //   printf("\n");
      // }
      // print mult and g
      // for (size_t i = 0; i < K; i++)
      //   printf("%.2f ", mult[i]);
      // printf("\n");
      // for (size_t i = 0; i < K; i++)
      //   printf("%.2f ", g[i]);
      // printf("\n");
      // printf("denom: %.2f\n", denom);

      if (LB_old[0] > LB[0])
        lrate /= 2;
      if (abs(LB_old[0] - LB[0]) / LB[0] < 1E-6)
        terminate = true;
      if (LB[0] > UB + 1 && t < 5)
      {
        DLog(debug, "Initial Step size too large, restart with smaller step size\n");
        lrate /= 2;
        t = 0;
        LB[0] = 0;
        restart = true;
      }
    }
    __syncthreads();
    if (terminate)
      break;
  }
}
__global__ void g_subgrad_solver(const problem_info *pinfo, subgrad_space *space, float UB)
{

  subgrad_solver_block(pinfo, space, UB);
}