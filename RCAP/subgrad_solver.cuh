#pragma once

#include "../utils/logger.cuh"
#include "stdio.h"
#include "gurobi_solver.h"
#include <sstream>

#include "../LAP/Hung_lap.cuh"

template <typename cost_type, typename weight_type>
cost_type subgrad_solver(cost_type *costs, cost_type UB, weight_type *weights, weight_type *budgets, uint N, uint K, uint dev_ = 0)
{
  Log(info, "Starting subgrad solver");
  // For root nodew
  cost_type LB = 0;
  weight_type *mult = new weight_type[K];
  std::fill(mult, mult + K, 0);

  weight_type *lap_costs = new weight_type[N * N];
  // std::fill(lap_costs, lap_costs + N * N, 0);
  // for (size_t i = 0; i < N; i++)
  // {
  //   for (size_t j = 0; j < N; j++)
  //   {
  //     weight_type sum = 0 for (size_t k = 0; k < K; k++)
  //     {
  //       sum += mult[k] * weights[i * N * N + j * N + k];
  //     }
  //     lap_costs[i * N + j] = costs[i * N + j] + sum;
  //   }
  // }

  // Move LAP costs to device
  weight_type *d_lap_costs;
  cudaMalloc(&d_lap_costs, N * N * sizeof(weight_type));
  cudaMemcpy(d_lap_costs, costs, N * N * sizeof(weight_type), cudaMemcpyHostToDevice);
  Log(debug, "LAP costs moved to device");
  LAP<weight_type> lap = LAP<weight_type>(d_lap_costs, N, dev_);
  Log(debug, "LAP created");
  lap.solve();

  return LB;
}
