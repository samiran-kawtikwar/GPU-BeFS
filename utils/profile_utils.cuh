#pragma once
#include "../defs.cuh"
#include "cuda_utils.cuh"

struct Counters
{
  unsigned long long int tmp[NUM_COUNTERS];
  unsigned long long int totalTime[NUM_COUNTERS];
  unsigned long long int total;
  float percentTime[NUM_COUNTERS];
};

__managed__ Counters *counters;

static __device__ void initializeCounters(Counters *counters)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
    {
      counters->totalTime[i] = 0;
    }
  }
  __syncthreads();
}

static __device__ void startTime(CounterName counterName, Counters *counters)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    counters->tmp[counterName] = clock64();
  }
  __syncthreads();
}

static __device__ void endTime(CounterName counterName, Counters *counters)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    counters->totalTime[counterName] += clock64() - counters->tmp[counterName];
  }
  __syncthreads();
}

__host__ void allocateCounters(Counters **counters, const uint nworkers)
{
  GRID_DIM_X = nworkers;
  CUDA_RUNTIME(cudaMallocManaged(counters, nworkers * sizeof(Counters)));
  CUDA_RUNTIME(cudaDeviceSynchronize());
}

__host__ void freeCounters(Counters *counters)
{
  CUDA_RUNTIME(cudaFree(counters));
}

__host__ void normalizeCounters(Counters *counters)
{
  for (uint t = 0; t < GRID_DIM_X; t++)
  {
    counters[t].total = 0;
    for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
    {
      counters[t].total += counters[t].totalTime[i];
    }
    for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
    {
      if (counters[t].total == 0)
        counters[t].percentTime[i] = 0;
      else
        counters[t].percentTime[i] = (counters[t].totalTime[i] * 100.0f) / counters[t].total;
    }
  }
}

__host__ void fixOverLappingCounters(Counters *counters)
{
  for (uint t = 0; t < GRID_DIM_X; t++)
  {
    assert(counters[t].totalTime[UPDATE_LB] >= counters[t].totalTime[GET_Z]);
    counters[t].totalTime[UPDATE_LB] -= counters[t].totalTime[GET_Z];

    assert(counters[t].totalTime[UPDATE_LB] >= counters[t].totalTime[SOLVE_Z]);
    counters[t].totalTime[UPDATE_LB] -= counters[t].totalTime[SOLVE_Z];

    assert(counters[t].totalTime[WAITING] >= counters[t].totalTime[WAITING_UNDERFLOW]);
    counters[t].totalTime[WAITING] -= counters[t].totalTime[WAITING_UNDERFLOW];
  }
  Log(debug, "Fixed overlapping counters");
}

__host__ void printCounters(Counters *counters, bool print_blockwise_stats = false)
{
  fixOverLappingCounters(counters);
  normalizeCounters(counters);
  printf(", ");
  for (unsigned int i = 0; i < NUM_COUNTERS; i++)
  {
    printf("%s, ", CounterName_text[i]);
  }
  printf("\n");
  // block wise stats
  if (print_blockwise_stats)
  {
    for (uint t = 0; t < GRID_DIM_X; t++)
    {
      printf("%d, ", t);
      for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
      {
        printf("%.2f, ", counters[t].percentTime[i]);
      }
      printf("\n");
    }
  }
  // aggregate stats
  float grand_total = 0;
  float col_mean[NUM_COUNTERS] = {0};
  for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
  {
    for (uint t = 1; t < GRID_DIM_X; t++)
    {
      col_mean[i] += counters[t].percentTime[i] / GRID_DIM_X;
    }
    grand_total += col_mean[i];
  }
  // print grand_total
  printf("Mean, ");
  for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
    printf("%.2f, ", (col_mean[i] * 100.0f) / grand_total);
  printf("\n");

  printf("Variance/mean, ");
  for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
  {
    float variance = 0;
    for (uint t = 1; t < GRID_DIM_X; t++)
    {
      variance += (counters[t].percentTime[i] - (col_mean[i])) * (counters[t].percentTime[i] - (col_mean[i]));
    }
    printf("%.2f, ", variance / GRID_DIM_X / col_mean[i]);
  }
  printf("\n");
}