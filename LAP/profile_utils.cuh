#pragma once
#include "../defs.cuh"
#include "../utils/cuda_utils.cuh"

struct LAPCounters
{
  unsigned long long int tmp[NUM_LAP_COUNTERS];
  unsigned long long int totalTime[NUM_LAP_COUNTERS];
  unsigned long long int total;
  float percentTime[NUM_LAP_COUNTERS];
};

__managed__ LAPCounters lap_counters[GRID_DIM_X];

static __device__ void initializeCounters(LAPCounters *counters)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    for (unsigned int i = STEP1; i < NUM_LAP_COUNTERS; ++i)
    {
      counters->totalTime[i] = 0;
    }
  }
  __syncthreads();
}

static __device__ void startTime(LAPCounterName counterName, LAPCounters *counters)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    counters->tmp[counterName] = clock64();
  }
  __syncthreads();
}

static __device__ void endTime(LAPCounterName counterName, LAPCounters *counters)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    counters->totalTime[counterName] += clock64() - counters->tmp[counterName];
  }
  __syncthreads();
}

__host__ void normalizeCounters(LAPCounters *counters)
{
  for (uint t = 0; t < GRID_DIM_X; t++)
  {
    counters[t].total = 0;
    for (unsigned int i = STEP1; i < NUM_LAP_COUNTERS; ++i)
    {
      counters[t].total += counters[t].totalTime[i];
    }
    for (unsigned int i = STEP1; i < NUM_LAP_COUNTERS; ++i)
    {
      if (counters[t].total == 0)
        counters[t].percentTime[i] = 0;
      else
        counters[t].percentTime[i] = (counters[t].totalTime[i] * 100.0f) / counters[t].total;
    }
  }
}

__host__ void printCounters(LAPCounters *counters, bool print_blockwise_stats = false)
{
  normalizeCounters(counters);
  printf("\n, ");
  for (unsigned int i = 0; i < NUM_LAP_COUNTERS - NUM_COUNTERS; i++)
  {
    printf("%s, ", LAPCounterName_text[i]);
  }
  printf("\n");
  // block wise stats
  if (print_blockwise_stats)
  {
    for (uint t = 0; t < GRID_DIM_X; t++)
    {
      printf("%d, ", t);
      for (unsigned int i = STEP1; i < NUM_LAP_COUNTERS; ++i)
      {
        printf("%.2f, ", counters[t].percentTime[i]);
      }
      printf("\n");
    }
  }
  // aggregate stats
  unsigned long long int grand_total = 0;
  float col_mean[NUM_LAP_COUNTERS] = {0};
  for (unsigned int i = STEP1; i < NUM_LAP_COUNTERS; ++i)
  {
    for (uint t = 1; t < GRID_DIM_X; t++)
    {
      col_mean[i] += counters[t].percentTime[i] / GRID_DIM_X;
    }
    grand_total += col_mean[i];
  }

  printf("LAP Mean, ");
  for (unsigned int i = STEP1; i < NUM_LAP_COUNTERS; ++i)
    printf("%.2f, ", (col_mean[i] * 100.0f) / grand_total);
  printf("\n");

  printf("LAP Variance/mean, ");
  for (unsigned int i = STEP1; i < NUM_LAP_COUNTERS; ++i)
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