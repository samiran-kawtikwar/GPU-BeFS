#pragma once
#include "../../utils/cuda_utils.cuh"

#define MAX_DATA float(1e6)
typedef unsigned long long int uint64;
#define eps 1e-6

__managed__ __device__ int zeros_size;     // The number fo zeros
__managed__ __device__ int n_matches;      // Used in step 3 to count the number of matches found
__managed__ __device__ bool goto_5;        // After step 4, goto step 5?
__managed__ __device__ bool repeat_kernel; // Needs to repeat the step 2 and step 4 kernel?

enum MemoryLoc
{
  INTERNAL,
  EXTERNAL
};

template <typename data = int>
struct PARTITION_HANDLE
{
  data *cost;
  data *slack;
  data *min_in_rows;
  data *min_in_cols;
  data *objective;

  size_t *zeros;
  int *row_of_star_at_column;
  int *column_of_star_at_row; // In unified memory
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;
  uint *tail;

  // from shared handle
  int zeros_size, n_matches;
  bool goto_5, repeat_kernel;

  // internal shared variables
  bool s_found, repeat, s_repeat_kernel;

  data *max_in_mat_row, *max_in_mat_col, *d_min_in_mat;
  int row_mask;
};

template <typename data = int>
struct TILED_HANDLE
{
  MemoryLoc memoryloc;
  data *cost;
  data *slack;
  data *min_in_rows;
  data *min_in_cols;
  data *objective;

  size_t *zeros;
  int *row_of_star_at_column;
  int *column_of_star_at_row; // In unified memory
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;

  data *d_min_in_mat;
  int row_mask;

  void clear()
  {
    Log(debug, "Clearing TH memory");
    CUDA_RUNTIME(cudaFree(min_in_rows));
    CUDA_RUNTIME(cudaFree(min_in_cols));
    CUDA_RUNTIME(cudaFree(row_of_star_at_column));
    // CUDA_RUNTIME(cudaFree(cost));
    CUDA_RUNTIME(cudaFree(slack));
    CUDA_RUNTIME(cudaFree(zeros));
    CUDA_RUNTIME(cudaFree(column_of_star_at_row));
    CUDA_RUNTIME(cudaFree(cover_row));
    CUDA_RUNTIME(cudaFree(cover_column));
    CUDA_RUNTIME(cudaFree(column_of_prime_at_row));
    CUDA_RUNTIME(cudaFree(row_of_green_at_column));
    CUDA_RUNTIME(cudaFree(d_min_in_mat));
    CUDA_RUNTIME(cudaFree(objective));
  };
};

__device__ void print_cost_matrix(float *cost, int n, int m)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
    {
      if (cost[i * m + j] < 1000)
        printf("%d ", (int)cost[i * m + j]);
      else
        printf("inf ");
    }
    printf("\n");
  }
}

__device__ void print_cost_matrix(int *cost, int n, int m)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
      printf("%d ", cost[i * m + j]);
    printf("\n");
  }
}