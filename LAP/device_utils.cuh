#pragma once
#include "../utils/cuda_utils.cuh"

#define MAX_DATA float(1e6)
typedef unsigned long long int uint64;
#define eps 1e-6
#define __DEBUG__D true

#define checkpoint()                                       \
  {                                                        \
    __syncthreads();                                       \
    if (__DEBUG__D)                                        \
    {                                                      \
      if (threadIdx.x == 0)                                \
        printf("\nReached %s:%u\n\n", __FILE__, __LINE__); \
    }                                                      \
    __syncthreads();                                       \
  }

__managed__ __device__ int zeros_size;     // The number fo zeros
__managed__ __device__ int n_matches;      // Used in step 3 to count the number of matches found
__managed__ __device__ bool goto_5;        // After step 4, goto step 5?
__managed__ __device__ bool repeat_kernel; // Needs to repeat the step 2 and step 4 kernel?

enum MemoryLoc
{
  INTERNAL,
  EXTERNAL
};

template <typename cost_type = float>
struct GLOBAL_HANDLE
{
  cost_type *cost;
  cost_type *slack;
  cost_type *min_in_rows;
  cost_type *min_in_cols;
  cost_type *objective;

  size_t *zeros, *zeros_size_b;
  int *row_of_star_at_column;
  int *column_of_star_at_row; // In unified memory
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;

  cost_type *d_min_in_mat_vect, *d_min_in_mat;
  int row_mask;
  uint nb4;

  void clear()
  {
    CUDA_RUNTIME(cudaFree(slack));
    CUDA_RUNTIME(cudaFree(min_in_rows));
    CUDA_RUNTIME(cudaFree(min_in_cols));
    CUDA_RUNTIME(cudaFree(zeros));
    CUDA_RUNTIME(cudaFree(zeros_size_b));
    CUDA_RUNTIME(cudaFree(row_of_star_at_column));
    CUDA_RUNTIME(cudaFree(column_of_star_at_row));
    CUDA_RUNTIME(cudaFree(cover_row));
    CUDA_RUNTIME(cudaFree(cover_column));
    CUDA_RUNTIME(cudaFree(column_of_prime_at_row));
    CUDA_RUNTIME(cudaFree(row_of_green_at_column));
    CUDA_RUNTIME(cudaFree(d_min_in_mat_vect));
    CUDA_RUNTIME(cudaFree(d_min_in_mat));
  };
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

  size_t *zeros, *zeros_size_b;
  int *row_of_star_at_column;
  int *column_of_star_at_row; // In unified memory
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;
  // uint *tail; // Only difference between TILED and GLOBAL //Not needed

  data *d_min_in_mat_vect, *d_min_in_mat;
  int row_mask;
  uint nb4;

  void clear()
  {
    // CUDA_RUNTIME(cudaFree(cost));  //Already cleared to save memory

    CUDA_RUNTIME(cudaFree(min_in_rows));
    CUDA_RUNTIME(cudaFree(min_in_cols));
    CUDA_RUNTIME(cudaFree(row_of_star_at_column));
    CUDA_RUNTIME(cudaFree(slack));
    CUDA_RUNTIME(cudaFree(zeros));
    CUDA_RUNTIME(cudaFree(zeros_size_b));
    CUDA_RUNTIME(cudaFree(column_of_star_at_row));
    CUDA_RUNTIME(cudaFree(cover_row));
    CUDA_RUNTIME(cudaFree(cover_column));
    CUDA_RUNTIME(cudaFree(column_of_prime_at_row));
    CUDA_RUNTIME(cudaFree(row_of_green_at_column));
    CUDA_RUNTIME(cudaFree(d_min_in_mat_vect));
    CUDA_RUNTIME(cudaFree(d_min_in_mat));
    // CUDA_RUNTIME(cudaFree(tail));
  };
};

struct SHARED_HANDLE
{
  int zeros_size, n_matches;
  bool goto_5, repeat_kernel;
};

__device__ void print_cost_matrix(float *cost, int n, int m)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
      printf("%.3f ", cost[i * m + j]);
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