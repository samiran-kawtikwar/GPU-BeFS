#pragma once
#include "../defs.cuh"
#include "../utils/logger.cuh"
#include "../utils/timer.h"
#include "block_lap_kernels.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template <typename cost_type = float>
class TLAP
{
private:
  uint nprob_;
  int dev_, maxtile;
  size_t size_, h_nrows, h_ncols;
  cost_type *Tcost_;
  uint num_blocks_4;
  // uint num_blocks_reduction;

public:
  // Blank constructor
  TILED_HANDLE<cost_type> th;
  TLAP(uint nproblem, size_t size, int dev = 0)
      : nprob_(nproblem), dev_(dev), size_(size)
  {
    th.memoryloc = EXTERNAL;
    allocate(nproblem, size, dev);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }
  TLAP(uint nproblem, cost_type *tcost, size_t size, int dev = 0)
      : nprob_(nproblem), Tcost_(tcost), dev_(dev), size_(size)
  {
    th.memoryloc = INTERNAL;
    allocate(nproblem, size, dev);
    th.cost = Tcost_;
    // initialize slack
    CUDA_RUNTIME(cudaMemcpy(th.slack, Tcost_, nproblem * size * size * sizeof(cost_type), cudaMemcpyDefault));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  };

  void allocate(uint nproblem, size_t size, int dev)
  {
    h_nrows = size;
    h_ncols = size;
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NPROB, &nprob_, sizeof(NPROB)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &size, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(nrows, &h_nrows, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(ncols, &h_ncols, sizeof(SIZE)));

    maxtile = nproblem;
    Log(debug, "Allocating memory for %d problems", maxtile);

    // external memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.slack, maxtile * size * size * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.column_of_star_at_row, maxtile * h_nrows * sizeof(int)));

    // internal memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.zeros, maxtile * h_nrows * h_ncols * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.cover_row, maxtile * h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.cover_column, maxtile * h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.column_of_prime_at_row, maxtile * h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.row_of_green_at_column, maxtile * h_ncols * sizeof(int)));

    // CUDA_RUNTIME(cudaMalloc((void **)&th.d_min_in_mat_vect, maxtile * num_blocks_reduction * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.d_min_in_mat, maxtile * 1 * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.min_in_rows, maxtile * h_nrows * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.min_in_cols, maxtile * h_ncols * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.row_of_star_at_column, maxtile * h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.objective, maxtile * 1 * sizeof(cost_type)));
    // CUDA_RUNTIME(cudaMalloc((void **)&th.tail, 1 * sizeof(uint)));
    // CUDA_RUNTIME(cudaMemset(th.tail, 0, sizeof(uint)));

    if (th.memoryloc == INTERNAL)
    {
      Log(info, "Allocating internal memory for %d problems", nproblem);
    }
  }
};
