#pragma once
#include "../defs.cuh"
#include "../utils/logger.cuh"
#include "../utils/timer.h"
#include "lap_kernels.cuh"
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
  // destructor
  ~TLAP()
  {
    // th.clear();
  }
  /*
    void solve()
    {
      if (th.memoryloc == EXTERNAL)
      {
        Log(critical, "Unassigned external memory, exiting...");
        return;
      }
      int nblocks = maxtile;
      Log(debug, "nblocks: %d\n", nblocks);
      Timer t;
      execKernel((THA<cost_type, nthr>), nblocks, nthr, dev_, true, th);
      auto time = t.elapsed();
      Log(info, "kernel time %f s\n", time);
    }

    void solve(cost_type *costs, int *row_ass, cost_type *row_duals, cost_type *col_duals, cost_type *obj)
    {
      if (th.memoryloc == INTERNAL)
      {
        Log(debug, "Doubly assigned external memory, exiting...");
        return;
      }
      th.cost = costs;
      th.row_of_star_at_column = row_ass;
      th.min_in_rows = row_duals;
      th.min_in_cols = col_duals;
      th.objective = obj;
      int nblocks = maxtile;
      CUDA_RUNTIME(cudaMemcpy(th.slack, th.cost, nprob_ * size_ * size_ * sizeof(cost_type), cudaMemcpyDefault));
      CUDA_RUNTIME(cudaMemset(th.objective, 0, nprob_ * sizeof(cost_type)));
      CUDA_RUNTIME(cudaMemset(th.min_in_rows, 0, nprob_ * size_ * sizeof(cost_type)));
      CUDA_RUNTIME(cudaMemset(th.min_in_cols, 0, nprob_ * size_ * sizeof(cost_type)));
      // Log(debug, "nblocks from external solve: %d\n", nblocks);
      Timer t;
      execKernel((THA<cost_type, nthr>), nblocks, nthr, dev_, false, th);
      auto time = t.elapsed();
      // Log(info, "kernel time %f s\n", time);
    }
  */
  void allocate(uint nproblem, size_t size, int dev)
  {
    h_nrows = size;
    h_ncols = size;
    CUDA_RUNTIME(cudaSetDevice(dev_));
    Log(critical, "Allocating space for TLAP");
    CUDA_RUNTIME(cudaMemcpyToSymbol(NPROB, &nprob_, sizeof(NPROB)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &size, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(nrows, &h_nrows, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(ncols, &h_ncols, sizeof(SIZE)));
    num_blocks_4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    // num_blocks_reduction = min(size, 512UL);
    CUDA_RUNTIME(cudaMemcpyToSymbol(NB4, &num_blocks_4, sizeof(NB4)));
    // CUDA_RUNTIME(cudaMemcpyToSymbol(NBR, &num_blocks_reduction, sizeof(NBR)));
    // const uint temp1 = ceil(size / num_blocks_reduction);
    // CUDA_RUNTIME(cudaMemcpyToSymbol(n_rows_per_block, &temp1, sizeof(n_rows_per_block)));
    // CUDA_RUNTIME(cudaMemcpyToSymbol(n_cols_per_block, &temp1, sizeof(n_rows_per_block)));
    const uint temp2 = (uint)ceil(log2(size_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_n, &temp2, sizeof(log2_n)));
    uint max_active_blocks = 108;
    maxtile = min(nproblem, max_active_blocks);
    th.row_mask = (1 << temp2) - 1;
    // Log(debug, "log2_n %d", temp2);
    // Log(debug, "row mask: %d", th.row_mask);
    th.nb4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_blocks_step_4, &th.nb4, sizeof(n_blocks_step_4)));
    const uint temp4 = columns_per_block_step_4 * pow(2, ceil(log2(size_)));
    // Log(debug, "dbs: %u", temp4);
    CUDA_RUNTIME(cudaMemcpyToSymbol(data_block_size, &temp4, sizeof(data_block_size)));
    const uint temp5 = temp2 + (uint)ceil(log2(columns_per_block_step_4));
    // Log(debug, "l2dbs: %u", temp5);
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_data_block_size, &temp5, sizeof(log2_data_block_size)));
    // external memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.slack, maxtile * size * size * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.column_of_star_at_row, maxtile * h_nrows * sizeof(int)));

    // internal memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.zeros, maxtile * h_nrows * h_ncols * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.zeros_size_b, maxtile * num_blocks_4 * sizeof(size_t)));

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
