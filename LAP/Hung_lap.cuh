#pragma once
#include "../defs.cuh"
#include "../utils/logger.cuh"
#include "../utils/timer.h"
#include "lap_kernels.cuh"
#include "block_lap_kernels.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template <typename cost_type>
class LAP
{
private:
  int dev_;
  size_t size_, h_nrows, h_ncols;
  cost_type *cost_;
  uint num_blocks_4, num_blocks_reduction;

public:
  cost_type objective;
  GLOBAL_HANDLE<cost_type> gh;
  // constructor
  LAP(cost_type *cost, size_t size, int dev = 0) : cost_(cost), dev_(dev), size_(size)
  {
    h_nrows = size;
    h_ncols = size;

    // constant memory copies
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &size, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(nrows, &h_nrows, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(ncols, &h_ncols, sizeof(SIZE)));
    num_blocks_4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    num_blocks_reduction = min(size, 256UL);
    CUDA_RUNTIME(cudaMemcpyToSymbol(NB4, &num_blocks_4, sizeof(NB4)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NBR, &num_blocks_reduction, sizeof(NBR)));
    const uint temp1 = ceil(size / num_blocks_reduction);
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_rows_per_block, &temp1, sizeof(n_rows_per_block)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_cols_per_block, &temp1, sizeof(n_rows_per_block)));
    const uint temp2 = (uint)ceil(log2(size_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_n, &temp2, sizeof(log2_n)));
    gh.row_mask = (1 << temp2) - 1;
    // Log(debug, "log2_n %d", temp2);
    // Log(debug, "row mask: %d", gh.row_mask);
    gh.nb4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_blocks_step_4, &gh.nb4, sizeof(n_blocks_step_4)));
    const uint temp4 = columns_per_block_step_4 * pow(2, ceil(log2(size_)));
    // Log(debug, "dbs: %u", temp4);
    CUDA_RUNTIME(cudaMemcpyToSymbol(data_block_size, &temp4, sizeof(data_block_size)));
    const uint temp5 = temp2 + (uint)ceil(log2(columns_per_block_step_4));
    // Log(debug, "l2dbs: %u", temp5);
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_data_block_size, &temp5, sizeof(log2_data_block_size)));
    // Log(debug, " nb4: %u\n nbr: %u\n dbs: %u\n l2dbs %u\n", gh.nb4, num_blocks_reduction, temp4, temp5);
    // memory allocations
    // CUDA_RUNTIME(cudaMalloc((void **)&gh.cost, size * size * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.slack, size * size * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.min_in_rows, h_nrows * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.min_in_cols, h_ncols * sizeof(cost_type)));

    CUDA_RUNTIME(cudaMalloc((void **)&gh.zeros, h_nrows * h_ncols * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.zeros_size_b, num_blocks_4 * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.row_of_star_at_column, h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&gh.column_of_star_at_row, h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.cover_row, h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.cover_column, h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.column_of_prime_at_row, h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.row_of_green_at_column, h_ncols * sizeof(int)));

    CUDA_RUNTIME(cudaMalloc((void **)&gh.d_min_in_mat_vect, num_blocks_reduction * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&gh.d_min_in_mat, 1 * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&gh.objective, 1 * sizeof(cost_type)));

    CUDA_RUNTIME(cudaMemcpy(gh.slack, cost_, size * size * sizeof(cost_type), cudaMemcpyDefault));
    // CUDA_RUNTIME(cudaMemcpy(gh.cost, cost_, size * size * sizeof(cost_type), cudaMemcpyDefault));
    gh.cost = cost_;
    CUDA_RUNTIME(cudaDeviceSynchronize());
  };

  // destructor
  ~LAP()
  {
    // Log(debug, "LAP Destructor called");
    gh.clear();
  };
  cost_type full_solve()
  {
    const uint n_threads = (uint)min(size_, 64UL);
    const uint n_threads_full = (uint)min(size_, 512UL);

    const size_t n_blocks = (size_t)ceil((size_ * 1.0) / n_threads);
    const size_t n_blocks_full = (size_t)ceil((size_ * size_ * 1.0) / n_threads_full);

    execKernel(init, n_blocks, n_threads, dev_, false, gh);

    execKernel(calc_row_min, size_, n_threads_reduction, dev_, false, gh);
    execKernel(row_sub, n_blocks_full, n_threads_full, dev_, false, gh);

    execKernel(calc_col_min, size_, n_threads_reduction, dev_, false, gh);
    execKernel(col_sub, n_blocks_full, n_threads_full, dev_, false, gh);

    execKernel(compress_matrix, n_blocks_full, n_threads_full, dev_, false, gh);

    // use thrust instead of add reduction
    // execKernel((add_reduction), 1, (uint)ceil(max(size_ / columns_per_block_step_4, 1)), dev_, false, gh);
    zeros_size = thrust::reduce(thrust::device, gh.zeros_size_b, gh.zeros_size_b + num_blocks_4);

    // printDebugArray(gh.zeros_size_b, gh.nb4, "zeros array");
    // Log(debug, "Zeros size: %d", zeros_size);
    do
    {
      repeat_kernel = false;
      uint temp_blockdim = (gh.nb4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size;
      execKernel(step_2, gh.nb4, temp_blockdim, dev_, false, gh);
    } while (repeat_kernel);

    // printDebugArray(gh.row_of_star_at_column, size_, "row ass");
    // printDebugArray(gh.column_of_star_at_row, size_, "col ass");

    // needed for cub reduce
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes1 = 0, temp_storage_bytes2 = 0;

    CUDA_RUNTIME(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes1,
                                           gh.d_min_in_mat_vect, gh.d_min_in_mat,
                                           num_blocks_reduction, cub::Min(), MAX_DATA));
    CUDA_RUNTIME(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes2, gh.zeros_size_b, &zeros_size, num_blocks_4));
    size_t temp_storage_bytes = max(temp_storage_bytes1, temp_storage_bytes2);
    CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    bool first = true;
    while (1)
    {
      execKernel(step_3_init, n_blocks, n_threads, dev_, false, gh);
      execKernel(step_3, n_blocks, n_threads, dev_, false, gh);
      if (first)
      {
        first = false;
      }
      if (n_matches >= h_ncols)
        break;

      execKernel(step_4_init, n_blocks, n_threads, dev_, false, gh);

      // printDebugArray(gh.cover_column, size_, "Cover column");
      // printDebugArray(gh.cover_row, size_, "Cover Rows");
      // printDebugArray(gh.zeros_size_b, num_blocks_4, "zeros size per block");
      // exit(-1);
      while (1)
      {
        do
        {
          goto_5 = false;
          repeat_kernel = false;
          CUDA_RUNTIME(cudaDeviceSynchronize());

          uint temp_blockdim = (gh.nb4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size;
          execKernel(step_4, gh.nb4, temp_blockdim, dev_, false, gh);
        } while (repeat_kernel && !goto_5);

        // exit(-1);
        if (goto_5)
          break;

        // step 6
        // printDebugArray(gh.cover_column, size_, "Column cover");
        // printDebugArray(gh.cover_row, size_, "Row cover");
        execKernel((min_reduce_kernel1<cost_type, n_threads_reduction>),
                   num_blocks_reduction, n_threads_reduction, dev_, false,
                   gh.slack, gh.d_min_in_mat_vect, h_nrows * h_ncols, gh);

        // printDebugArray(gh.d_min_in_mat_vect, num_blocks_reduction, "min vector");
        // printDebugArray(gh.cover_column, size_, "Column cover");
        // printDebugArray(gh.cover_row, size_, "Row cover");
        // exit(-1);
        // finding minimum with cub
        CUDA_RUNTIME(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes1,
                                               gh.d_min_in_mat_vect, gh.d_min_in_mat,
                                               num_blocks_reduction, cub::Min(), MAX_DATA));

        if (!passes_sanity_test(gh.d_min_in_mat))
          exit(-1);

        execKernel(step_6_init, ceil(num_blocks_4 * 1.0 / 256), 256, dev_, false, gh);
        execKernel(step_6_add_sub_fused_compress_matrix, n_blocks_full, n_threads_full, dev_, false, gh);

        // add_reduction

        // printDebugArray(gh.zeros_size_b, num_blocks_4);
        CUDA_RUNTIME(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes2, gh.zeros_size_b, &zeros_size, num_blocks_4));

      } // repeat step 4 and 6

      execKernel(step_5a, n_blocks, n_threads, dev_, false, gh);
      execKernel(step_5b, n_blocks, n_threads, dev_, false, gh);
    } // repeat steps 3 to 6

    CUDA_RUNTIME(cudaFree(d_temp_storage));

    // find objective
    execKernel(get_objective, 1, n_threads, dev_, false, gh);
    // for (uint r = 0; r < h_nrows; r++)
    // {
    //   int c = gh.column_of_star_at_row[r];
    //   if (c >= 0)
    //     objective += cost_[c * h_nrows + r];
    //   // printf("r = %d, c = %d\n", r, c);
    // }
    return gh.objective[0];
  };
  void print_solution()
  {
    for (uint r = 0; r < h_nrows; r++)
    {
      int c = gh.column_of_star_at_row[r];
      if (c >= 0)
        printf("c = %d, r = %d\n", r, c);
    }
  }
  void get_X(int *X)
  {
    std::fill(X, X + h_nrows * h_nrows, 0);
    for (uint r = 0; r < h_nrows; r++)
    {
      int c = gh.column_of_star_at_row[r];
      assert(c >= 0);
      X[h_nrows * c + r] = 1;
    }
  }
  bool passes_sanity_test(cost_type *d_min)
  {
    cost_type temp;
    CUDA_RUNTIME(cudaMemcpy(&temp, d_min, 1 * sizeof(cost_type), cudaMemcpyDeviceToHost));
    if (temp <= 0)
    {
      Log(critical, "minimum element in matrix is non positive => infinite loop condition !!!");
      Log(critical, "%d", temp);
      return false;
    }
    else
      return true;
  }
  void reset_costs(cost_type *cost)
  {
    CUDA_RUNTIME(cudaMemcpy(gh.slack, cost, size_ * size_ * sizeof(cost_type), cudaMemcpyDefault));
    gh.cost = cost;
  }
};

template <typename cost_type = float>
class TLAP
{
private:
  uint nprob_;
  int dev_, maxtile;
  size_t size_, h_nrows, h_ncols;
  cost_type *Tcost_;
  uint num_blocks_4, num_blocks_reduction;

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
    th.clear();
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
    CUDA_RUNTIME(cudaMemcpyToSymbol(NPROB, &nprob_, sizeof(NPROB)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &size, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(nrows, &h_nrows, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(ncols, &h_ncols, sizeof(SIZE)));
    num_blocks_4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    num_blocks_reduction = min(size, 512UL);
    CUDA_RUNTIME(cudaMemcpyToSymbol(NB4, &num_blocks_4, sizeof(NB4)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NBR, &num_blocks_reduction, sizeof(NBR)));
    const uint temp1 = ceil(size / num_blocks_reduction);
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_rows_per_block, &temp1, sizeof(n_rows_per_block)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_cols_per_block, &temp1, sizeof(n_rows_per_block)));
    const uint temp2 = (uint)ceil(log2(size_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_n, &temp2, sizeof(log2_n)));
    uint max_active_blocks = 108;
    maxtile = min(nproblem, max_active_blocks);
    // Log(debug, "Grid dimension %d\n", maxtile);
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

    CUDA_RUNTIME(cudaMalloc((void **)&th.max_in_mat_row, maxtile * h_nrows * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.max_in_mat_col, maxtile * h_ncols * sizeof(cost_type)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.d_min_in_mat_vect, maxtile * num_blocks_reduction * sizeof(cost_type)));
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
