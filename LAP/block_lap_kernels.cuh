#pragma once
#include <cooperative_groups.h>
#include "../utils/cuda_utils.cuh"
#include "device_utils.cuh"
#include "../defs.cuh"
namespace cg = cooperative_groups;

#ifdef TIMER
#include "profile_utils.cuh"
#endif

#define fundef template <typename data = float> \
__forceinline__ __device__

__constant__ size_t SIZE;
__constant__ uint NPROB;
__constant__ size_t nrows;
__constant__ size_t ncols;

#define WARP_SIZE 32

#if TileSize <= WARP_SIZE
template <typename Op>
__forceinline__ __device__ float tileReduce(cg::thread_block_tile<TileSize> tile, float value, Op operation)
{
  // Intra-tile reduction
  typedef cub::WarpReduce<float, TileSize> WR;
  __shared__ typename WR::TempStorage temp_storage[TilesPerBlock];
  value = WR(temp_storage[tile.meta_group_rank()]).Reduce(value, operation);
  return value;
}
#elif TileSize == BlockSize
template <typename Op>
__forceinline__ __device__ float tileReduce(cg::thread_block_tile<TileSize> tile, float value, Op operation)
{
  // perform blockReduce with cub
  typedef cub::BlockReduce<float, BlockSize> BR;
  __shared__ typename BR::TempStorage temp_storage;
  value = BR(temp_storage).Reduce(value, operation);
  return value;
}
#else
template <typename Op>
__device__ float warpReduce(cg::thread_block_tile<WARP_SIZE> tile, float value, Op operation)

{
  // Intra-tile reduction
  for (int offset = tile.size() / 2; offset > 0; offset /= 2)
  {
    value = operation(value, tile.shfl_down(value, offset));
  }
  return value;
}

// Generalized tile-based reduction function
template <typename Op>
__device__ float tileReduce(cg::thread_block_tile<TileSize> tile, float value, Op operation)
{
  __shared__ float val[TilesPerBlock][TileSize];
  val[tile.meta_group_rank()][tile.thread_rank()] = value;

  cg::thread_block_tile<WARP_SIZE> subtile = cg::tiled_partition<WARP_SIZE>(tile);
  __shared__ float red_val[TilesPerBlock];
  tile.sync();
  if (subtile.meta_group_rank() == 0)
  {
    for (uint i = WARP_SIZE + subtile.thread_rank(); i < TileSize; i += WARP_SIZE)
      value = operation(value, val[tile.meta_group_rank()][i]);
    subtile.sync();

    value = warpReduce(subtile, value, operation);

    subtile.sync();
    if (subtile.thread_rank() == 0)
    {
      red_val[tile.meta_group_rank()] = value;
    }
    subtile.sync();
  }
  tile.sync();
  return red_val[tile.meta_group_rank()];
}
#endif

fundef void init(TILE tile, PARTITION_HANDLE<data> &ph) // with single block
{
  // initializations
  // for step 2
  for (size_t i = tile.thread_rank(); i < SIZE; i += TileSize)
  {
    ph.cover_row[i] = 0;
    ph.column_of_star_at_row[i] = -1;
    ph.cover_column[i] = 0;
    ph.row_of_star_at_column[i] = -1;
  }
  // initialize slack with cost
  for (size_t i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
  {
    ph.slack[i] = ph.cost[i];
    ph.zeros[i] = size_t(0); // Reset old zero indices
  }
}

fundef void calc_col_min(TILE tile, PARTITION_HANDLE<data> &ph) // with single block
{
  for (size_t col = 0; col < SIZE; col++)
  {
    size_t i = (size_t)tile.thread_rank() * SIZE + col;
    data thread_min = (data)MAX_DATA;

    while (i <= SIZE * (SIZE - 1) + col)
    {
      thread_min = min(thread_min, ph.slack[i]);
      i += (size_t)TileSize * SIZE;
    }
    tile.sync();

    thread_min = tileReduce(tile, thread_min, cub::Min());
    if (tile.thread_rank() == 0)
    {
      ph.min_in_cols[col] = thread_min;
    }
    tile.sync();
  }
}

fundef void col_sub(TILE tile, PARTITION_HANDLE<data> &ph) // with single block
{
  // uint i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
  {
    size_t l = i % SIZE;
    ph.slack[i] = ph.slack[i] - ph.min_in_cols[l]; // subtract the minimum in col from that col
  }
}

fundef void calc_row_min(TILE tile, PARTITION_HANDLE<data> &ph) // with single block
{

  // size_t i = (size_t)blockIdx.x * SIZE + (size_t)threadIdx.x;
  for (size_t row = 0; row < SIZE; row++)
  {
    data thread_min = MAX_DATA;
    for (size_t i = tile.thread_rank() + row * SIZE; i < SIZE * (row + 1); i += TileSize)
    {
      thread_min = min(thread_min, ph.slack[i]);
    }
    tile.sync();
    thread_min = tileReduce(tile, thread_min, cub::Min());
    if (tile.thread_rank() == 0)
    {
      ph.min_in_rows[row] = thread_min;
    }
    tile.sync();
  }
}

fundef void row_sub(TILE tile, PARTITION_HANDLE<data> &ph) // with single block
{
  for (size_t i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
  {
    size_t c = i / SIZE;
    ph.slack[i] = ph.slack[i] - ph.min_in_rows[c]; // subtract the minimum in row from that row
  }
  if (tile.thread_rank() == 0)
    ph.zeros_size = 0;
}

fundef bool near_zero(data val)
{
  return ((val < eps) && (val > -eps));
}

fundef void compress_matrix(TILE tile, PARTITION_HANDLE<data> &ph) // with single block
{
  // size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
  {
    if (near_zero(ph.slack[i]))
    {
      size_t j = (size_t)atomicAdd(&ph.zeros_size, 1);
      ph.zeros[j] = i; // saves index of zeros in slack matrix per block
    }
  }
}

fundef void step_2(TILE tile, PARTITION_HANDLE<data> &ph)
{
  uint i = tile.thread_rank();

  if (i == 0)
    ph.s_repeat_kernel = false;

  do
  {
    tile.sync();
    if (i == 0)
      ph.repeat = false;
    tile.sync();

    for (int j = i; j < ph.zeros_size; j += TileSize)
    {
      uint z = ph.zeros[j];
      uint l = z % nrows;
      uint c = z / nrows;
      if (ph.cover_row[l] == 0 &&
          ph.cover_column[c] == 0)
      {
        if (!atomicExch((int *)&(ph.cover_row[l]), 1))
        {
          // only one thread gets the line
          if (!atomicExch((int *)&(ph.cover_column[c]), 1))
          {
            // only one thread gets the column
            ph.row_of_star_at_column[c] = l;
            ph.column_of_star_at_row[l] = c;
          }
          else
          {
            ph.cover_row[l] = 0;
            ph.repeat = true;
            ph.s_repeat_kernel = true;
          }
        }
      }
    }
    tile.sync();
  } while (ph.repeat);
  if (ph.s_repeat_kernel)
    ph.repeat_kernel = true;
}

fundef void step_3_init(TILE tile, PARTITION_HANDLE<data> &ph) // For single block
{
  for (size_t i = tile.thread_rank(); i < nrows; i += TileSize)
  {
    ph.cover_row[i] = 0;
    ph.cover_column[i] = 0;
  }
  if (tile.thread_rank() == 0)
    ph.n_matches = 0;
}

fundef void step_3(TILE tile, PARTITION_HANDLE<data> &ph) // For single block
{
  // size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = tile.thread_rank(); i < nrows; i += TileSize)
  {
    // printf("i %lu, rosc %d\n", i, gh.row_of_star_at_column[i]);
    if (ph.row_of_star_at_column[i] >= 0)
    {
      ph.cover_column[i] = 1;
      atomicAdd((int *)&ph.n_matches, 1);
    }
  }
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

fundef void step_4_init(TILE tile, PARTITION_HANDLE<data> &ph)
{
  for (size_t i = tile.thread_rank(); i < SIZE; i += TileSize)
  {
    ph.column_of_prime_at_row[i] = -1;
    ph.row_of_green_at_column[i] = -1;
  }
}

fundef void step_4(TILE tile, PARTITION_HANDLE<data> &ph)
{
  const size_t i = tile.thread_rank();
  volatile int *v_cover_row = ph.cover_row;
  volatile int *v_cover_column = ph.cover_column;
  if (i == 0)
  {
    ph.goto_5 = false;
    ph.repeat_kernel = false;
  }
  tile.sync();
  do
  {
    tile.sync();
    if (i == 0)
      ph.s_found = false;
    tile.sync();
    for (size_t j = tile.thread_rank(); j < ph.zeros_size; j += TileSize)
    {
      // each thread picks a zero!
      size_t z = ph.zeros[j];
      int l = z % nrows; // row
      int c = z / nrows; // column
      int c1 = ph.column_of_star_at_row[l];
      // printf("j %lu, z %lu, l %d, c %d, c1 %d\n", j, z, l, c, c1);
      if (!v_cover_column[c] && !v_cover_row[l])
      {
        ph.s_found = true; // find uncovered zero
        ph.repeat_kernel = true;
        ph.column_of_prime_at_row[l] = c; // prime the uncovered zero

        if (c1 >= 0)
        {
          v_cover_row[l] = 1; // cover row
          __threadfence();
          v_cover_column[c1] = 0; // uncover column
        }
        else
        {
          ph.goto_5 = true;
        }
      }
    } // for(int j
    tile.sync();
  } while (ph.s_found && !ph.goto_5);
}

fundef void min_reduce_kernel1(TILE tile, const size_t n, PARTITION_HANDLE<data> &ph)
{
  data myval = MAX_DATA;
  size_t i = tile.thread_rank();
  size_t gridSize = (size_t)TileSize * 2;
  while (i < n)
  {
    size_t i1 = i;
    size_t i2 = i + TileSize;
    size_t l1 = i1 % nrows; // local index within the row
    size_t c1 = i1 / nrows; // Row number
    data g1 = MAX_DATA, g2 = MAX_DATA;
    if (ph.cover_row[l1] == 1 || ph.cover_column[c1] == 1)
      g1 = MAX_DATA;
    else
      g1 = ph.slack[i1];
    if (i2 < n)
    {
      size_t l2 = i2 % nrows;
      size_t c2 = i2 / nrows;
      if (ph.cover_row[l2] == 1 || ph.cover_column[c2] == 1)
        g2 = MAX_DATA;
      else
        g2 = ph.slack[i2];
    }
    myval = min(myval, min(g1, g2));
    i += gridSize;
  }
  tile.sync();

  data minimum = tileReduce(tile, myval, cub::Min());
  if (tile.thread_rank() == 0)
    *ph.d_min_in_mat = minimum;
}

fundef void step_6_init(TILE tile, PARTITION_HANDLE<data> &ph)
{
  // size_t id = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x;
  if (tile.thread_rank() == 0)
    ph.zeros_size = 0;
  for (uint i = tile.thread_rank(); i < SIZE; i += TileSize)
  {
    if (ph.cover_column[i] == 0)
      ph.min_in_rows[i] += ph.d_min_in_mat[0] / 2;
    else
      ph.min_in_rows[i] -= ph.d_min_in_mat[0] / 2;
    if (ph.cover_row[i] == 0)
      ph.min_in_cols[i] += ph.d_min_in_mat[0] / 2;
    else
      ph.min_in_cols[i] -= ph.d_min_in_mat[0] / 2;
  }
  tile.sync();
}

/* STEP 5:
Construct a series of alternating primed and starred zeros as
follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0(if any).
Let Z2 denote the primed zero in the row of Z1(there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/

// Eliminates joining paths
fundef void step_5a(TILE tile, PARTITION_HANDLE<data> &ph)
{
  // size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = tile.thread_rank(); i < SIZE; i += TileSize)
  {
    int r_Z0, c_Z0;

    c_Z0 = ph.column_of_prime_at_row[i];
    if (c_Z0 >= 0 && ph.column_of_star_at_row[i] < 0) // if primed and not covered
    {
      ph.row_of_green_at_column[c_Z0] = i; // mark the column as green

      while ((r_Z0 = ph.row_of_star_at_column[c_Z0]) >= 0)
      {
        c_Z0 = ph.column_of_prime_at_row[r_Z0];
        ph.row_of_green_at_column[c_Z0] = r_Z0;
      }
    }
  }
}

// Applies the alternating paths
fundef void step_5b(cg::thread_block_tile<TileSize> tile, PARTITION_HANDLE<data> &ph)
{
  // size_t j = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t j = tile.thread_rank(); j < SIZE; j += TileSize)
  {
    int r_Z0, c_Z0, c_Z2;

    r_Z0 = ph.row_of_green_at_column[j];

    if (r_Z0 >= 0 && ph.row_of_star_at_column[j] < 0)
    {

      c_Z2 = ph.column_of_star_at_row[r_Z0];

      ph.column_of_star_at_row[r_Z0] = j;
      ph.row_of_star_at_column[j] = r_Z0;

      while (c_Z2 >= 0)
      {
        r_Z0 = ph.row_of_green_at_column[c_Z2]; // row of Z2
        c_Z0 = c_Z2;                            // col of Z2
        c_Z2 = ph.column_of_star_at_row[r_Z0];  // col of Z4

        // star Z2
        ph.column_of_star_at_row[r_Z0] = c_Z0;
        ph.row_of_star_at_column[c_Z0] = r_Z0;
      }
    }
  }
}

fundef void step_6_add_sub_fused_compress_matrix(TILE tile, PARTITION_HANDLE<data> &ph) // For single tile
{
  // STEP 6:
  /*STEP 6: Add the minimum uncovered value to every element of each covered
  row, and subtract it from every element of each uncovered column.
  Return to Step 4 without altering any stars, primes, or covered lines. */
  // const size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
  {
    const size_t l = i % nrows;
    const size_t c = i / nrows;
    auto reg = ph.slack[i];
    switch (ph.cover_row[l] + ph.cover_column[c])
    {
    case 2:
      reg += ph.d_min_in_mat[0];
      ph.slack[i] = reg;
      break;
    case 0:
      reg -= ph.d_min_in_mat[0];
      ph.slack[i] = reg;
      break;
    default:
      break;
    }

    // compress matrix
    if (near_zero(reg))
    {
      int j = atomicAdd(&ph.zeros_size, 1);
      ph.zeros[j] = i;
    }
  }
}

fundef void get_objective(TILE tile, PARTITION_HANDLE<data> &ph)
{
  data obj = 0;
  for (uint c = tile.thread_rank(); c < SIZE; c += TileSize)
  {
    obj += ph.cost[c * SIZE + ph.row_of_star_at_column[c]];
    // printf("r: %u, c: %u, obj: %u\n", c, gh.row_of_star_at_column[c], obj);
  }
  obj = tileReduce(tile, obj, cub::Sum());
  if (tile.thread_rank() == 0)
    ph.objective[0] = obj;

  tile.sync();
}

fundef void PHA(TILE tile, PARTITION_HANDLE<data> &ph, const uint problemID = blockIdx.x)
{
  init(tile, ph);
  tile.sync();
  calc_row_min(tile, ph);
  tile.sync();
  row_sub(tile, ph);
  tile.sync();

  calc_col_min(tile, ph);
  tile.sync();
  col_sub(tile, ph);
  tile.sync();

  compress_matrix(tile, ph);
  tile.sync();
  do
  {
    tile.sync();
    if (tile.thread_rank() == 0)
      ph.repeat_kernel = false;
    tile.sync();
    step_2(tile, ph);
    tile.sync();
  } while (ph.repeat_kernel);
  tile.sync();

  while (1)
  {

    tile.sync();
    step_3_init(tile, ph);
    tile.sync();
    step_3(tile, ph);
    tile.sync();

    if (ph.n_matches >= SIZE)
      break;

    step_4_init(tile, ph);
    tile.sync();

    while (1)
    {
      do
      {
        tile.sync();
        if (tile.thread_rank() == 0)
        {
          ph.goto_5 = false;
          ph.repeat_kernel = false;
        }
        tile.sync();
        step_4(tile, ph);
        tile.sync();
      } while (ph.repeat_kernel && !ph.goto_5);
      tile.sync();

      if (ph.goto_5)
        break;
      tile.sync();

      min_reduce_kernel1(tile, SIZE * SIZE, ph);
      tile.sync();

#if __DEBUG__D == true
      {
        if (ph.d_min_in_mat[0] <= eps)
        {
          tile.sync();
          if (tile.thread_rank() == 0)
          {
            printf("minimum element in problemID %u is non positive: %.3f\n", problemID, (float)ph.d_min_in_mat[0]);
            printf("Cost\n");
            print_cost_matrix(ph.cost, SIZE, SIZE);

            printf("Slack\n");
            print_cost_matrix(ph.slack, SIZE, SIZE);

            printf("Row cover\n");
            print_cost_matrix(ph.cover_row, 1, SIZE);

            printf("Column cover\n");
            print_cost_matrix(ph.cover_column, 1, SIZE);

            printf("min rows\n");
            print_cost_matrix(ph.min_in_rows, 1, SIZE);

            printf("Min column\n");
            print_cost_matrix(ph.min_in_cols, 1, SIZE);

            printf("Printing finished by block %u\n", problemID);
            assert(false);
          }
          return;
        }
        tile.sync();
      }
#endif

      step_6_init(tile, ph); // Also does dual update
      tile.sync();

      step_6_add_sub_fused_compress_matrix(tile, ph);
      tile.sync();
    }

    tile.sync();
    // checkpoint();
    step_5a(tile, ph);
    tile.sync();
    step_5b(tile, ph);
    tile.sync();
  }
  tile.sync();
  return;
}

fundef void PHA_fa(TILE tile, PARTITION_HANDLE<data> &ph, int *row_fa, int *col_fa, const int caller = 0)
{
  if (row_fa != nullptr && col_fa != nullptr)
  {
    for (uint i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
    {
      uint c = i % SIZE;
      uint r = i / SIZE;
      if (row_fa[r] != 0 && row_fa[r] != c + 1)
      {
        ph.cost[i] = (data)MAX_DATA;
      }
      if (col_fa[c] != 0 && col_fa[c] != r + 1)
      {
        ph.cost[i] = (data)MAX_DATA;
      }
    }
  }
  tile.sync();
  PHA(tile, ph);
  tile.sync();
}

fundef void get_X(TILE tile, PARTITION_HANDLE<data> &ph, int *X)
{
  for (uint i = tile.thread_rank(); i < SIZE * SIZE; i += TileSize)
  {
    X[i] = 0;
  }
  tile.sync();
  for (uint r = tile.thread_rank(); r < SIZE; r += TileSize)
  {
    int c = ph.column_of_star_at_row[r];
    // assert(c >= 0);
    X[c * SIZE + r] = 1;
  }
  tile.sync();
}

fundef void set_handles(TILE tile, PARTITION_HANDLE<data> &ph, TILED_HANDLE<data> &th)
{
  const uint tile_id = tile.meta_group_rank();
  const uint worker_id = blockIdx.x * TilesPerBlock + tile_id;

  if (tile.thread_rank() == 0)
  {
    ph.slack = &th.slack[worker_id * SIZE * SIZE];
    ph.column_of_star_at_row = &th.column_of_star_at_row[worker_id * SIZE];

    ph.zeros = &th.zeros[worker_id * SIZE * SIZE];

    ph.cover_row = &th.cover_row[worker_id * SIZE];
    ph.cover_column = &th.cover_column[worker_id * SIZE];
    ph.column_of_prime_at_row = &th.column_of_prime_at_row[worker_id * SIZE];
    ph.row_of_green_at_column = &th.row_of_green_at_column[worker_id * SIZE];
    ph.d_min_in_mat = &th.d_min_in_mat[worker_id];
    ph.min_in_rows = &th.min_in_rows[worker_id * SIZE];
    ph.min_in_cols = &th.min_in_cols[worker_id * SIZE];
    ph.row_of_star_at_column = &th.row_of_star_at_column[worker_id * SIZE];
    ph.objective = &th.objective[worker_id];
    ph.objective[0] = 0;

    // from shared handles
    ph.repeat_kernel = false;
    ph.goto_5 = false;
    ph.zeros_size = 0;
    ph.n_matches = 0;
  }
  __syncthreads();
}
