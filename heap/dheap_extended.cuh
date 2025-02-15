#pragma once

#include <vector>
#include <iostream>
#include <omp.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "dheap.cuh"
#include "../queue/queue.cuh" // For HHEAP, etc.

// Forward declarations
__global__ void link_node_fa(node_info *node_space, int *fa_space, const size_t size_limit, const uint psize, const size_t start);
__global__ void check_duplicates(int *where, int *bitmap, int *d_result, const size_t size_limit);
__device__ __forceinline__ void cpy(node_info *source, node_info *dest, const uint psize);

// Extended class: Inherits all the basic heap operations from DHEAP
// and adds functions for standardizing, printing, and moving the heap.
template <typename NODE>
class DHEAPExtended : public DHEAP<NODE>
{
public:
  // Inherit variables from base
  using DHEAP<NODE>::psize;
  using DHEAP<NODE>::dev_;
  using DHEAP<NODE>::d_heap;
  using DHEAP<NODE>::d_node_space;
  using DHEAP<NODE>::d_fixed_assignment_space;
  using DHEAP<NODE>::d_size_limit;
  using DHEAP<NODE>::d_size;

  // Extended class variables
  size_t *d_trigger_size; // size at which the heap is triggered to move to host

  // constructors
  __host__ DHEAPExtended(size_t size_limit, uint problem_size, int device_id = 0)
      : DHEAP<NODE>(size_limit, problem_size, device_id)
  {
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_trigger_size, sizeof(size_t)));
  }

  // Destructor
  __host__ void free_memory()
  {
    CUDA_RUNTIME(cudaFree(d_trigger_size));
    DHEAP<NODE>::free_memory();
  }

  // --- Host helper functions ---
  void to_host(HHEAP<NODE> &h_bheap)
  {
    size_t size_limit = d_size_limit[0];
    size_t heap_size = d_size[0];
    auto &h_heap = h_bheap.heap;
    auto &h_node_space = h_bheap.node_space;
    auto &h_fixed_assignment_space = h_bheap.fixed_assignment_space;

    h_heap.resize(heap_size);
    h_node_space.resize(size_limit);
    h_fixed_assignment_space.resize(psize * size_limit);

    CUDA_RUNTIME(cudaMemcpy(h_heap.data(), d_heap, sizeof(NODE) * heap_size, cudaMemcpyDeviceToHost));
    CUDA_RUNTIME(cudaMemcpy(h_node_space.data(), d_node_space, sizeof(node_info) * size_limit, cudaMemcpyDeviceToHost));
    CUDA_RUNTIME(cudaMemcpy(h_fixed_assignment_space.data(), d_fixed_assignment_space,
                            psize * size_limit * sizeof(int), cudaMemcpyDeviceToHost));

// move non standard offsets correctly
#pragma omp parallel for
    for (size_t i = 0; i < heap_size; i++)
    {
      size_t offset = h_heap[i].value - d_node_space;
      h_heap[i].value = &h_node_space[offset];
      h_heap[i].value->fixed_assignments = &h_fixed_assignment_space[offset * psize];
    }
    h_bheap.update_size();
  }

  // sort d_heap in ascending order with thrust
  void sort()
  {
    Log(debug, "Started sorting device heap");
    if (d_size[0] != 0)
    {
      thrust::device_ptr<NODE> dev_ptr(d_heap);
      thrust::sort(dev_ptr, dev_ptr + d_size[0]);
    }
    Log(debug, "Finished sorting the device heap");
  }

  /* Convert the heap into standard format, defined as:
  1. The heap is sorted in ascending order of keys
  2. The heap node values in device memory are in a continuous order:
      i.e. d_heap[i].value at location d_node_space[i] for all i < d_size
  This format would allow efficient batch operations on the heap when the heap is moved to host
  */
  void standardize(int grid_dim)
  {
    sort(); // sort the heap
    Log(debug, "Started standardizing device heap");
    // define where arrray
    // where[i] = j -> information at i should be stored at j
    // where[i] = -1 -> There is no information at i
    int *where;
    int *visited; // To lock visited nodes along a cycle, important when doing things in parallel
    size_t size_limit = d_size_limit[0];
    CUDA_RUNTIME(cudaMalloc((void **)&where, size_limit * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&visited, size_limit * sizeof(int)));
    // CUDA_RUNTIME(cudaMemset(visited, 0, size_limit * sizeof(int)));

    int block_dim = 32;
    int grid_dim_where = std::min(int(size_limit + block_dim - 1) / block_dim, grid_dim);
    assert(grid_dim_where > 0);
    execKernel(set_where, grid_dim_where, block_dim, dev_, false, *this, where, visited);
    execKernel(update_where, grid_dim_where, block_dim, dev_, false, *this, where);
    // printDeviceArray(where, size_limit, "where");
    // printDeviceArray(visited, size_limit, "visited");
    // assert(sanity_check(where, size_limit));

    // Create temp memory for node_info and fixed_assignments for each worker: to be used during swap
    node_info *temp_node_space;
    int *temp_fa_space;
    int grid_dim_link = std::min(grid_dim, (2 * grid_dim + block_dim - 1) / block_dim);
    CUDA_RUNTIME(cudaMalloc((void **)&temp_node_space, 2 * grid_dim * sizeof(node_info)));
    CUDA_RUNTIME(cudaMalloc((void **)&temp_fa_space, 2 * grid_dim * psize * sizeof(int)));

    CUDA_RUNTIME(cudaMemset(temp_node_space, 0, 2 * grid_dim * sizeof(node_info)));
    CUDA_RUNTIME(cudaMemset(temp_fa_space, 0, 2 * grid_dim * psize * sizeof(int)));
    execKernel(link_node_fa, grid_dim_link, block_dim, dev_, false,
               temp_node_space, temp_fa_space, 2 * grid_dim, psize);

    // printDeviceMatrix(d_fixed_assignment_space, size_limit, psize, "Fixed assignments:");
    execKernel(rearrange, grid_dim, block_dim, dev_, true, *this,
               where, visited, temp_node_space, psize);
    // printDeviceArray(where, size_limit, "where");
    // printDeviceMatrix(d_fixed_assignment_space, d_size_limit[0], psize, "Fixed assignments:");

    CUDA_RUNTIME(cudaFree(where));
    CUDA_RUNTIME(cudaFree(visited));
    CUDA_RUNTIME(cudaFree(temp_node_space));
    CUDA_RUNTIME(cudaFree(temp_fa_space));
    Log(debug, "Finished standardizing device heap");
  }

  // This function is used to move the later half of the heap to host to free up space on device
  void move_tail(HHEAP<NODE> &h_bheap, const float frac)
  {
    Log(info, "Launching move tail");
    auto &h_heap = h_bheap.heap;
    auto &h_node_space = h_bheap.node_space;
    auto &h_fixed_assignment_space = h_bheap.fixed_assignment_space;
    Log(debug, "Moving tail to host");
    d_trigger_size[0] = d_size[0];
    Log(debug, "Trigger size: %lu", d_trigger_size[0]);
    uint nelements = (d_size[0] - d_size_limit[0] / 2) > (frac * d_size[0]) ? (d_size[0] - d_size_limit[0] / 2) : (frac * d_size[0]);
    uint last = d_size[0] - nelements;
    h_bheap.attach(d_heap, d_node_space, d_fixed_assignment_space, last, nelements);
    d_size[0] = last;
    Log(info, "Host heap size: %lu", h_bheap.size);
  }

  // Move the first half of the host heap to device
  void move_front(HHEAP<NODE> &h_bheap, const uint nelements)
  {
    auto &h_heap = h_bheap.heap;
    auto &h_node_space = h_bheap.node_space;
    auto &h_fixed_assignment_space = h_bheap.fixed_assignment_space;
    size_t start = d_size[0];
    assert(start == 0);
    CUDA_RUNTIME(cudaMemcpy(d_heap + start, h_heap.data(), sizeof(NODE) * nelements, cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMemcpy(d_node_space + start, h_node_space.data(), sizeof(node_info) * nelements, cudaMemcpyHostToDevice));
    CUDA_RUNTIME(cudaMemcpy(d_fixed_assignment_space + start * psize, h_fixed_assignment_space.data(), nelements * psize * sizeof(int), cudaMemcpyHostToDevice));
    d_size[0] += nelements;
    uint block_dim = 1024;
    uint grid_dim = (size_t(nelements) + block_dim - 1) / block_dim;
    execKernel(link_value_node, grid_dim, block_dim, dev_, false,
               d_heap, d_node_space, d_fixed_assignment_space, d_size[0], psize, start);

    h_heap.erase(h_heap.begin(), h_heap.begin() + nelements);
    h_node_space.erase(h_node_space.begin(), h_node_space.begin() + nelements);
    h_fixed_assignment_space.erase(h_fixed_assignment_space.begin(), h_fixed_assignment_space.begin() + nelements * psize);
    h_bheap.update_size();
    h_bheap.update_pointers();
    Log(debug, "Finished moving front to device");
    // format_print("Device heap after moving front");
    // h_bheap.print("Host moving front");
  }

  // Utility functions
  void simple_print()
  {
    size_t *h_size = (size_t *)malloc(sizeof(size_t));
    CUDA_RUNTIME(cudaMemcpy(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    NODE *h_heap = (NODE *)malloc(sizeof(NODE) * h_size[0]);
    CUDA_RUNTIME(cudaMemcpy(h_heap, d_heap, sizeof(NODE) * h_size[0], cudaMemcpyDeviceToHost));

    printf("heap size: %lu\n", h_size[0]);
    for (size_t i = 0; i < h_size[0]; i++)
    {
      printf("%u, ", (uint)h_heap[i].key);
    }
    printf("\n");
  }

  void format_print(std::string name = NULL)
  {
    HHEAP<NODE> h_heap(psize);
    this->to_host(h_heap);
    h_heap.print(name);

    // free memory
    h_heap.cleanup();
  }

  void print_node_space(const char *label)
  {
    size_t size = d_size_limit[0];
    std::vector<node_info> h_node_space(size); // Allocate host memory
    cudaMemcpy(h_node_space.data(), d_node_space, size * sizeof(node_info), cudaMemcpyDeviceToHost);
    // Print the values
    std::cout << label << " (size = " << size << "):\n";
    for (int i = 0; i < size; i++)
    {
      std::cout << "Index " << i << " -> LB: " << h_node_space[i].LB
                << ", Level: " << h_node_space[i].level
                << ", ID: " << h_node_space[i].id << "\n";
    }
  }

  bool sanity_check(int *where, const size_t size_limit)
  {
    int block_dim = 32;
    int grid_dim = (size_limit + block_dim - 1) / block_dim;
    int *d_bitmap;
    int *d_result, h_result;
    CUDA_RUNTIME(cudaMalloc((void **)&d_bitmap, size_limit * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&d_result, sizeof(int)));
    CUDA_RUNTIME(cudaMemset(d_bitmap, 0, size_limit * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(d_result, 0, sizeof(int)));
    execKernel(check_duplicates, grid_dim, block_dim, dev_, false, where, d_bitmap, d_result, size_limit);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    CUDA_RUNTIME(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_RUNTIME(cudaFree(d_bitmap));
    CUDA_RUNTIME(cudaFree(d_result));
    return h_result == 0;
  }
};

// Standardize kernels
template <typename NODE>
__global__ void set_where(DHEAP<NODE> heap, int *where, int *visited)
{
  size_t size = heap.d_size_limit[0];
  NODE *d_heap = heap.d_heap;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < size)
  {
    where[i] = -1;
    visited[i] = -1;
    i += blockDim.x * gridDim.x;
  }
}

template <typename NODE>
__global__ void update_where(const DHEAP<NODE> heap, int *where)
{
  size_t size = heap.d_size[0];
  NODE *d_heap = heap.d_heap;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < size)
  {
    where[d_heap[i].value->id] = i;
    i += blockDim.x * gridDim.x;
  }
}

template <typename NODE>
__global__ void link_value_node(NODE *heap, node_info *node_space, int *fa_space, const size_t size_limit, const uint psize, const size_t start = 0)
{
  size_t g_id = blockIdx.x * blockDim.x + threadIdx.x + start;
  for (size_t i = g_id; i < size_limit; i += blockDim.x * gridDim.x)
  {
    heap[i].value = &node_space[i];
    node_space[i].id = i;
    node_space[i].fixed_assignments = &fa_space[i * psize];
  }
}

__global__ void link_node_fa(node_info *node_space, int *fa_space, const size_t size_limit, const uint psize, const size_t start = 0)
{
  size_t g_id = blockIdx.x * blockDim.x + threadIdx.x + start;
  for (size_t i = g_id; i < size_limit; i += blockDim.x * gridDim.x)
  {
    node_space[i].fixed_assignments = &fa_space[i * psize];
  }
}

__device__ __forceinline__ LockStatus lock_try(int *loc, uint index) // For cases where cycling is possible
{
  DLog(debug, "Block %u: Locking %u\n", blockIdx.x, index);
  int old;
  uint ns = 1;
  uint counter = 0;
  do
  {
    old = atomicCAS(&loc[index], -1, int(blockIdx.x));
    ns = my_sleep(ns);
    if (old > int(blockIdx.x))
    {
      counter++;
      if (counter >= 20 * 3)
      {
        DLog(critical, "Block %u: giving up on Locking %u\t locked by: %d\n", blockIdx.x, index, old);
        return GIVE_UP;
      }
    }
  } while (old != -1 && old != int(blockIdx.x));
  DLog(debug, "Block %u: Locked %u\n", blockIdx.x, index);
  return SUCCESS;
}

__device__ __forceinline__ void lock(int *loc, uint index)
{
  DLog(debug, "Block %u: Locking %u\n", blockIdx.x, index);
  int old;
  uint ns = 1;
  do
  {
    old = atomicCAS(&loc[index], -1, int(blockIdx.x));
    ns = my_sleep(ns);
  } while (old != -1 && old != int(blockIdx.x));
  DLog(debug, "Block %u: Locked %u\n", blockIdx.x, index);
}

__device__ __forceinline__ void unlock(int *loc, uint index)
{
  DLog(debug, "Block %u: Unlocking %u\n", blockIdx.x, index);
  uint ns = 0;
  int old;
  do
  {
    old = atomicCAS(&loc[index], int(blockIdx.x), -1);
    ns = my_sleep(ns);
  } while (old != int(blockIdx.x) && old != -1);
  DLog(debug, "Block %u: Unlocked %u\n", blockIdx.x, index);
}

template <typename NODE>
__global__ void rearrange(DHEAP<NODE> heap, int *where, int *visited, node_info *temp_space, const uint psize)
{
  size_t heap_size = heap.d_size[0];
  NODE *d_heap = heap.d_heap;
  node_info *node_space = heap.d_node_space;

  __shared__ node_info *temp1, *temp2;
  __shared__ int temp1_dest, temp2_dest;
  __shared__ int my_id, my_dest, where_status;
  __shared__ LockStatus lock_status;

  if (threadIdx.x == 0)
  {
    temp1 = &temp_space[blockIdx.x];
    temp2 = &temp_space[gridDim.x + blockIdx.x];
  }
  __syncthreads();

  // Use block level parallelism for data movement
  for (size_t i = blockIdx.x; i < heap_size; i += gridDim.x)
  {
    __syncthreads();
    if (threadIdx.x == 0)
    {
      my_id = int(d_heap[i].value->id);
      // acquire lock on where[my_id]
      lock(visited, my_id);
      my_dest = atomicRead(&where[my_id]);
      lock_status = lock_try(visited, my_dest); // Cycling possible here
    }
    __syncthreads();
    if (lock_status == GIVE_UP)
    {
      if (threadIdx.x == 0)
        unlock(visited, my_id);
      continue;
    }
    if (threadIdx.x == 0)
    {
      where_status = atomicRead(&where[my_dest]);
      DLog(debug, "Block: %lu, my_id: %d, my_dest: %d, where[dest] %d\n", i, my_id, my_dest, where_status);
    }
    __syncthreads();
    if (my_id == my_dest) // case 1: no need to move
    {
      if (threadIdx.x == 0)
      {
        unlock(visited, my_id);
      }
      continue;
    }
    else if (where_status == -1) // case 2: destination is empty
    {
      cpy(&node_space[my_id], &node_space[my_dest], psize); // Thread safe since noone else would try to access my_dest
      if (threadIdx.x == 0)
      {
        d_heap[i].value = &node_space[my_dest];
        atomicExch(&where[my_id], -1);
        unlock(visited, my_id);
        atomicExch(&where[my_dest], my_dest);
        unlock(visited, my_dest);
      }
    }
    else
    {

      // Init cycle
      cpy(&node_space[my_dest], temp1, psize);
      cpy(&node_space[my_id], &node_space[my_dest], psize);
      if (threadIdx.x == 0)
      {
        atomicExch(&where[my_id], int(-1));
        unlock(visited, my_id);

        d_heap[i].value = &node_space[my_dest];
        temp1_dest = atomicRead(&where[my_dest]);
        assert(temp1_dest >= 0);

        lock(visited, temp1_dest);
        atomicExch(&where[my_dest], my_dest);
        unlock(visited, my_dest);
      }
      __syncthreads();

      // Cycle
      if (threadIdx.x == 0)
        where_status = atomicRead(&where[temp1_dest]);
      __syncthreads();
      while (where_status != -1)
      {
        cpy(&node_space[temp1_dest], temp2, psize);
        cpy(temp1, &node_space[temp1_dest], psize);
        if (threadIdx.x == 0)
        {
          d_heap[temp1_dest].value = &node_space[temp1_dest];
          temp2_dest = atomicRead(&where[temp1_dest]);
          atomicExch(&where[temp1_dest], temp1_dest);
          unlock(visited, temp1_dest);
        }
        __syncthreads();
        cpy(temp2, temp1, psize);
        if (threadIdx.x == 0)
        {
          temp1_dest = temp2_dest;
          lock(visited, temp1_dest);
          where_status = atomicRead(&where[temp1_dest]);
        }
        __syncthreads();
      }

      // End cycle
      cpy(temp1, &node_space[temp1_dest], psize);
      if (threadIdx.x == 0)
      {
        d_heap[temp1_dest].value = &node_space[temp1_dest];
        atomicExch(&where[temp1_dest], temp1_dest);
        unlock(visited, temp1_dest);
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void cpy(node_info *source, node_info *dest, const uint psize)
{
  if (threadIdx.x == 0)
  {
    dest->LB = source->LB;
    dest->level = source->level;
  }
  for (uint j = threadIdx.x; j < psize; j += blockDim.x)
  {
    dest->fixed_assignments[j] = source->fixed_assignments[j];
  }
  __syncthreads();
}

__global__ void check_duplicates(int *where, int *bitmap, int *d_result, const size_t size_limit)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < size_limit)
  {
    int val = where[i];
    if (val >= int(size_limit))
    {
      printf("Thread %lu incountered high value: %d\n", i, val);
      atomicAdd(d_result, 1);
    }
    if (val > -1 && val < size_limit)
    {
      bool old = bool(atomicOr(&bitmap[val], int(true)));
      if (old)
      {
        printf("Thread %lu incountered duplicate value: %d\n", i, val);
        atomicAdd(d_result, 1);
      }
    }
    i += blockDim.x * gridDim.x;
  }
}