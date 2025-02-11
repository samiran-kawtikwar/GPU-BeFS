#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../defs.cuh"
#include "../queue/queue.cuh"
#include "../utils/logger.cuh"
#include "../utils/cuda_utils.cuh"

template <typename NODE>
class HHEAP; // Forward declaration

/*
DHEAP: A device heap implementation
1. This is a min-heap maintained as an array of NODEs
2. All heap operations are implemented as device functions: they are NOT thread-safe
3. Heap operations must be performed by master
*/
template <typename NODE>
class DHEAP
{

public:
  uint psize; // problem size
  uint dev_;
  NODE *d_heap;
  node_info *d_node_space;
  int *d_fixed_assignment_space;
  size_t *d_size;       // live size of the device heap
  size_t *d_max_size;   // max size of the device heap during kernel execution
  size_t *d_size_limit; // max allowed size of the device heap

  // Constructors
  __host__ DHEAP(size_t size_limit, uint problem_size, int device_id = 0)
  {
    psize = problem_size;
    dev_ = device_id;
    CUDA_RUNTIME(cudaSetDevice(device_id));
    CUDA_RUNTIME(cudaMalloc((void **)&d_heap, size_limit * sizeof(NODE)));
    CUDA_RUNTIME(cudaMalloc((void **)&d_node_space, size_limit * sizeof(node_info)));
    CUDA_RUNTIME(cudaMalloc((void **)&d_fixed_assignment_space, size_limit * psize * sizeof(int)));

    CUDA_RUNTIME(cudaMallocManaged((void **)&d_size, sizeof(size_t)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_max_size, sizeof(size_t)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_size_limit, sizeof(size_t)));

    CUDA_RUNTIME(cudaMemset((void *)d_node_space, 0, size_limit * sizeof(node_info)));
    CUDA_RUNTIME(cudaMemset((void *)d_fixed_assignment_space, 0, size_limit * psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(d_size, 0, sizeof(size_t)));
    d_max_size[0] = 0;
    d_size_limit[0] = size_limit;
  }

  __device__ DHEAP();

  // Destructors
  __host__ void free_memory()
  {
    CUDA_RUNTIME(cudaFree(d_heap));
    CUDA_RUNTIME(cudaFree(d_node_space));
    CUDA_RUNTIME(cudaFree(d_fixed_assignment_space));

    CUDA_RUNTIME(cudaFree(d_size));
    CUDA_RUNTIME(cudaFree(d_max_size));
    CUDA_RUNTIME(cudaFree(d_size_limit));
  }
};

// Heap operations: push, pop, batch_push
template <typename NODE>
__device__ void pop(DHEAP<NODE> heap, NODE &min)
{
  if (threadIdx.x == 0)
  {
    NODE *h = heap.d_heap;
    size_t size = heap.d_size[0];
    if (size == 0)
    {
      printf("heap underflow!!\n");
      min = NODE(0, 0);
    }
    min = h[0];
    h[0] = h[size - 1];
    size_t i = 0;
    // Down heapiy the heap to maintain min heap property
    while (2 * i + 1 < size)
    {
      size_t j = 2 * i + 1;
      if (j + 1 < size && h[j + 1].key < h[j].key)
      {
        j++;
      }
      if (h[i].key < h[j].key)
      {
        break;
      }
      NODE temp = h[i];
      h[i] = h[j];
      h[j] = temp;
      i = j;
    }

    heap.d_size[0]--;
  }
  __syncthreads();
};

template <typename NODE>
__device__ void push(DHEAP<NODE> bheap, NODE new_node)
{
  if (threadIdx.x == 0)
  {
    NODE *heap = bheap.d_heap;
    size_t size = bheap.d_size[0];
    if (size >= bheap.d_size_limit[0])
    {
      DLog(critical, "heap overflow!!\n");
      return;
    }
    heap[size] = new_node;
    // Up heapify the heap to maintain min heap property
    size_t i = size;
    while (i > 0 && heap[i].key < heap[(i - 1) / 2].key)
    {
      NODE temp = heap[i];
      heap[i] = heap[(i - 1) / 2];
      heap[(i - 1) / 2] = temp;
      i = (i - 1) / 2;
    }
    bheap.d_size[0]++;
    bheap.d_max_size[0] = max(bheap.d_max_size[0], bheap.d_size[0]);
    // printf("pushed: %f: heap size: %lu\n", new_node.key, bheap.d_size[0]);
  }
  return;
};

template <typename NODE>
__device__ void batch_push(DHEAP<NODE> heap, NODE *new_nodes, size_t num_nodes)
{
  for (int i = 0; i < num_nodes; i++)
  {
    push(heap, new_nodes[i]);
  }
};

// Driver functions
template <typename NODE>
__global__ void parse_instr(DHEAP<NODE> heap, d_instruction *ins_list, size_t INS_LEN, size_t MAX_BATCH)
{
  if (blockIdx.x == 0)
  {
    NODE min;
    for (uint iter = 0; iter < INS_LEN; iter++)
    {
      switch (ins_list[iter].type)
      {
      case PUSH:
        push(heap, ins_list[iter].values[0]);
        break;
      case POP:
        min = pop(heap);
        if (threadIdx.x == 0)
          printf("popped: min: %f\n", min.key);
        break;
      case BATCH_PUSH:
        batch_push(heap, ins_list[iter].values, ins_list[iter].num_values);
        break;
      default:
        printf("Reached default\n");
        break;
      }
      __syncthreads();
    }
  }
}
