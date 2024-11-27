#pragma once

#include "../utils/logger.cuh"
#include "../utils/cuda_utils.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../defs.cuh"
#include "../queue/queue.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

template <typename NODE>
class BHEAP
{
public:
  NODE *d_heap;
  size_t *d_size, *d_max_size, *d_size_limit;
  // Constructors
  __host__ BHEAP(size_t size_limit, int device_id = 0)
  {
    CUDA_RUNTIME(cudaSetDevice(device_id));
    CUDA_RUNTIME(cudaMalloc((void **)&d_heap, sizeof(NODE) * size_limit));
    CUDA_RUNTIME(cudaMalloc((void **)&d_size, sizeof(size_t)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_max_size, sizeof(size_t)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_size_limit, sizeof(size_t)));

    CUDA_RUNTIME(cudaMemset(d_size, 0, sizeof(size_t)));
    d_max_size[0] = 0;
    d_size_limit[0] = size_limit;
  }
  __device__ BHEAP();

  // Destructors
  __host__ void free_memory()
  {
    CUDA_RUNTIME(cudaFree(d_heap));
    CUDA_RUNTIME(cudaFree(d_size));
    CUDA_RUNTIME(cudaFree(d_max_size));
    CUDA_RUNTIME(cudaFree(d_size_limit));
  }

  void print()
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
  void print_size()
  {
    size_t *h_size = (size_t *)malloc(sizeof(size_t));
    CUDA_RUNTIME(cudaMemcpy(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    Log(info, "Heap size at termination: %lu", h_size[0]);
  }
  void sort()
  {
    Log(info, "Sorting the heap");
    size_t *h_size = (size_t *)malloc(sizeof(size_t));
    CUDA_RUNTIME(cudaMemcpy(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    // sort d_heap in ascending order with CUB
    if (h_size[0] == 0)
    {
      // No elements to sort
      free(h_size);
      return;
    }
    // Use thrust::device_ptr to wrap the device memory
    thrust::device_ptr<NODE> dev_ptr(d_heap);
    // Call thrust::sort to sort the heap in-place
    thrust::sort(dev_ptr, dev_ptr + h_size[0]);
    // copy the sorted heap back to host
    free(h_size); // Free the host-side memory
  }
  void move_tail(const float frac)
  {
    // This function is used to move the later half of the heap to host to free up space on device
    size_t *h_size = (size_t *)malloc(sizeof(size_t));
    CUDA_RUNTIME(cudaMemcpy(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    uint nelements = (uint)h_size[0] * frac;
    uint last = h_size[0] - nelements;
    NODE *h_heap = (NODE *)malloc(sizeof(NODE) * nelements);
    CUDA_RUNTIME(cudaMemcpy(h_heap, d_heap + last, sizeof(NODE) * nelements, cudaMemcpyDeviceToHost));
    CUDA_RUNTIME(cudaMemcpy(d_size, &last, sizeof(size_t), cudaMemcpyHostToDevice));
    free(h_size);
  }
};

// Heap operations: push, pop, batch_push
template <typename NODE>
__device__ void pop(BHEAP<NODE> heap, NODE &min)
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
__device__ void push(BHEAP<NODE> bheap, NODE new_node)
{
  if (threadIdx.x == 0)
  {
    NODE *heap = bheap.d_heap;
    size_t size = bheap.d_size[0];
    if (size >= bheap.d_size_limit[0])
    {
      printf("heap overflow!!\n");
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
__device__ void batch_push(BHEAP<NODE> heap, NODE *new_nodes, size_t num_nodes)
{
  for (int i = 0; i < num_nodes; i++)
  {
    push(heap, new_nodes[i]);
  }
};

// Driver functions
template <typename NODE>
__global__ void parse_instr(BHEAP<NODE> heap, d_instruction *ins_list, size_t INS_LEN, size_t MAX_BATCH /*, pass the queue*/)
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
