#pragma once

#include "../utils/logger.cuh"
#include "../utils/cuda_utils.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../defs.cuh"
#include "../queue/queue.cuh"
#include "hheap.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

/*
DHEAP: A device heap implementation
1. This is a min-heap maintained as an array of NODEs
2. All heap operations are implemented as device functions: they are NOT thread-safe
3. Heap operations must be performed by master
*/
template <typename NODE>
class DHEAP
{
private:
  uint psize; // problem size

public:
  NODE *d_heap;
  node_info *d_node_space;
  int *d_fixed_assignment_space;
  size_t *d_size;         // live size of the device heap
  size_t *d_max_size;     // max size of the device heap during kernel execution
  size_t *d_size_limit;   // max allowed size of the device heap
  size_t *d_trigger_size; // size at which the heap is triggered to move to host

  // Constructors
  __host__ DHEAP(size_t size_limit, uint problem_size, int device_id = 0)
  {
    psize = problem_size;
    CUDA_RUNTIME(cudaSetDevice(device_id));
    CUDA_RUNTIME(cudaMalloc((void **)&d_heap, size_limit * sizeof(NODE)));
    CUDA_RUNTIME(cudaMalloc((void **)&d_node_space, size_limit * sizeof(node_info)));
    CUDA_RUNTIME(cudaMalloc((void **)&d_fixed_assignment_space, size_limit * psize * sizeof(int)));

    CUDA_RUNTIME(cudaMallocManaged((void **)&d_size, sizeof(size_t)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_max_size, sizeof(size_t)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_size_limit, sizeof(size_t)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_trigger_size, sizeof(size_t)));

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
    CUDA_RUNTIME(cudaFree(d_trigger_size));
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

  // sort d_heap in ascending order with thrust
  void sort()
  {
    Log(info, "Sorting the heap");
    if (d_size[0] != 0)
    {
      thrust::device_ptr<NODE> dev_ptr(d_heap);
      thrust::sort(dev_ptr, dev_ptr + d_size[0]);
    }
  }

  /* Convert the heap into standard format, defined as:
  1. The heap is sorted in ascending order of keys
  2. The heap node values in device memory are in a continuous order:
      i.e. d_heap[i].value at location d_node_space[i] for all i < d_size
  This format would allow efficient batch operations on the heap when the heap is moved to host
  */
  void standardize()
  {
    sort(); // sort the heap
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
    uint nelements = max((int)(d_size[0] - d_size_limit[0] / 2), (uint)(frac * d_size[0]));
    uint last = d_size[0] - nelements;
    d_size[0] = last;

    h_heap.resize(h_bheap.size + nelements);
    // h_node_space.resize(h_bheap.size + nelements);
    // h_fixed_assignment_space.resize(h_bheap.size + nelements * psize);
    auto correct_ptr = h_heap.data() + h_bheap.size;
    CUDA_RUNTIME(cudaMemcpy(correct_ptr, d_heap + last, sizeof(NODE) * nelements, cudaMemcpyDeviceToHost));

    // copy heap node values to host; can do this in parallel with cpu threads
    for (size_t i = 0; i < nelements; i++)
    {
      // create memory for node_info
      node_info *temp_node = (node_info *)malloc(sizeof(node_info));
      int *temp_fa = (int *)malloc(psize * sizeof(int));
      // copy corresponding device pointer to host
      CUDA_RUNTIME(cudaMemcpy(temp_node, h_heap[h_heap.size() - nelements + i].value, sizeof(node_info), cudaMemcpyDeviceToHost));
      CUDA_RUNTIME(cudaMemcpy(temp_fa, temp_node->fixed_assignments, psize * sizeof(int), cudaMemcpyDeviceToHost));
      temp_node->fixed_assignments = temp_fa;
      h_heap[h_heap.size() - nelements + i].value = temp_node;
      h_heap[h_heap.size() - nelements + i].location = HOST;
    }
    h_bheap.update_size();
    Log(info, "Host heap size: %lu", h_bheap.size);
  }

  // Move the first half of the host heap to device
  void move_front(HHEAP<NODE> &h_bheap,
                  const uint *id,
                  const uint nelements)
  {
    Log(info, "Launching move front");
    auto &h_heap = h_bheap.heap;
    assert(d_size[0] == 0);
    // print();
    for (size_t i = 0; i < nelements; i++)
    {
      const uint d_id = id[i];
      // copy fixed assignments to device
      CUDA_RUNTIME(cudaMemcpy(&d_fixed_assignment_space[d_id * psize], h_heap[i].value->fixed_assignments,
                              psize * sizeof(int), cudaMemcpyHostToDevice));
      // remap the fixed assignments location
      free(h_heap[i].value->fixed_assignments);
      h_heap[i].value->fixed_assignments = &d_fixed_assignment_space[d_id * psize];
      // copy node info to device
      CUDA_RUNTIME(cudaMemcpy(&d_node_space[d_id], h_heap[i].value,
                              sizeof(node_info), cudaMemcpyHostToDevice));
      // remap the node info location
      free(h_heap[i].value);
      h_heap[i].value = &d_node_space[d_id];
      cudaMemcpy(d_heap, h_heap.data(), sizeof(NODE) * nelements, cudaMemcpyHostToDevice);
      // free memory at h_heap[i]
    }
    d_size[0] = nelements;
    h_heap.erase(h_heap.begin(), h_heap.begin() + nelements);

    Log(debug, "Finished moving front to device");
    // exit(0);
    // copy heap node value to device
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
__device__ void batch_push(DHEAP<NODE> heap, NODE *new_nodes, size_t num_nodes)
{
  for (int i = 0; i < num_nodes; i++)
  {
    push(heap, new_nodes[i]);
  }
};

// Driver functions
template <typename NODE>
__global__ void parse_instr(DHEAP<NODE> heap, d_instruction *ins_list, size_t INS_LEN, size_t MAX_BATCH /*, pass the queue*/)
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
