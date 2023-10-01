#include "logger.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "defs.cuh"

template <typename Node>
class BHEAP
{
public:
  Node *d_heap;
  size_t *d_size;
  void print()
  {
    size_t *h_size = (size_t *)malloc(sizeof(size_t));
    CUDA_RUNTIME(cudaMemcpy(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    Node *h_heap = (Node *)malloc(sizeof(Node) * h_size[0]);
    CUDA_RUNTIME(cudaMemcpy(h_heap, d_heap, sizeof(Node) * h_size[0], cudaMemcpyDeviceToHost));

    printf("heap size: %lu\n", h_size[0]);
    for (size_t i = 0; i < h_size[0]; i++)
    {
      printf("%f, ", h_heap[i]);
    }
    printf("\n");
  }
};

template <typename Node>
__device__ Node pop(BHEAP<Node> heap)
{
  __shared__ Node min;
  if (threadIdx.x == 0)
  {
    Node *h = heap.d_heap;
    size_t size = heap.d_size[0];
    if (size == 0)
    {
      printf("heap underflow!!\n");
      min = NULL;
    }
    min = h[0];
    h[0] = h[size - 1];
    size_t i = 0;
    // Down heapiy the heap to maintain min heap property
    while (2 * i + 1 < size)
    {
      size_t j = 2 * i + 1;
      if (j + 1 < size && h[j + 1] < h[j])
      {
        j++;
      }
      if (h[i] < h[j])
      {
        break;
      }
      Node temp = h[i];
      h[i] = h[j];
      h[j] = temp;
      i = j;
    }

    heap.d_size[0]--;
  }
  __syncthreads();
  return min;
};

template <typename Node>
__device__ void push(BHEAP<Node> bheap, Node new_Node)
{
  if (threadIdx.x == 0)
  {
    Node *heap = bheap.d_heap;
    size_t size = bheap.d_size[0];
    if (size >= MAX_HEAP_SIZE)
    {
      printf("heap overflow!!\n");
      return;
    }
    heap[size] = new_Node;
    // Up heapify the heap to maintain min heap property
    size_t i = size;
    while (i > 0 && heap[i] < heap[(i - 1) / 2])
    {
      Node temp = heap[i];
      heap[i] = heap[(i - 1) / 2];
      heap[(i - 1) / 2] = temp;
      i = (i - 1) / 2;
    }
    bheap.d_size[0]++;
  }
  return;
};

template <typename Node>
__device__ void batch_push(BHEAP<Node> heap, Node *new_Nodes, size_t num_Nodes)
{
  for (int i = 0; i < num_Nodes; i++)
  {
    push(heap, new_Nodes[i]);
  }
};

template <typename Node>
__global__ void parse_queue(BHEAP<Node> heap, d_instruction *ins_list, size_t INS_LEN, size_t MAX_BATCH /*, pass the queue*/)
{
  if (blockIdx.x == 0)
  {
    for (uint iter = 0; iter < INS_LEN; iter++)
    {
      switch (ins_list[iter].type)
      {
      case PUSH:
        push(heap, ins_list[iter].values[0]);
        break;
      case POP:
        Node min = pop(heap);
        break;
      case BATCH_PUSH:
        batch_push(heap, ins_list[iter].values, MAX_BATCH);
        break;
      default:
        printf("Reached default\n");
        break;
      }
      __syncthreads();
    }
  }
}
