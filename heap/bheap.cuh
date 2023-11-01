#include "../utils/logger.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../defs.cuh"
#include "../queue/queue.cuh"

__device__ uint count = 0;

template <typename NODE>
class BHEAP
{
public:
  NODE *d_heap;
  size_t *d_size;
  void print()
  {
    size_t *h_size = (size_t *)malloc(sizeof(size_t));
    CUDA_RUNTIME(cudaMemcpy(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    NODE *h_heap = (NODE *)malloc(sizeof(NODE) * h_size[0]);
    CUDA_RUNTIME(cudaMemcpy(h_heap, d_heap, sizeof(NODE) * h_size[0], cudaMemcpyDeviceToHost));

    printf("heap size: %lu\n", h_size[0]);
    for (size_t i = 0; i < h_size[0]; i++)
    {
      printf("%f, ", h_heap[i].key);
    }
    printf("\n");
  }
};

template <typename NODE>
__device__ NODE pop(BHEAP<NODE> heap)
{
  __shared__ NODE min;
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
  return min;
};

template <typename NODE>
__device__ void push(BHEAP<NODE> bheap, NODE new_node)
{
  if (threadIdx.x == 0)
  {
    NODE *heap = bheap.d_heap;
    size_t size = bheap.d_size[0];
    if (size >= MAX_HEAP_SIZE)
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

template <typename NODE>
__global__ void parse_instr(BHEAP<NODE> heap, d_instruction *ins_list, size_t INS_LEN, size_t MAX_BATCH /*, pass the queue*/)
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
        NODE min = pop(heap);
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

template <typename NODE>
__device__ void process_requests(size_t INS_LEN,
                                 queue_callee(queue, tickets, head, tail),
                                 cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready,
                                 uint32_t queue_size,
                                 BHEAP<NODE> heap, queue_info *queue_space)
{
  if (blockIdx.x == 0)
  {
    __shared__ bool fork;
    __shared__ uint32_t qidx, dequeued_idx;
    __shared__ TaskType task_type;

    while (count < INS_LEN)
    {
      // Dequeue here
      if (threadIdx.x == 0)
        fork = false;
      __syncthreads();
      if (threadIdx.x == 0)
      {
        // try dequeue
        queue_dequeue(queue, tickets, head, tail, queue_size, fork, qidx, N_RECEPIENTS);
      }
      __syncthreads();
      if (fork)
      {
        if (threadIdx.x == 0)
        {
          for (uint iter = 0; iter < N_RECEPIENTS; iter++)
          {
            queue_wait_ticket(queue, tickets, head, tail, queue_size, qidx, dequeued_idx);

            // TODO: copy memory from queue space[dequeued_idx] to queue_space[own_idx]
            task_type = queue_space[dequeued_idx].type;
            queue_space[blockIdx.x].type = task_type;
            queue_space[blockIdx.x].batch_size = queue_space[dequeued_idx].batch_size;
            for (uint i = 0; i < queue_space[dequeued_idx].batch_size; i++)
              queue_space[blockIdx.x].values[i] = queue_space[dequeued_idx].values[i];
            queue_space[dequeued_idx].already_occupied = int(false);

            work_ready[dequeued_idx].store(1, cuda::memory_order_relaxed);
            qidx++;
          }
        }

        __syncthreads();
        if (blockIdx.x == 0)
        {
          switch (task_type)
          {
            {
            case PUSH:
              push(heap, queue_space[blockIdx.x].values[0]);
              break;
            case POP:
              NODE min = pop(heap);
              break;
            case BATCH_PUSH:
              batch_push(heap, queue_space[blockIdx.x].values, queue_space[blockIdx.x].batch_size);
              break;
            default:
              printf("Reached default\n");
              break;
            }
          }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
          atomicAdd(&(count), 1);
        }
      }
    }
    return;
  }
}

template <typename NODE>
__device__ void generate_requests(d_instruction *ins_list, size_t INS_LEN,
                                  queue_callee(queue, tickets, head, tail),
                                  cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready,
                                  uint32_t queue_size,
                                  queue_info *queue_space)
{
  if (blockIdx.x != 0)
  {
    for (uint iter = blockIdx.x - 1; iter < INS_LEN; iter += gridDim.x - 1)
    {
      // uint id = iter + 1;
      __shared__ bool space_free;

      while (true)
      {
        if (threadIdx.x == 0)
        {
          if (atomicOr(&queue_space[blockIdx.x].already_occupied, 1) == 0)
            space_free = true;
          else
            space_free = false;
        }
        __syncthreads();
        if (space_free)
        {
          if (threadIdx.x == 0)
          {
            queue_space[blockIdx.x].type = ins_list[iter].type;
            queue_space[blockIdx.x].batch_size = ins_list[iter].num_values;
          }
          for (uint i = threadIdx.x; i < ins_list[iter].num_values; i += blockDim.x)
            queue_space[blockIdx.x].values[i] = ins_list[iter].values[i];
          break;
        }
        else
        {
          // sleep block for some time and check again
        }
        __syncthreads();
      }
      __syncthreads();
      if (threadIdx.x == 0)
      {
        queue_enqueue(queue, tickets, head, tail, queue_size, blockIdx.x);
      }
      __syncthreads();
    }
  }
}

template <typename NODE>
__global__ void request_manager(d_instruction *ins_list, size_t INS_LEN,
                                queue_callee(queue, tickets, head, tail),
                                cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready,
                                uint32_t queue_size,
                                BHEAP<NODE> heap, queue_info *queue_space)
{
  generate_requests<NODE>(ins_list, INS_LEN, queue_caller(queue, tickets, head, tail), work_ready, queue_size, queue_space);

  process_requests<NODE>(INS_LEN, queue_caller(queue, tickets, head, tail), work_ready, queue_size, heap, queue_space);
}