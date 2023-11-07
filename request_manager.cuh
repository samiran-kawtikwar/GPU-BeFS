#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "queue/queue.cuh"
#include "defs.cuh"
#include "utils/logger.cuh"
#include "heap/bheap.cuh"

template <typename NODE>
__device__ void process_requests(size_t INS_LEN,
                                 queue_callee(queue, tickets, head, tail),
                                 cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready,
                                 uint32_t queue_size,
                                 BHEAP<NODE> heap, queue_info *queue_space)
{
  __shared__ bool fork;
  __shared__ uint qidx, dequeued_idx;
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
        NODE min;
        switch (task_type)
        {
          {
          case PUSH:
            push(heap, queue_space[blockIdx.x].values[0]);
            break;
          case POP:
            min = pop(heap);
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
        atomicAdd(&(count), 1);
    }
    __syncthreads();
  }
  return;
}

template <typename NODE>
__device__ void generate_requests(d_instruction *ins_list, size_t INS_LEN,
                                  queue_callee(queue, tickets, head, tail),
                                  cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready,
                                  uint32_t queue_size,
                                  queue_info *queue_space)
{

  for (uint iter = blockIdx.x - 1; iter < INS_LEN; iter += gridDim.x - 1)
  {
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

template <typename NODE>
__global__ void request_manager(d_instruction *ins_list, size_t INS_LEN,
                                queue_callee(queue, tickets, head, tail),
                                cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready,
                                uint32_t queue_size,
                                BHEAP<NODE> heap, queue_info *queue_space)
{
  if (blockIdx.x == 0)
    process_requests<NODE>(INS_LEN, queue_caller(queue, tickets, head, tail), work_ready, queue_size, heap, queue_space);
  else
    generate_requests<NODE>(ins_list, INS_LEN, queue_caller(queue, tickets, head, tail), work_ready, queue_size, queue_space);
}
