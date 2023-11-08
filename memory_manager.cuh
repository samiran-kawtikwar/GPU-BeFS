#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "queue/queue.cuh"
#include "defs.cuh"
#include "utils/logger.cuh"
#include "heap/bheap.cuh"

__global__ void fill_memory_queue(queue_callee(queue, tickets, head, tail),
                                  uint memory_queue_len,
                                  cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready)
{
  if (threadIdx.x == 0)
  {
    queue_enqueue(queue, tickets, head, tail, memory_queue_len, blockIdx.x);
  }
}

// Should always be called by single block
__device__ void check_queue(queue_callee(queue, tickets, head, tail),
                            cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready,
                            uint memory_queue_len)
{
  if (threadIdx.x == 0)
  {
    bool full = queue_full(queue, tickets, head, tail, memory_queue_len);
    printf("Queue full: %s\n", full ? "true" : "false");
  }
}

// Should always be called by single block
__device__ void queue_length(queue_callee(queue, tickets, head, tail))
{
  if (threadIdx.x == 0)
  {
    uint size = *tail_queue - *head_queue;
    printf("Queue size: %u\n", size);
  }
}

// Should always be called by single block
__device__ void get_memory(queue_callee(queue, tickets, head, tail),
                           uint queue_size,
                           uint n_tokens,
                           uint *dequeued_idx,
                           cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready)
{
  __shared__ bool fork;
  __shared__ uint qidx, count;
  // try dequeue
  if (threadIdx.x == 0)
    count = 0;
  __syncthreads();
  while (count < n_tokens)
  {
    if (threadIdx.x == 0)
    {
      queue_dequeue(queue, tickets, head, tail, queue_size, fork, qidx, n_tokens);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      if (fork)
      {
        // do work
        for (uint iter = 0; iter < n_tokens; iter++)
        {
          queue_wait_ticket(queue, tickets, head, tail, queue_size, qidx, dequeued_idx[iter]);
          qidx++;
          count++;
        }
      }
      else
      {
        uint size = *tail_queue - *head_queue;
        if (size == 0)
          printf("Memory queue empty, ran out of space :(\n Queue size: %u\n", size);
      }
    }
    __syncthreads();
    // sleep block here if needed
  }
}

// Should always be called by single block
__device__ void free_memory(queue_callee(queue, tickets, head, tail),
                            uint queue_size,
                            uint free_index)
{
  // Add free_index back to memory queue
  if (threadIdx.x == 0)
    queue_enqueue(queue, tickets, head, tail, queue_size, free_index);
  __syncthreads();
}

// For testing purposes
__global__ void get_memory_global(queue_callee(queue, tickets, head, tail),
                                  uint queue_size,
                                  uint n_tokens,
                                  uint *dequeued_idx,
                                  cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready)
{
  get_memory(queue_caller(queue, tickets, head, tail), queue_size, n_tokens,
             &dequeued_idx[blockIdx.x * MAX_TOKENS], work_ready);
}

__global__ void free_memory_global(queue_callee(queue, tickets, head, tail),
                                   uint queue_size,
                                   uint *free_index)
{
  free_memory(queue_caller(queue, tickets, head, tail), queue_size, free_index[blockIdx.x]);
}

__global__ void check_queue_global(queue_callee(queue, tickets, head, tail),
                                   uint queue_size,
                                   cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready)
{
  if (blockIdx.x == 0)
    check_queue(queue_caller(queue, tickets, head, tail), work_ready, queue_size);
}

__global__ void get_queue_length_global(queue_callee(queue, tickets, head, tail))
{
  if (blockIdx.x == 0)
    queue_length(queue_caller(queue, tickets, head, tail));
}

__global__ void memory_test(queue_callee(queue, tickets, head, tail),
                            uint queue_size,
                            uint n_tokens, uint *dequeued_idx,
                            uint *free_index,
                            cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready)
{
  if (threadIdx.x == 0)
    printf("Block %u, starting get memory\n", blockIdx.x);

  get_memory(queue_caller(queue, tickets, head, tail), queue_size, n_tokens,
             &dequeued_idx[blockIdx.x * MAX_TOKENS], work_ready);
  __syncthreads();
  if (threadIdx.x == 0)
    printf("Block %u, finished get memory\n", blockIdx.x);

  free_memory(queue_caller(queue, tickets, head, tail), queue_size, free_index[blockIdx.x]);
}
