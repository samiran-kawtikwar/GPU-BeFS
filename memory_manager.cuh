#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "queue/queue.cuh"
#include "defs.cuh"
#include "utils/logger.cuh"
#include "heap/bheap.cuh"

__global__ void fill_memory_queue(queue_callee(queue, tickets, head, tail),
                                  node_info *node_space,
                                  uint memory_queue_len)
{
  if (threadIdx.x == 0)
  {
    queue_enqueue(queue, tickets, head, tail, memory_queue_len, blockIdx.x);
    node_space[blockIdx.x].id = blockIdx.x;
  }
}

// Should always be called by single block
__device__ void check_queue(queue_callee(queue, tickets, head, tail),
                            uint len)
{
  if (threadIdx.x == 0)
  {
    bool full = queue_full(queue, tickets, head, tail, len);
    DLog(info, "Queue full: %s\n", full ? "true" : "false");
  }
}

// Should always be called by single block
__device__ void queue_length(queue_callee(queue, tickets, head, tail))
{
  if (threadIdx.x == 0)
  {
    uint size = *tail_queue - *head_queue;
    DLog(info, "Queue size: %u\n", size);
  }
}

// Should always be called by single block
__device__ void get_memory(queue_callee(queue, tickets, head, tail),
                           uint queue_size,
                           uint n_tokens,
                           uint *dequeued_idx)
{
  __shared__ bool fork;
  __shared__ uint qidx, count;
  // try dequeue
  if (threadIdx.x == 0)
    count = 0;
  __syncthreads();
  while (count < n_tokens && !heap_overflow.load(cuda::memory_order_relaxed))
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
        uint size = tail_queue->load(cuda::memory_order_relaxed) - head_queue->load(cuda::memory_order_relaxed);
        if (size < n_tokens)
        {
          DLog(warn, "Memory queue empty, block %u ran out of space :(\n", blockIdx.x);
          heap_overflow.store(true, cuda::memory_order_release);
        }
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
                                  uint *dequeued_idx)
{
  get_memory(queue_caller(queue, tickets, head, tail), queue_size, n_tokens,
             &dequeued_idx[blockIdx.x * MAX_TOKENS]);
}

__global__ void free_memory_global(queue_callee(queue, tickets, head, tail),
                                   uint queue_size,
                                   uint *free_index)
{
  free_memory(queue_caller(queue, tickets, head, tail), queue_size, free_index[blockIdx.x]);
}

__global__ void check_queue_global(queue_callee(queue, tickets, head, tail),
                                   uint queue_size)
{
  if (blockIdx.x == 0)
    check_queue(queue_caller(queue, tickets, head, tail), queue_size);
}

__global__ void get_queue_length_global(queue_callee(queue, tickets, head, tail))
{
  if (blockIdx.x == 0)
    queue_length(queue_caller(queue, tickets, head, tail));
}

__global__ void memory_test(queue_callee(queue, tickets, head, tail),
                            uint queue_size,
                            uint n_tokens, uint *dequeued_idx,
                            uint *free_index)
{
  if (threadIdx.x == 0)
    DLog(debug, "Block %u, starting get memory\n", blockIdx.x);

  get_memory(queue_caller(queue, tickets, head, tail), queue_size, n_tokens,
             &dequeued_idx[blockIdx.x * MAX_TOKENS]);
  __syncthreads();
  if (threadIdx.x == 0)
    DLog(debug, "Block %u, finished get memory\n", blockIdx.x);

  free_memory(queue_caller(queue, tickets, head, tail), queue_size, free_index[blockIdx.x]);
}
