#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "queue/queue.cuh"
#include "defs.cuh"
#include "utils/logger.cuh"
#include "heap/dheap.cuh"

__global__ void fill_memory_queue(queue_callee(queue, tickets, head, tail),
                                  node_info *node_space,
                                  uint memory_queue_len)
{
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < memory_queue_len)
  {
    queue_enqueue(queue, tickets, head, tail, memory_queue_len, id);
    node_space[id].id = id;
  }
}

// Add free indexes back to memory queue
__global__ void refill_tail(queue_callee(queue, tickets, head, tail),
                            uint memory_queue_len,
                            DHEAPExtended<node> bheap)
{
  size_t global_id = blockIdx.x * blockDim.x + threadIdx.x + bheap.d_size[0];
  if (global_id < bheap.d_trigger_size[0])
  {
    uint id = bheap.d_heap[global_id].value->id; // id in node space
    queue_enqueue(queue, tickets, head, tail, memory_queue_len, id);
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
                           uint *dequeued_idx,
                           bool *overflow_flag = nullptr)
{
  __shared__ bool fork;
  __shared__ uint qidx, sh_count;
  if (overflow_flag == nullptr)
  {
    __shared__ bool o_flag; // define overflow flag when not provided
    overflow_flag = &o_flag;
  }
  // try dequeue
  if (threadIdx.x == 0)
  {
    sh_count = 0;
    *overflow_flag = false;
    if (heap_overflow.load(cuda::memory_order_relaxed))
      *overflow_flag = true;
  }
  __syncthreads();
  while (sh_count < n_tokens && !*overflow_flag)
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
          sh_count++;
        }
      }
      else
      {
        uint size = tail_queue->load(cuda::memory_order_relaxed) - head_queue->load(cuda::memory_order_relaxed);
        if (size < n_tokens)
        {
          DLog(debug, "no space for block %u: available: %u, needed: %u\n", blockIdx.x, size, n_tokens);
          heap_overflow.store(true, cuda::memory_order_release);
          *overflow_flag = true;
        }
      }
    }
    __syncthreads();
    // sleep block here if needed
  }
  if (threadIdx.x == 0 && !*overflow_flag)
    DLog(debug, "Block %u got memory\n", blockIdx.x, n_tokens);
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
