#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "queue/queue.cuh"
#include "defs.cuh"
#include "utils/logger.cuh"
#include "heap/bheap.cuh"

__device__ cuda::atomic<bool, cuda::thread_scope_device> opt_reached;
__device__ cuda::atomic<bool, cuda::thread_scope_device> heap_overflow;
__device__ cuda::atomic<bool, cuda::thread_scope_device> heap_underflow; // For infeasibility check

template <typename NODE>
__device__ void process_requests_bnb(queue_callee(queue, tickets, head, tail),
                                     uint32_t queue_size,
                                     BHEAP<NODE> heap, queue_info *queue_space,
                                     bool *hold_status)
{
  __shared__ bool fork, opt_flag, overflow_flag, underflow_flag;
  __shared__ uint qidx, dequeued_idx, count, invalid_count;
  __shared__ TaskType task_type;
  if (threadIdx.x == 0)
  {
    invalid_count = 0;
    count = 0;
    opt_flag = false;
    overflow_flag = false;
    underflow_flag = false;
    if (opt_reached.load(cuda::memory_order_relaxed))
      opt_flag = true;
    if (heap_overflow.load(cuda::memory_order_relaxed))
      overflow_flag = true;
    if (heap_underflow.load(cuda::memory_order_relaxed))
      underflow_flag = true;
  }
  __syncthreads();
  while (!opt_flag && !overflow_flag && !underflow_flag)
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
            queue_space[blockIdx.x].nodes[i] = queue_space[dequeued_idx].nodes[i];

          qidx++;
        }
      }

      __syncthreads();
      __shared__ NODE min;
      __shared__ int request_status;
      if (blockIdx.x == 0)
      {
        __syncthreads();
        // if (threadIdx.x == 0)
        //   printf("Block %u is processing %s request for block %u\n", blockIdx.x, getTextForEnum(task_type), dequeued_idx);
        if (threadIdx.x == 0)
          request_status = DONE;
        __syncthreads();
        switch (task_type)
        {
          {
          case PUSH:
            push(heap, queue_space[blockIdx.x].nodes[0]);
            break;
          case POP:
            if (heap.d_size[0] > 0)
            {
              pop(heap, min);
            }
            else
            {
              if (threadIdx.x == 0)
              {
                request_status = INVALID;
                if (hold_status[dequeued_idx] == false)
                {
                  // DLog(debug, "Holding pop request from block %u\n", dequeued_idx);
                  queue_space[dequeued_idx].req_status.store(INVALID, cuda::memory_order_release);
                  invalid_count++;
                }
                hold_status[dequeued_idx] = true;
                // send the pop request back to the queue
                queue_enqueue(queue, tickets, head, tail, queue_size, dequeued_idx);
              }
              __syncthreads();
            }
            break;
          case BATCH_PUSH:
            batch_push(heap, queue_space[blockIdx.x].nodes, queue_space[blockIdx.x].batch_size);
            break;
          default:
            printf("Reached default\n");
            break;
          }
        }
      }
      __syncthreads();

      if (request_status == DONE)
      {
        if (threadIdx.x == 0)
        {
          uint size = tail_queue->load(cuda::memory_order_relaxed) - head_queue->load(cuda::memory_order_relaxed);
          if (task_type == POP)
            queue_space[dequeued_idx].nodes[0] = min;
          queue_space[dequeued_idx].req_status.store(DONE, cuda::memory_order_release);
          atomicAdd(&(count), 1);
          if (count % 100000 == 0)
            DLog(debug, "Processed %u requests\n", count);
          if (task_type == POP && hold_status[dequeued_idx] == true)
          {
            invalid_count--;
            hold_status[dequeued_idx] = false;
          }
          // DLog(debug, "Block %u processed %s request for block %u, queue-size: %u\n", blockIdx.x, getTextForEnum(task_type), dequeued_idx, size);
        }
        // __syncthreads();
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && invalid_count >= gridDim.x - 1)
    {
      heap_underflow.store(true, cuda::memory_order_release); // heap empty
      DLog(critical, "Heap underflow detected at invalid_count: %u\n", invalid_count);
      // print hold status of all blocks
      for (uint i = 0; i < gridDim.x; i++)
        DLog(debug, "Block %u hold status: %s\n", i, hold_status[i] == true ? "true" : "false");
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      if (opt_reached.load(cuda::memory_order_relaxed))
        opt_flag = true;
      if (heap_overflow.load(cuda::memory_order_relaxed))
        overflow_flag = true;
      if (heap_underflow.load(cuda::memory_order_relaxed))
        underflow_flag = true;
    }
    __syncthreads();
  }
  return;
}

// For testing
template <typename NODE>
__device__ void process_requests(uint INS_LEN,
                                 queue_callee(queue, tickets, head, tail),
                                 uint32_t queue_size,
                                 BHEAP<NODE> heap, queue_info *queue_space)
{
  __shared__ bool fork;
  __shared__ uint qidx, dequeued_idx, count, invalid_count;
  __shared__ TaskType task_type;
  if (threadIdx.x == 0)
  {
    invalid_count = 0;
    count = 0;
  }
  __syncthreads();
  while (
      count < INS_LEN &&
      //  head_queue->load(cuda::memory_order_relaxed) != tail_queue->load(cuda::memory_order_relaxed) &&
      // !opt_reached.load(cuda::memory_order_relaxed) &&
      invalid_count < 10)
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
            queue_space[blockIdx.x].nodes[i] = queue_space[dequeued_idx].nodes[i];

          qidx++;
        }
      }

      __syncthreads();
      __shared__ NODE min;
      __shared__ bool request_valid;
      if (blockIdx.x == 0)
      {
        if (threadIdx.x == 0)
          request_valid = true;
        __syncthreads();
        switch (task_type)
        {
          {
          case PUSH:
            push(heap, queue_space[blockIdx.x].nodes[0]);
            break;
          case POP:
            if (heap.d_size[0] > 0)
            {
              pop(heap, min);
            }
            else
            {
              if (threadIdx.x == 0)
              {
                // DLog(debug, "Holding pop request from block %u\n", dequeued_idx);
                request_valid = false;
                invalid_count++;
                // send the pop request back to the queue
                queue_enqueue(queue, tickets, head, tail, queue_size, dequeued_idx);
              }
              __syncthreads();
            }
            break;
          case BATCH_PUSH:
            batch_push(heap, queue_space[blockIdx.x].nodes, queue_space[blockIdx.x].batch_size);
            break;
          default:
            DLog(error, "Reached default\n");
            break;
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0 && request_valid)
      {
        if (task_type == POP)
          queue_space[dequeued_idx].nodes[0] = min;
        queue_space[dequeued_idx].req_status.store(DONE, cuda::memory_order_release);
        // printf("Set %u occupied to false\n", dequeued_idx);
        atomicAdd(&(count), 1);
        invalid_count = 0;
      }
    }
    __syncthreads();
  }
  return;
}

template <typename NODE>
__device__ void generate_request_block(const d_instruction ins,
                                       queue_callee(queue, tickets, head, tail),
                                       uint32_t queue_size,
                                       queue_info *queue_space)
{
  __shared__ bool space_free, termination_flag;

  if (threadIdx.x == 0)
  {
    space_free = false;
    termination_flag = false;
  }
  __syncthreads();
  // wait for previous request to be processed (i.e. space_free)
  uint ns = 8;
  while (true)
  {
    if (threadIdx.x == 0)
    {
      if (queue_space[blockIdx.x].req_status.load(cuda::memory_order_relaxed) == DONE)
      {
        if (queue_space[blockIdx.x].req_status.load(cuda::memory_order_acquire) == DONE)
        {
          queue_space[blockIdx.x].req_status.store(PROCESSING, cuda::memory_order_release);
          space_free = true;
        }
      }
    }
    __syncthreads();
    if (space_free)
      break;

    // optimality reached while a block is waiting for a request
    if (threadIdx.x == 0)
    {
      if (opt_reached.load(cuda::memory_order_relaxed) || heap_overflow.load(cuda::memory_order_relaxed))
      {
        DLog(debug, "Termination reached while waiting to send %s for block %u\n", getTextForEnum(ins.type), blockIdx.x);
        termination_flag = true;
      }
    }
    __syncthreads();
    if (termination_flag)
      return;
    ns = my_sleep(ns);
  }
  __syncthreads();
  if (space_free)
  {
    for (uint i = threadIdx.x; i < ins.num_values; i += blockDim.x)
      queue_space[blockIdx.x].nodes[i] = ins.values[i];
    if (threadIdx.x == 0)
    {
      queue_space[blockIdx.x].type = ins.type;
      queue_space[blockIdx.x].batch_size = ins.num_values;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    queue_enqueue(queue, tickets, head, tail, queue_size, blockIdx.x);
  }
  __syncthreads();
}

template <typename NODE>
__device__ void generate_requests(d_instruction *ins_list, size_t INS_LEN,
                                  queue_callee(queue, tickets, head, tail),
                                  uint32_t queue_size,
                                  queue_info *queue_space)
{

  for (uint iter = blockIdx.x - 1; iter < INS_LEN; iter += gridDim.x - 1)
  {
    generate_request_block<NODE>(ins_list[iter], queue_caller(queue, tickets, head, tail), queue_size, queue_space);
  }
}

__device__ void send_requests(TaskType req_type, size_t req_size, node *nodes,
                              queue_callee(queue, tickets, head, tail),
                              uint32_t queue_size, queue_info *queue_space)
{
  const d_instruction ins = d_instruction(req_type, req_size, nodes);
  generate_request_block<node>(ins, queue_caller(queue, tickets, head, tail), queue_size, queue_space);
}

template <typename NODE>
__global__ void request_manager(d_instruction *ins_list, size_t INS_LEN,
                                queue_callee(queue, tickets, head, tail),
                                uint32_t queue_size,
                                BHEAP<NODE> heap, queue_info *queue_space)
{
  if (blockIdx.x == 0)
    process_requests<NODE>(INS_LEN, queue_caller(queue, tickets, head, tail), queue_size, heap, queue_space);
  else
    generate_requests<NODE>(ins_list, INS_LEN, queue_caller(queue, tickets, head, tail), queue_size, queue_space);
}

__device__ bool wait_for_pop(queue_info *queue_space)
{
  uint ns = 8;
  __shared__ int pop_status;
  __shared__ bool terminate_signal, first_invalid;
  if (threadIdx.x == 0)
  {
    pop_status = PROCESSING;
    terminate_signal = false;
    first_invalid = true;
  }
  __syncthreads();
  while (true)
  {
    if (threadIdx.x == 0)
      pop_status = queue_space[blockIdx.x].req_status.load(cuda::memory_order_relaxed);
    __syncthreads();
    if (pop_status == INVALID && first_invalid)
    {
      START_TIME(WAITING_UNDERFLOW);
      if (threadIdx.x == 0)
      {
        // DLog(debug, "Block %u's pop request was invalid\n", blockIdx.x);
        first_invalid = false;
      }
    }
    __syncthreads();
    if (pop_status == DONE)
    {
      if (!first_invalid)
      {
        END_TIME(WAITING_UNDERFLOW);
      }
      __syncthreads();
      return true;
    }

    // optimality reached while a block is waiting for a pop
    if (threadIdx.x == 0)
    {
      if (opt_reached.load(cuda::memory_order_relaxed) ||
          heap_overflow.load(cuda::memory_order_relaxed) ||
          heap_underflow.load(cuda::memory_order_relaxed))
        terminate_signal = true;
    }
    __syncthreads();
    if (terminate_signal)
      return false;

    ns = my_sleep(ns);
  }
}