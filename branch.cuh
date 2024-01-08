#pragma once

#include "utils/cuda_utils.cuh"
#include "memory_manager.cuh"
#include "request_manager.cuh"
#include "queue/queue.cuh"
#include "defs.cuh"

__global__ void initial_branching(queue_callee(memory_queue, tickets, head, tail), uint memory_queue_size,
                                  uint psize, uint *addresses_space, node_info *node_space,
                                  const cost_type *original_cost, uint max_node_length,
                                  queue_callee(request_queue, tickets, head, tail), uint request_queue_size,
                                  queue_info *queue_space, work_info *work_space, BHEAP<node> bheap,
                                  const cost_type UB)
{
  const uint bId = blockIdx.x;
  uint *my_addresses = &addresses_space[bId * max_node_length];
  if (bId > 0)
  {
    __shared__ uint nchild, child_index;
    if (threadIdx.x == 0)
    {
      nchild = 0;
      child_index = 0;
    }
    __syncthreads();
    for (uint i = threadIdx.x; i < psize; i += blockDim.x)
    {
      cost_type LB = original_cost[i * psize];
      if (LB < UB)
      {
        atomicAdd(&nchild, 1);
      }
    }
    __syncthreads();
    get_memory(queue_caller(memory_queue, tickets, head, tail), memory_queue_size, nchild,
               my_addresses);
    node *a = work_space[bId].nodes;

    for (uint i = threadIdx.x; i < psize; i += blockDim.x)
    {
      uint my_index = 0;
      cost_type LB = original_cost[i * psize];
      if (LB < UB)
      {
        my_index = atomicAdd(&child_index, 1);
        node_info *b = &node_space[my_addresses[my_index]];
        b->fixed_assignments[i] = 1; // fix row i to column lvl + 1.
        b->LB = (float)original_cost[i * psize];
        b->level = 1;
        a[my_index].value = b;
        a[my_index].key = (float)original_cost[i * psize];
      }
      // printf("Key: %f\n", a[i].key);
    }
    __syncthreads();
    send_requests(BATCH_PUSH, nchild, a,
                  queue_caller(request_queue, tickets, head, tail),
                  request_queue_size, queue_space);
  }
  else
  {
    process_requests(1, queue_caller(request_queue, tickets, head, tail), request_queue_size,
                     bheap, queue_space);
  }
}

__global__ void branch_n_bound(queue_callee(memory_queue, tickets, head, tail), uint memory_queue_size,
                               uint psize, uint *addresses_space, node_info *node_space,
                               const cost_type *original_cost, uint max_node_length,
                               queue_callee(request_queue, tickets, head, tail), uint request_queue_size,
                               queue_info *queue_space, work_info *work_space, BHEAP<node> bheap,
                               const cost_type UB)
{

  if (blockIdx.x > 0)
  {
    uint ns = 8;
    uint *my_addresses = &addresses_space[blockIdx.x * max_node_length];
    while (!opt_reached.load(cuda::memory_order_relaxed) &&
           !heap_overflow.load(cuda::memory_order_relaxed))
    {
      // pop a node from the bheap
      send_requests(POP, 0, NULL,
                    queue_caller(request_queue, tickets, head, tail),
                    request_queue_size, queue_space);
      __syncthreads();

      // Wait for POP to be done
      // __shared__ bool pop_reset; // To print the "waiting for pop statement" only once
      // if (threadIdx.x == 0)
      // {
      //   pop_reset = true;
      // }
      // __syncthreads();
      do
      {
        if (queue_space[blockIdx.x].req_status.load(cuda::memory_order_relaxed) == int(false))
        {
          // if (threadIdx.x == 0)
          // {
          //   printf("Pop for block: %u, LB: %f at level: %u\n", blockIdx.x, queue_space[blockIdx.x].nodes[0].value->LB, queue_space[blockIdx.x].nodes[0].value->level);
          // }
          // __syncthreads();
          break;
        }
        __syncthreads();
        // optimality reached while a block is waiting for a pop
        // if (threadIdx.x == 0 && pop_reset)
        // {
        //   pop_reset = false;
        // printf("block %u is waiting for pop\n", blockIdx.x);
        // }
        // __syncthreads();
        if (opt_reached.load(cuda::memory_order_relaxed) || heap_overflow.load(cuda::memory_order_relaxed))
        {
          if (threadIdx.x == 0)
            printf("Termination reached while waiting for pop for block %u\n", blockIdx.x);
          __syncthreads();
          return;
        }
      } while (ns = my_sleep(ns));
      __syncthreads();
      // copy from queue space to work space
      node *a = work_space[blockIdx.x].nodes;
      if (threadIdx.x == 0)
      {
        a[0] = queue_space[blockIdx.x].nodes[0];
        a[0].value->LB = 0;
      }
      __syncthreads();
      uint lvl = a[0].value->level;
      // Update bounds of the popped node
      for (uint i = threadIdx.x; i < psize; i += blockDim.x)
      {
        if (a[0].value->fixed_assignments[i] != 0)
          atomicAdd(&a[0].value->LB, original_cost[i * psize + (a[0].value->fixed_assignments[i] - 1)]);
      }
      __syncthreads();

      if (a[0].value->LB < UB)
      {
        // branch on popped node and copy bounds
        get_memory(queue_caller(memory_queue, tickets, head, tail), memory_queue_size, psize - lvl,
                   my_addresses);

        if (!heap_overflow.load(cuda::memory_order_relaxed))
        {
          __shared__ uint nfail;
          if (threadIdx.x == 0)
            nfail = 0;
          __syncthreads();
          for (uint i = threadIdx.x; i < psize - lvl; i += blockDim.x)
          {
            node_info *b = &node_space[my_addresses[i]];
            for (uint j = 0; j < psize; j++)
              b->fixed_assignments[j] = a[0].value->fixed_assignments[j];
            b->LB = a[0].value->LB;

            // fix further assignments
            if (b->fixed_assignments[i] == 0)
            {
              b->fixed_assignments[i] = lvl + 1;
            }
            else
            {
              uint offset = atomicAdd(&nfail, 1);
              // find appropriate index
              uint prog = 0, index = psize - lvl;
              for (uint j = psize - lvl; j < psize; j++)
              {
                if (b->fixed_assignments[j] == 0)
                {
                  if (prog == offset)
                  {
                    index = j;
                    break;
                  }
                  prog++;
                }
              }
              b->fixed_assignments[index] = lvl + 1;
            }
            b->level = lvl + 1;
            a[i].value = b;
            a[i].key = b->LB;
          }
          __syncthreads();
          // push children to bheap
          // if (threadIdx.x == 0)
          //   printf("Pushing for block %u with bound %f\n", blockIdx.x, a[0].key);
          // __syncthreads();
          send_requests(BATCH_PUSH, psize - lvl, a,
                        queue_caller(request_queue, tickets, head, tail),
                        request_queue_size, queue_space);
        }
      }
      else if (a[0].value->LB == UB)
      {
        if (threadIdx.x == 0)
        {
          printf("\033[1;31mOptimal solution reached with cost %f\033[0m\n", a[0].value->LB);
          opt_reached.store(true, cuda::memory_order_release);
        }
      }
      // else
      // {
      //   printf("Node with key %f is pruned\n", a[0].value->LB);
      // }
    }
  }
  else
  {
    process_requests_bnb(queue_caller(request_queue, tickets, head, tail), request_queue_size,
                         bheap, queue_space);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    printf("Block %u is done\n", blockIdx.x);
}

__global__ void dummy_vector_add(const cost_type *a, uint N2)
{
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N2)
  {
    printf("a[%u] = %u\n", i, a[i]);
  }
}