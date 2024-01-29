#pragma once

#include "utils/cuda_utils.cuh"
#include "memory_manager.cuh"
#include "request_manager.cuh"
#include "queue/queue.cuh"
#include "defs.cuh"
#include "RCAP/rcap_kernels.cuh"

__global__ void initial_branching(queue_callee(memory_queue, tickets, head, tail), uint memory_queue_size,
                                  uint *addresses_space, node_info *node_space,
                                  const problem_info *pinfo, uint max_node_length,
                                  queue_callee(request_queue, tickets, head, tail), uint request_queue_size,
                                  queue_info *queue_space, work_info *work_space, BHEAP<node> bheap,
                                  const cost_type UB)
{
  const uint bId = blockIdx.x, psize = pinfo->psize;
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
      cost_type LB = pinfo->costs[i * psize];
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
      cost_type LB = pinfo->costs[i * psize];
      if (LB < UB)
      {
        my_index = atomicAdd(&child_index, 1);
        node_info *b = &node_space[my_addresses[my_index]];
        b->fixed_assignments[i] = 1; // fix row i to column lvl + 1.
        b->LB = (float)pinfo->costs[i * psize];
        b->level = 1;
        a[my_index].value = b;
        a[my_index].key = (float)pinfo->costs[i * psize];
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
                               uint *addresses_space, node_info *node_space,
                               const problem_info *pinfo, uint max_node_length,
                               queue_callee(request_queue, tickets, head, tail), uint request_queue_size,
                               queue_info *queue_space, work_info *work_space, BHEAP<node> bheap,
                               const cost_type UB,
                               bnb_stats *stats)
{
  const uint psize = pinfo->psize;
  if (blockIdx.x > 0)
  {
    uint *my_addresses = &addresses_space[blockIdx.x * max_node_length];
    __shared__ uint popped_index;
    while (!opt_reached.load(cuda::memory_order_relaxed) &&
           !heap_overflow.load(cuda::memory_order_relaxed))
    {
      // pop a node from the bheap
      send_requests(POP, 0, NULL,
                    queue_caller(request_queue, tickets, head, tail),
                    request_queue_size, queue_space);
      __syncthreads();

      // Wait for POP to be done
      wait_for_pop(queue_space);

      // copy from queue space to work space
      node *a = work_space[blockIdx.x].nodes;
      if (threadIdx.x == 0)
      {
        a[0] = queue_space[blockIdx.x].nodes[0];
        a[0].value->LB = 0;
        popped_index = a[0].value->id;
      }
      __syncthreads();
      uint lvl = a[0].value->level;

      // Check feasibility for budget constraints
      __shared__ bool feasible;
      feas_check(pinfo, a, stats, feasible);
      if (feasible)
      {
        // Update bounds of the popped node
        update_bounds(pinfo, a);
        if (a[0].value->LB < UB)
        {
          if (threadIdx.x == 0)
          {
            atomicAdd(&stats->nodes_explored, 1);
          }
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
              // Copy info from popped node to node space
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
            //   DLog(debug, "Pushing for block %u with bound %f\n", blockIdx.x, a[0].key);
            // __syncthreads();
            send_requests(BATCH_PUSH, psize - lvl, a,
                          queue_caller(request_queue, tickets, head, tail),
                          request_queue_size, queue_space);
          }
        }
        else if (lvl == psize && a[0].value->LB <= UB)
        {
          if (threadIdx.x == 0)
          {
            DLog(critical, "Optimal solution reached with cost %f\n", a[0].value->LB);
            opt_reached.store(true, cuda::memory_order_release);
          }
        }
        else
        {
          // DLog(debug, "Node with key %f is pruned\n", a[0].value->LB);
          if (threadIdx.x == 0)
            atomicAdd(&stats->nodes_pruned_incumbent, 1);
        }
      }
      __syncthreads();
      // free the popped node from node space
      free_memory(queue_caller(memory_queue, tickets, head, tail), memory_queue_size,
                  popped_index);
    }
  }
  else
  {
    process_requests_bnb(queue_caller(request_queue, tickets, head, tail), request_queue_size,
                         bheap, queue_space);
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    if (blockIdx.x == 3)
    {
      DLog(critical, "Block %u is done\n", blockIdx.x);
    }
    else
    {
      DLog(debug, "Block %u is done\n", blockIdx.x);
    }
  }
}

__global__ void dummy_vector_add(const cost_type *a, uint N2)
{
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N2)
  {
    printf("a[%u] = %u\n", i, a[i]);
  }
}