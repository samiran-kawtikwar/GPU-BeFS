#pragma once

#include <cooperative_groups.h>
#include "utils/cuda_utils.cuh"
#include "memory_manager.cuh"
#include "request_manager.cuh"
#include "queue/queue.cuh"
#include "defs.cuh"
#include "QAP/GLB_solver.cuh"

namespace cg = cooperative_groups;

__launch_bounds__(BlockSize, 2048 / BlockSize)
    __global__ void initial_branching(queue_callee(memory_queue, tickets, head, tail), uint memory_queue_size,
                                      node_info *node_space, const problem_info *pinfo, glb_space *glb_space,
                                      queue_callee(request_queue, tickets, head, tail), uint request_queue_size,
                                      queue_info *queue_space, worker_info *work_space, BHEAP<node> bheap,
                                      bool *hold_status, const cost_type UB)
{
  const uint bId = blockIdx.x, psize = pinfo->psize;
  if (bId > 0)
  {
    uint *my_addresses = work_space[bId].address_space;
    worker_info *my_space = &work_space[bId];
    __shared__ int *fa, *la; // Ownership: Block

    if (threadIdx.x == 0)
    {
      fa = my_space->fixed_assignments;
      la = glb_space[bId].la;
    }
    __syncthreads();
    // update bounds
    glb_solve(glb_space[bId], fa, la, pinfo, my_space->LB[0]);
    __syncthreads();
    if (threadIdx.x == 0)
      my_space->pushable[0] = (my_space->LB[0] <= UB) ? true : false;
    __syncthreads();

    // Get nchild addresses
    if (my_space->pushable[0])
    {
      get_memory(queue_caller(memory_queue, tickets, head, tail), memory_queue_size, 1, my_addresses);
      // construct a for sending to queue
      node *a = work_space[bId].nodes;
      __syncthreads();

      __shared__ node_info *b;

      if (threadIdx.x == 0)
      {
        b = &node_space[my_addresses[0]];
        b->LB = my_space->LB[0];
        b->level = my_space->level[0];
        a[0].value = b;
        a[0].key = my_space->LB[0];
      }
      __syncthreads();
      // copy fixed assignments
      for (uint j = threadIdx.x; j < psize; j += blockDim.x)
        b->fixed_assignments[j] = my_space->fixed_assignments[j]; // fix row i to column lvl + 1.
      __syncthreads();
      send_requests(BATCH_PUSH, 1, a,
                    queue_caller(request_queue, tickets, head, tail),
                    request_queue_size, queue_space);
    }
    __syncthreads();
    // reset my_space
    for (uint i = threadIdx.x; i < psize; i += blockDim.x)
    {

      my_space->LB[i] = 0;
      my_space->pushable[i] = false;
      my_space->level[i] = 0;
      my_space->fixed_assignments[i] = -1;
      la[i] = -1;
    }
    __syncthreads();
  }
  else
  {
    process_requests(1, queue_caller(request_queue, tickets, head, tail), request_queue_size,
                     bheap, queue_space);

    for (uint i = threadIdx.x; i < request_queue_size; i += blockDim.x)
      hold_status[i] = false;
  }
}

// Add launch bounds
__launch_bounds__(BlockSize, 2048 / BlockSize)
    __global__ void branch_n_bound(queue_callee(memory_queue, tickets, head, tail), uint memory_queue_size,
                                   node_info *node_space, glb_space *glb_space, const problem_info *pinfo,
                                   queue_callee(request_queue, tickets, head, tail), uint request_queue_size,
                                   queue_info *queue_space, worker_info *work_space, BHEAP<node> bheap,
                                   bool *hold_status,
                                   const cost_type global_UB,
                                   bnb_stats *stats)
{
  const uint psize = pinfo->psize;

  if (blockIdx.x > 0)
  {
    INIT_TIME(counters);
    INIT_TIME(lap_counters);
    START_TIME(INIT);

    __shared__ uint *my_addresses;    // Ownership: Block
    __shared__ worker_info *my_space; // Ownership: Block
    __shared__ int *fa, *la;          // Ownership: Block

    if (threadIdx.x == 0)
    {
      my_space = &work_space[blockIdx.x];
      my_addresses = my_space->address_space;
      la = glb_space[blockIdx.x].la;
    }
    // Needed for feasibility check
    __shared__ node popped_node;                    // Ownership: Block
    __shared__ uint popped_index, nchild_feas, lvl; // Ownership: Block
    __shared__ bool opt_flag, overflow_flag;        // Ownership: Block

    if (threadIdx.x == 0)
    {
      opt_flag = false;
      overflow_flag = false;
      if (opt_reached.load(cuda::memory_order_relaxed))
        opt_flag = true;
      if (heap_overflow.load(cuda::memory_order_relaxed))
        overflow_flag = true;
    }
    __syncthreads();

    END_TIME(INIT);

    while (!opt_flag && !overflow_flag)
    {
      START_TIME(QUEUING);
      // pop a node from the bheap
      send_requests(POP, 0, NULL,
                    queue_caller(request_queue, tickets, head, tail),
                    request_queue_size, queue_space);
      __syncthreads();
      END_TIME(QUEUING);

      START_TIME(WAITING);
      // Wait for POP to be done  -- Entire block waits
      if (wait_for_pop(queue_space) == false)
        break;

      END_TIME(WAITING);

      START_TIME(TRANSFER)
      // copy from queue space to work space
      if (threadIdx.x == 0)
      {
        popped_node = queue_space[blockIdx.x].nodes[0];
        nchild_feas = 0;
        popped_index = popped_node.value->id;
        lvl = popped_node.value->level;
      }
      __syncthreads();
      END_TIME(TRANSFER);
      // iterate over all children
      for (uint i = 0; i < psize - lvl; i++)
      {
        START_TIME(BRANCH);
        if (threadIdx.x == 0)
          fa = &my_space->fixed_assignments[i * psize];
        // Get popped_node info in the worker space
        __syncthreads();
        for (uint j = threadIdx.x; j < psize; j += blockDim.x)
          fa[j] = popped_node.value->fixed_assignments[j];
        __syncthreads();

        if (threadIdx.x == 0)
        {
          // DLog(info, "\nPopped fa: ");
          // for (uint j = 0; j < psize; j++)
          //   printf("%d ", fa[j]);
          // printf("\n");
          uint counter = 0;
          for (uint index = 0; index < psize; index++)
          {
            if (counter == i && fa[index] == -1)
            {
              fa[index] = lvl; // fixes the assignment at counter
              break;
            }
            if (fa[index] == -1)
              counter++;
          }
          my_space->level[i] = lvl + 1;
        }
        __syncthreads();
        END_TIME(BRANCH);
        START_TIME(UPDATE_LB);
        glb_solve(glb_space[blockIdx.x], fa, la, pinfo, my_space->LB[i]);
        __syncthreads();
        END_TIME(UPDATE_LB);
        START_TIME(BRANCH);
        if (threadIdx.x == 0)
        {
          // DLog(info, "Processed node \t");
          // DLog(warn, "LB: %u\n", my_space->LB[i]);
          if (lvl + 1 == psize && my_space->LB[i] <= global_UB)
          {
            DLog(critical, "Optimal solution reached with cost %u\n", my_space->LB[i]);
            opt_reached.store(true, cuda::memory_order_release);
          }
          else if (my_space->LB[i] <= global_UB)
          {
            atomicAdd(&stats->nodes_explored, 1);
            atomicAdd(&nchild_feas, 1);
            my_space->pushable[i] = true;
          }
          else
          {
            atomicAdd(&stats->nodes_pruned_incumbent, 1);
            my_space->pushable[i] = false;
          }
        }
        __syncthreads();
        END_TIME(BRANCH);
      }
      __syncthreads();

      START_TIME(QUEUING);
      // free the popped node from node space
      free_memory(queue_caller(memory_queue, tickets, head, tail), memory_queue_size,
                  popped_index);
      __syncthreads();
      if (opt_reached.load(cuda::memory_order_relaxed))
      {
        if (threadIdx.x == 0)
          opt_flag = true;
      }
      __syncthreads();

      if (nchild_feas > 0 && !opt_flag)
      {
        // get nchild addresses
        get_memory(queue_caller(memory_queue, tickets, head, tail), memory_queue_size, nchild_feas, my_addresses);

        if (threadIdx.x == 0)
        {
          if (heap_overflow.load(cuda::memory_order_relaxed))
            overflow_flag = true;
        }
        __syncthreads();
        if (!overflow_flag)
        {
          node *a = my_space->nodes;
          __shared__ uint index; // Ownership: Block
          if (threadIdx.x == 0)
            index = 0;
          __syncthreads();
          for (uint i = threadIdx.x; i < psize - lvl; i += blockDim.x)
          {
            if (my_space->pushable[i])
            {
              uint ind = atomicAdd(&index, 1);
              node_info *b = &node_space[my_addresses[ind]];
              b->LB = my_space->LB[i];
              b->level = my_space->level[i];
              for (uint j = 0; j < psize; j++)
                b->fixed_assignments[j] = my_space->fixed_assignments[i * psize + j];
              a[ind].value = b;
              a[ind].key = my_space->LB[i];
            }
          }
          __syncthreads();
          // if (threadIdx.x == 0)
          // DLog(debug, "Block %u is pushing %u nodes\n", blockIdx.x, nchild_feas);
          // __syncthreads();
          send_requests(BATCH_PUSH, nchild_feas, a,
                        queue_caller(request_queue, tickets, head, tail),
                        request_queue_size, queue_space);
        }
      }
      __syncthreads();
      END_TIME(QUEUING);

      START_TIME(INIT);
      if (threadIdx.x == 0)
      {
        if (heap_overflow.load(cuda::memory_order_relaxed))
          overflow_flag = true;
      }
      __syncthreads();
      END_TIME(INIT);
    }
  }
  else
  {
    process_requests_bnb(queue_caller(request_queue, tickets, head, tail), request_queue_size,
                         bheap, queue_space,
                         hold_status);
  }
  __syncthreads();
  // if (threadIdx.x == 0)
  // {
  //   DLog(debug, "Block %u is done\n", blockIdx.x);
  // }
}
