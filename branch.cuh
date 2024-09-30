#pragma once

#include "utils/cuda_utils.cuh"
#include "memory_manager.cuh"
#include "request_manager.cuh"
#include "queue/queue.cuh"
#include "defs.cuh"
#include "RCAP/rcap_kernels.cuh"

#include "LAP/Hung_Tlap.cuh"

__global__ void initial_branching(queue_callee(memory_queue, tickets, head, tail), uint memory_queue_size,
                                  node_info *node_space, const problem_info *pinfo,
                                  queue_callee(request_queue, tickets, head, tail), uint request_queue_size,
                                  queue_info *queue_space, worker_info *work_space, BHEAP<node> bheap,
                                  bool *hold_status, const cost_type UB)
{
  const uint bId = blockIdx.x, psize = pinfo->psize;
  if (bId > 0)
  {
    uint *my_addresses = work_space[bId].address_space;
    worker_info *my_space = &work_space[bId];
    __shared__ uint nchild, child_index;
    if (threadIdx.x == 0)
    {
      nchild = 0;
      child_index = 0;
    }
    __syncthreads();
    // get nchildren and their bounds
    for (uint child_id = 0; child_id < psize; child_id++)
    {
      if (threadIdx.x == 0)
      {
        my_space->fixed_assignments[psize * child_id + child_id] = 1;
        my_space->level[child_id] = 1;
        my_space->LB[child_id] = (float)pinfo->costs[child_id * psize];
      }
      __syncthreads();
      if (threadIdx.x == 0)
      {
        if (my_space->LB[child_id] < UB)
        {
          my_space->feasible[child_id] = true;
          atomicAdd(&nchild, 1);
        }
        else
          my_space->feasible[child_id] = false;
      }
    }
    __syncthreads();

    // Get nchild addresses
    get_memory(queue_caller(memory_queue, tickets, head, tail), memory_queue_size, nchild,
               my_addresses);

    // construct a for sending to queue
    node *a = work_space[bId].nodes;
    __syncthreads();
    for (uint child_id = threadIdx.x; child_id < psize; child_id += blockDim.x)
    {
      if (my_space->feasible[child_id])
      {
        uint my_index = atomicAdd(&child_index, 1);
        node_info *b = &node_space[my_addresses[my_index]];

        // copy fixed assignments
        for (uint j = 0; j < psize; j++)
          b->fixed_assignments[j] = my_space->fixed_assignments[psize * child_id + j]; // fix row i to column lvl + 1.

        b->LB = my_space->LB[child_id];
        b->level = my_space->level[child_id];
        a[my_index].value = b;
        a[my_index].key = my_space->LB[child_id];
      }
    }

    __syncthreads();
    send_requests(BATCH_PUSH, nchild, a,
                  queue_caller(request_queue, tickets, head, tail),
                  request_queue_size, queue_space);
    __syncthreads();

    // reset my_space
    for (uint i = threadIdx.x; i < psize; i += blockDim.x)
    {
      my_space->LB[i] = 0;
      my_space->feasible[i] = false;
      my_space->level[i] = 0;
      for (uint j = 0; j < psize; j++)
        my_space->fixed_assignments[psize * i + j] = 0;
    }
  }
  else
  {
    process_requests(1, queue_caller(request_queue, tickets, head, tail), request_queue_size,
                     bheap, queue_space);

    for (uint i = threadIdx.x; i < request_queue_size; i += blockDim.x)
    {
      hold_status[i] = false;
    }
  }
}
// Add launch bounds
__launch_bounds__(n_threads, 2048 / n_threads)
    __global__ void branch_n_bound(queue_callee(memory_queue, tickets, head, tail), uint memory_queue_size,
                                   node_info *node_space, subgrad_space *subgrad_space, const problem_info *pinfo,
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

    __shared__ uint *my_addresses;
    __shared__ int *col_fa;
    __shared__ float *lap_costs;
    __shared__ worker_info *my_space;
    if (threadIdx.x == 0)
    {
      my_addresses = work_space[blockIdx.x].address_space;
      // Needed for feasibility check
      col_fa = &subgrad_space->col_fixed_assignments[blockIdx.x * psize];
      lap_costs = &subgrad_space->lap_costs[blockIdx.x * psize * psize]; // subgradient always works with floats
      my_space = &work_space[blockIdx.x];
    }
    __shared__ GLOBAL_HANDLE<float> gh;
    __shared__ SHARED_HANDLE sh;
    __shared__ float UB;

    set_handles(gh, subgrad_space->T.th);
    __shared__ node popped_node;
    __shared__ uint popped_index, nchild_feas, lvl;
    __shared__ bool opt_flag, overflow_flag;
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
      // Wait for POP to be done
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
        UB = float(global_UB); // Reset UB
        lvl = popped_node.value->level;
      }
      __syncthreads();
      END_TIME(TRANSFER);

      // start branching to get children
      __shared__ uint nfail;
      if (threadIdx.x == 0)
        nfail = 0;
      __syncthreads();
      // Check feasibility and update bounds of each child
      for (uint i = 0; i < psize - lvl; i++)
      {
        START_TIME(BRANCH);
        __shared__ node current_node;
        __shared__ node_info current_node_info;
        // Get popped_node info in the worker space
        for (uint j = threadIdx.x; j < psize; j += blockDim.x)
          my_space->fixed_assignments[i * psize + j] = popped_node.value->fixed_assignments[j];
        __syncthreads();

        if (threadIdx.x == 0)
        {
          if (my_space->fixed_assignments[i * psize + i] == 0)
          {
            my_space->fixed_assignments[i * psize + i] = lvl + 1;
          }
          else
          {
            uint offset = nfail;
            nfail++;
            // find appropriate index
            uint prog = 0, index = psize - lvl;
            for (uint j = psize - lvl; j < psize; j++)
            {
              if (my_space->fixed_assignments[i * psize + j] == 0)
              {
                if (prog == offset)
                {
                  index = j;
                  break;
                }
                prog++;
              }
            }
            my_space->fixed_assignments[i * psize + index] = lvl + 1;
          }
          my_space->level[i] = lvl + 1;
          current_node_info = node_info(&my_space->fixed_assignments[i * psize], 0, lvl + 1);
          current_node.value = &current_node_info;
        }
        __syncthreads();
        END_TIME(BRANCH);
        START_TIME(FEAS_CHECK);
        feas_check(pinfo, &current_node, col_fa, lap_costs, stats, my_space->feasible[i], gh, sh);
        __syncthreads();
        END_TIME(FEAS_CHECK);
        // update bounds if the child is feasible
        if (my_space->feasible[i])
        {
          START_TIME(UPDATE_LB);
          update_bounds_subgrad(pinfo, subgrad_space, UB, &current_node, col_fa, gh, sh);
          __syncthreads();
          END_TIME(UPDATE_LB);
          START_TIME(BRANCH);
          if (lvl + 1 == psize && current_node.value->LB <= global_UB)
          {
            if (threadIdx.x == 0)
            {
              printf("Optimal solution reached with cost %f\n", popped_node.value->LB);
              opt_reached.store(true, cuda::memory_order_release);
            }
          }
          else if (current_node.value->LB <= global_UB)
          {
            if (threadIdx.x == 0)
            {
              atomicAdd(&stats->nodes_explored, 1);
              nchild_feas++;
              my_space->feasible[i] = true;
              my_space->LB[i] = current_node.value->LB;
            }
          }
          else
          {
            if (threadIdx.x == 0)
            {
              atomicAdd(&stats->nodes_pruned_incumbent, 1);
              my_space->feasible[i] = false;
            }
          }
          END_TIME(BRANCH);
        }
        else
        {
          START_TIME(BRANCH);
          if (threadIdx.x == 0)
            atomicAdd(&stats->nodes_pruned_infeasible, 1);
          __syncthreads();
          END_TIME(BRANCH);
        }
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
          __shared__ uint index;
          if (threadIdx.x == 0)
            index = 0;
          __syncthreads();
          for (uint i = threadIdx.x; i < psize - lvl; i += blockDim.x)
          {
            if (my_space->feasible[i])
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
          send_requests(BATCH_PUSH, nchild_feas, a,
                        queue_caller(request_queue, tickets, head, tail),
                        request_queue_size, queue_space);
        }
        __syncthreads();
      }
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
  if (threadIdx.x == 0)
  {
    DLog(debug, "Block %u is done\n", blockIdx.x);
  }
}
