#pragma once

#include <cooperative_groups.h>
#include "utils/cuda_utils.cuh"
#include "memory_manager.cuh"
#include "request_manager.cuh"
#include "queue/queue.cuh"
#include "defs.cuh"
#include "RCAP/rcap_kernels.cuh"
#include "RCAP/problem_info.h"
#include "RCAP/LAP/Hung_Tlap.cuh"

namespace cg = cooperative_groups;

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

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);
    const uint tile_id = tile.meta_group_rank();
    const uint local_id = tile.thread_rank();

    // get nchildren and their bounds
    for (uint child_id = tile_id; child_id < psize; child_id += TilesPerBlock)
    {
      if (local_id == 0)
      {
        my_space->fixed_assignments[psize * child_id + child_id] = 1;
        my_space->level[child_id] = 1;
        my_space->LB[child_id] = (float)pinfo->costs[child_id * psize];
      }
      sync(tile);
      if (local_id == 0)
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

    __shared__ node_info *b[TilesPerBlock];
    for (uint child_id = tile_id; child_id < psize; child_id += TilesPerBlock)
    {
      if (my_space->feasible[child_id])
      {
        if (local_id == 0)
        {
          uint my_index = atomicAdd(&child_index, 1);
          b[tile_id] = &node_space[my_addresses[my_index]];
          b[tile_id]->LB = my_space->LB[child_id];
          b[tile_id]->level = my_space->level[child_id];
          a[my_index].value = b[tile_id];
          a[my_index].key = my_space->LB[child_id];
        }
        sync(tile);
        // copy fixed assignments
        for (uint j = local_id; j < psize; j += TileSize)
          b[tile_id]->fixed_assignments[j] = my_space->fixed_assignments[psize * child_id + j]; // fix row i to column lvl + 1.
        sync(tile);
      }
    }

    __syncthreads();
    send_requests(BATCH_PUSH, nchild, a,
                  queue_caller(request_queue, tickets, head, tail),
                  request_queue_size, queue_space);
    __syncthreads();

    // reset my_space
    for (uint i = tile_id; i < psize; i += TilesPerBlock)
    {
      if (local_id == 0)
      {
        my_space->LB[i] = 0;
        my_space->feasible[i] = false;
        my_space->level[i] = 0;
      }
      for (uint j = local_id; j < psize; j += TileSize)
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
__launch_bounds__(BlockSize, 2048 / BlockSize)
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
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);
    const uint tile_id = tile.meta_group_rank();
    const uint local_id = tile.thread_rank();

    INIT_TIME(counters);
    INIT_TIME(lap_counters);
    START_TIME(INIT);

    __shared__ uint *my_addresses;              // Ownership: Block
    __shared__ worker_info *my_space;           // Ownership: Block
    __shared__ int *col_fa[TilesPerBlock];      // Ownership: Tile
    __shared__ float *lap_costs[TilesPerBlock]; // Ownership: Tile
    if (local_id == 0)
    {
      if (threadIdx.x == 0)
      {
        my_addresses = work_space[blockIdx.x].address_space;
        my_space = &work_space[blockIdx.x];
      }
      // Needed for feasibility check
      col_fa[tile_id] = &subgrad_space->col_fixed_assignments[(blockIdx.x * TilesPerBlock + tile_id) * psize];
      lap_costs[tile_id] = &subgrad_space->lap_costs[(blockIdx.x * TilesPerBlock + tile_id) * psize * psize]; // subgradient always works with floats
    }
    __shared__ PARTITION_HANDLE<float> ph[TilesPerBlock]; // needed for LAP. ownership: Tile
    __shared__ float UB[TilesPerBlock];                   // needed for subgradient, ownership: Tile

    set_handles(tile, ph[tile_id], subgrad_space->T.th);
    __shared__ node popped_node;                           // Ownership: Block
    __shared__ uint popped_index, nchild_feas, lvl, nfail; // Ownership: Block
    __shared__ bool opt_flag, overflow_flag;               // Ownership: Block

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
      if (local_id == 0)
        UB[tile_id] = float(global_UB); // Reset UB
      __syncthreads();
      END_TIME(TRANSFER);

      // start branching to get children
      __shared__ uint child_id, my_i[TilesPerBlock];
      if (threadIdx.x == 0)
      {
        nfail = 0; // Owned by Block
        child_id = 0;
      }
      __syncthreads();
      // Check feasibility and update bounds of each child
      __shared__ node current_node[TilesPerBlock];           // Ownership: Tile
      __shared__ node_info current_node_info[TilesPerBlock]; // Ownership: Tile
      START_TIME(UPDATE_LB);
      if (local_id == 0)
        my_i[tile_id] = atomicAdd(&child_id, 1);
      sync(tile);
      uint i = my_i[tile_id];
      while (i < psize - lvl)
      {
        // Get popped_node info in the worker space
        for (uint j = local_id; j < psize; j += TileSize)
          my_space->fixed_assignments[i * psize + j] = popped_node.value->fixed_assignments[j];
        sync(tile);

        if (local_id == 0)
        {
          if (my_space->fixed_assignments[i * psize + i] == 0)
          {
            my_space->fixed_assignments[i * psize + i] = lvl + 1;
          }
          else
          {
            uint offset = atomicAdd(&nfail, 1);
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
          current_node_info[tile_id] = node_info(&my_space->fixed_assignments[i * psize], 0, lvl + 1);
          current_node[tile_id].value = &current_node_info[tile_id];
        }
        sync(tile);
        feas_check(pinfo, tile,
                   &current_node[tile_id], col_fa[tile_id],
                   lap_costs[tile_id], stats, my_space->feasible[i],
                   ph[tile_id]);

        sync(tile);
        // update bounds if the child is feasible
        if (my_space->feasible[i])
        {
          update_bounds_subgrad(pinfo, tile, subgrad_space, UB[tile_id],
                                &current_node[tile_id], col_fa[tile_id], ph[tile_id]);
          sync(tile);

          if (lvl + 1 == psize && current_node[tile_id].value->LB <= global_UB)
          {
            if (local_id == 0)
            {
              printf("Optimal solution reached with cost %f\n", popped_node.value->LB);
              opt_reached.store(true, cuda::memory_order_release);
            }
          }
          else if (current_node[tile_id].value->LB <= global_UB)
          {
            if (local_id == 0)
            {
              atomicAdd(&stats->nodes_explored, 1);
              atomicAdd(&nchild_feas, 1);
              my_space->feasible[i] = true;
              my_space->LB[i] = current_node[tile_id].value->LB;
            }
          }
          else
          {
            if (local_id == 0)
            {
              atomicAdd(&stats->nodes_pruned_incumbent, 1);
              my_space->feasible[i] = false;
            }
          }
        }
        else
        {
          if (local_id == 0)
            atomicAdd(&stats->nodes_pruned_infeasible, 1);
          sync(tile);
        }
        if (local_id == 0)
          my_i[tile_id] = atomicAdd(&child_id, 1);
        sync(tile);
        i = my_i[tile_id];
      }
      __syncthreads();
      END_TIME(UPDATE_LB);
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
  if (threadIdx.x == 0)
  {
    DLog(debug, "Block %u is done\n", blockIdx.x);
  }
}
