#include <stdio.h>
#include <cmath>
#include <vector>
#include "heap/bheap.cuh"
#include "utils/logger.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/timer.h"
#include "queue/queue.cuh"
#include "request_manager.cuh"
#include "memory_manager.cuh"
#include "defs.cuh"
#include "branch.cuh"

#include "RCAP/config.h"
#include "RCAP/cost_generator.h"
#include "RCAP/gurobi_solver.h"
#include "RCAP/subgrad_solver.cuh"

#include "cudaProfiler.h"
#include "cuda_profiler_api.h"

__global__ void get_exit_code(ExitCode *ec)
{

  ec[0] = opt_reached.load(cuda::memory_order_consume)      ? ExitCode::OPTIMAL
          : heap_overflow.load(cuda::memory_order_consume)  ? ExitCode::HEAP_FULL
          : heap_underflow.load(cuda::memory_order_consume) ? ExitCode::INFEASIBLE
                                                            : ExitCode::UNKNOWN_ERROR;
}

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);
  int dev_ = config.deviceId;
  CUDA_RUNTIME(cudaSetDevice(dev_));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev_);
  problem_info *pinfo = generate_problem<cost_type>(config, config.seed);
  // print(pinfo, true, true, false);
  printConfig(config);

  Timer t = Timer();
  uint psize = config.user_n, ncommodities = config.user_ncommodities;
  // Solve RCAP for getting UB (Can be taken from a library if not interesting in using gurobi)
  cost_type UB = solve_with_gurobi<cost_type, weight_type>(pinfo->costs, pinfo->weights, pinfo->budgets, psize, ncommodities);
  Log(info, "RCAP solved with GUROBI: objective %u\n", (uint)UB);
  Log(info, "Time taken by Gurobi: %f sec", t.elapsed());

  opt_reached.store(false, cuda::memory_order_release);
  heap_overflow.store(false, cuda::memory_order_release);
  heap_underflow.store(false, cuda::memory_order_release);

  Log(debug, "Solving RCAP with Branching");
  t.reset();

  // Create space for queue
  // Find max concurrent blocks for the branch_n_bound kernel

  int nw1, nb1;
  cudaOccupancyMaxPotentialBlockSize(&nw1, &nb1, branch_n_bound, 0, 0);
  Log(info, "Max potential block size: %d", nb1);
  Log(info, "Max potential grid size: %d", nw1);

  assert(nb1 >= BlockSize);
  int nworkers; // equals grid dimension of request manager
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nworkers, branch_n_bound, BlockSize, 0);
  Log(debug, "Max concurrent blocks per SM: %d", nworkers);
  nworkers *= deviceProp.multiProcessorCount;

  // Create space for bound computation storing and branching
  Log(debug, "Creating scratch space for workers");
  worker_info *d_worker_space = nullptr; // managed by each worker
  Log(debug, "Allocating space for %u workers with psize %u", nworkers, psize);
  worker_info::allocate_all(d_worker_space, nworkers, psize);

  int nsubworkers = BlockSize / TileSize;
  Log(debug, "Creating space for subgrad solver");
  subgrad_space *d_subgrad_space = nullptr; // managed by each subworker
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_subgrad_space, nsubworkers * nworkers * sizeof(subgrad_space)));
  d_subgrad_space->allocate(psize, ncommodities, nsubworkers * nworkers, dev_);

  Log(debug, "Creating space for request queue");
  queue_info *d_queue_space = nullptr;
  queue_info::allocate_all(d_queue_space, nworkers);

  // Create MPMC queue for handling heap requests
  queue_declare(request_queue, tickets, head, tail);
  queue_init(request_queue, tickets, head, tail, nworkers, dev_);

  // Get memory queue length based on available memory
  size_t free, total;
  CUDA_RUNTIME(cudaMemGetInfo(&free, &total));
  Log(info, "Occupied memory: %.3f%%", ((total - free) * 1.0) / total * 100);
  size_t memory_queue_weight = (sizeof(node_info) + sizeof(node) + psize * sizeof(int) + sizeof(queue_type) + sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>));
  size_t memory_queue_len = (free * 0.95) / memory_queue_weight; // Keeping 5% headroom
  Log(info, "Memory queue length: %lu", memory_queue_len);

  // Create space for node_info
  node_info *d_node_space;
  CUDA_RUNTIME(cudaMalloc((void **)&d_node_space, memory_queue_len * sizeof(node_info)));
  CUDA_RUNTIME(cudaMemset((void *)d_node_space, 0, memory_queue_len * sizeof(node_info)));
  // space for fixed assignments in node_info
  int *d_fixed_assignment_space;
  CUDA_RUNTIME(cudaMalloc((void **)&d_fixed_assignment_space, memory_queue_len * psize * sizeof(int)));
  CUDA_RUNTIME(cudaMemset((void *)d_fixed_assignment_space, 0, memory_queue_len * psize * sizeof(int)));

  // Set fixed assignment pointers to d_fixed_assignment_space
  uint block_dimension = 1024;
  uint grid_dimension = min(size_t(deviceProp.maxGridSize[0]), (memory_queue_len - 1) / block_dimension + 1);
  execKernel(set_fixed_assignment_pointers, grid_dimension, block_dimension, dev_, true,
             d_node_space, d_fixed_assignment_space, psize, memory_queue_len);

  // create space for hold_status
  bool *d_hold_status; // Managed by Workers
  CUDA_RUNTIME(cudaMalloc((void **)&d_hold_status, nworkers * sizeof(bool)));
  CUDA_RUNTIME(cudaMemset((void *)d_hold_status, 0, nworkers * sizeof(bool)));

  // Create BHEAP on device
  BHEAP<node> d_bheap = BHEAP<node>(memory_queue_len, dev_);

  // Create bnb-stats object on device
  bnb_stats *stats;
  CUDA_RUNTIME(cudaMallocManaged((void **)&stats, sizeof(bnb_stats)));
  stats->initialize();

  // Create MPMC queue for handling memory requests
  queue_declare(memory_queue, tickets, head, tail);
  queue_init(memory_queue, tickets, head, tail, memory_queue_len, dev_);

  CUDA_RUNTIME(cudaMemGetInfo(&free, &total));
  Log(info, "Occupied memory: %.3f%%", ((total - free) * 1.0) / total * 100);

#ifdef TIMER
  allocateCounters(&counters, nworkers);
  Log(debug, "Allocated regular counters");
  allocateCounters(&lap_counters, nworkers);
  Log(debug, "Allocated lap counters");
#endif

  // Populate memory queue and node_space IDs
  execKernel(fill_memory_queue, grid_dimension, block_dimension, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), d_node_space,
             memory_queue_len);

  Log(warn, "TileSize: %u", TileSize);
  // Frist kernel to create L1 nodes
  execKernel(initial_branching, 2, BlockSize, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), memory_queue_len,
             d_node_space, pinfo,
             queue_caller(request_queue, tickets, head, tail), nworkers,
             d_queue_space, d_worker_space, d_bheap,
             d_hold_status, UB);
  cuProfilerStart();
  execKernel(branch_n_bound, nworkers, BlockSize, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), memory_queue_len,
             d_node_space, d_subgrad_space, pinfo,
             queue_caller(request_queue, tickets, head, tail), nworkers,
             d_queue_space, d_worker_space, d_bheap,
             d_hold_status,
             UB, stats);
  cuProfilerStop();
  Log(warn, "BnB Terminated");

#ifdef TIMER
  printCounters(counters, false);
  // printCounters(lap_counters, false);
  freeCounters(counters);
  freeCounters(lap_counters);
#endif

  // Get exit code
  ExitCode exit_code, *d_exit_code;
  CUDA_RUNTIME(cudaMalloc((void **)&d_exit_code, sizeof(ExitCode)));
  execKernel(get_exit_code, 1, 1, dev_, false, d_exit_code);
  CUDA_RUNTIME(cudaMemcpy(&exit_code, d_exit_code, sizeof(ExitCode), cudaMemcpyDeviceToHost));
  CUDA_RUNTIME(cudaFree(d_exit_code));

  d_bheap.print_size();
  Log(info, "Max heap size during execution: %lu", d_bheap.d_max_size[0]);
  Log(info, "Nodes Explored: %u, Incumbent: %u, Infeasible: %u", stats->nodes_explored, stats->nodes_pruned_incumbent, stats->nodes_pruned_infeasible);
  Log(info, "Total time taken: %f sec", t.elapsed());

  // Free device memory
  d_bheap.free_memory();
  CUDA_RUNTIME(cudaFree(d_node_space));
  CUDA_RUNTIME(cudaFree(d_fixed_assignment_space));
  CUDA_RUNTIME(cudaFree(stats));
  CUDA_RUNTIME(cudaFree(d_hold_status));

  worker_info::free_all(d_worker_space, nworkers);
  queue_info::free_all(d_queue_space);
  d_subgrad_space->clear();
  CUDA_RUNTIME(cudaFree(d_subgrad_space));
  CUDA_RUNTIME(cudaFree(pinfo));

  queue_free(request_queue, tickets, head, tail);
  queue_free(memory_queue, tickets, head, tail);

  // print exit code message and return
  Log(info, "Exit code: %s", ExitCode_text[exit_code]);
  return int(exit_code);
}
