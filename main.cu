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
#include "LAP/device_utils.cuh"
#include "LAP/Hung_Tlap.cuh"
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

__global__ void set_fixed_assignment_pointers(node_info *nodes, int *fixed_assignment_space, const uint size, const uint len)
{
  size_t g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = g_tid; i < len; i += gridDim.x * blockDim.x)
    nodes[i].fixed_assignments = &fixed_assignment_space[i * size];
}

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);
  printConfig(config);
  int dev_ = config.deviceId;
  uint psize = config.user_n;
  uint ncommodities = config.user_ncommodities;
  if (psize > 100)
  {
    Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 100");
    exit(-1);
  }
  CUDA_RUNTIME(cudaDeviceReset());
  CUDA_RUNTIME(cudaSetDevice(dev_));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev_);
  problem_info *h_problem_info = generate_problem<cost_type>(config, config.seed);

  Timer t = Timer();
  // Solve RCAP
  cost_type UB = solve_with_gurobi<cost_type, weight_type>(h_problem_info->costs, h_problem_info->weights, h_problem_info->budgets, psize, ncommodities);
  Log(info, "RCAP solved with GUROBI: objective %u\n", (uint)UB);
  // print time
  Log(info, "Time taken by Gurobi: %f sec", t.elapsed());
  // print(h_problem_info, true, true, false);

  // Copy problem info to device
  problem_info *d_problem_info;
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_problem_info, sizeof(problem_info)));
  d_problem_info->psize = psize;
  d_problem_info->ncommodities = ncommodities;
  CUDA_RUNTIME(cudaMalloc((void **)&d_problem_info->costs, psize * psize * sizeof(cost_type)));
  CUDA_RUNTIME(cudaMemcpy(d_problem_info->costs, h_problem_info->costs, psize * psize * sizeof(cost_type), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMalloc((void **)&d_problem_info->weights, ncommodities * psize * psize * sizeof(weight_type)));
  CUDA_RUNTIME(cudaMemcpy(d_problem_info->weights, h_problem_info->weights, ncommodities * psize * psize * sizeof(weight_type), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMalloc((void **)&d_problem_info->budgets, ncommodities * sizeof(weight_type)));
  CUDA_RUNTIME(cudaMemcpy(d_problem_info->budgets, h_problem_info->budgets, ncommodities * sizeof(weight_type), cudaMemcpyHostToDevice));

  // weight_type LB = subgrad_solver<cost_type, weight_type>(h_problem_info->costs, UB, h_problem_info->weights, h_problem_info->budgets, psize, ncommodities);
  // Log(info, "RCAP solved with Subgradient: objective %u\n", (uint)LB);

  opt_reached.store(false, cuda::memory_order_release);
  heap_overflow.store(false, cuda::memory_order_release);
  heap_underflow.store(false, cuda::memory_order_release);

  Log(debug, "Solving RCAP with Branching");
  t.reset();

  // Create space for queue
  int nworkers, nsubworkers; // equals grid dimension of request manager
  // Find max concurrent blocks for the branch_n_bound kernel
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nworkers, branch_n_bound, BlockSize, 0);
  Log(debug, "Max concurrent blocks per SM: %d", nworkers);
  nworkers *= deviceProp.multiProcessorCount;
  nsubworkers = BlockSize / TileSize;

  int nw1, nb1;
  cudaOccupancyMaxPotentialBlockSize(&nw1, &nb1, branch_n_bound, 0, 0);
  Log(info, "Max potential block size: %d", nb1);
  Log(info, "Max potential grid size: %d", nw1);

  // Create space for bound computation storing and branching
  Log(debug, "Creating scratch space for workers");
  worker_info *d_worker_space; // managed by each worker
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_worker_space, nworkers * sizeof(worker_info)));
  worker_info::allocate_all(d_worker_space, nworkers, psize);

  Log(debug, "Creating space for subgrad solver");
  subgrad_space *d_subgrad_space; // managed by each subworker
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_subgrad_space, nsubworkers * nworkers * sizeof(subgrad_space)));
  d_subgrad_space->allocate(psize, ncommodities, nsubworkers * nworkers, dev_);

  // Call subgrad_solver Block
  // execKernel(g_subgrad_solver, 1, BlockSize, dev_, true, d_problem_info, d_subgrad_space, UB); // block dimension >=256
  // printf("Exiting...\n");
  // exit(0);

  queue_info *d_queue_space, *h_queue_space; // Queue is managed by workers
  CUDA_RUNTIME(cudaMalloc((void **)&d_queue_space, nworkers * sizeof(queue_info)));
  h_queue_space = (queue_info *)malloc(nworkers * sizeof(queue_info));
  memset(h_queue_space, 0, nworkers * sizeof(queue_info));
  for (size_t i = 0; i < nworkers; i++)
  {
    h_queue_space[i].req_status.store(DONE, cuda::memory_order_release);
    h_queue_space[i].batch_size = 0;
    h_queue_space[i].id = (uint32_t)i;
  }
  CUDA_RUNTIME(cudaMemcpy(d_queue_space, h_queue_space, nworkers * sizeof(queue_info), cudaMemcpyHostToDevice));
  delete[] h_queue_space;

  // Create MPMC queue for handling heap requests
  queue_declare(request_queue, tickets, head, tail);
  queue_init(request_queue, tickets, head, tail, nworkers, dev_);

  // Get memory queue length based on available memory
  size_t free, total;
  CUDA_RUNTIME(cudaMemGetInfo(&free, &total));
  Log(info, "Occupied memory: %.3f%%", ((total - free) * 1.0) / total * 100);
  size_t memory_queue_weight = (sizeof(node_info) + sizeof(node) + psize * sizeof(int) + sizeof(queue_type) + sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>));
  size_t memory_queue_len = (free * 0.15) / memory_queue_weight; // Keeping 5% headroom
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

  // Populate memory queue and node_space IDs
  execKernel(fill_memory_queue, grid_dimension, block_dimension, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), d_node_space,
             memory_queue_len);

  Log(warn, "TileSize: %u", TileSize);
  // Frist kernel to create L1 nodes
  execKernel(initial_branching, 2, BlockSize, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), memory_queue_len,
             d_node_space, d_problem_info,
             queue_caller(request_queue, tickets, head, tail), nworkers,
             d_queue_space, d_worker_space, d_bheap,
             d_hold_status, UB);
  cuProfilerStart();
  execKernel(branch_n_bound, nworkers, BlockSize, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), memory_queue_len,
             d_node_space, d_subgrad_space, d_problem_info,
             queue_caller(request_queue, tickets, head, tail), nworkers,
             d_queue_space, d_worker_space, d_bheap,
             d_hold_status,
             UB, stats);
  cuProfilerStop();
  printf("\n");

#ifdef TIMER
  printCounters(counters, false);
  // printCounters(lap_counters, false);
#endif

  // Get exit code
  ExitCode exit_code, *d_exit_code;
  CUDA_RUNTIME(cudaMalloc((void **)&d_exit_code, sizeof(ExitCode)));
  execKernel(get_exit_code, 1, 1, dev_, false, d_exit_code);
  CUDA_RUNTIME(cudaMemcpy(&exit_code, d_exit_code, sizeof(ExitCode), cudaMemcpyDeviceToHost));
  CUDA_RUNTIME(cudaFree(d_exit_code));

  d_bheap.print_size();
  Log(info, "Max heap size during execution: %lu", d_bheap.d_max_size[0]);
  Log(info, "Nodes Explored: %u, Incumbant: %u, Infeasible: %u", stats->nodes_explored, stats->nodes_pruned_incumbent, stats->nodes_pruned_infeasible);
  Log(info, "Total time taken: %f sec", t.elapsed());

  // Free device memory
  d_bheap.free_memory();
  CUDA_RUNTIME(cudaFree(d_queue_space));
  CUDA_RUNTIME(cudaFree(d_node_space));
  CUDA_RUNTIME(cudaFree(d_fixed_assignment_space));
  CUDA_RUNTIME(cudaFree(stats));
  CUDA_RUNTIME(cudaFree(d_problem_info->costs));
  CUDA_RUNTIME(cudaFree(d_problem_info->weights));
  CUDA_RUNTIME(cudaFree(d_problem_info->budgets));
  CUDA_RUNTIME(cudaFree(d_problem_info));
  CUDA_RUNTIME(cudaFree(d_hold_status));
  worker_info::free_all(d_worker_space, nworkers);
  CUDA_RUNTIME(cudaFree(d_worker_space));
  d_subgrad_space->clear();
  CUDA_RUNTIME(cudaFree(d_subgrad_space));

  delete[] h_problem_info->costs;
  delete[] h_problem_info->weights;
  delete[] h_problem_info->budgets;
  delete[] h_problem_info;

  queue_free(request_queue, tickets, head, tail);
  queue_free(memory_queue, tickets, head, tail);

  // print exit code message and return
  Log(info, "Exit code: %s", ExitCode_text[exit_code]);
  return int(exit_code);
}
