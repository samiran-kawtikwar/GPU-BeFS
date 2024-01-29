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
#include "LAP/Hung_lap.cuh"
#include "LAP/lap_kernels.cuh"
#include "branch.cuh"

#include "RCAP/config.h"
#include "RCAP/cost_generator.h"
#include "RCAP/gurobi_solver.h"

__global__ void get_exit_code(ExitCode *ec)
{

  ec[0] = opt_reached.load(cuda::memory_order_consume)     ? ExitCode::OPTIMAL
          : heap_overflow.load(cuda::memory_order_consume) ? ExitCode::HEAP_FULL
                                                           : ExitCode::UNKNOWN_ERROR;
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
  problem_info *h_problem_info = generate_problem<cost_type>(config, config.seed);

  /*
    Log(info, "Costs: ");
    for (size_t i = 0; i < psize; i++)
    {
      for (size_t j = 0; j < psize; j++)
      {
        printf("%u, ", h_problem_info->costs[i * psize + j]);
      }
      printf("\n");
    }

    Log(info, "Weights: ");
    for (size_t k = 0; k < ncommodities; k++)
    {
      printf("Commodity: %lu\n", k);
      for (size_t i = 0; i < psize; i++)
      {
        for (size_t j = 0; j < psize; j++)
        {
          printf("%u, ", h_problem_info->weights[k * psize * psize + i * psize + j]);
        }
        printf("\n");
      }
      printf("\n");
    }

    Log(info, "Budgets: ");
    for (size_t k = 0; k < ncommodities; k++)
    {
      printf("%u, ", h_problem_info->budgets[k]);
    }
    printf("\n");
  */
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

  // Solve RCAP
  const cost_type UB = solve_with_gurobi<cost_type, weight_type>(h_problem_info->costs, h_problem_info->weights, h_problem_info->budgets, psize, ncommodities);

  // Log(info, "RCAP solved succesfully, objective %u\n", (uint)UB);
  // printf("Exiting...\n");
  // exit(0);

  Log(debug, "Solving RCAP with Branching");
  Timer t = Timer();

  size_t free, total;
  CUDA_RUNTIME(cudaMemGetInfo(&free, &total));
  Log(info, "Occupied memory: %f %", ((total - free) * 1.0) / total * 100);

  // Create space for queue
  size_t queue_size = psize + 1; // To be changed later -- equals grid dimension of request manager
  // size_t num_nodes = psize;      // To be changed later -- equals maximum multiplication factor
  queue_info *d_queue_space, *h_queue_space;
  CUDA_RUNTIME(cudaMalloc((void **)&d_queue_space, queue_size * sizeof(queue_info)));
  h_queue_space = (queue_info *)malloc(queue_size * sizeof(queue_info));
  memset(h_queue_space, 0, queue_size * sizeof(queue_info));
  for (size_t i = 0; i < queue_size; i++)
  {
    // CUDA_RUNTIME(cudaMalloc((void **)&h_queue_space[i].nodes, num_nodes * sizeof(node)));
    // std::fill(h_queue_space[i].nodes, 0, num_nodes * sizeof(node));
    // std::fill(h_queue_space[i].nodes, h_queue_space[i].nodes + num_nodes, 0);
    h_queue_space[i].req_status.store(0, cuda::memory_order_release);
    h_queue_space[i].batch_size = 0;
    h_queue_space[i].id = (uint32_t)i;
  }
  CUDA_RUNTIME(cudaMemcpy(d_queue_space, h_queue_space, queue_size * sizeof(queue_info), cudaMemcpyHostToDevice));
  delete[] h_queue_space;

  // Create space for bound computation and branching
  work_info *d_work_space;
  CUDA_RUNTIME(cudaMalloc((void **)&d_work_space, queue_size * sizeof(work_info)));
  CUDA_RUNTIME(cudaMemset((void *)d_work_space, 0, queue_size * sizeof(work_info)));

  // Create MPMC queue for handling heap requests
  queue_declare(request_queue, tickets, head, tail);
  queue_init(request_queue, tickets, head, tail, queue_size, dev_);

  // Create space for node_info and addresses
  size_t max_node_length = min(MAX_TOKENS, psize); // To be changed later -- equals problem size
  uint max_workers = psize + 1;
  node_info *d_node_space;

  uint *d_address_space; // To store dequeued addresses
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_address_space, max_workers * max_node_length * sizeof(uint)));
  CUDA_RUNTIME(cudaMemset((void *)d_address_space, 0, max_workers * max_node_length * sizeof(uint)));

  // uint memory_queue_len = MAX_HEAP_SIZE;
  // Get memory queue length based on available memory
  // size_t free, total;
  CUDA_RUNTIME(cudaMemGetInfo(&free, &total));
  Log(info, "Occupied memory: %f %", ((total - free) * 1.0) / total * 100);
  size_t memory_queue_len = (free * 0.95) / (sizeof(node_info) + sizeof(node)); // Keeping 5% headroom
  Log(info, "Memory queue length: %lu", memory_queue_len);

  CUDA_RUNTIME(cudaMalloc((void **)&d_node_space, memory_queue_len * sizeof(node_info)));
  CUDA_RUNTIME(cudaMemset((void *)d_node_space, 0, memory_queue_len * sizeof(node_info)));

  CUDA_RUNTIME(cudaMemGetInfo(&free, &total));
  Log(info, "Occupied memory: %f %", ((total - free) * 1.0) / total * 100);
  // Create BHEAP on device
  BHEAP<node> d_bheap = BHEAP<node>(memory_queue_len, dev_);

  // Create bnb-stats object on device
  bnb_stats *stats;
  CUDA_RUNTIME(cudaMallocManaged((void **)&stats, sizeof(bnb_stats)));
  stats->nodes_explored = 1; // for root node
  stats->nodes_pruned_incumbent = 0;
  stats->nodes_pruned_infeasible = 0;

  CUDA_RUNTIME(cudaMemGetInfo(&free, &total));
  Log(info, "Occupied memory: %f %", ((total - free) * 1.0) / total * 100);
  // Create MPMC queue for handling memory requests
  queue_declare(memory_queue, tickets, head, tail);
  queue_init(memory_queue, tickets, head, tail, memory_queue_len, dev_);

  // Populate memory queue and node_space IDs
  execKernel(fill_memory_queue, memory_queue_len, 32, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), d_node_space,
             memory_queue_len);
  execKernel(check_queue_global, 1, 1, dev_, false, queue_caller(memory_queue, tickets, head, tail),
             memory_queue_len);

  // Frist kernel to create L1 nodes
  execKernel(initial_branching, 2, 32, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), memory_queue_len,
             d_address_space, d_node_space,
             d_problem_info, max_node_length,
             queue_caller(request_queue, tickets, head, tail), queue_size,
             d_queue_space, d_work_space, d_bheap,
             UB);

  execKernel(branch_n_bound, psize + 1, 32, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), memory_queue_len,
             d_address_space, d_node_space,
             d_problem_info, max_node_length,
             queue_caller(request_queue, tickets, head, tail), queue_size,
             d_queue_space, d_work_space, d_bheap,
             UB, stats);

  printf("\n");

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
  CUDA_RUNTIME(cudaFree(d_address_space));
  CUDA_RUNTIME(cudaFree(d_work_space));
  CUDA_RUNTIME(cudaFree(stats));
  CUDA_RUNTIME(cudaFree(d_problem_info->costs));
  CUDA_RUNTIME(cudaFree(d_problem_info->weights));
  CUDA_RUNTIME(cudaFree(d_problem_info->budgets));
  CUDA_RUNTIME(cudaFree(d_problem_info));

  delete[] h_problem_info->costs;
  delete[] h_problem_info->weights;
  delete[] h_problem_info->budgets;
  delete[] h_problem_info;

  queue_free(request_queue, tickets, head, tail);
  queue_free(memory_queue, tickets, head, tail);

  return int(exit_code);
}
