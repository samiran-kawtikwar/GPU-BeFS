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
#include "LAP/config.h"
#include "LAP/cost_generator.h"
#include "LAP/device_utils.cuh"
#include "LAP/Hung_lap.cuh"
#include "LAP/lap_kernels.cuh"
#include "branch.cuh"

__global__ void get_exit_code(bool *optimal, bool *heap_full)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    optimal[0] = opt_reached.load(cuda::memory_order_relaxed);
    heap_full[0] = heap_overflow.load(cuda::memory_order_relaxed);
  }
}

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);
  printConfig(config);
  int dev_ = config.deviceId;
  uint psize = config.user_n;
  if (psize > 100)
  {
    Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 100");
    exit(-1);
  }
  CUDA_RUNTIME(cudaDeviceReset());
  CUDA_RUNTIME(cudaSetDevice(dev_));

  cost_type *h_costs = generate_cost<cost_type>(config, config.seed);

  // print h_costs
  // for (size_t i = 0; i < psize; i++)
  // {
  //   for (size_t j = 0; j < psize; j++)
  //   {
  //     printf("%u, ", h_costs[i * psize + j]);
  //   }
  //   printf("\n");
  // }
  cost_type *d_costs;

  CUDA_RUNTIME(cudaMalloc((void **)&d_costs, psize * psize * sizeof(cost_type)));
  CUDA_RUNTIME(cudaMemcpy(d_costs, h_costs, psize * psize * sizeof(cost_type), cudaMemcpyHostToDevice));

  LAP<cost_type> *lap = new LAP<cost_type>(h_costs, psize, dev_);
  lap->solve();
  const cost_type UB = lap->objective;

  Log(info, "LAP solved succesfully, objective %u\n", (uint)UB);
  lap->print_solution();
  delete lap;
  Log(debug, "solving LAP with branching");

  // Create BHEAP on device
  BHEAP<node> d_bheap;
  CUDA_RUNTIME(cudaMalloc((void **)&d_bheap.d_heap, MAX_HEAP_SIZE * sizeof(node)));
  CUDA_RUNTIME(cudaMalloc((void **)&d_bheap.d_size, sizeof(size_t)));
  CUDA_RUNTIME(cudaMemset((void *)d_bheap.d_size, 0, sizeof(size_t)));

  // Create space for queue
  size_t queue_size = 100; // To be changed later -- equals grid dimension of request manager
  size_t num_nodes = 100;  // To be changed later -- equals maximum multiplication factor
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
  uint memory_queue_len = MAX_HEAP_SIZE;
  uint max_workers = queue_size;
  node_info *d_node_space;
  CUDA_RUNTIME(cudaMalloc((void **)&d_node_space, memory_queue_len * sizeof(node_info)));
  CUDA_RUNTIME(cudaMemset((void *)d_node_space, 0, memory_queue_len * sizeof(node_info)));

  uint *d_address_space; // To store dequeued addresses
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_address_space, max_workers * max_node_length * sizeof(uint)));
  CUDA_RUNTIME(cudaMemset((void *)d_address_space, 0, max_workers * max_node_length * sizeof(uint)));

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
             psize, d_address_space, d_node_space,
             d_costs, max_node_length,
             queue_caller(request_queue, tickets, head, tail), queue_size,
             d_queue_space, d_work_space, d_bheap,
             UB);

  // print d_address_space directly from unified memory
  for (size_t i = 0; i < 2; i++)
  {
    printf("Worker %d: \n", i);
    for (size_t j = 0; j < max_node_length; j++)
    {
      printf("%u, ", d_address_space[i * max_node_length + j]);
    }
    printf("\n\n");
  }
  d_bheap.print();

  execKernel(branch_n_bound, psize + 1, 32, dev_, true,
             queue_caller(memory_queue, tickets, head, tail), memory_queue_len,
             psize, d_address_space, d_node_space,
             d_costs, max_node_length,
             queue_caller(request_queue, tickets, head, tail), queue_size,
             d_queue_space, d_work_space, d_bheap,
             UB);

  printf("\n");

  // Get exit code
  bool *optimal, *heap_full;
  CUDA_RUNTIME(cudaMallocManaged((void **)&optimal, sizeof(bool)));
  CUDA_RUNTIME(cudaMallocManaged((void **)&heap_full, sizeof(bool)));
  optimal[0] = false;
  heap_full[0] = false;
  execKernel(get_exit_code, 1, 1, dev_, false, optimal, heap_full);
  Log(critical, "Optimal: %s, Heap full: %s", optimal[0] ? "true" : "false", heap_full[0] ? "true" : "false");
  d_bheap.print_size();
  /*
  // execKernel(free_memory_global, max_workers, 32, dev_, true, queue_caller(memory_queue, tickets, head, tail),
  //            memory_queue_len, d_address_space);
  // execKernel(get_queue_length_global, 1, 1, dev_, true, queue_caller(memory_queue, tickets, head, tail));

  ins_len = ilist.tasks.size();

  execKernel((request_manager<node>), max_workers + 1, 32, dev_, true,
             d_ilist, ins_len, queue_caller(request_queue, tickets, head, tail), request_state,
             queue_size, d_bheap, d_queue_space);
  */

  // Free device memory
  CUDA_RUNTIME(cudaFree(d_bheap.d_heap));
  CUDA_RUNTIME(cudaFree(d_bheap.d_size));
  CUDA_RUNTIME(cudaFree(d_queue_space));
  CUDA_RUNTIME(cudaFree(d_node_space));
  queue_free(request_queue, tickets, head, tail);
  queue_free(memory_queue, tickets, head, tail);

  if (optimal[0])
  {
    return 0;
  }
  else
  {
    return 1;
  }
}
