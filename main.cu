#include <stdio.h>
#include <cmath>
#include <vector>
#include "utils/logger.cuh"
#include "instr_parser.cuh"
#include "heap/bheap.cuh"
#include "utils/cuda_utils.cuh"
#include "queue/queue.cuh"
#include "request_manager.cuh"
#include "memory_manager.cuh"
#include "defs.cuh"

int main(int argc, char **argv)
{
  Log(debug, "Starting program");
  uint dev_ = 0;
  CUDA_RUNTIME(cudaSetDevice(dev_));
  const char *fileName = argv[1];
  Log(debug, "File name: %s", fileName);

  FILE *fptr = fopen(fileName, "r");
  if (fptr == NULL)
  {
    Log(error, "%s file failed to open.", fileName);
    exit(-1);
  }
  INSTRUCTIONS ilist;
  ilist.populate_ins_from_file(fptr);
  ilist.print();
  d_instruction *d_ilist = ilist.to_device_array();

  // create BHEAP on device
  BHEAP<node> d_bheap;
  CUDA_RUNTIME(cudaMalloc((void **)&d_bheap.d_heap, MAX_HEAP_SIZE * sizeof(node)));
  CUDA_RUNTIME(cudaMalloc((void **)&d_bheap.d_size, sizeof(size_t)));

  size_t ins_len = ilist.tasks.size();
  const size_t max_batch = ilist.get_max_batch_size();

  // Create space for queue
  size_t queue_size = 100; // To be changed later -- equals grid dimension of request manager
  size_t num_nodes = 100;  // To be changed later -- equals maximum multiplication factor
  queue_info *d_queue_space, *h_queue_space;
  CUDA_RUNTIME(cudaMalloc((void **)&d_queue_space, queue_size * sizeof(queue_info)));
  h_queue_space = (queue_info *)malloc(queue_size * sizeof(queue_info));
  for (size_t i = 0; i < queue_size; i++)
  {
    CUDA_RUNTIME(cudaMalloc((void **)&h_queue_space[i].values, num_nodes * sizeof(node)));
    CUDA_RUNTIME(cudaMemset(h_queue_space[i].values, 0, num_nodes * sizeof(node)));
    h_queue_space[i].batch_size = 0;
    h_queue_space[i].already_occupied = int(false);
    h_queue_space[i].id = (uint32_t)i;
  }
  CUDA_RUNTIME(cudaMemcpy(d_queue_space, h_queue_space, queue_size * sizeof(queue_info), cudaMemcpyHostToDevice));
  delete[] h_queue_space;

  // Create space for node_info
  size_t max_node_length = 100; // To be changed later -- equals problem size
  uint memory_queue_len = MAX_HEAP_SIZE;
  nodetype **d_node_space;
  CUDA_RUNTIME(cudaMalloc((void **)&d_node_space, memory_queue_len * max_node_length * sizeof(nodetype)));
  CUDA_RUNTIME(cudaMemset(d_node_space, 0, memory_queue_len * max_node_length * sizeof(nodetype)));

  // Create MPMC queue for handling memory requests
  cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready_memory = nullptr;
  queue_declare(memory_queue, tickets, head, tail);
  queue_init(memory_queue, tickets, head, tail, memory_queue_len, dev_);
  CUDA_RUNTIME(cudaMalloc((void **)&work_ready_memory, memory_queue_len * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
  CUDA_RUNTIME(cudaMemset((void *)work_ready_memory, 0, memory_queue_len * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));

  // Populate memory queue
  execKernel(populate_memory_queue, memory_queue_len, 32, dev_, true, queue_caller(memory_queue, tickets, head, tail), memory_queue_len, work_ready_memory);
  // execKernel(check_queue, 1, 1, dev_, true, queue_caller(memory_queue, tickets, head, tail), work_ready_memory, memory_queue_len);

  // Create MPMC queue for handling heap requests
  cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready_requests = nullptr;
  queue_declare(request_queue, tickets, head, tail);
  queue_init(request_queue, tickets, head, tail, ins_len, dev_);
  CUDA_RUNTIME(cudaMalloc((void **)&work_ready_requests, ins_len * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
  CUDA_RUNTIME(cudaMemset((void *)work_ready_requests, 0, ins_len * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));

  ins_len = ilist.tasks.size();

  execKernel((request_manager<node>), 3 + 1, 32, dev_, true,
             d_ilist, ins_len, queue_caller(request_queue, tickets, head, tail), work_ready_requests,
             queue_size, d_bheap, d_queue_space);
  d_bheap.print();
  // Free device memory
  CUDA_RUNTIME(cudaFree(d_bheap.d_heap));
  CUDA_RUNTIME(cudaFree(d_ilist));
  CUDA_RUNTIME(cudaFree(d_bheap.d_size));
  CUDA_RUNTIME(cudaFree(d_queue_space));
  CUDA_RUNTIME(cudaFree(work_ready_requests));
  CUDA_RUNTIME(cudaFree(work_ready_memory));
  queue_free(request_queue, tickets, head, tail);
  queue_free(memory_queue, tickets, head, tail);

  return 0;
}