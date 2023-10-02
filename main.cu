#include <stdio.h>
#include <cmath>
#include <vector>
#include "logger.cuh"
#include "instr_parser.cuh"
#include "bheap.cuh"
#include "cuda_utils.cuh"

int main(int argc, char **argv)
{
  Log(debug, "Starting program");

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
  CUDA_RUNTIME(cudaMalloc((void **)&d_bheap.d_heap, MAX_HEAP_SIZE * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc((void **)&d_bheap.d_size, sizeof(size_t)));

  const size_t ins_len = ilist.tasks.size();
  const size_t max_batch = ilist.get_max_batch_size();
  execKernel((parse_queue<node>), 1, 32, 0, true, d_bheap, d_ilist, ins_len, max_batch);
  d_bheap.print();
}