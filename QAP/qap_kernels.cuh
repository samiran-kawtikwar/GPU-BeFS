#pragma once
#include "../defs.cuh"

// called by a worker block
__device__ __forceinline__ void branch(int *fa, worker_info *my_space,
                                       const node &popped_node, const uint i, const uint lvl, const uint psize)
{
  for (uint j = threadIdx.x; j < psize; j += blockDim.x)
    fa[j] = popped_node.value->fixed_assignments[j];
  __syncthreads();
  if (threadIdx.x == 0)
  {
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
}