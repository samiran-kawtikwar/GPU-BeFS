#pragma once

#include <execution>
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include "../queue/queue.cuh"

/*
HHEAP: A host heap implementation
1. This is a min-heap maintained as a vector of NODEs
2. All heap operations are implemented as host functions: they are NOT thread-safe
3. Heap operations must be performed by single thread
*/
template <typename NODE>
class HHEAP
{
public:
  std::vector<NODE> heap;
  std::vector<node_info> node_space;
  std::vector<int> fixed_assignment_space;
  uint psize;       // problem size
  size_t size;      // live size of the host heap
  size_t footprint; // Live memory footprint of the heap with all its node associated data

  // Constructors
  __host__ HHEAP(uint problem_size)
  {
    psize = problem_size;
    size = 0;
    footprint = 0;
    heap = std::vector<NODE>();
    node_space = std::vector<node_info>();
    fixed_assignment_space = std::vector<int>();
  }

  // Destructors
  __host__ void cleanup()
  {
    heap.clear();
    node_space.clear();
    fixed_assignment_space.clear();
  }

  void update_footprint()
  {
    footprint = 0;
    footprint += heap.size() * sizeof(NODE);
    footprint += node_space.size() * sizeof(node_info);
    footprint += fixed_assignment_space.size() * psize * sizeof(int);
  }

  void update_size()
  {
    size = heap.size();
  }

  // Sort in ascending order
  void sort()
  {
    std::sort(std::execution::par, heap.begin(), heap.end());
  }
};