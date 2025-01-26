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
  std::vector<NODE> h_heap;
  std::vector<node_info> h_node_space;
  std::vector<int> h_fixed_assignment_space;
  uint psize;       // problem size
  size_t size;      // live size of the host heap
  size_t footprint; // Live memory footprint of the heap with all its node associated data

  // Constructors
  __host__ HHEAP(uint problem_size)
  {
    psize = problem_size;
    size = 0;
    footprint = 0;
    h_heap = std::vector<NODE>();
    h_node_space = std::vector<node_info>();
    h_fixed_assignment_space = std::vector<int>();
  }

  // Destructors
  __host__ void cleanup()
  {
    h_heap.clear();
    h_node_space.clear();
    h_fixed_assignment_space.clear();
  }

  void update_footprint()
  {
    footprint = 0;
    footprint += h_heap.size() * sizeof(NODE);
    footprint += h_node_space.size() * sizeof(node_info);
    footprint += h_fixed_assignment_space.size() * psize * sizeof(int);
  }

  // Sort in ascending order
  void sort()
  {
    std::sort(std::execution::par, h_heap.begin(), h_heap.end());
  }
};