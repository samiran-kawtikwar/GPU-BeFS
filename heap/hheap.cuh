#include <execution>
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include "../queue/queue.cuh"

template <typename NODE>
class DHEAP; // Forward declaration

/*
HHEAP: A host heap implementation
1. This is a min-heap maintained as a vector of NODEs
2. All heap operations are implemented as host functions: they are NOT thread-safe
3. Heap operations must be performed by single thread
*/
template <typename NODE>
class HHEAP
{
private:
  uint psize; // problem size
public:
  std::vector<NODE> heap;
  std::vector<node_info> node_space;
  std::vector<int> fixed_assignment_space;
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

  /* Convert the heap into standard format, defined as:
  1. The heap is sorted in ascending order of keys
  2. The heap node values in device memory are in a continuous order:
      i.e. d_heap[i].value at location d_node_space[i] for all i < d_size
  This format would allow efficient batch operations on the heap when the heap is moved to host and vice versa
  */
  void standardize()
  {
    // sort();
    Log(debug, "Started standardizing heap");
    std::vector<int> where(node_space.size(), -1);
    // where[i] = j -> information at i should be stored at j
    for (size_t i = 0; i < heap.size(); i++)
      where[heap[i].value->id] = i;

    node_info temp1 = node_info(psize), temp2 = node_info(psize);
    int temp1_dest, temp2_dest;

    for (size_t i = 0; i < heap.size(); i++)
    {
      int my_id = heap[i].value->id;
      int my_dest = where[my_id];

      if (my_id == my_dest) // case 1: no need to move
        continue;

      if (where[my_dest] == -1) // case 2: destination is empty
      {
        cpy(node_space[my_id], node_space[my_dest]);
        heap[i].value = &node_space[my_dest];
        where[my_dest] = my_dest;
        where[my_id] = -1;
      }

      else // case 3: Move until an empty location is found
      {
        cpy(node_space[my_dest], temp1);
        cpy(node_space[my_id], node_space[my_dest]);
        heap[i].value = &node_space[my_dest];

        // update where[] array
        temp1_dest = where[my_dest];
        where[my_dest] = my_dest;
        where[my_id] = -1;
        assert(temp1_dest >= 0);
        while (where[temp1_dest] != -1)
        {
          cpy(node_space[temp1_dest], temp2);
          cpy(temp1, node_space[temp1_dest]);
          heap[temp1_dest].value = &node_space[temp1_dest];

          temp2_dest = where[temp1_dest];
          where[temp1_dest] = temp1_dest;
          cpy(temp2, temp1);
          temp1_dest = temp2_dest;
        }

        cpy(temp1, node_space[temp1_dest]);
        heap[temp1_dest].value = &node_space[temp1_dest];
        where[temp1_dest] = temp1_dest;
      }
    }
    where.clear();
    temp1.clear();
    temp2.clear();
  }

  void to_device(DHEAP<NODE> &d_heap)
  {
    size_t size_limit = node_space.size();
    size_t heap_size = heap.size();
    assert(d_heap.d_size_limit[0] >= size_limit);
    cudaMemcpy(d_heap.d_node_space, node_space.data(), size_limit * sizeof(node_info), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heap.d_fixed_assignment_space, fixed_assignment_space.data(),
               psize * size_limit * sizeof(int), cudaMemcpyHostToDevice);
    int grid_dim = (size_limit + 31) / 32;
    execKernel(link_node_fa, grid_dim, 32, 0, true,
               d_heap.d_node_space, d_heap.d_fixed_assignment_space, size_limit, psize);
    // Update the heap to use device addresses for the `value` field
    for (size_t i = 0; i < heap_size; i++)
    {
      if (heap[i].value != nullptr)
      {
        size_t offset = heap[i].value - node_space.data(); // Offset in host node_space
        heap[i].value = d_heap.d_node_space + offset;      // Point to the corresponding device address
      }
    }
    cudaMemcpy(d_heap.d_heap, heap.data(), heap_size * sizeof(node), cudaMemcpyHostToDevice);
    d_heap.d_size[0] = heap_size;
    d_heap.d_max_size[0] = max(d_heap.d_max_size[0], heap_size);
  }

  // print heap in custom format for debugging
  void print(std::string name = NULL)
  {
    if (name != "NULL")
      Log(critical, "%s", name.c_str());
    for (size_t i = 0; i < heap.size(); i++)
    {
      Log<nun>(info, "Node: %d, LB: %.0f, level: %d ----> \t", i, heap[i].key, heap[i].value->level);
      for (size_t j = 0; j < psize - 1; j++)
      {
        Log<comma>(debug, "%d", heap[i].value->fixed_assignments[j]);
      }
      Log<nun>(debug, "%d", heap[i].value->fixed_assignments[psize - 1]);
      Log<nun>(warn, "\t--- stored at ID: %u\n", heap[i].value->id);
    }
  }

  void print(std::vector<int> &where)
  {
    Log(critical, "Where array:");
    for (size_t i = 0; i < node_space.size(); i++)
    {
      Log<comma>(info, "%d", where[i]);
    }
    Log(info, "\n");
  }

  // Sort in ascending order
  void sort()
  {
    std::sort(std::execution::par, heap.begin(), heap.end());
  }

private:
  void cpy(node_info &source, node_info &dest)
  {
    dest.LB = source.LB;
    dest.level = source.level;
    for (size_t j = 0; j < psize; j++)
    {
      dest.fixed_assignments[j] = source.fixed_assignments[j];
    }
  }
};
