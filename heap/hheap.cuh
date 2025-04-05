#include <tbb/parallel_sort.h>
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include "../queue/queue.cuh"
#include <thread>
#include <vector>
#include <list>
#include <algorithm>
#include <execution>

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

  std::thread standardize_thread;
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

  void resize(size_t new_size)
  {
    heap.resize(new_size);
    node_space.resize(new_size);
    fixed_assignment_space.resize(new_size * psize);
  }

  void update_pointers()
  {
#pragma omp parallel for
    for (size_t i = 0; i < heap.size(); i++)
    {
      heap[i].value = &node_space[i];
      heap[i].value->fixed_assignments = &fixed_assignment_space[i * psize];
      heap[i].value->id = i;
    }
  }

  void append(NODE *d_heap, node_info *d_node_space, int *d_fixed_assignment_space, size_t last, size_t d_nele)
  {
    Log(debug, "Appending elements: %lu to %lu", last, last + d_nele);
    // Append nelements to the heap
    resize(size + d_nele);
    cudaMemcpy(&heap[size], &d_heap[last], d_nele * sizeof(NODE), cudaMemcpyDeviceToHost);
    cudaMemcpy(&node_space[size], &d_node_space[last], d_nele * sizeof(node_info), cudaMemcpyDeviceToHost);
    cudaMemcpy(&fixed_assignment_space[size * psize], &d_fixed_assignment_space[(size_t)last * psize], d_nele * psize * sizeof(int), cudaMemcpyDeviceToHost);

    Log(debug, "Appended %lu elements to host heap", d_nele);
    update_pointers();
    update_size();
  }

  void merge(DHEAPExtended<NODE> &d_bheap, size_t &host_nele, size_t &dev_nele, const float frac = 0.5)
  {
    host_nele = 0;
    dev_nele = 0;
    auto dev_ = d_bheap.dev_;
    auto d_heap = d_bheap.d_heap;
    auto d_size = d_bheap.d_size[0];
    auto d_size_limit = d_bheap.d_size_limit[0];

    // Create a copy of host heap
    std::vector<NODE> heap_temp(size + d_size);
    std::vector<NODE> device_heap(d_size);
    cudaMemcpy(device_heap.data(), d_heap, d_size * sizeof(NODE), cudaMemcpyDeviceToHost);
    std::merge(std::execution::par, device_heap.begin(), device_heap.end(),
               heap.begin(), heap.end(),
               heap_temp.begin());
    Log(debug, "Merge sorted heap size: %lu", heap_temp.size());
    size_t d_nele = d_size > d_size_limit * frac ? d_size_limit * frac : d_size;
    Log(debug, "Cutoff at %lu", d_nele);
    assert((d_nele > 0) && (d_nele < heap_temp.size()));

    size_t add = d_nele - 1;
    int loc = getPointerAtt(heap_temp[add].value);
    Log(debug, "Target pointer location: %s", loc >= 0 ? "DEVICE" : "HOST");
    if (loc == -1)
    {
      host_nele = (size_t)heap_temp[add].value->id + 1;
      for (int64_t i = add - 1; i >= 0; i--)
      {
        if (getPointerAtt(heap_temp[i].value) == dev_)
        {
          dev_nele = heap_temp[i].value - d_bheap.d_node_space + 1;
          break;
        }
      }
    }
    else if (loc == dev_)
    {
      dev_nele = heap_temp[add].value - d_bheap.d_node_space + 1;
      for (int64_t i = add - 1; i >= 0; i--)
      {
        if (getPointerAtt(heap_temp[i].value) == -1)
        {
          host_nele = (size_t)heap_temp[i].value->id + 1;
          break;
        }
      }
    }
    else
    {
      Log(critical, "Get attributes failed");
      exit(1);
    }
    Log(info, "Host nele: %lu, Dev nele: %lu", host_nele, dev_nele);
    assert(host_nele + dev_nele == d_nele);
    heap_temp.clear();
    device_heap.clear();
  }

  /* Convert the heap into standard format, defined as:
  1. The heap is sorted in ascending order of keys
  2. The heap node values in device memory are in a continuous order:
      i.e. d_heap[i].value at location d_node_space[i] for all i < d_size
  This format would allow efficient batch operations on the heap when the heap is moved to host and vice versa
  */
  void rearrange()
  {
    // sort();
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

  void rearrange_p()
  {
    std::vector<int> where(heap.size(), -1);
    std::vector<node_info> cache_node(node_space.size());
    std::vector<int> cache_int(node_space.size());
// where[i] = j -> information at i should be stored at j
#pragma omp parallel
    {
      // Fill where array
#pragma omp for schedule(static)
      for (size_t i = 0; i < heap.size(); i++)
        where[i] = heap[i].value->id;

// #pragma omp for schedule(static)
#pragma omp for schedule(static)
      for (size_t i = 0; i < heap.size(); i++)
      {
        cache_node[i] = node_space[where[i]]; // gather the node values
        // printf("%lu <- %d \t level %u\n", i, my_id, cache_node[i].level);
      }
      // for (size_t i = 0; i < heap.size(); i++)
      //   printf("%lu, %u\n", i, cache_node[i].level);

#pragma omp for schedule(static)
      for (size_t i = 0; i < heap.size(); i++)
        node_space[i] = cache_node[i];
      for (size_t j = 0; j < psize; j++)
      {
#pragma omp for schedule(static)
        for (size_t i = 0; i < heap.size(); i++)
        {
          cache_int[i] = fixed_assignment_space[where[i] * psize + j];
        }
#pragma omp for schedule(static)
        for (size_t i = 0; i < heap.size(); i++)
        {
          fixed_assignment_space[i * psize + j] = cache_int[i];
        }
      }
#pragma omp for schedule(static)
      for (size_t i = 0; i < heap.size(); i++)
      {
        heap[i].value = &node_space[i];
        heap[i].value->fixed_assignments = &fixed_assignment_space[i * psize];
        heap[i].value->id = i;
      }
    }
    where.clear();
    cache_node.clear();
    cache_int.clear();
  }

  void
  to_device(DHEAP<NODE> &d_heap)
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

  void standardize()
  {
    Log(warn, "Started standardizing host heap");
    check_std("Checking host before sorting", false, false);
    sort();
    check_std("Checking host before rearrange", false, false);
    Log(debug, "Launching rearrange_p");
    rearrange_p();
    check_std("Checking host after rearrange", true, false);
    Log(warn, "Finished standardizing host heap");
  }

  void standardize_async()
  {
    if (standardize_thread.joinable())
      standardize_thread.join();
    Log(info, "Launching asynchronous standardization of host heap");
    standardize_thread = std::thread(&HHEAP::standardize, this);
  }

  // print heap in custom format for debugging
  void print(std::string name = NULL)
  {
    if (name != "NULL")
      Log(critical, "%s", name.c_str());
    for (size_t i = 0; i < heap.size(); i++)
    {
      Log<nun>(info, "Node: %d, LB: %f, level: %d ----> \t", i, heap[i].key, heap[i].value->level);
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

  void check_std(std::string name = NULL, bool check_id = false, bool print_heap = true)
  {
#ifdef __DEBUG__
    bool failed = false;
    if (name != "NULL")
      Log(info, "%s", name.c_str());

    for (size_t i = 0; i < heap.size(); i++)
    {
      uint count = 0;
      for (size_t j = 0; j < psize; j++)
      {
        if (heap[i].value->fixed_assignments[j] > 0)
          count++;
      }
      if (count != heap[i].value->level)
      {
        Log(critical, "Level mismatch at index %lu", i);
        failed = true;
        break;
      }
      if (check_id && heap[i].value->id != i)
      {
        Log(critical, "ID mismatch at index %lu", i);
        failed = true;
        break;
      }
    }
    if (print_heap || failed)
      print("Host heap");
    assert(failed == false);
#endif
  }

  // Sort in ascending order
  void sort(std::vector<NODE> *heap_ = nullptr)
  { // Change parameter to a pointer
    if (heap_ == nullptr)
      heap_ = &heap; // Assign the address of the member vector

    // assign stability_index

    Log(debug, "Started sorting host heap");
    std::sort(std::execution::par, heap_->begin(), heap_->end());
    Log(debug, "Finished sorting host heap");
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
