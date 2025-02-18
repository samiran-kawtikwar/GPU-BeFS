#include <tbb/parallel_sort.h>
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include "../queue/queue.cuh"
#include <thread>

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

  void append(NODE &node)
  {
    heap.push_back(node);
    node_space.push_back(node.value);
    for (size_t i = 0; i < psize; i++)
    {
      fixed_assignment_space.push_back(node.value->fixed_assignments[i]);
    }
  }

  void update_pointers()
  {
#pragma omp paralle for
    for (size_t i = 0; i < size; i++)
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
    update_size();
    update_pointers();

    standardize();
  }

  void extended_append(DHEAP<NODE> &d_bheap, std::vector<node_extended> &heap_temp, size_t d_nele)
  {
    // Append nelements to the heap
    auto d_heap = d_bheap.d_heap;
    NODE *heap_from_device = new NODE[d_bheap.d_size[0]];
    cudaMemcpy(heap_from_device, d_heap, d_bheap.d_size[0] * sizeof(NODE), cudaMemcpyDeviceToHost);
#pragma omp parallel for
    for (size_t i = 0; i < heap_temp.size(); i++)
    {
      if (i < size)
      {
        heap_temp[i].key = heap[i].key;
        heap_temp[i].value = heap[i].value;
        heap_temp[i].stability_index = i;
      }
      else
      {
        heap_temp[i].key = heap_from_device[i - size].key;
        heap_temp[i].value = heap_from_device[i - size].value;
        heap_temp[i].stability_index = i;
      }
    }
    delete[] heap_from_device;
  }

  void merge(DHEAPExtended<NODE> &d_bheap, size_t &host_last, size_t &dev_last, const float frac = 0.5)
  {
    host_last = 0;
    dev_last = 0;
    auto dev_ = d_bheap.dev_;
    auto d_heap = d_bheap.d_heap;
    auto d_size = d_bheap.d_size[0];
    auto d_size_limit = d_bheap.d_size_limit[0];
    // Create a copy of host heap
    std::vector<node_extended> heap_temp(size + d_size);
    // Append all nodes to host heap
    extended_append(d_bheap, heap_temp, d_size);
    Log(debug, "Appended %d elements to host heap", d_size);
    Log(debug, "Merged heap size: %lu", heap_temp.size());
    // for (size_t i = 0; i < heap_temp.size(); i++)
    //   std::cout << i << "\t" << heap_temp[i].key << "\t" << heap_temp[i].value << std::endl;
    stable_sort(&heap_temp);
    // for (size_t i = 0; i < heap_temp.size(); i++)
    //   std::cout << i << "\t" << heap_temp[i].key << "\t" << heap_temp[i].value << std::endl;
    size_t d_nele = d_size > d_size_limit * frac ? d_size_limit * frac : d_size;

    Log(debug, "Cutoff at %lu", d_nele);
    assert(d_nele < heap_temp.size());
    int loc = getPointerAtt(heap_temp[d_nele].value);
    Log(warn, "Target pointer location: %d", loc);
    if (loc == -1)
    {
      host_last = (size_t)heap_temp[d_nele].value->id;
      dev_last = 0;
      for (int64_t i = d_nele; i >= 0; i--)
      {
        if (getPointerAtt(heap_temp[i].value) == dev_)
        {
          dev_last = heap_temp[i].value - d_bheap.d_node_space;
          break;
        }
      }
    }
    else if (loc == dev_)
    {
      dev_last = heap_temp[d_nele].value - d_bheap.d_node_space;
      host_last = 0;
      for (int64_t i = d_nele; i >= 0; i--)
      {
        if (getPointerAtt(heap_temp[i].value) == -1)
        {
          host_last = (size_t)heap_temp[i].value->id;
          break;
        }
      }
    }
    else
    {
      Log(critical, "Get attributes failed");
      exit(1);
    }
    Log(info, "Host last: %lu, Dev last: %lu", host_last, dev_last);
    if (host_last + dev_last > d_nele)
    {
      Log(critical, "Merge failed");
      // print("Host heap");
      // d_bheap.format_print("Device heap");
      // float *temp_keys = new float[heap_temp.size()];
      // for (size_t i = 0; i < heap_temp.size(); i++)
      //   temp_keys[i] = heap_temp[i].key;
      // printHostArray(temp_keys, heap_temp.size(), "Merged heap");
      // delete[] temp_keys;
      exit(1);
    }
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
    std::vector<int> where(node_space.size(), -1);
    std::vector<node_info> cache_node(node_space.size());
    std::vector<int> cache_int(node_space.size());
// where[i] = j -> information at i should be stored at j
#pragma omp parallel
    {
      // Fill where array
#pragma omp for schedule(static)
      for (size_t i = 0; i < heap.size(); i++)
        where[heap[i].value->id] = i;

#pragma omp for schedule(static)
      for (size_t i = 0; i < heap.size(); i++)
      {
        int my_id = heap[i].value->id;
        cache_node[i] = node_space[my_id];
      }
#pragma omp for schedule(static)
      for (size_t i = 0; i < heap.size(); i++)
        node_space[i] = cache_node[i];
      for (size_t j = 0; j < psize; j++)
      {
#pragma omp for schedule(static)
        for (size_t i = 0; i < heap.size(); i++)
        {
          int my_id = heap[i].value->id;
          cache_int[i] = fixed_assignment_space[my_id * psize + j];
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

  void standardize()
  {
    Log(warn, "Started standardizing host heap");
    sort();
    rearrange_p();
    Log(warn, "Finished standardizing host heap");
  }

  void standardize_async()
  {
    if (standardize_thread.joinable())
      standardize_thread.join();

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

  void check_std(std::string name = NULL)
  {
    bool failed = false;
    if (name != "NULL")
      Log(critical, "%s", name.c_str());
    for (size_t i = 0; i < heap.size(); i++)
    {
      if (heap[i].value->id != i)
      {
        Log(critical, "Standardization failed at index %lu", i);
        failed = true;
        break;
      }
    }
    if (failed)
      print("Standardized heap");
  }

  // Sort in ascending order
  void sort(std::vector<NODE> *heap_ = nullptr)
  { // Change parameter to a pointer
    if (heap_ == nullptr)
      heap_ = &heap; // Assign the address of the member vector

    // assign stability_index

    Log(debug, "Started sorting host heap");
    tbb::parallel_sort(heap_->begin(), heap_->end());
    Log(debug, "Finished sorting host heap");
  }

  // Uses stability index to maintain order of elements with same key
  void stable_sort(std::vector<node_extended> *heap_)
  {
// Reassign stability index
#pragma omp parallel for
    for (size_t i = 0; i < heap_->size(); i++)
      (*heap_)[i].stability_index = i;

    Log(debug, "Started stable sorting host heap");
    tbb::parallel_sort(heap_->begin(), heap_->end());
    Log(debug, "Finished stable sorting host heap");
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
