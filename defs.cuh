#pragma once
#include "queue/queue.cuh"
#include "RCAP/enums.cuh"
#include "RCAP/config.h"

// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_ITER 100
// #define TIMER

const uint N_RECEPIENTS = 1; // Don't change

#define BlockSize 64U
#define TileSize 64U
#define TilesPerBlock (BlockSize / TileSize)
#define TILE cg::thread_block_tile<TileSize>

uint GRID_DIM_X;

struct node_info
{
  int *fixed_assignments;
  float LB;
  uint level;
  uint id; // For mapping with memory queue; DON'T UPDATE
  __host__ __device__ node_info() {};
  __host__ __device__ node_info(int *fa, float lb, uint lvl, uint address = 0) : fixed_assignments(fa), LB(lb), level(lvl), id(address) {};
};

struct node
{
  float key;
  node_info *value;
  __host__ __device__ node() {};
  __host__ __device__ node(float a, node_info *b) : key(a), value(b) {};
};

struct d_instruction
{
  TaskType type;
  node *values;
  size_t num_values; // For batch push
  __host__ __device__ d_instruction(TaskType req_type, size_t req_len, node *nodes) { type = req_type, num_values = req_len, values = nodes; };
};

struct worker_info
{
  node nodes[MAX_TOKENS];
  float LB[MAX_TOKENS];
  uint level[MAX_TOKENS];
  bool feasible[MAX_TOKENS];
  int *fixed_assignments; // To temporarily store fixed assignments
  uint *address_space;    // To temporarily store dequeued addresses

  static void allocate_all(worker_info *&d_worker_space, size_t nworkers, size_t psize)
  {
    CUDA_RUNTIME(cudaMallocManaged((void **)&d_worker_space, nworkers * sizeof(worker_info)));
    for (size_t i = 0; i < nworkers; i++)
    {
      d_worker_space[i].allocate(psize);
    }
  }

  // Static function to free memory for an array of work_info instances
  static void free_all(worker_info *d_worker_space, size_t nworkers)
  {
    for (size_t i = 0; i < nworkers; i++)
    {
      d_worker_space[i].free();
    }
    CUDA_RUNTIME(cudaFree(d_worker_space));
  }
  // Function to allocate memory for this instance
  void allocate(size_t psize)
  {
    CUDA_RUNTIME(cudaMalloc((void **)&fixed_assignments, psize * psize * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&address_space, psize * sizeof(uint)));
    CUDA_RUNTIME(cudaMemset(fixed_assignments, 0, psize * psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(address_space, 0, psize * sizeof(uint)));
  }
  // Function to free allocated memory for this instance
  void free()
  {
    if (fixed_assignments && address_space)
    {
      CUDA_RUNTIME(cudaFree(fixed_assignments));
      CUDA_RUNTIME(cudaFree(address_space));
    }
  }
};

struct bnb_stats
{
  uint nodes_explored;
  uint nodes_pruned_incumbent;
  uint nodes_pruned_infeasible;
  void initialize()
  {
    nodes_explored = 1; // for root node
    nodes_pruned_incumbent = 0;
    nodes_pruned_infeasible = 0;
  }
};

#ifdef TIMER
#include "utils/profile_utils.cuh"
#include "RCAP/LAP/profile_utils.cuh"
#define INIT_TIME(counters) initializeCounters(&counters[blockIdx.x]);

#define START_TIME(countername)                                                       \
  {                                                                                   \
    if (countername < NUM_COUNTERS)                                                   \
      startTime(static_cast<CounterName>(countername), &counters[blockIdx.x]);        \
    else                                                                              \
      startTime(static_cast<LAPCounterName>(countername), &lap_counters[blockIdx.x]); \
  }

#define END_TIME(countername)                                                       \
  {                                                                                 \
    if (countername < NUM_COUNTERS)                                                 \
      endTime(static_cast<CounterName>(countername), &counters[blockIdx.x]);        \
    else                                                                            \
      endTime(static_cast<LAPCounterName>(countername), &lap_counters[blockIdx.x]); \
  }

#else
#define INIT_TIME(counters)
#define START_TIME(countername)
#define END_TIME(countername)
#endif
