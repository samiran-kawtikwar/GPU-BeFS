#pragma once
#include "queue/queue.cuh"
#include "utils/cuda_utils.cuh"

// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_ITER 100
// #define TIMER

const uint N_RECEPIENTS = 1; // Don't change
typedef unsigned int uint;
typedef uint cost_type;
typedef uint weight_type;

#define BlockSize 64U
#define TileSize 64U
#define TilesPerBlock (BlockSize / TileSize)
#define TILE cg::thread_block_tile<TileSize>

uint GRID_DIM_X;

enum TaskType
{
  PUSH = 0,
  POP,
  BATCH_PUSH,
  TOP
};

enum ExitCode
{
  OPTIMAL = 0,
  HEAP_FULL,
  INFEASIBLE,
  UNKNOWN_ERROR
};

enum Location
{
  HOST = -1,
  DEVICE
};

const char *ExitCode_text[] = {
    "OPTIMAL",
    "HEAP_FULL",
    "INFEASIBLE",
    "UNKNOWN_ERROR"};

__device__ __forceinline__ const char *
getTextForEnum(int enumVal)
{
  return (const char *[]){
      "PUSH",
      "POP",
      "BATCH_PUSH",
      "TOP",
  }[enumVal];
}

const char *enum_to_str(TaskType type)
{
  if (type == PUSH)
    return "push";
  else if (type == POP)
    return "pop";
  else if (type == BATCH_PUSH)
    return "batch_push";
  else if (type == TOP)
    return "top";
  else
    return "unknown";
};

struct problem_info
{
  uint psize, ncommodities;
  cost_type *costs;     // cost of assigning
  weight_type *weights; // weight of each commodity
  weight_type *budgets; // capacity of each commodity
};

struct node_info
{
  int *fixed_assignments;
  float LB;
  uint level;
  uint id; // For mapping with memory queue; DON'T UPDATE
  __host__ __device__ node_info() {};
  __host__ __device__ node_info(int *fa, float lb, uint lvl, uint address = 0) : fixed_assignments(fa), LB(lb), level(lvl), id(address) {};
  __host__ node_info(uint psize)
  {
    fixed_assignments = new int[psize];
  }
  void clear()
  {
    if (fixed_assignments)
    {
      delete[] fixed_assignments;
      fixed_assignments = nullptr;
    }
  }
};

struct node
{
  float key;
  Location location; // To identify where the value is stored
  node_info *value;
  __host__ __device__ node() {};
  __host__ __device__ node(float a, node_info *b)
  {
    key = a;
#ifdef __CUDA_ARCH__
    location = DEVICE; // Compiling for device
#else
    location = HOST; // Compiling for host
#endif
    value = b;
  }

  // Comparison operator
  __host__ __device__ bool operator<(const node &other) const
  {
    return key < other.key; // Compare based on the key
  }
};

struct d_instruction
{
  TaskType type;
  node *values;
  size_t num_values; // For batch push
  __host__ __device__ d_instruction(TaskType req_type, size_t req_len, node *nodes) { type = req_type, num_values = req_len, values = nodes; };
};

struct queue_info
{
  TaskType type;
  node nodes[MAX_TOKENS];
  uint batch_size;                                              // For batch push
  cuda::atomic<uint32_t, cuda::thread_scope_device> req_status; // 0 done 1 pending 2 invalid
  uint id;                                                      // For mapping with request queue; DON'T UPDATE
};

struct worker_info
{
  node nodes[MAX_TOKENS];
  float LB[MAX_TOKENS];
  uint level[MAX_TOKENS];
  bool feasible[MAX_TOKENS];
  int *fixed_assignments; // To temporarily store fixed assignments
  uint *address_space;    // To temporarily store dequeued addresses

  static void allocate_all(worker_info *d_worker_space, size_t nworkers, size_t psize)
  {
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

enum CounterName
{
  INIT = 0,
  QUEUING,
  WAITING,
  WAITING_UNDERFLOW,
  TRANSFER,
  // FEAS_CHECK,
  UPDATE_LB,
  // SOLVE_LAP_FEAS,
  // SOLVE_LAP_SUBGRAD,
  // BRANCH,
  NUM_COUNTERS
};

enum LAPCounterName
{
  STEP1 = static_cast<int>(NUM_COUNTERS),
  STEP2,
  STEP3,
  STEP4,
  STEP5,
  STEP6,
  NUM_LAP_COUNTERS
};

enum RequestStatus
{
  DONE = 0,
  PROCESSING,
  INVALID
};

const char *CounterName_text[] = {
    "INIT",
    "QUEUING",
    "WAITING",
    "WAITING_UNDERFLOW",
    "TRANSFER",
    // "FEAS_CHECK",
    "UPDATE_LB",
    // "SOLVE_LAP_FEAS",
    // "SOLVE_LAP_SUBGRAD",
    // "BRANCH",
    "NUM_COUNTERS"};

const char *LAPCounterName_text[] = {
    "STEP1",
    "STEP2",
    "STEP3",
    "STEP4",
    "STEP5",
    "STEP6",
    "NUM_LAP_COUNTERS"};

#ifdef TIMER
#include "utils/profile_utils.cuh"
#include "LAP/profile_utils.cuh"
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
