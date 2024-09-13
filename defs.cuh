#pragma once
#include "queue/queue.cuh"

// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_ITER 100
#define TIMER

const uint N_RECEPIENTS = 1; // Don't change
typedef unsigned int uint;
typedef uint cost_type;
typedef uint weight_type;

const uint n_threads = 512;
const uint GRID_DIM_X = (512 / n_threads) * 108;

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

struct queue_info
{
  TaskType type;
  node nodes[100];
  uint batch_size; // For batch push
  cuda::atomic<uint32_t, cuda::thread_scope_device> req_status;
  uint id; // For mapping with request queue; DON'T UPDATE
};

struct work_info
{
  uint batch_size;
  node nodes[100];
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
  TRANSFER,
  FEAS_CHECK,
  UPDATE_LB,
  SOLVE_LAP_FEAS,
  SOLVE_LAP_SUBGRAD,
  BRANCH,
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

const char *CounterName_text[] = {
    "INIT",
    "QUEUING",
    "WAITING",
    "TRANSFER",
    "FEAS_CHECK",
    "UPDATE_LB",
    "SOLVE_LAP_FEAS",
    "SOLVE_LAP_SUBGRAD",
    "BRANCH",
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
#define INIT_TIME(counters) initializeCounters(counters);

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
