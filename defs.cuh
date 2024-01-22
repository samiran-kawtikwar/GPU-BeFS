#pragma once
#include "queue/queue.cuh"

#define __DEBUG__
// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_DATA 0xffffffff
#define eps 1e-6

const uint N_RECEPIENTS = 1; // Don't change
typedef unsigned long long int uint64;
typedef unsigned int uint;
typedef uint cost_type;
typedef uint weight_type;

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

__forceinline__ __device__ const char *
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
  cost_type *costs;     // cost of assigning
  weight_type *weights; // weight of each commodity
  weight_type *budgets; // capacity of each commodity
};

struct node_info
{
  int fixed_assignments[100]; // To be changed later using appropriate partitions
  float LB;
  uint level;
  uint id; // For mapping with memory queue; DON'T UPDATE
  __host__ __device__ node_info() { std::fill(fixed_assignments, fixed_assignments + 100, -1); };
};

struct node
{
  float key;
  node_info *value;
  __host__ __device__ node(){};
  __host__ __device__ node(float a, node_info *b) : key(a), value(b){};
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
  uint nodes_pruned;
};