#pragma once
#include "queue/queue.cuh"
#include "LAP/Hung_lap.cuh"
#include "LAP/Hung_Tlap.cuh"

// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_ITER 100

const uint N_RECEPIENTS = 1; // Don't change
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

struct subgrad_space
{
  float *mult, *g, *lap_costs, *LB, *real_obj, *max_LB;
  int *X;
  int *col_fixed_assignments;
  TLAP<float> T;
  __host__ void allocate(uint N, uint K, uint nworkers = 0, uint devID = 0)
  {
    nworkers = (nworkers == 0) ? N : nworkers;
    // allocate space for mult, g, lap_costs, LB, LB_old, X, and th
    CUDA_RUNTIME(cudaMalloc((void **)&mult, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&g, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&lap_costs, nworkers * N * N * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&X, nworkers * N * N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&LB, nworkers * MAX_ITER * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&max_LB, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&real_obj, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_fixed_assignments, nworkers * N * sizeof(int)));

    CUDA_RUNTIME(cudaMemset(mult, 0, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(g, 0, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(lap_costs, 0, nworkers * N * N * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(X, 0, nworkers * N * N * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(LB, 0, nworkers * MAX_ITER * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(max_LB, 0, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(real_obj, 0, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(col_fixed_assignments, 0, nworkers * N * sizeof(int)));

    Log(debug, "Allocating space for %u LAPs", nworkers);
    T = TLAP<float>(nworkers, N, devID);
    // T.allocate(nworkers, N, devID);
  };
  __host__ void clear()
  {
    CUDA_RUNTIME(cudaFree(mult));
    CUDA_RUNTIME(cudaFree(g));
    CUDA_RUNTIME(cudaFree(lap_costs));
    CUDA_RUNTIME(cudaFree(LB));
    CUDA_RUNTIME(cudaFree(real_obj));
    CUDA_RUNTIME(cudaFree(max_LB));
    CUDA_RUNTIME(cudaFree(X));
    CUDA_RUNTIME(cudaFree(col_fixed_assignments));
    T.th.clear();
  }
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