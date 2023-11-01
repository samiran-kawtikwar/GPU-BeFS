#pragma once
#include <vector>
#define MAX_HEAP_SIZE 1000000
const uint N_RECEPIENTS = 1; // Don't change
typedef unsigned long long int uint64;
typedef unsigned int uint;

enum TaskType
{
  PUSH = 0,
  POP,
  BATCH_PUSH,
  TOP
};

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

struct node
{
  float key;
  float value;
  node(){};
  __host__ __device__ node(float a, float b) : key(a), value(b){};
};

struct d_instruction
{
  TaskType type;
  node *values;
  size_t num_values;
};

struct queue_info
{
  TaskType type;
  node *values;
  uint32_t batch_size;  // For batch push
  int already_occupied; // For extra overwriting checks (defined as int for atomicOr operation)
  uint32_t id;          // For mapping with queue DON'T UPDATE
};
