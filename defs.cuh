#pragma once

#include <vector>
#define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
const uint N_RECEPIENTS = 1; // Don't change
typedef unsigned long long int uint64;
typedef unsigned int uint;
typedef float nodetype; // To be changed from float to required data type

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
  nodetype *value;
  node(){};
  __host__ __device__ node(float a, nodetype *b) : key(a), value(b){};
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
  uint batch_size;      // For batch push
  int already_occupied; // For extra overwriting checks (defined as int for atomicOr operation)
  uint id;              // For mapping with queue DON'T UPDATE
};
