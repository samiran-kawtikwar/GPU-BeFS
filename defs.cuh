#pragma once
#include <vector>
#define __DEBUG__
#define MAX_HEAP_SIZE 1000000
typedef unsigned long long int uint64;
typedef unsigned int uint;

enum LogPriorityEnum
{
  critical,
  warn,
  error,
  info,
  debug,
  none
};

template <typename T>
struct Node
{
  float key;
  T value;
};

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
