#pragma once

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

enum CounterName
{
  INIT = 0,
  QUEUING,
  WAITING,
  WAITING_UNDERFLOW,
  TRANSFER,
  UPDATE_LB,
  GET_Z,
  SOLVE_Z,
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
    "UPDATE_LB",
    "GET_Z",
    "SOLVE_Z",
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
