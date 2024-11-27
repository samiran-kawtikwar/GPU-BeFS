#pragma once

#include <iostream>
#include <string>
// #define __DEBUG__
#define DEBUG_D

namespace logger
{
  __device__ __forceinline__ uint32_t my_sleep(uint32_t ns)
  {
    __nanosleep(ns);
    if (ns < (1 << 20))
      ns <<= 1;
    return ns;
  }
}

const char newline[] = "\n";
const char comma[] = ", ";
const char colon[] = ": ";
const char nun[] = "";

enum LogPriorityEnum
{
  critical,
  warn,
  error,
  info,
  debug,
  none
};

template <const char *END = newline, typename... Args>
__host__ __forceinline__ void Log(LogPriorityEnum l, const char *f, Args... args)
{

  bool print = true;
#ifndef __DEBUG__
  if (l == debug)
  {
    print = false;
  }
#endif // __DEBUG__

  if (print)
  {
    // Line Color Set
    switch (l)
    {
    case critical:
      printf("\033[1;31m"); // Set the text to the color red.
      break;
    case warn:
      printf("\033[1;33m"); // Set the text to the color brown.
      break;
    case error:
      printf("\033[1;31m"); // Set the text to the color red.
      break;
    case info:
      printf("\033[1;32m"); // Set the text to the color green.
      break;
    case debug:
      printf("\033[1;34m"); // Set the text to the color blue.
      break;
    default:
      printf("\033[0m"); // Resets the text to default color.
      break;
    }

    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    if (END == newline)
      printf("[%02d:%02d:%02d] ", timeinfo->tm_hour, timeinfo->tm_min,
             timeinfo->tm_sec);

    printf(f, args...);
    printf(END);

    printf("\033[0m");
  }
}

template <typename... Args>
__forceinline__ __device__ void DLog(LogPriorityEnum l, const char *f, Args... args)
{
#ifdef DEBUG_D
  {
    bool print = true;
    static int logging_flag = int(false);
#ifndef __DEBUG__
    if (l == debug)
    {
      print = false;
    }
#endif // __DEBUG__

    if (print)
    {
      uint ns = 8;
      do
      {
        if (atomicCAS(&logging_flag, int(false), int(true)) == int(false))
        {

          // Line Color Set
          const char *prefix = (l == debug)                    ? "\033[1;34m" // blue
                               : (l == info)                   ? "\033[1;32m" // green
                               : (l == warn)                   ? "\033[1;33m" // brown
                               : (l == error || l == critical) ? "\033[1;31m" // red
                                                               : "\033[0m";   // default

          printf(prefix);
          printf(f, args...);
          printf("\033[0m");
          atomicCAS(&logging_flag, int(true), int(false));
          break;
        }
      } while (ns = logger::my_sleep(ns));
    }
  }
#else
  {
  }
#endif
}

template <typename cost_type = int>
void printDeviceArray(const cost_type *d_array, size_t len, std::string name = NULL)
{

  using namespace std;
  cost_type *temp = new cost_type[len];

  if (name != "NULL")
  {
    if (len < 1)
      Log(debug, "%s", name.c_str());
    else
      Log<colon>(debug, "%s", name.c_str());
  }
  if (len >= 1)
  {
    CUDA_RUNTIME(cudaMemcpy(temp, d_array, len * sizeof(cost_type), cudaMemcpyDefault));
    for (size_t i = 0; i < len - 1; i++)
    {
      cout << temp[i] << ',';
    }
    cout << temp[len - 1] << '.' << endl;
  }
  delete[] temp;
}

template <typename cost_type = uint>
void printDeviceMatrix(const cost_type *array, size_t nrows, size_t ncols, std::string name = NULL)
{
  using namespace std;
  cost_type *temp = new cost_type[nrows * ncols];
  CUDA_RUNTIME(cudaMemcpy(temp, array, nrows * ncols * sizeof(cost_type), cudaMemcpyDefault));

  if (name != "NULL")
  {
    Log(debug, "%s", name.c_str());
  }
  for (size_t j = 0; j < nrows; j++)
  {
    cost_type *temp2 = &temp[j * ncols];
    for (size_t i = 0; i < ncols - 1; i++)
    {
      cout << temp2[i] << ", ";
    }
    cout << temp2[ncols - 1] << endl;
    // for (size_t i = 0; i < ncols; i++)
    // {
    //   if (temp2[i] >= (int)ncols)
    //     cout << "Problem at row: " << i << " assignment: " << temp2[i] << endl;
    // }
  }
  delete[] temp;
}