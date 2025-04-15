#pragma once

#include <iostream>
#include <string>
#include <stdint.h>
// #define __DEBUG__

#ifdef __CUDACC__

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

#endif

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

#ifdef __CUDACC__
template <const char *END = newline, typename... Args>
__host__ __forceinline__ void Log(LogPriorityEnum l, const char *f, Args... args)
#else
template <const char *END = newline, typename... Args>
void Log(LogPriorityEnum l, const char *f, Args... args)
#endif // __CUDACC__
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

template <typename cost_type = int>
void printHostArray(const cost_type *h_array, size_t len, std::string name = "NULL")
{
  using namespace std;
  if (name != "NULL")
  {
    if (len < 1)
      Log(debug, "%s", name.c_str());
    else
      Log<colon>(debug, "%s", name.c_str());
  }
  if (len >= 1)
  {
    for (size_t i = 0; i < len - 1; i++)
    {
      cout << h_array[i] << ',';
    }
    cout << h_array[len - 1] << '.' << endl;
  }
}

template <typename cost_type = int>
void printHostMatrix(const cost_type *matrix, size_t nrows, size_t ncols, std::string name = "NULL")
{
  using namespace std;
  if (name != "NULL")
  {
    Log(debug, "%s", name.c_str());
  }
  for (size_t j = 0; j < nrows; j++)
  {
    const cost_type *array = &matrix[j * ncols];
    for (size_t i = 0; i < ncols; i++)
    {
      if (array[i] == cost_type(1e10))
        cout << "M";
      else
        cout << array[i];
      if (i < ncols - 1)
        cout << ", ";
    }
    cout << endl;
  }
}

#ifdef __CUDACC__

template <typename... Args>
__device__ __forceinline__ void DLog(LogPriorityEnum l, const char *f, Args... args)
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

template <typename cost_type = int>
void printDeviceArray(const cost_type *d_array, size_t len, std::string name = "NULL")
{
  if (name != "NULL")
  {
    if (len < 1)
      Log(debug, "%s", name.c_str());
    else
      Log<colon>(debug, "%s", name.c_str());
  }
  if (len >= 1)
  {
    cost_type *temp = new cost_type[len];
    cudaMemcpy(temp, d_array, len * sizeof(cost_type), cudaMemcpyDefault);
    printHostArray(temp, len, "NULL");
    delete[] temp;
  }
}

template <typename cost_type = uint>
void printDeviceMatrix(const cost_type *array, size_t nrows, size_t ncols, std::string name = "NULL")
{
  using namespace std;
  cost_type *temp = new cost_type[nrows * ncols];
  CUDA_RUNTIME(cudaMemcpy(temp, array, nrows * ncols * sizeof(cost_type), cudaMemcpyDefault));
  printHostMatrix(temp, nrows, ncols, name);
  delete[] temp;
}
#endif