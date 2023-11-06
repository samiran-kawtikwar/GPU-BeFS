#pragma once

#include <iostream>
#include <string>
#define __DEBUG__

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
void Log(LogPriorityEnum l, const char *f, Args... args)
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

// #define Log(l_, f_, ...)printf((f_), __VA_ARGS__);

template <typename data = int>
void printDeviceArray(const data *d_array, size_t len, std::string name = NULL)
{

  using namespace std;
  data *temp = new data[len];

  if (name != "NULL")
  {
    if (len < 1)
      Log(debug, "%s", name.c_str());
    else
      Log<colon>(debug, "%s", name.c_str());
  }
  if (len >= 1)
  {
    CUDA_RUNTIME(cudaMemcpy(temp, d_array, len * sizeof(data), cudaMemcpyDefault));
    for (size_t i = 0; i < len - 1; i++)
    {
      cout << temp[i] << ',';
    }
    cout << temp[len - 1] << '.' << endl;
  }
  delete[] temp;
}

template <typename data = uint>
void printDeviceMatrix(const data *array, size_t nrows, size_t ncols, std::string name = NULL)
{
  using namespace std;
  data *temp = new data[nrows * ncols];
  CUDA_RUNTIME(cudaMemcpy(temp, array, nrows * ncols * sizeof(data), cudaMemcpyDefault));

  if (name != "NULL")
  {
    Log(debug, "%s", name.c_str());
  }
  for (size_t j = 0; j < nrows; j++)
  {
    data *temp2 = &temp[j * ncols];
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