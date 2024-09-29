#pragma once

#include <iostream>
#include <string>

namespace host_log
{
  // #define __DEBUG__ ;
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
}