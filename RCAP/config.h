#pragma once
#include <stdio.h>
#include <string>
#include <unistd.h>

typedef unsigned int uint;
typedef uint cost_type;
typedef uint weight_type;

struct Config
{
  uint user_n, user_ncommodities;
  double frac;
  int deviceId;
  int seed;
};

static void usage()
{
  fprintf(stderr,
          "\nUsage: <Description> [Default]\n"
          "\n-n <size of the problem> [10]"
          "\n-k <number of commodities> [10]"
          "\n-f <range-fraction> [10.0]"
          "\n-d <deviceId> [0]"
          "\n-s <seed-value> [45345]"
          "\n");
}

static void printConfig(Config config)
{
  printf("  size: %u\n", config.user_n);
  printf("  ncommodities: %u\n", config.user_ncommodities);
  printf("  frac: %f\n", config.frac);
  printf("  Device: %u\n", config.deviceId);
  printf("  seed value: %d\n", config.seed);
}

static Config parseArgs(int argc, char **argv)
{
  Config config;
  config.user_n = 10;
  config.frac = 10.0;
  config.deviceId = 0;
  config.seed = 45345;
  config.user_ncommodities = 10;

  int opt;
  while ((opt = getopt(argc, argv, "n:f:d:s:h:k:")) >= 0)
  {
    switch (opt)
    {
    case 'n':
      config.user_n = atoi(optarg);
      break;
    case 'f':
      config.frac = std::stod(optarg);
      break;
    case 'd':
      config.deviceId = atoi(optarg);
      break;
    case 's':
      config.seed = atoi(optarg);
      break;
    case 'k':
      config.user_ncommodities = atoi(optarg);
      break;
    case 'h':
      usage();
      exit(0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return config;
}
