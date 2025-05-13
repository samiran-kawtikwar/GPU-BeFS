#pragma once
#include <random>
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <thread>
#include "config.h"
#include "problem_info.h"
#include "gurobi_solver.h"
#include "../utils/timer.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"

using namespace std;

// Download a URL to local_path (using wget), then open it in an ifstream.
// Throws std::runtime_error on failure.

std::ifstream download_and_open(const std::string &raw_url,
                                const std::string &local_path)
{
  const int max_retries = 3;
  int delay_sec = 2;

  for (int attempt = 1; attempt <= max_retries; ++attempt)
  {
    // Use --timeout and --tries to avoid hanging
    std::string cmd = "wget --timeout=10 --tries=1 -q -O " + local_path + " " + raw_url;
    int rc = std::system(cmd.c_str());

    if (rc == 0)
    {
      std::ifstream in(local_path, std::ios::binary);
      if (!in.is_open())
        throw std::runtime_error("Failed to open file \"" + local_path + "\" after download");

      // Check file isn't empty
      in.seekg(0, std::ios::end);
      if (in.tellg() == 0)
        throw std::runtime_error("Downloaded file is empty: " + local_path);
      in.seekg(0);

      // Attempt to remove the file immediately
      if (std::remove(local_path.c_str()) != 0)
        std::perror(("Warning: could not delete temp file " + local_path).c_str());

      return in;
    }

    std::cerr << "[Attempt " << attempt << "] Download failed, retrying in " << delay_sec << "s...\n";
    std::this_thread::sleep_for(std::chrono::seconds(delay_sec));
    delay_sec *= 2;
  }

  throw std::runtime_error("Download failed after " + std::to_string(max_retries) + " attempts");
}

// Convenience overload: just pass the instance name (e.g. "chr12a.dat")
// and it will format the raw‑GitHub URL for your repo.
std::ifstream download_instance(const std::string &instance_name)
{
  const std::string base =
      "https://raw.githubusercontent.com/"
      "samiran-kawtikwar/QAPLIB-instances/"
      "main/instances/";
  std::string url = base + instance_name;
  return download_and_open(url, instance_name);
}

template <typename cost_type = uint>
problem_info *generate_problem(Config &config, const int seed = 45345)
{
  problem_info *pinfo = nullptr;
  CUDA_RUNTIME(cudaMallocManaged((void **)&pinfo, sizeof(problem_info)));
  double frac = 10;
  cost_type *distances, *flows;

  if (config.problemType == generated)
  {
    size_t user_n = config.user_n;
    pinfo->psize = user_n;
    distances = new cost_type[user_n * user_n];
    flows = new cost_type[user_n * user_n];
    if (user_n > 50)
    {
      Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 50");
      exit(-1);
    }
    // Generate random distances and flows
    default_random_engine generator(seed);
    generator.discard(1);
    uniform_int_distribution<cost_type> distribution(0, frac * user_n - 1);
    for (size_t i = 0; i < user_n; i++)
    {
      for (size_t j = 0; j < user_n; j++)
      {
        if (i == j)
        {
          distances[user_n * i + j] = 0;
          flows[user_n * i + j] = 0;
        }
        else
        {
          distances[user_n * i + j] = (cost_type)distribution(generator);
          flows[user_n * i + j] = (cost_type)distribution(generator);
        }
      }
    }

    // solve the problem to get the optimal objective
    pinfo->opt_objective = solve_with_gurobi(distances, flows, user_n);
  }
  else if (config.problemType == qaplib)
  {
    // Read distances and flows from file
    ifstream infile = download_instance(config.inputfile);
    if (!infile.is_open())
    {
      Log(error, "Could not open input file: %s", config.inputfile);
      exit(1);
    }
    while (infile.is_open() && infile.good())
    {
      infile >> pinfo->psize;
      uint n = pinfo->psize;
      config.user_n = n;

      distances = new cost_type[n * n];
      flows = new cost_type[n * n];
      infile >> pinfo->opt_objective;
      int counter = 0;
      cost_type *ptr1 = flows;
      cost_type *ptr2 = distances;
      while (!infile.eof())
      {
        if (counter < n * n)
          infile >> *(ptr1++);

        else
          infile >> *(ptr2++);

        counter++;
      }

      if (counter < 2 * n * n)
      {
        std::cerr << "Error: input size mismatch: " << config.inputfile << std::endl;
        exit(1);
      }

      infile.close();
    }
    infile.close();
  }
  else
  {
    Log(error, "Invalid problem type");
    exit(1);
  }
  // Copy distances and flows to info
  pinfo->allocate();
  uint n = pinfo->psize;
  CUDA_RUNTIME(cudaMemcpy(pinfo->distances, distances, n * n * sizeof(cost_type), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemcpy(pinfo->flows, flows, n * n * sizeof(cost_type), cudaMemcpyHostToDevice));
  delete[] distances;
  delete[] flows;
  if (pinfo->psize > 50)
  {
    Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 50");
    exit(-1);
  }
  return pinfo;
}

template <typename cost_type = uint>
void print(problem_info *pinfo, bool print_distances = true, bool print_flows = true)
{
  uint N = pinfo->psize;
  Log(info, "Optimal objective: %u", pinfo->opt_objective);
  if (print_distances)
  {
    Log(warn, "Distances: ");
    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        printf("%u, ", pinfo->distances[i * N + j]);
      }
      printf("\n");
    }
  }
  if (print_flows)
  {
    Log(warn, "Flows: ");
    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        printf("%u, ", pinfo->flows[i * N + j]);
      }
      printf("\n");
    }
  }
}