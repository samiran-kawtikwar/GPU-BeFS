#include <string>
#include "defs.cuh"
#include <vector>
#include <stdio.h>
#include "utils/logger.cuh"
#include "utils/cuda_utils.cuh"

struct instruction
{
  TaskType type;
  std::vector<node> values;
  instruction()
  {
    values = std::vector<node>();
  };
};

class INSTRUCTIONS
{
public:
  std::vector<instruction> tasks;
  FILE *ins_file;
  // Default blank constructor
  INSTRUCTIONS(){};
  INSTRUCTIONS(FILE *file)
  {
    ins_file = file;
  };
  instruction parse_instruction(char *line)
  {
    instruction ins;
    char *instr = strtok(line, " ,\n");
    while (instr != NULL)
    {
      // Log(debug, "instr: %s", instr);
      if (strcmp(instr, "push") == 0 || strcmp(instr, "PUSH") == 0)
        ins.type = TaskType(PUSH);
      else if (strcmp(instr, "pop") == 0 || strcmp(instr, "POP") == 0)
        ins.type = TaskType(POP);
      else if (strcmp(instr, "batch_push") == 0 || strcmp(instr, "BATCH_PUSG") == 0)
        ins.type = TaskType(BATCH_PUSH);
      else if (strcmp(instr, "top") == 0 || strcmp(instr, "TOP") == 0)
        ins.type = TaskType(TOP);
      else
        ins.values.push_back(node(float(atof(instr)), 0));
      instr = strtok(NULL, " ,\n");
    }
    return ins;
  };
  void populate_ins_from_file(FILE *filename = NULL)
  {
    if (filename == NULL)
      filename = ins_file;
    char line[100];
    while (fgets(line, 100, filename))
    {
      instruction ins = parse_instruction(line);
      this->tasks.push_back(ins);
    }
  }
  // iterate over ilist and print out the INSTRUCTIONS with values
  void print()
  {
    for (int i = 0; i < this->tasks.size(); i++)
    {
      instruction ins = this->tasks.at(i);
      Log(debug, "ins - type: %s, len: %lu", enum_to_str(ins.type), ins.values.size());
      for (size_t j = 0; j < ins.values.size(); j++)
      {
        Log<comma>(debug, "%f", ins.values.at(j).key);
      }
      Log<nun>(debug, "\n");
    }
  }
  uint get_max_batch_size()
  {
    uint max_batch_size = 1;
    for (int i = 0; i < this->tasks.size(); i++)
    {
      instruction ins = this->tasks.at(i);
      if (ins.type == TaskType(BATCH_PUSH))
      {
        if (ins.values.size() > max_batch_size)
          max_batch_size = ins.values.size();
      }
    }
    return max_batch_size;
  }

  d_instruction *to_device_array()
  {
    d_instruction *d_tasks, *h_tasks;
    h_tasks = (d_instruction *)malloc(sizeof(d_instruction) * tasks.size());
    Log(debug, "tasks size %lu", tasks.size());
    uint max_batch_size = get_max_batch_size();
    for (int i = 0; i < tasks.size(); i++)
    {
      CUDA_RUNTIME(cudaMalloc((void **)&h_tasks[i].values, sizeof(node) * max_batch_size));
      CUDA_RUNTIME(cudaMemset(h_tasks[i].values, 0, sizeof(node) * max_batch_size));
      node *h_values = new node[tasks.at(i).values.size()];
      std::copy(tasks.at(i).values.begin(), tasks.at(i).values.end(), h_values);
      h_tasks[i].type = tasks.at(i).type;
      CUDA_RUNTIME(cudaMemcpy(h_tasks[i].values, h_values, sizeof(node) * tasks.at(i).values.size(), cudaMemcpyHostToDevice));
      h_tasks[i].num_values = tasks.at(i).values.size();
      delete[] h_values;
    }

    CUDA_RUNTIME(cudaMalloc((void **)&d_tasks, sizeof(d_instruction) * tasks.size()));
    CUDA_RUNTIME(cudaMemcpy(d_tasks, h_tasks, sizeof(d_instruction) * tasks.size(), cudaMemcpyHostToDevice));
    delete[] h_tasks;

    return d_tasks;
  }
};