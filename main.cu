#include <stdio.h>
#include <cmath>
#include <vector>
#include "logger.cuh"
#include "instr_parser.cuh"

int main(int argc, char **argv)
{
  Log(debug, "Starting program");

  const char *fileName = argv[1];
  Log(debug, "File name: %s", fileName);

  FILE *fptr = fopen(fileName, "r");
  if (fptr == NULL)
  {
    Log(error, "%s file failed to open.", fileName);
    exit(-1);
  }
  instructions ilist;
  ilist.populate_ins_from_file(fptr);
  ilist.print();
  d_instruction *d_ilist = ilist.to_device_array();
}