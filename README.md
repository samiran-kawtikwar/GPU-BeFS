# GPU-BeFS

A GPU-Accelerated framework for performing Branch and Bound Search on GPUs with Best First Search (BeFS) order. This project is part of the research paper "A framework for GPU-Accelerated Branch-And-Bound with Best First Search" by Samiran Kawtikwar, Izzat El Hajj, and Rakesh Nagi. [DOI Pending]

### Description

This repository contains the implementation of a GPU-accelerated framework for performing Branch and Bound Search on GPUs with Best First Search (BeFS) order. The framework is designed to efficiently solve combinatorial optimization problems using the Branch and Bound algorithm, leveraging the parallel processing capabilities of modern GPUs.
The framework is implemented in C++ and CUDA, and it provides a flexible and extensible architecture for solving various combinatorial optimization problems. The code is organized into several modules, each responsible for a specific aspect of the Branch and Bound algorithm.

### Features

- GPU-accelerated Branch and Bound framework (extensible to discrete optimization problems, user would have to implement problem-specific branching and bounding functions)
- Best First Search (BeFS) order for efficient exploration of the search space
- Works with a known upper bound, no feasibility search heuristics implemented
- Efficiently combines node and tree parallelism
- BnB tree is stored as an array of nodes in GPU global memory, subroutines available for offloading and onloading from host memory

## Requirements

- CUDA Toolkit (version 12.0 or higher)
- C++ compiler (g++, version 13 or higher)
- NVIDIA GPU with SM version 8.0 or higher (e.g. NVIDIA Ampere series or later, RTX30+ series)
- OS: Any recent linux (e.g. Ubuntu 22+, SLES 15+, RHEL 8+)

### Libraries used

- OMP (Open Multi-Processing) for multi-threading
- Gurobi (optional, for getting tight upper bounds)
- NVIDIA CUB (Part of CUDA Toolkit) for parallel algorithms and data structures
- CUDA cooperative groups (for abstract subworker parallelism)
- C++ STL (Standard Template Library) for data structures and algorithms

## Installation

1. Install the required libraries (CUDA Toolkit, C++ compiler, gurobi, openmp)
2. Clone the repository
3. Edit the SM version in the makefile according to your GPU architecture or use the get_SM.sh script by making it executable.
4. Link libraries accordingly in the makefile (mainly for Gurobi)
5. Compile the code using the 'make all' command
6. Refer to example section for running the code

### Configurations

The framework is designed to be flexible and extensible, allowing users to configure various aspects of the Branch and Bound algorithm. The following configurations are available: (P.S. the framework only supports 1 dimensional thread blocks and grids)

- **BlockSize**: The size of independent workers (Can range from 32 -- 1024)
- **TileSize**: The size of the tile used for processing individual nodes on the GPU (Can range from 2 -- BlockSize, subject to shared and register memory constraints)
- **TIMER**: Defining this flag will enable time measurement for different processes in the BnB kernel, disable it when performance testing
- **DEBUG**: Defining this flag will enable debug level prints from host and device side for the BnB kernel, disable it when performance testing (device side prints cause significant register usage).
- **cost_type**: The type of objective costs, **weight_type**: The type of coefficients, the available options are:
  - _uint_: Unsigned Integer coefficients [Default]
  - _float_: Floating-point coefficients
  - _double_: Double-precision floating-point coefficients (May restrict occupancy)

## Structure

The codebase is modular, with each component handling a distinct aspect of the GPU-based Branch and Bound (BnB) framework.

#### Core Modules

- **main.cpp**  
  Entry point of the framework. Handles initialization and kernel launch.

- **defs.cuh**  
  Defines core data structures such as BnB nodes and `node_info`, as well as configuration parameters like `cost_type` and `weight_type`.

- **branch.cuh**  
  Implements the global `branch_n_bound` kernel and invokes problem-specific branching and bounding functions.

- **request_manager.cuh**  
  Manages inkernel push/pop operations between GPU workers and the BnB tree in global memory.

- **memory_manager.cuh**  
  Handles GPU memory allocation for the tree, including overflow/underflow protection and host-device transfer routines.

- **queue/**  
  Lock-free multi-producer multi-consumer (MPMC) queue for inter-worker coordination, adapted from [1].

#### Problem-Specific Modules

- **LAP/**  
  GPU-accelerated solver for the Linear Assignment Problem using tile-based primitives. Based on [2].

- **RCAP/**  
  Implements subgradient-based bounding and feasibility routines for the Resource-Constrained Assignment Problem (RCAP), using the LAP module. Based on [3].

- **QAP/** _(experimental)_  
  Implements Gilmore-Lawler bounds for the Quadratic Assignment Problem. Not included in the main branch.

#### Utilities

- **heap/**  
  Contains a standard binary heap used in the device-resident priority queue.

- **utils/**  
  Helper routines for logging, device operations, timing, and profiling.

### File Tree (Partial View)

```
|--- main.cu
|--- branch.cuh
|--- defs.cuh
|--- memory_manager.cuh
|--- request_manager.cuh
├── RCAP/
│   ├── subgrad_solver.cuh
│   ├── rcap_kernels.cuh
│   ├── LAP/
│   │   ├── block_lap_kernels.cuh
│   │   ├── Hung_Tlap.cuh
├── queue/
│   ├── queue.cuh
│   ├── queue_utils.cuh
├── heap/
│   └── bheap.cuh
└── utils/
    ├── logger.cuh
    ├── cuda_utils.cuh
```

## Examples

Once compiled, the executable will be located under build directory with the name `main.exe`
The executable for RCAP takes following command line arguments with default values in square brackets:

```
-n <size of the problem> [10]
-k <number of commodities> [10]
-f <range-fraction> [10.0]
-d <deviceId> [0]
-s <seed-value> [45345]
```

The executable for QAP takes following command line arguments with default values in square brackets:

```
-n <size of the problem> [10]
-d <deviceId> [0]
-s <seed-value> [45345]
-i <input filename> [None, uses randomly generated problem]
```

When QAP is run with a generated problem, it will use gurobi to get the tight upper bound.
When run with an input filename, it will download the problem from a publicly hosted github repository which includes problem instances from QAPLIB. The file will also have the upper bound (in the standard QAPLIB format), so there will be no Gurobi call.
Some of the example input filenames are: `nug12.dat`, `nug14.dat`, `els19.dat` (etc)

## References

<a id="1">[1]</a> Almasri, M., Chang, Y.-H., El Hajj, I., Nagi, R., Xiong, J., & Hwu, W.-m. (2024). _Parallelizing Maximal Clique Enumeration on GPUs_ In _Proceedings of the 32nd International Conference on Parallel Architectures and Compilation Techniques (PACT '23)_, IEEE Press, pp. 162–175. [DOI](https://doi.org/10.1109/PACT58117.2023.00022)

<a id="2">[2]</a> Kawtikwar S. and Nagi, R. (2024). *HyLAC: Hybrid Linear Assignment solver in CUDA”, *Journal of Parallel and Distributed Computing\*, vol. 187, p. 104838. [DOI](https://doi.org/10.1016/j.jpdc.2024.104838)

<a id="3">[3]</a> Reynen O. H. (2020). *GPU-Accelerated algorithms for the resource-constrained assignment problem”, *M.S. Thesis, University of Illinois at Urbana-Champaign\*, May 2020. [DOI](https://hdl.handle.net/2142/108143)
