#include "logger.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename Node>
class BHEAP
{
  Node *d_heap;
  int *d_size;
};

template <typename Node>
__global__ __device__ void pop(BHEAP<Node> heap){

};

template <typename Node>
__global__ __device__ void push(BHEAP<Node> heap, Node *new_Node){

};

template <typename Node>
__global__ __device__ void batch_push(BHEAP<Node> heap, Node *new_Nodes, int num_Nodes){

};

template <typename Node>
__global__ __device__ void parse_queue(BHEAP<Node> heap /*, pass the queue*/){

};