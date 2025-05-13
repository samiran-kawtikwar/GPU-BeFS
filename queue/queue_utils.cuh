#pragma once

#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

typedef uint32_t queue_type;
typedef std::chrono::system_clock::time_point timepoint;

static std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
	return std::chrono::high_resolution_clock::now();
}

static timepoint stime()
{
	return std::chrono::system_clock::now();
}

static double elapsedSec(timepoint start)
{
	return (std::chrono::system_clock::now() - start).count() / 1e9;
}

/*Device function that returns how many SMs are there in the device/arch - it can be more than the maximum readable SMs*/
__device__ __forceinline__ unsigned int getnsmid()
{
	unsigned int r;
	asm("mov.u32 %0, %%nsmid;"
			: "=r"(r));
	return r;
}

/*Device function that returns the current SMID of for the block being run*/
__device__ __forceinline__ unsigned int getsmid()
{
	unsigned int r;
	asm("mov.u32 %0, %%smid;"
			: "=r"(r));
	return r;
}

/*Device function that returns the current warpid of for the block being run*/
__device__ __forceinline__ unsigned int getwarpid()
{
	unsigned int r;
	asm("mov.u32 %0, %%warpid;"
			: "=r"(r));
	return r;
}

/*Device function that returns the current laneid of for the warp in the block being run*/
__device__ __forceinline__ unsigned int getlaneid()
{
	unsigned int r;
	asm("mov.u32 %0, %%laneid;"
			: "=r"(r));
	return r;
}

template <typename T>
__host__ __device__ void swap_ele(T &a, T &b)
{
	T temp = a;
	a = b;
	b = temp;
}

__device__ __forceinline__ uint32_t my_sleep(uint32_t ns)
{
	__nanosleep(ns);
	if (ns < (1 << 20))
		ns <<= 1;
	return ns;
}