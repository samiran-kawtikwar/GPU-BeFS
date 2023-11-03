#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>
#include "queue_utils.cuh"
typedef uint32_t queue_type;

__device__ __forceinline__ uint32_t my_sleep(uint32_t ns)
{
    __nanosleep(ns);
    if (ns < (1 << 20))
        ns <<= 1;
    return ns;
}

#define queue_enqueue(queue, tickets, head, tail, queue_size, val)                        \
    do                                                                                    \
    {                                                                                     \
        const uint32_t ticket = tail_##queue->fetch_add(1, cuda::memory_order_relaxed);   \
        const uint32_t target = ticket % queue_size;                                      \
        const uint32_t ticket_target = ticket / queue_size * 2;                           \
        uint32_t ns = 8;                                                                  \
        while (tickets_##queue[target].load(cuda::memory_order_relaxed) != ticket_target) \
            ns = my_sleep(ns);                                                            \
        while (tickets_##queue[target].load(cuda::memory_order_acquire) != ticket_target) \
            ns = my_sleep(ns);                                                            \
        queue[target] = val;                                                              \
        tickets_##queue[target].store(ticket_target + 1, cuda::memory_order_release);     \
    } while (0)

#define queue_dequeue(queue, tickets, head, tail, queue_size, fork, qidx, count)                        \
    do                                                                                                  \
    {                                                                                                   \
        qidx = head_##queue->load(cuda::memory_order_relaxed);                                          \
        fork = false;                                                                                   \
        if (tail_##queue->load(cuda::memory_order_relaxed) - qidx >= count)                             \
            fork = head_##queue->compare_exchange_weak(qidx, qidx + count, cuda::memory_order_relaxed); \
    } while (0)

#define queue_wait_ticket(queue, tickets, head, tail, queue_size, qidx, res)              \
    do                                                                                    \
    {                                                                                     \
        const uint32_t target = qidx % queue_size;                                        \
        const uint32_t ticket_target = qidx / queue_size * 2 + 1;                         \
        uint32_t ns = 8;                                                                  \
        while (tickets_##queue[target].load(cuda::memory_order_relaxed) != ticket_target) \
            ns = my_sleep(ns);                                                            \
        while (tickets_##queue[target].load(cuda::memory_order_acquire) != ticket_target) \
            ns = my_sleep(ns);                                                            \
        res = queue[target];                                                              \
        tickets_##queue[target].store(ticket_target + 1, cuda::memory_order_release);     \
    } while (0)

#define queue_full(queue, tickets, head, tail, queue_size) \
    (tail_##queue->load(cuda::memory_order_relaxed) - head_##queue->load(cuda::memory_order_relaxed) == queue_size)

#define queue_declare(queue, tickets, head, tail)                                                                                   \
    cuda::atomic<uint32_t, cuda::thread_scope_device> *tickets_##queue = nullptr, *head_##queue = nullptr, *tail_##queue = nullptr; \
    queue_type *queue = nullptr

#define queue_init(queue, tickets, head, tail, queue_size, dev)                                                                       \
    do                                                                                                                                \
    {                                                                                                                                 \
        CUDA_RUNTIME(cudaMalloc((void **)&queue, queue_size * sizeof(queue_type)));                                                   \
        CUDA_RUNTIME(cudaMalloc((void **)&tickets_##queue, queue_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));  \
        CUDA_RUNTIME(cudaMemset((void *)tickets_##queue, 0, queue_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>))); \
        CUDA_RUNTIME(cudaMalloc((void **)&head_##queue, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                  \
        CUDA_RUNTIME(cudaMemset((void *)head_##queue, 0, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                 \
        CUDA_RUNTIME(cudaMalloc((void **)&tail_##queue, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                  \
        CUDA_RUNTIME(cudaMemset((void *)tail_##queue, 0, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                 \
    } while (0)

#define queue_free(queue, tickets, head, tail)               \
    do                                                       \
    {                                                        \
        if (queue != nullptr)                                \
        {                                                    \
            CUDA_RUNTIME(cudaFree((void *)queue));           \
            CUDA_RUNTIME(cudaFree((void *)tickets_##queue)); \
            CUDA_RUNTIME(cudaFree((void *)head_##queue));    \
            CUDA_RUNTIME(cudaFree((void *)tail_##queue));    \
        }                                                    \
    } while (0)

#define queue_caller(queue, tickets, head, tail) queue, tickets_##queue, head_##queue, tail_##queue

#define queue_callee(queue, tickets, head, tail)                            \
    queue_type *queue,                                                      \
        cuda::atomic<uint32_t, cuda::thread_scope_device> *tickets_##queue, \
        cuda::atomic<uint32_t, cuda::thread_scope_device> *head_##queue,    \
        cuda::atomic<uint32_t, cuda::thread_scope_device> *tail_##queue
