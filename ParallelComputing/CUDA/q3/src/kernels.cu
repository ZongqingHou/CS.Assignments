#include "kernels.h"

__global__ void multiply(float *a, float *b, int n)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    a[global_id] = a[global_id] * b[global_id % (n * blockDim.x)];
}

__global__ void reduce(float *a, float *c, int n)
{
    unsigned int block_size = blockDim.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    unsigned int chunk_size = block_size * 2;

    unsigned int block_start = block_id * chunk_size;

    unsigned int left;  // holds index of left operand
    unsigned int right; // holds index or right operand
    unsigned int threads = block_size; // number of active threads (on current iteration)
    for (unsigned int stride = 1; stride < chunk_size; stride *= 2, threads /= 2)
    {
        left = block_start + thread_id * (stride * 2);
        right = left + stride;

        if (thread_id < threads && right < n * (block_id + 1)) 
        {
            a[left] += a[right];
        }

        __syncthreads();
    }

    if (!thread_id)
    {
        c[block_id] = a[block_start];
    }
}
