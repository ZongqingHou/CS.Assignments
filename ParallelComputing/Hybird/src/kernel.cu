#include "kernel.h"

__global__ void vec_add(float *a, float *b, float *c, int n)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n)
    {
        c[global_id] = a[global_id] + b[global_id];
    }
}

