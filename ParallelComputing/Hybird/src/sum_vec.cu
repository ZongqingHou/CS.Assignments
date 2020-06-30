#include "sum_vec.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int get_max_block_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    status = cudaGetDevice(&dev_num);

    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);

    return max_threads;
}

void add(float* buffer_a, float* buffer_b, float* buffer_c, int n)
{
    cudaError_t statusA; // records status of operations on A vec
    cudaError_t statusB; // for B vec
    cudaError_t statusC; // for C vec

    size_t size = n * sizeof(float); //size in bytes

    float *dev_a;
    float *dev_b;
    float *dev_c;
    statusA = cudaMalloc(&dev_a, size);
    statusB = cudaMalloc(&dev_b, size);
    statusC = cudaMalloc(&dev_c, size);

    statusA = cudaMemcpy(dev_a, buffer_a, size, cudaMemcpyHostToDevice);
    statusB = cudaMemcpy(dev_b, buffer_b, size, cudaMemcpyHostToDevice);

    int block_size = get_max_block_threads();
    int blocks = n / block_size + (n % block_size > 0 ? 1 : 0);

    vec_add<<<blocks, block_size>>>(dev_a, dev_b, dev_c, n);
    
    statusC = cudaMemcpy(buffer_c, dev_c, size, cudaMemcpyDeviceToHost);
}
