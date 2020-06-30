#ifndef _KERNEL_H
#define _KERNEL_H

__global__ void multiply(float *a, float *b, int n);
__global__ void reduce(float *a, float *c, int n);

#endif
