/* Comp 4510 - CUDA Vector Addition Example
 * This code uses a simple kernel to add two vectors of size n.
 * The result is transfered back to the host and (optionally) printed out,
 * along with some stats.
 *
 * Notes:
 * - This program takes 1 command line arg: an integer i, which is used to compute n = 2^i.
 * - If you'd like to see the vectors (and the result), just set the PRINT_VECS constant in vec_add.h to 1.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "main.h"
#include "kernels.h"
#include "timer.h"
#include "utils.h"

void print_results(
    int n,
    int block_size,
    int blocks,
    Timer *d2h_timer,
    Timer *h2d_timer,
    Timer *kernel_timer,
    Timer *total_timer
    )
{
    printf("-------\n");
    printf("Results\n");
    printf("-------\n");
    
    // Print the resulting C vector and the timing stats
    printf("Data size: %d\n", n * n);
    printf("blocks_size: %d\n", block_size);
    printf("blocks: %d\n", blocks);
    printf("\n");

    // note: these times are in milliseconds
    float transfer_time = get_time(d2h_timer) + get_time(h2d_timer);
    float kernel_time = get_time(kernel_timer);
    float total_time = get_time(total_timer);
    printf("%-*s: %f ms\n", RESULTS_WIDTH, "Total time", total_time);
    printf("%-*s: %f ms\n", RESULTS_WIDTH, "Data transfer time", transfer_time);
    printf("%-*s: %f ms\n", RESULTS_WIDTH, "Kernel time", kernel_time);
}

int main(int argc, char *argv[])
{   
    int index;
    const int n = 16;
    
    cudaError_t statusA; // records status of operations on A vec
    cudaError_t statusB; // for B vec
    cudaError_t statusC; // for C vec

    Timer d2h_timer = create_timer();
    Timer h2d_timer = create_timer();
    Timer kernel_timer = create_timer();
    Timer total_timer = create_timer();
    
    size_t size = n * sizeof(float); //size in bytes
    float *host_a = (float *) malloc(size * n);
    float *host_b = (float *) malloc(size);
    float *host_c = (float *) malloc(size);

    for(index = 0; index < n; index ++ ){
        init_vec(&host_a[index * n], n);
    }

    init_vec(host_b, n);
    print_vec("A vector:\n", host_a, n * n);
    print_vec("B vector:\n", host_b, n);

    float *dev_a;
    float *dev_b;
    float *dev_c;
    statusA = cudaMalloc(&dev_a, size * n);
    check_error(statusA, "Error allocating dev buffer A.");
    statusB = cudaMalloc(&dev_b, size);
    check_error(statusB, "Error allocating dev buffer B.");
    statusC = cudaMalloc(&dev_c, size);
    check_error(statusC, "Error allocating dev buffer C.");

    start_timer(&total_timer);
    
    start_timer(&d2h_timer);
    statusA = cudaMemcpy(dev_a, host_a, size * n, cudaMemcpyHostToDevice);
    statusB = cudaMemcpy(dev_b, host_b, size, cudaMemcpyHostToDevice);
    stop_timer(&d2h_timer);
    check_error(statusA, "Error on CPU->GPU cudaMemcpy for A.");
    check_error(statusB, "Error on CPU->GPU cudaMemcpy for B.");

    start_timer(&kernel_timer);
    multiply<<<1, n * n>>>(dev_a, dev_b);
        statusC = cudaMemcpy(host_a, dev_a, size * n, cudaMemcpyDeviceToHost);

    reduce<<<n, n>>>(dev_a, dev_c, n);
    stop_timer(&kernel_timer);
    check_error( cudaGetLastError(), "Error in kernel.");
    
    start_timer(&h2d_timer);
    statusC = cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);
    stop_timer(&h2d_timer);
    check_error(statusC, "Error on GPU->CPU cudaMemcpy for C.");

    stop_timer(&total_timer);
        
    cudaEventSynchronize(total_timer.stop);

    print_vec("C vector:\n", host_c, n);
    print_results(
        n,
        n,
        n,
        &d2h_timer,
        &h2d_timer,
        &kernel_timer,
        &total_timer
        );

    free(host_a);
    free(host_b);
    free(host_c);
    destroy_timer(&d2h_timer);
    destroy_timer(&h2d_timer);
    destroy_timer(&kernel_timer);
    destroy_timer(&total_timer);

    statusA = cudaFree(dev_a);
    statusB = cudaFree(dev_b);
    statusC = cudaFree(dev_c);
    check_error(statusA, "Error calling cudaFree on dev_a buffer" );
    check_error(statusB, "Error calling cudaFree on dev_b buffer" );
    check_error(statusC, "Error calling cudaFree on dev_c buffer" );
    
    return EXIT_SUCCESS;
}
