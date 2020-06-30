#include "utils.h"
#include <stdio.h>
#include <time.h>
#include "main.h"

// Reads the value of i from the command line array and returns n = 2^i
int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./reduce <i, where n = 2^i>\n");
        exit(1);
    }
    
    return atoi(argv[1]) * 512;
}

void init_vec(float *vec, int len)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand(time(NULL));
        seeded = 1;
    }
    
    int i;
    for (i = 0; i < len; i++)
    {
        *(vec + i) = (float) rand() / RAND_MAX;
    }    
}

void print_vec(const char *label, float *vec, int len)
{
#if PRINT_VECS
    printf("%s", label);
    
    int i;
    for (i = 0; i < len; i++)
    {
        printf("%f ", vec[i]);
    }
    printf("\n\n");
#endif
}

void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); // bail out immediately (makes debugging easier)
    }
}

int get_max_block_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    status = cudaGetDevice(&dev_num);
    check_error(status, "Error querying device number.");

    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
    check_error(status, "Error querying max block threads.");

    return max_threads;
}