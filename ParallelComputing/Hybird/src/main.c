// Note: this program requires one command line arg: an integer i.
// The vectors are then created to be of size 2^i.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "sum_vec.h"

#define PRINT_VECS 0
#define MAX_RAND 10

int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./reduce <i, where n = 2^i>\n");
        exit(1);
    }

    return (int) pow(2, atoi(argv[1]));
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

void print_hostname()
{
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(proc_name, &len);
    printf("Using system: %s\n", proc_name);
}

int main(int argc, char *argv[])
{
    int my_rank;
    int num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = parse_args(argc, argv);
    int i;
    print_hostname();

    float host_a[n];
    float host_b[n];
    float host_c[n];

    float buffer_a[n / 2];
    float buffer_b[n / 2];
    float buffer_c[n / 2];

    if (!my_rank)
    {

        init_vec(host_a, n);
        init_vec(host_b, n);
        printf("A vector\n");
        for (i = 0; i < n; i++)
        {
            printf("%f ", host_a[i]);
        }
        printf("\n\n");
        printf("B vector\n");
        for (i = 0; i < n; i++)
        {
            printf("%f ", host_b[i]);
        }
        printf("\n\n");
    }

    MPI_Scatter(
        host_a,
        n / 2,
        MPI_FLOAT,
        buffer_a,
        n / 2,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatter(
        host_b,
        n / 2,
        MPI_FLOAT,
        buffer_b,
        n / 2,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    add(buffer_a, buffer_b, buffer_c, n / 2);

    MPI_Gather(buffer_c, n / 2, MPI_FLOAT,
               host_c, n / 2, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (!my_rank)
    {
        printf("C vector\n");
        for (i = 0; i < n; i++)
        {
            printf("%f ", host_c[i]);
        }
        printf("\n\n");
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
