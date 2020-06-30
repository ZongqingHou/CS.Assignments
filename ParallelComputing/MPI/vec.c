
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"


#define N (2 << 4)
#define PRINT_VECS 1
#define MAX_RAND 100

void init_vec(int *vec, int len);
void print_vec(const char *label, int *vec, int len);
int sum_chunk(int *vec, int low_index, int high_index);


void init_vec(int *vec, int len)
{
    int i;
    for (i = 0; i < len; i++)
    {
        vec[i] = rand() % MAX_RAND;
    }
}

void print_vec(const char *label, int *vec, int len)
{
#if PRINT_VECS
    printf("%s", label);

    int i;
    for (i = 0; i < len; i++)
    {
        printf("%d ", vec[i]);
    }
    printf("\n\n");
#endif
}

int sum_chunk(int *vec, int low_index, int high_index)
{
    int i;
    int my_sum = 0;
    for (i = low_index; i < high_index; i++)
    {
        my_sum += vec[i];
    }

    return my_sum;
}

int main(int argc, char *argv[])
{
    int my_rank;
    int num_procs;
    int total;

    total = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int vec[N];
    int recvbuf[N];
    double start_time;
    double stop_time;

    int chunk_size = N / num_procs;

    if (!my_rank)
    {
        printf("Number of processes: %d\n", num_procs);
        printf("N: %d\n", N);
        printf("Chunk size: %d\n", chunk_size);
        srand(time(NULL));
        init_vec(vec, N);
        print_vec("Initial vector:\n", vec, N);
        start_time = MPI_Wtime();
    }

    MPI_Scatter(vec, chunk_size, MPI_INT, recvbuf, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    int my_sum = sum_chunk(recvbuf, 0, chunk_size);
    printf("Result from process %d: %d\n", my_rank, my_sum);

    MPI_Reduce(&my_sum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!my_rank)
    {
        stop_time = MPI_Wtime();
        printf("Final result from process %d: %d\n", my_rank, total);
        printf("Total time (sec): %f\n", stop_time - start_time);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;;
}