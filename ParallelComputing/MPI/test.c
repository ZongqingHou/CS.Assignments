#include <stdio.h>
#include <mpi.h>

int main(int argc, char const *argv[])
{
	int my_rank;
    int processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
	int ii[] = {1,2,3};
	int buf[3] = {0};
	MPI_Reduce(ii, buf, 3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}