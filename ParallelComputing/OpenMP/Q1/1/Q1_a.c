#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char const *argv[])
{
	int myrank;
	int size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int result_value;
	int index;
	int innerindex;
	int length;
	double start_time;
	double stop_time;

	result_value = 0;
	length = size;

	int matrix_a[length][length];
	int submatrix_a[length];
	int matrix_b[length];
	int matrix_result[length];

	if (!myrank) {
		srand(1);
		for (index = 0; index < length; index++) {
			for (innerindex = 0; innerindex < length; innerindex++) {
				matrix_a[index][innerindex] = rand() % 23;
			}
		}

		for (index = 0; index < length; index++) {
			matrix_b[index] = rand() % 23;
		}

		start_time = MPI_Wtime();
	}

	MPI_Scatter(matrix_a, length, MPI_INT, submatrix_a, length, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(matrix_b, length, MPI_INT, 0, MPI_COMM_WORLD);

	for (index = 0; index < length; index++) {
		result_value += submatrix_a[index] * matrix_b[index];
	}

	MPI_Gather(&result_value, 1, MPI_INT, matrix_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (!myrank) {
		printf("%d  %d\n", length, size);

		stop_time = MPI_Wtime();
		printf("%f\n", stop_time - start_time);

		for (index = 0; index < length; index++) {
			printf("%d\n", matrix_result[index]);
		}
	}

	MPI_Finalize();

	return 0;
}