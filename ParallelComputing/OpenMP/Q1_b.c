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

	int index;
	int innerindex;
	int length;
	int sublength;
	int pro_number;
	double start_time;
	double stop_time;

	length = atoi(argv[1]);
	sublength = length / size;

	int matrix_a[length][length];
	int matrix_b[length];
	int submatrix_a[length][sublength];
	int submatrix_b[sublength];
	int submatrix_b_buf[sublength];
	int result_value[sublength];
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

	MPI_Scatter(matrix_a, length * sublength, MPI_INT, submatrix_a, length * sublength, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(matrix_b, sublength, MPI_INT, submatrix_b, sublength, MPI_INT, 0, MPI_COMM_WORLD);

	for (pro_number = 0; pro_number < size; pro_number++) {
		if (myrank == index) {
			for (index = 0; index < sublength; index++) {
				submatrix_b_buf[index] = submatrix_b[index];
			}
		}
		
		MPI_Bcast(submatrix_b_buf, sublength, MPI_INT, pro_number, MPI_COMM_WORLD);

		for (index = 0; index < sublength; index++) {
			for (innerindex = 0; innerindex < sublength; innerindex++) {
				result_value[index] += submatrix_a[index][innerindex] * matrix_b[innerindex];
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Gather(result_value, sublength, MPI_INT, matrix_result, sublength, MPI_INT, 0, MPI_COMM_WORLD);

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