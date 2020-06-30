#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "omp.h"

#define MATRIX_SIZE 3
#define RANDOM_SEED 1
#define PRIME 23

int main(int argc, char const *argv[])
{
	struct timeval tv;

	int matrix_a[MATRIX_SIZE][MATRIX_SIZE], matrix_b[MATRIX_SIZE][MATRIX_SIZE];
	int matrix_result[MATRIX_SIZE][MATRIX_SIZE];
	int row, colume;
	int index;
	int temp;

	time_t start, end;

	temp = 0;
	index = 0;
	srand(RANDOM_SEED);

	for (row = 0; row < MATRIX_SIZE; row++) {
		for (colume = 0; colume < MATRIX_SIZE; colume++) {
			matrix_a[row][colume] = rand() % PRIME;
			matrix_b[row][colume] = rand() % PRIME;
		}
	}

	gettimeofday(&tv, NULL);
	start = tv.tv_usec;
	#pragma omp parallel for private(row, index, colume)
	for (row = 0; row < MATRIX_SIZE; row++) {
		// #pragma omp parallel for firstprivate(matrix_a, matrix_b, row) private(index)
		for (colume = 0; colume < MATRIX_SIZE; colume++) {
			matrix_result[row][colume] = 0;
			// temp = matrix_result[row][colume];
			// #pragma omp parallel for firstprivate(matrix_a, matrix_b) reduction(+:temp)
			for (index = 0; index < MATRIX_SIZE; index++) {
				matrix_result[row][colume] += matrix_a[row][index] * matrix_b[index][colume];
				// temp += matrix_a[row][index] * matrix_b[index][colume];
			}
			// matrix_result[row][colume] = temp;
		}
	}
	gettimeofday(&tv, NULL);

	end = tv.tv_usec;
	printf("%ld\n", end - start);
	for (row = 0; row < MATRIX_SIZE; row++) {
		for (colume = 0; colume < MATRIX_SIZE; colume++) {
			printf("%d\n", matrix_result[row][colume]);
		}
	}
	return 0;
}