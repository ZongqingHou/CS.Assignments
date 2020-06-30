#include <stdio.h>
#include <stdlib.h>

#include "omp.h"

#define MATRIX_SIZE 3
#define RANDOM_SEED 1
#define PRIME 23

int main(int argc, char const *argv[])
{
	int matrix_a[MATRIX_SIZE][MATRIX_SIZE], matrix_b[MATRIX_SIZE][MATRIX_SIZE];
	int matrix_result[MATRIX_SIZE][MATRIX_SIZE];
	int row, colume;
	int index;

	index = 0;
	srand(RANDOM_SEED);

	for (row = 0; row < MATRIX_SIZE; row++) {
		for (colume = 0; colume < MATRIX_SIZE; colume++) {
			matrix_a[row][colume] = rand() % PRIME;
			matrix_b[row][colume] = rand() % PRIME;
		}
	}

	#pragma omp parallel for private(row, colume, index) collapse(2)
	for (row = 0; row < MATRIX_SIZE; row++) {
		for (colume = 0; colume < MATRIX_SIZE; colume++) {
			matrix_result[row][colume] = 0;
			for (index = 0; index < MATRIX_SIZE; index++) {
				matrix_result[row][colume] += matrix_a[row][index] * matrix_b[index][colume];
			}
		}
	}

	for (row = 0; row < MATRIX_SIZE; row++) {
		for (colume = 0; colume < MATRIX_SIZE; colume++) {
			printf("%d\n", matrix_result[row][colume]);
		}
	}
	return 0;
}