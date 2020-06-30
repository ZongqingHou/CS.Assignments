//-----------------------------------------
// NAME: Zongqing Hou 
// STUDENT NUMBER: 7729727
// COURSE: COMP 4510, SECTION: A01
// INSTRUCTOR: Parimala Thulasiraman
// ASSIGNMENT: assignment #1, QUESTION: question #2
// 
//-----------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
#include "qdbmp.h"
#include "julia.h"

void toRGB(unsigned int val, unsigned char *r, unsigned char *g, unsigned char *b)
{
	*r = (val & 0xff);
	val >>= 8;
	*b = (val & 0xff);
	val >>= 8;
	*g = (val & 0xff);
}

unsigned int sum_array(unsigned int *array, int len)
{
	unsigned int total = 0;
	for (int i = 0; i < len; i++)
	{
		total += array[i];
	}

	return total;
}

void hist_eq(unsigned int *data, unsigned int *hist)
{
	unsigned int total = sum_array(hist, MAX_ITER);
	unsigned int val;

	float cache[MAX_ITER];
	float hue = 0.0;
	for (unsigned int i = 0; i < MAX_ITER; i++)
	{
		cache[i] = hue;
		hue += (float) hist[i] / total;
	}


	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			val = data[y * WIDTH + x];

			if (val < MAX_ITER)
			{
				hue = cache[val];
			}
			else
			{
				hue = cache[MAX_ITER - 1];
				for (unsigned int i = MAX_ITER; i < val; i++)
				{
					hue += (float) hist[i] / total;
				}
			}

			data[y * WIDTH + x] = (unsigned int) (hue * MAX_COLOUR);
		}
	}
}

void write_bmp(unsigned int *data, char *fname)
{
	BMP *bmp = BMP_Create((UINT) WIDTH, (UINT) HEIGHT, (USHORT) DEPTH);
	unsigned char r, g, b;
	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			toRGB(data[y * WIDTH + x], &r, &g, &b);
			BMP_SetPixelRGB(bmp, (UINT) x, (UINT) y,
			                (UCHAR) r, (UCHAR) g, (UCHAR) b);
		}
	}

	BMP_WriteFile(bmp, FNAME);
	BMP_Free(bmp);
}

unsigned int julia_iters(float complex z)
{
	unsigned int iter = 0;
	while (fabsf(cimag(z)) < LIMIT && iter < MAX_ITER)
	{
		z = C * csin(z);
		iter++;
	}

	return iter;
}

void compute_row(int my_rank, int row, unsigned int *data, unsigned int *hist)
{
	float complex z;
	float series_row;
	float series_col;
	unsigned int iters;

	for (int col = 0; col < WIDTH; col++)
	{
		series_row = row - HEIGHT / 2;
		series_col = col - WIDTH / 2;
		z = series_col / RES_FACTOR + (I / RES_FACTOR) * series_row;
		z *= SCALE;
		iters = julia_iters(z);
		data[(row - (my_rank - 1) * N)* WIDTH + col] = iters;
		hist[iters]++;
	}
}

int main(int argc, char *argv[])
{
	int my_rank;
	int processes;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);

	int chunk_size;

	unsigned int hist_buf[MAX_ITER] = {0};

	if (HEIGHT % (processes - 1 ) == 0) {
		N = HEIGHT / (processes - 1);
		chunk_size = N * WIDTH;
	} else {
		printf("Error\n");
		MPI_Finalize();
		exit(1);
	}

	if (!my_rank) {
		// supervisor process

		MPI_Status status;

		printf("Beginning julia set computation...\n");

		for (int i = 1; i < processes; i++) {
			// wait the datas from workers
			MPI_Recv(&data[chunk_size*(i-1)], chunk_size, MPI_UNSIGNED, i, i, MPI_COMM_WORLD, &status);
		}

	} else {
		// worker processes

		unsigned int data_buf[chunk_size];

		for (int row = (my_rank - 1) * N; row < my_rank * N; row++) {
			// excution
			compute_row(my_rank, row, data_buf, hist_buf);
		}

		// send datas to supervisor process
		MPI_Send(data_buf, chunk_size, MPI_UNSIGNED, 0, my_rank, MPI_COMM_WORLD);
	}

	// add values of hist array into supervisor process
	MPI_Reduce(hist_buf, hist, MAX_ITER, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

	if (!my_rank) {
		// drawing

		hist_eq(data, hist);
		write_bmp(data, FNAME);
		printf("Done!\n");
	}
	MPI_Finalize();

	return EXIT_SUCCESS;
}