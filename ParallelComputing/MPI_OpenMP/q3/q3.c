#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "omp.h"

#define N 2048

double f (int i) {
	double x;
	x = (double) i / (double) N;
	return 4.0 / (1.0 + x * x);
}

int main (int argc, char const *argv[]) {
	double start, end;
	double area;
	int i;
	area = f(0) - f(N);

	omp_set_num_threads(16);

	start = clock();

	#pragma omp parallel 
	{
		#pragma omp for reduction(+:area) private(i)  schedule(guided,32) 
		for (i = 1; i <=  N / 2; i++) {
			area += 4.0 * f(2 * i - 1) + 2 * f(2 * i);
		}
	}
	end = clock();

	printf("%f\n", area);
	printf("time : %f\n", (end - start));
	area /= (3.0 * N);
	printf("Approximation of pi: % 13.11f\n", area);

	return EXIT_SUCCESS;
}