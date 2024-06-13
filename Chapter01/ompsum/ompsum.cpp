#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cxtimers.h"

inline float sinsum(float x, int terms) // sin(x) = x - x^3/3! + x^5/5! ...
{
	float term = x;   // first term of series
	float sum = term; // sum of terms so far
	float x2 = x * x;
	for (int n = 1; n < terms; n++) {
		term *= -x2 / (float)(2 * n * (2 * n + 1));
		sum += term;
	}
	return sum;
}
int main(int argc, char *argv[])
{
	int steps = (argc > 1) ? atoi(argv[1]) : 1000000;
	int terms = (argc > 2) ? atoi(argv[2]) : 1000;
	int threads = (argc > 3) ? atoi(argv[3]) : 8;
	double pi = 3.14159265358979323;
	double step_size = pi / (steps - 1); // n-1 steps

	cx::timer tim;
	double omp_sum = 0.0;

	omp_set_num_threads(threads);  // OpenMP
#pragma omp parallel for reduction (+:omp_sum) // OpenMP
	for (int step = 0; step < steps; step++) {
		float x = step_size * step;
		omp_sum += sinsum(x, terms);   // sum of Taylor series
	}
	double cpu_time = tim.lap_ms();    // elapsed time
	// Trapezoidal Rule correction
	omp_sum -= 0.5 * (sinsum(0.0, terms) + sinsum(pi, terms));
	omp_sum *= step_size;
	printf("omp sum = %.10f, steps %d terms %d time %.3f ms\n",
		omp_sum, steps, terms, cpu_time);
	return 0;
}