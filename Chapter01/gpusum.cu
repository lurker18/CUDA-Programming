#include "cxtimers.h"
#include "cx.h"

__host__ __device__ inline float sinsum(float x, int terms)
{
	float x2 = x * x;
	float term = x;    // first term of series
	float sum = term;  // sum of terms so far
	for (int n = 1; n < terms; n++) {
		term *= -x2 / (2 * n * (2 * n + 1));  // build factorial
		sum += term;
	}
	return sum;
}

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size)
{
	// unique thread ID
	int step = blockIdx.x * blockDim.x + threadIdx.x;
	if (step < steps) {
		float x = step_size * step;
		sums[step] = sinsum(x, terms);  // store sums
	}
}

int main(int argc, char* argv[])
{
	// get command line arguments
	int steps = (argc > 1) ? atoi(argv[1]) : 1000000;
	int terms = (argc > 2) ? atoi(argv[2]) : 1000;
	int threads = 256;
	int blocks = (steps + threads - 1) / threads;   // round up

	double pi = 3.14159265358979323;
	double step_size = pi / (steps - 1); // NB n-1
	// allocate GPU buffer and get pointer
	thrust::device_vector<float> dsums(steps);    // GPU buffer
	float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer
	cx::timer tim;
	gpu_sin<<<blocks, threads>>>(dptr, steps, terms, (float)step_size);
	double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());
	double gpu_time = tim.lap_ms(); // get elapsed time
	// Trapezoidal Rule correction
	gpu_sum -= 0.5 * (sinsum(0.0, terms) + sinsum(pi, terms));
	gpu_sum *= step_size;
	printf("gpusum = %.10f, steps %d terms %d time %.3f ms\n",
		gpu_sum, steps, terms, gpu_time);
	return 0;
}