
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

const int N = 1500;

__global__ void kernel(float *x, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		x[i] = sqrt(pow(3.14159, i));
	}
}

int main()
{
	const int num_streams = 8;

	cudaStream_t streams[num_streams];
	float *data[num_streams];

	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);

		cudaMalloc((void**)&data[i], N * sizeof(float));

		// launch one worker kernel per stream
		kernel << <1, 64, 0, streams[i] >> >(data[i], N);

		// launch a dummy kernel on the default stream
		kernel << <1, 1 >> >(0, 0);
	}


	for (int i = 0; i < num_streams; i++)
	{
		cudaFree(data[i]);
	}

	cudaDeviceReset();

	return 0;
}