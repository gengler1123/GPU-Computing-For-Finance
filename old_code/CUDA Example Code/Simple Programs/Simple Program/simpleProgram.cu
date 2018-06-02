
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define N 256

__global__ void addOne(
	float *d_a)
{
	int tid = threadIdx.x;

	d_a[tid] += 1;
}

int main()
{
	float *h_a;
	float *d_a;

	h_a = new float[N];

	cudaMalloc((void**)&d_a, N * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
	}

	std::cout << h_a[0] << std::endl;

	cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);

	addOne <<<1, N >>>(d_a);

	cudaMemcpy(h_a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << h_a[0] << std::endl;

	delete[] h_a;
	cudaFree(d_a);

	return 0;
}