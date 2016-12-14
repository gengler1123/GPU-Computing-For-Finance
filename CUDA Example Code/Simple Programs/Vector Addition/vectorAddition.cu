
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define N 8000
#define numThreads 512

__global__ void addVectors(
	int size,
	float *d_a,
	float *d_b,
	float *d_c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < size)
	{ 	
		d_c[tid] = d_a[tid] + d_b[tid];
	}
}

int main()
{
	float *h_a,*h_b,*h_c;
	float *d_a, *d_b, *d_c;

	h_a = new float[N];
	h_b = new float[N];
	h_c = new float[N];

	cudaMalloc((void**)&d_a, N * sizeof(float));
	cudaMalloc((void**)&d_b, N * sizeof(float));
	cudaMalloc((void**)&d_c, N * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
		h_b[i] = i+1;
	}


	cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

	addVectors <<<(N + numThreads - 1)/numThreads, numThreads >>>(
		N,
		d_a,
		d_b,
		d_c);

	cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << h_c[0] << std::endl;

	delete[] h_a; delete[] h_b; delete[] h_c;
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}