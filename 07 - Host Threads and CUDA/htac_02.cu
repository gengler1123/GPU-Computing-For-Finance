
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thread>
#include <iostream>


__global__ void addArray(
	float *d_a, float *d_b, float *d_c, int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= size)
	{
		return;
	}
	d_c[i] = d_a[i] + d_b[i];
}


void callKernel(
	int numThreads_,
	int size,
	float *d_a, float *d_b, float *d_c)
{
	addArray <<< (size + numThreads_ - 1)/numThreads_, numThreads_ >>>(
		d_a, d_b, d_c, size);
}


unsigned int numThreads = 512;
unsigned int N = 100000;


int main()
{
    
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;

	cudaMallocHost((void**)&h_a, N*sizeof(float));
	cudaMallocHost((void**)&h_b, N*sizeof(float));
	cudaMallocHost((void**)&h_c, N*sizeof(float));

	cudaMalloc((void**)&d_a, N*sizeof(float));
	cudaMalloc((void**)&d_b, N*sizeof(float));
	cudaMalloc((void**)&d_c, N*sizeof(float));

	for (int i = 0; i < N; i++)
	{
		h_a[i] = float(i);
		h_b[i] = float(i);
	}

	cudaStream_t s1, s2;
	cudaStreamCreate(&s1); cudaStreamCreate(&s2);


	cudaMemcpyAsync(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice, s1);
	cudaMemcpyAsync(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice, s2);

	std::thread t(callKernel, numThreads, N, d_a, d_b, d_c);

	t.join();

	cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 100; i++)
	{
		std::cout << h_c[i] << std::endl;
	}


	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	cudaDeviceReset();

    return 0;
}
