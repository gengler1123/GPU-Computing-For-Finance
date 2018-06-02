// w01_01.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <random>
#include <ctime>


#define numThreads 1024


__global__ void addVectors(
	float *d_a,
	float *d_b,
	float *d_c,
	int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_c[i] = d_a[i] + d_b[i];
	}
}

int main()
{
	
	int N = 10000;

	std::mt19937 rng;  // Create Random Number Generator
	rng.seed(std::random_device()()); // Set RNG seed value;
	std::uniform_real_distribution<float> dist(0.0, 1.0);  // Create uniform distribution

	float *h_a, *h_b, *h_c; // Creating host pointers
	h_a = new float[N];
	h_b = new float[N];
	h_c = new float[N];

	for (int i = 0; i < N; i++)
	{
		h_a[i] = dist(rng);
		h_b[i] = dist(rng);
	}

	float *d_a, *d_b, *d_c; // Creating Device Pointers

	// Allocating space on GPU for arrays
	cudaMalloc((void**)&d_a, N*sizeof(float));
	cudaMalloc((void**)&d_b, N*sizeof(float));
	cudaMalloc((void**)&d_c, N*sizeof(float));

	// Copying Data To Device Pointers From Host Pointers
	cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);

	addVectors<<<(N + numThreads - 1)/numThreads, numThreads>>(
		d_a,
		d_b,
		d_c,
		N);

	cudaDeviceSynchronize();

	cudaMemcpy(h_C2, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete[] h_a; delete[] h_b; delete[] h_c;

	cudaDeviceReset();

	return 0;
}