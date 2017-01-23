
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <random>
#include <ctime>


#define numThreads 1024


__global__ void addVectors(
	float *d_A,
	float *d_B,
	float *d_C,
	int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_C[i] = d_A[i] + d_B[i];
	}
}

int main(int argc, char ** argv)
{
	int N = 10000000;
	if (argc == 2)
	{
		N = std::stoi(argv[1]);
	}
	std::cout << "Timing Vector Addition" << std::endl;

	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	float *h_A, *h_B, *h_C1, *h_C2;

	h_A = new float[N];
	h_B = new float[N];
	h_C1 = new float[N];
	h_C2 = new float[N];

	for (int i = 0; i < N; i++)
	{
		h_A[i] = dist(rng);
		h_B[i] = dist(rng);
	}

	std::cout << "Adding Vectors Using C++ on the CPU" << std::endl;

	clock_t begin = clock();

	for (int i = 0; i < N; i++)
	{
		h_C1[i] = h_A[i] + h_B[i];
	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / float(CLOCKS_PER_SEC);

	std::cout << "CPU Version Took " << elapsed_secs << " To Complete." << std::endl;

	std::cout << "Now Timing GPU Version." << std::endl;

	float *d_A, *d_B, *d_C;

	clock_t beginGPU = clock();

	cudaMalloc((void**)&d_A, N*sizeof(float));
	cudaMalloc((void**)&d_B, N*sizeof(float));
	cudaMalloc((void**)&d_C, N*sizeof(float));

	cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);

	clock_t beginGPUNoOver = clock();

	addVectors << <(N + numThreads - 1) / numThreads, numThreads >> >(
		d_A,
		d_B,
		d_C,
		N);

	cudaDeviceSynchronize();

	clock_t endGPUNoOver = clock();

	cudaMemcpy(h_C2, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	clock_t endGPU = clock();

	double elapsed_secs_GPU = double(endGPU - beginGPU) / float(CLOCKS_PER_SEC);
	double elapsed_secs_NoOver = double(endGPUNoOver - beginGPUNoOver) / float(CLOCKS_PER_SEC);

	std::cout << "Elapsed Times for:" << std::endl;
	std::cout << "No Overhead Involved " << elapsed_secs_NoOver << std::endl;
	std::cout << "Overhead Calculated " << elapsed_secs_GPU << std::endl;

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	delete[] h_A; delete[] h_B; delete[] h_C1; delete[] h_C2;

    return 0;
}

