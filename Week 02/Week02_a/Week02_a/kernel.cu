
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>

#define numThreads 512

__global__ void addVectors(
	int size,
	float *d_a,
	float *d_b,
	float *d_c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= size)
	{
		return;
	}
	d_c[i] = d_a[i] + d_b[i];
}


int main(int argc, char** argv)
{
	/* 
	Initialize and define an integer which will
	store the size of the vectors that we use.
	*/

	int size = 1000;

	/*
	Check for arguments into the main function,
	if they exist, the second one should denote
	the size of the vector.
	*/
	if (argc >= 2)
	{
		size = int(argv[1]);
	}

	/*
	Initialize three vectors with size 'size'
	*/
	std::vector<float> h_a(size), h_b(size), h_c(size);

	/*
	Initialize pseudo-random number generator
	and uniform distribution.
	*/

	std::mt19937 gen(time(NULL));
	std::uniform_real_distribution<float> dist(0.0, 1.0);


	/*
	Fill random numbers in the vectors.
	*/

	for (int i = 0; i < size; i++)
	{
		h_a[i] = dist(gen);
		h_b[i] = dist(gen);
	}
	

	/*
	Intialize float pointers for the device pointers.
	*/

	float *d_a, *d_b, *d_c;


	/*
	Allocate space on the GPU (device) for our arrays.
	*/

	cudaMalloc((void**)&d_a, size*sizeof(float));
	cudaMalloc((void**)&d_b, size*sizeof(float));
	cudaMalloc((void**)&d_c, size*sizeof(float));
	

	/*
	Copy data from the host vectors using the .data() member function
	of vectors to access the underlaying array.
	*/

	cudaMemcpy(d_a, h_a.data(), size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), size*sizeof(float), cudaMemcpyHostToDevice);


	/*
	Call kernel to handle vector addition.
	*/

	addVectors <<<(size + numThreads - 1) / numThreads, numThreads >>>(
		size,
		d_a,
		d_b,
		d_c);


	/*
	Copy data back from device to host.
	*/

	cudaMemcpy(h_c.data(), d_c, size*sizeof(float), cudaMemcpyDeviceToHost);

	/*
	Print out results.
	*/

	for (int i = 0; i < size; i++)
	{
		std::cout << h_c[i] << std::endl;
	}


	/*
	Free up allocated space on the device
	*/

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);


	cudaDeviceReset();

    return 0;
}