
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>


#define N 50000
#define numThreads 512


__global__ void init(
	unsigned int seed, 
	curandState_t *states,
	unsigned int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		curand_init(
			seed,
			i,
			0,
			&states[i]);
	}
}


__global__ void GeometricBrownianMotion(
	float *d_a,
	float mu,
	float sigma,
	float dt,
	curandState_t *states,
	unsigned int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_a[i] += d_a[i] * ( (dt*mu) + (sigma*sqrt(dt)*curand_normal(&states[i])));
	}
}


int main()
{
	float mu = 0;
	float sigma = 0.2;
	float dt = 1/252.0f;

	float h_a[N];
	float *d_a;

	for (int i = 0; i < N; i++)
	{
		h_a[i] = 100.0f;
	}
	cudaMalloc((void**)&d_a, N*sizeof(float));
	cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);

	curandState_t *states;
	cudaMalloc((void**)&states, N * sizeof(curandState_t));
	init << <(N + numThreads - 1)/numThreads, numThreads >> >(time(NULL), states, N);

	for (int t = 0; t < 252; t++)
	{
		GeometricBrownianMotion << < (N + numThreads - 1) / numThreads, numThreads >> >(
			d_a,
			mu,
			sigma,
			dt,
			states,
			N);
	}

	cudaMemcpy(h_a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost);

	float avg = 0;

	for (int i = 0; i < N; i++)
	{
		//std::cout << h_a[i] << std::endl;
		if ((h_a[i]- 100.0) > 0)
		{
			avg += (h_a[i]-100);
		}
	}

	avg /= float(N);

	std::cout << "The Average Value Is: " << avg << std::endl;

	cudaFree(d_a); cudaFree(states);

    return 0;
}
