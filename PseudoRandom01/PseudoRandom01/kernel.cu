
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>

#include <curand.h>
#include <curand_kernel.h>


#define N 100
#define numThreads 512


__global__ void initialize(
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


__global__ void uniDist(
	float *d_a,
	curandState_t *states,
	unsigned int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_a[i] = curand_uniform(&states[i]);
	}
}


__global__ void normalDist(
	float *d_a,
	curandState_t *states,
	unsigned int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_a[i] = curand_normal(&states[i]);
		d_a[i] = curand_
	}
}


int main()
{
	curandState_t *states;
	cudaMalloc((void**)&states, N*sizeof(curandState_t));

	initialize<< <1, numThreads >> >(time(NULL), states, N);

	float h_a[N], h_b[N];
	float *d_a, *d_b;

	cudaMalloc((void**)&d_a, N*sizeof(float));
	cudaMalloc((void**)&d_b, N*sizeof(float));

	uniDist<<<1,numThreads>>>(
		d_a,
		states,
		N);

	normalDist << <1, numThreads >> >(
		d_b,
		states,
		N);

	cudaMemcpy(h_a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::setw(10);
	std::cout << std::setprecision(4);
	for (int i = 0; i < N; i++)
	{
		std::cout << h_a[i] << " , " << h_b[i] << std::endl;
	}

	cudaFree(d_a); cudaFree(d_b); cudaFree(states);

    return 0;
}
