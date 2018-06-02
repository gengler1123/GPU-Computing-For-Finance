
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#define N 128

__global__ void initilize(
	unsigned int seed,
	curandState_t *states)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(
		seed,
		tid,
		0,
		&states[tid]);
}


__global__ void uniformRandom(
	curandState_t *states,
	float *d_values)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	d_values[tid] = curand_uniform(&states[tid]);
}


int main()
{

	curandState_t *d_states;
	cudaMalloc((void**)&d_states, N * sizeof(curandState_t));

	initilize << <1, N >> >(time(0), d_states);

	float *h_values;
	float *d_values;

	h_values = new float[N];
	
	cudaMalloc((void**)&d_values, N * sizeof(float));

	uniformRandom << <1, N >> >(
		d_states,
		d_values);

	cudaMemcpy(h_values, d_values, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		std::cout << h_values[i] << std::endl;
	}

	delete[] h_values;
	cudaFree(d_states);
	cudaFree(d_values);

	return 0;
}