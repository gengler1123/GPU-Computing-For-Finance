
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


#include <iostream>
#include <cmath>



#define mu 0.05f
#define sigma .2f
#define timespan 252.0f


#define TRIALS 10000
#define numThreads 512


__global__ void europeanOption(
	int size, int iterations,
	float *d_price, float initialPrice, float strikePrice,
	curandState_t *d_state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < size)
	{

		for (int i = 0; i < iterations; i++)
		{
			initialPrice *= 1 + mu / timespan + curand_normal(&d_state[tid])*sigma/sqrt(timespan);
		}

		d_price[tid] = initialPrice - strikePrice;
		if (d_price[tid] < 0)
		{
			d_price[tid] = 0;
		}
	}

}

__global__ void init(
	unsigned int seed,
	curandState_t *d_state)
{
	curand_init(
		seed,
		threadIdx.x + blockDim.x * blockIdx.x,
		0,
		&d_state[threadIdx.x + blockDim.x * blockIdx.x]);
}


int main()
{

	float *h_prices, *d_prices;

	h_prices = new float[TRIALS];
	cudaMalloc((void**)&d_prices, TRIALS*sizeof(float));

	curandState_t *d_state;
	cudaMalloc((void**)&d_state, TRIALS * sizeof(curandState_t));

	init <<< (TRIALS - numThreads - 1)/numThreads, numThreads >>>(time(0), d_state);

	europeanOption << <(TRIALS - numThreads - 1) / numThreads, numThreads >> >(
		TRIALS, 252,
		d_prices, 100.0f, 100.0f,
		d_state);

	cudaMemcpy(h_prices, d_prices, TRIALS*sizeof(float), cudaMemcpyDeviceToHost);

	float price = 0;

	int count = 0;

	for (int i = 0; i < TRIALS; i++)
	{
		price += h_prices[i];
		if (h_prices[i] != 0)
		{
			count += 1;
		}
	}

	price /= TRIALS;

	std::cout << "The Theoretical Price of the Option is " << price << "." << std::endl;

	std::cout << "Count is " << count << "." << std::endl;

	delete[] h_prices;
	cudaFree(d_state); cudaFree(d_prices);

	cudaDeviceReset();

	return 0;
}