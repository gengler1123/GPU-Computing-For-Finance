
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

#define imin(a,b) (a < b?a:b)

const int N = 1024;
const int numThreads = 256;
const int numBlocks = imin(32, (N + numThreads - 1) / numThreads);

__global__ void sumVector(float *d_a, float *d_b)
{
	__shared__ float cache[numThreads];
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int cidx = threadIdx.x;

	float temp = 0;

	cache[cidx] = 0;

	while (tidx < N)
	{
		cache[cidx] += d_a[tidx];
		tidx += blockDim.x * gridDim.x;
	}

	//cache[cidx] = temp;

	__syncthreads();


	int i = blockDim.x / 2;


	while (i != 0)
	{
		if (cidx < i)
		{
			cache[cidx] += cache[cidx + i];
		}
		__syncthreads();
		i /= 2;
	}
	
	if (cidx == 0)
	{
		d_b[blockIdx.x] = cache[0];
	}

}

int main()
{
    
	float *h_a = new float[N];
	float *d_a;

	float *h_b = new float[numBlocks];
	float *d_b;

	float S = 0;
	for (int i = 0; i < N; i++)
	{
		h_a[i] = i + 1;
		S += i + 1;
	}

	for (int i = 0; i < numBlocks; i++)
	{
		h_b[i] = 0;
	}


	cudaMalloc((void**)&d_a, N*sizeof(float));
	cudaMalloc((void**)&d_b, numBlocks*sizeof(float));

	cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, numBlocks*sizeof(float), cudaMemcpyHostToDevice);

	sumVector << < numBlocks, numThreads >> >(
		d_a,
		d_b);

	cudaMemcpy(h_b, d_b, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);

	int d_S = 0;
	
	for (int i = 0; i < numBlocks; i++)
	{
		d_S += h_b[i];
	}
	
	std::cout << S << "," << d_S << std::endl;

	delete[] h_a;
	delete[] h_b;

	cudaFree(d_a); cudaFree(d_b);

    return 0;
}
