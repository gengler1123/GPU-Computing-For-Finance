
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <iostream>

#define numThreads 512

const int N = pow(2, 20);

__global__ void stencil01(
	float *d_a,
	float *d_b,
	int size = N)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i == 0)
	{
		d_b[i] = d_a[i] + d_a[i + 1];
		d_b[i] /= 3.0f;
	}
	else if(i < size - 1)
	{
		d_b[i] = d_a[i - 1] + d_a[i] + d_a[i + 1];
		d_b[i] /= 3.0f;
	}
	else if (i == size - 1)
	{
		d_b[i] = d_a[i - 1] + d_a[i];
		d_b[i] /= 3.0f;
	}
}

__global__ void stencil02(
	float *d_a,
	int size = N)
{
	__shared__ float cache[numThreads + 2];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= size)
	{
		return;
	}

	int cid = threadIdx.x ;
	int c = cid + 1;
	if (tid == 0)
	{
		cache[0] = 0;
		cache[c] = d_a[tid];
	}
	else if (cid < numThreads - 1) // What is Missing?
	{
		cache[c] = d_a[tid];
	}
	else if (cid == numThreads - 1) // What is Missing?
	{
		cache[c] = d_a[tid];
		cache[c + 1] = d_a[tid + 1];
	}

	__syncthreads();

	d_a[tid] = cache[c - 1] + cache[c] + cache[c + 1];
	d_a[tid] /= 3.0f;
}


__device__ float func(float *a, int c)
{
	return (a[c - 1] + a[c] + a[c + 1])/3.0f;
}


__global__ void stencil03(
	float *d_a,
	int size = N)
{
	__shared__ float cache[numThreads + 2];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= size)
	{
		return;
	}

	int cid = threadIdx.x;
	int c = cid + 1;
	if (tid == 0)
	{
		cache[0] = 0;
		cache[c] = d_a[tid];
	}
	else if (cid < numThreads - 1) // What is Missing?
	{
		cache[c] = d_a[tid];
	}
	else if (cid == numThreads - 1) // What is Missing?
	{
		cache[c] = d_a[tid];
		cache[c + 1] = d_a[tid + 1];
	}

	__syncthreads();

	d_a[tid] = func(cache, c);

}



int main()
{
	float *h_a1 = new float[N];
	float *h_a2 = new float[N];
	float *h_a3 = new float[N];

	for (int i = 0; i < N; i++)
	{
		if (i % 3 == 0)
		{
			h_a1[i] = 1.0f;
			h_a2[i] = 1.0f;
			h_a3[i] = 1.0f;
		}
		else if (i % 3 == 1)
		{
			h_a1[i] = 2.0f;
			h_a2[i] = 2.0f;
			h_a3[i] = 2.0f;
		}
		else
		{
			h_a1[i] = 3.0f;
			h_a2[i] = 3.0f;
			h_a3[i] = 3.0f;
		}
	}

	float *d_a1, *d_a2, *d_a3, *d_a11;

	cudaMalloc((void**)&d_a1, N*sizeof(float));
	cudaMalloc((void**)&d_a2, N*sizeof(float));
	cudaMalloc((void**)&d_a3, N*sizeof(float));
	cudaMalloc((void**)&d_a11, N*sizeof(float));

	cudaMemcpy(d_a1, h_a1, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a2, h_a1, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a3, h_a1, N*sizeof(float), cudaMemcpyHostToDevice);

	stencil01 << < (N + numThreads - 1) / numThreads, numThreads >> >(d_a1, d_a11);
	stencil02 << < (N + numThreads - 1) / numThreads, numThreads >> >(d_a2);
	stencil03 << < (N + numThreads - 1) / numThreads, numThreads >> >(d_a3);

	cudaMemcpy(h_a1, d_a11, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_a2, d_a2, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_a3, d_a3, N*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		if (h_a1[i] != h_a2[i])
		{
			std::cout << i << "," << i%numThreads << ": " << h_a1[i] << "," << h_a2[i] << "," << h_a3[i] << std::endl;
		}
	}



	cudaFree(d_a1); cudaFree(d_a2); cudaFree(d_a3); cudaFree(d_a11);

	delete[] h_a1; delete[] h_a2; delete[] h_a3;

    return 0;
}
