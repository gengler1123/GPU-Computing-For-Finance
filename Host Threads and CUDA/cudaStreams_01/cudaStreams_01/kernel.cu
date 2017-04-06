
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


unsigned int N = 100000000;
unsigned int numThreads = 512;


__global__ void addArray(
	float *d_a,
	float *d_b,
	float *d_c,
	int size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= size)
	{
		return;
	}
	d_c[i] = d_a[i] + d_b[i];
}



int main()
{
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1); cudaStreamCreate(&stream2); cudaStreamCreate(&stream3);

	cudaError_t cudaStatus;

	float *h_a, *d_a;
	float *h_b, *d_b;
	float *h_c, *d_c;
	float *h_d, *d_d;
	float *h_e, *d_e;

	cudaStatus = cudaMallocHost(&h_a, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "1\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_a, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMallocHost(&h_b, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_b, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "4\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMallocHost(&h_c, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "5\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_c, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "6\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMallocHost(&h_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "7\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "8\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMallocHost(&h_e, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "9\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_e, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "10\n", cudaStatus);
		goto Error;
	}


	for (int i = 0; i < N; i++)
	{
		h_a[i] = float(i)/10000.0f;
		h_b[i] = float(i) / 10000.0f;
		h_b[i] = float(i) / 10000.0f;
	}


	cudaStatus = cudaMemcpyAsync(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice, stream1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "11\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpyAsync(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice, stream2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "12\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpyAsync(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice, stream3);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "13\n", cudaStatus);
		goto Error;
	}


	addArray << < (N + numThreads - 1) / numThreads, numThreads, 0, stream1 >> >(d_a, d_a, d_d, N);
	addArray << < (N + numThreads - 1) / numThreads, numThreads, 0, stream2 >> >(d_b, d_c, d_e, N);


	cudaStatus = cudaMemcpyAsync(h_d, d_d, N*sizeof(float), cudaMemcpyDeviceToHost, stream1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "14\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpyAsync(h_e, d_e, N*sizeof(float), cudaMemcpyDeviceToHost, stream2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "15\n", cudaStatus);
		goto Error;
	}

Error:

	cudaDeviceReset();

    return 0;
}
