
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>

const int numThreads = 512;

typedef struct SimPlan
{
	int device;
	int dataSize;
	int numBlocks;
	cudaStream_t streamID;

	float *h_a;
	float *h_b;
	float *h_c;

	float *d_a;
	float *d_b;
	float *d_c;

	bool needDestruct = false;

	~SimPlan()
	{
		if (needDestruct)
		{
			cudaFree(this->d_a);
			cudaFree(this->d_b);
			cudaFree(this->d_c);

			delete[] h_a;
			delete[] h_b;
			delete[] h_c;

		}
	}
};



void setData(SimPlan *plan)
{
	plan->h_a = new float[plan->dataSize];
	plan->h_b = new float[plan->dataSize];
	plan->h_c = new float[plan->dataSize];
	for (int i = 0; i < plan->dataSize; i++)
	{
		plan->h_a[i] = float(i / 10);
		plan->h_b[i] = float(i / 20);
	}

	cudaMemcpy(plan->d_a, plan->h_a, plan->dataSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(plan->d_b, plan->h_b, plan->dataSize*sizeof(float), cudaMemcpyHostToDevice);

}


void initializePlan(
	SimPlan *plan,
	int device_,
	int n_,
	cudaStream_t streamID_ = 0)
{
	plan->needDestruct = true;
	plan->device = device_;
	plan->dataSize = n_;
	plan->numBlocks = (plan->dataSize + numThreads - 1) / numThreads;

	if (streamID_ != 0)
	{
		plan->streamID = streamID_;
	}

	cudaMalloc((void**)&plan->d_a, plan->dataSize*sizeof(float));
	cudaMalloc((void**)&plan->d_b, plan->dataSize*sizeof(float));
	cudaMalloc((void**)&plan->d_c, plan->dataSize*sizeof(float));

	setData(plan);
}


__global__ void Kernel(
	SimPlan *plan)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= plan->dataSize)
	{
		return;
	}
	plan->d_c[i] = plan->d_a[i] + plan->d_b[i];
}


int main()
{
	SimPlan p1;

	cudaStream_t s1;

	cudaStreamCreate(&s1);

	initializePlan(
		&p1,
		0,
		10000,
		s1);

	Kernel << < p1.numBlocks, numThreads, 0, p1.streamID >> >(&p1);

	return 0;
}
