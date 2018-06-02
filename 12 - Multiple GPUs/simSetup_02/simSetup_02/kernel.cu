
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "curand_kernel.h"
#include "curand.h"

#include <stdio.h>

#include <iostream>

const int numThreads = 512;

typedef struct SimPlan
{
	int device;
	int dataSize;
	int numBlocks;
	cudaStream_t streamID;
	int timeLimit;
	int t = 0;
	float dt;
	unsigned int seed;

	float **h_V;
	float mu;
	float sigma;

	float *d_v0;
	float *d_v1;

	curandState_t *PRNG;

	bool needDestruct = false;

	~SimPlan()
	{
		if (needDestruct)
		{
			cudaFree(this->d_v0);
			cudaFree(this->d_v1);

			for (int i = 0; i < this->timeLimit; i++)
			{
				delete[] this->h_V[i];
			}
			delete[] this->h_V;

		}
	}
};



void setData(SimPlan *plan)
{
	
	plan->h_V = new float*[plan->timeLimit];

	for (int t = 0; t < plan->timeLimit; t++)
	{
		plan->h_V[t] = new float[plan->dataSize];
	}

	for (int i = 0; i < plan->dataSize; i++)
	{
		plan->h_V[0][i] = 100.0f;
	}

	cudaMemcpyAsync(plan->d_v0, plan->h_V[0], plan->dataSize*sizeof(float), cudaMemcpyHostToDevice, plan->streamID);

}


__global__ void initializePRNG(SimPlan *plan)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= plan->dataSize){ return; }
	curand_init(plan->seed, i, 0, &plan->PRNG[i]);
}


void initializePlan(
	SimPlan *plan,
	int device_,
	int n_,
	float mu_,
	float sigma_,
	int timeLimit_,
	cudaStream_t streamID_ = 0)
{
	plan->needDestruct = true;
	plan->device = device_;
	plan->dataSize = n_;
	plan->numBlocks = (plan->dataSize + numThreads - 1) / numThreads;
	plan->mu = mu_;
	plan->sigma = sigma_;
	plan->timeLimit = timeLimit_;
	plan->dt = 1.0f / float(timeLimit_);

	initializePRNG <<<plan->numBlocks, numThreads >>>(plan);

	if (streamID_ != 0)
	{
		plan->streamID = streamID_;
	}

	cudaMalloc((void**)&plan->d_v0, plan->dataSize*sizeof(float));
	cudaMalloc((void**)&plan->d_v1, plan->dataSize*sizeof(float));

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
	if (plan->t % 2 == 0)
	{
		plan->d_v1[i] = plan->d_v0[i] * (plan->mu * plan->dt + sqrtf(plan->dt) * plan->sigma * curand_normal(&plan->PRNG[i]));
	}
	else
	{
		plan->d_v0[i] = plan->d_v1[i] * (plan->mu * plan->dt + sqrtf(plan->dt) * plan->sigma * curand_normal(&plan->PRNG[i]));
	}
}


void runSim(SimPlan *plan)
{
	while (plan->t < plan->timeLimit - 1)
	{
		std::cout << plan->t << std::endl;
		Kernel <<< plan->numBlocks, numThreads >>>(plan);
		if (plan->t % 2 == 0)
		{
			cudaMemcpy(plan->h_V[plan->t + 1], plan->d_v1, plan->dataSize*sizeof(float), cudaMemcpyDeviceToHost);
		}
		else
		{
			cudaMemcpy(plan->h_V[plan->t + 1], plan->d_v0, plan->dataSize*sizeof(float), cudaMemcpyDeviceToHost);
		}
		plan->t += 1;
	}
}

int main()
{
	SimPlan p1;

	cudaStream_t s1;

	cudaStreamCreate(&s1);

	initializePlan(
		&p1,
		0,
		1000,
		0.05,
		0.2,
		252,
		s1);

	runSim(&p1);

	return 0;
}
