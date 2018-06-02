
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define numThreads 512


__device__ float explicitLocalStepHeat(
	float unjpo,
	float unjmo,
	float unj,
	float r)
{
	return (1 - 2 * r)*unj + r*unjmo + r * unjpo;
}


__global__ void explicitTimestepHeat(
	int size,
	float *d_currentVal,
	float *d_nextVal,
	float r
	)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		if (i < 2)
		{
			d_nextVal[i] == 0;
		}
		else if (i > size - 2)
		{
			d_nextVal[i] == 0;
		}
		else
		{
			d_nextVal[i] = explicitLocalStepHeat(
				d_currentVal[i + 1],
				d_currentVal[i - 1],
				d_currentVal[i],
				r);
		}
	}
}


int main()
{

	const unsigned int numSteps = pow(2, 4) + 3;
	float dx = 1 / float(numSteps-3);
	float dt = 1 / pow(2, 5);


	float *h_current, *h_next, *d_current, *d_next;

	h_current = new float[numSteps];
	h_next = new float[numSteps];

	std::cout << dx << std::endl;
	std::cout << dt << std::endl;

	std::cout << std::endl;

	for (int i = 0; i < numSteps; i++)
	{
		h_next[i] = 0;
		if (0 < i < numSteps -1)
		{
			float x = float(i - 1) * dx;
			h_current[i] = x*(1 - x);
			
		}
		if (i == 0 || i == numSteps-1)
		{
			h_current[i] = 0;
		}

		std::cout << h_current[i] << std::endl;
	}

	std::cout << std::endl;

	cudaMalloc((void**)&d_current, (numSteps) * sizeof(float));
	cudaMalloc((void**)&d_next, (numSteps)*sizeof(float));

	cudaMemcpy(d_current, h_current, (numSteps) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_next, h_next, (numSteps)*sizeof(float), cudaMemcpyHostToDevice);

	int T = 10 / dt;

	float r = dt / dx;

	float **h_Sim = new float*[T];
	for (int t = 0; t < T; t++)
	{
		h_Sim[t] = new float[numSteps];
	}

	for (int t = 0; t < T; t++)
	{
		if (t % 2 == 0)
		{
			explicitTimestepHeat << < (numSteps + numThreads - 1)/numThreads, numThreads >> >(
				numSteps + 3,
				d_current,
				d_next,
				r);
			cudaMemcpy(h_Sim[t], d_current, (numSteps)*sizeof(float), cudaMemcpyDeviceToHost);
		}
		else
		{
			explicitTimestepHeat << < (numSteps + numThreads - 1)/numThreads, numThreads >> >(
				numSteps + 3,
				d_next,
				d_current,
				r);
			cudaMemcpy(h_Sim[t], d_next, (numSteps)*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}


	for (int i = 0; i < T; i++)
	{
		std::cout << h_Sim[i][15] << std::endl;
	}

	for (int i = 0; i < T; i++)
	{
		delete[] h_Sim[i];
	}
	delete[] h_Sim;


	delete[] h_current;
	delete[] h_next;

	cudaFree(d_current);
	cudaFree(d_next);

	cudaDeviceReset();


	return 0;
}