
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <vector>
#include <ctime>
#include <iostream>

class stochasticNeuron
{
private:
	float t, dt, lastFire, p;

	float rate, maxRate;

	float c_x, c_y, tau;

	__device__ float genExpo(float p_, float lambda)
	{
		return -(1 / lambda)*(log( 1 - p_));
	};

	__device__ float genRate(float x_, float y_)
	{
		rate = maxRate * expf(- distSquare(x_,y_)*tau);
	};

	__device__ float distSquare(float x_, float y_)
	{
		return (c_x - x_)*(c_x - x_) + (c_y - y_) * (c_y - y_);
	};

public:

	int cnt;

	__host__ stochasticNeuron(
		float maxRate_,
		float x_, float y_,
		float dt_ = 1.0f,
		float tau_ = 100.0f)
	{
		t = 0.0f;
		dt = dt_;
		lastFire = -1.0f;
		maxRate = maxRate_;
		c_x = x_;
		c_y = y_;
		tau = tau_;
	};

	__device__ void genFired(float x_, float y_, curandState_t &state)
	{
		t += dt;
		cnt = 0;
		genRate(x_, y_);
		while (lastFire < t)
		{
			cnt++;
			p = curand_uniform(&state);
			lastFire += genExpo(p, rate);
		}
	};
};

__global__ void curandINIT(
	int size,
	curandState_t *d_states,
	unsigned int seed)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		curand_init(seed, i, 0, &d_states[i]);
	}
}


__global__ void runSim(
	int size,
	curandState_t *d_states,
	stochasticNeuron *d_sN,
	float x_, float y_,
	int *d_count)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_sN[i].genFired(x_,y_, d_states[i]);
		d_count[i] = d_sN[i].cnt;
	}
}


int main()
{
	float dX = 0.1;
	float x, y;
	std::vector<stochasticNeuron> Neurons;

	for (int i = 0; i < 11; i++)
	{
		x = float(i)*dX;
		for (int j = 0; j < 11; j++)
		{
			
			y = float(j)*dX;
			stochasticNeuron sN(15.0f,x, y);

			Neurons.push_back(sN);

		}

	}

	/*  */
	stochasticNeuron *d_sN;
	curandState_t *d_states;
	int *d_cnt;

	int numNeurons = Neurons.size();
	int numThreads = 512;
	int runTime = 100;
	
	int **spikeCounts = new int*[runTime];
	for (int i = 0; i < runTime; i++)
	{
		spikeCounts[i] = new int[numNeurons];
	}

	cudaMalloc((void**)&d_sN, numNeurons * sizeof(stochasticNeuron));
	cudaMalloc((void**)&d_states, numNeurons * sizeof(curandState_t));
	cudaMalloc((void**)&d_cnt, numNeurons * sizeof(int));

	cudaMemcpy(d_sN, Neurons.data(), numNeurons * sizeof(stochasticNeuron), cudaMemcpyHostToDevice);

	curandINIT<<<(numNeurons + numThreads - 1)/numThreads, numThreads>>>(numNeurons, d_states, time(NULL));

	for (int t = 0; t < runTime; t++)
	{
		runSim << <(numNeurons + numThreads - 1)/numThreads, numThreads >> >(
			numNeurons,
			d_states,
			d_sN,
			0.0,0.5,
			d_cnt);


		cudaMemcpy(spikeCounts[t], d_cnt, numNeurons*sizeof(int), cudaMemcpyDeviceToHost);
	}

	std::vector<float> avg(11);

	for (int t = 0; t < runTime; t++)
	{
		for (int i = 0; i < 11; i++)
		{
			avg[i] += spikeCounts[t][i];
		}
	}


	for (int i = 0; i < avg.size(); i++)
	{
		avg[i] /= float(runTime);
		std::cout << avg[i] << " ";
	}
	std::cout << std::endl;

	for (int t = 0; t < runTime; t++)
	{
		delete[] spikeCounts[t];
	}
	delete[] spikeCounts;


	cudaFree(d_cnt); cudaFree(d_sN); cudaFree(d_states);


	cudaDeviceReset();


    return 0;
}
