#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "neuronmodel.cuh"
#include "edges.cuh"

__global__ void Timestep(
	NeuronModel *d_nm,
	unsigned int size,
	bool *d_cf)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= size)
		return;

	d_nm[i].timestep();
	if (d_nm[i].fired)
	{
		d_cf[i] = true;
		d_nm[i].fired = false;
	}
	else
	{
		d_cf[i] = false;
	}
	d_nm[i].resetInput();

}


__global__ void DelayedInput(
	int size, int maxDelay, int t,
	NeuronModel *d_nm,
	float *d_di, float*d_inputs)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= size)
	{
		return;
	}
	int idx = maxDelay * i + t;
	d_nm[i].updateInput(d_di[idx]);
	d_nm[i].updateInput(d_inputs[i]);
	d_di[idx] = 0.0f;
}


__global__ void UpdateInputs(
	int size, int t, int maxDelay,
	Edge *d_ed,
	bool *d_cf,
	float *d_di)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= size)
	{
		return;
	}
	int s = d_ed[i].source;
	if (d_cf[s])
	{
		int tar = d_ed[i].target;
		int m = d_ed[i].delay;
		float w = d_ed[i].weight;
		d_di[maxDelay * tar + ((t + m) % maxDelay)] += w;
	}
}



#endif